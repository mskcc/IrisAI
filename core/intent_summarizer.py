"""Intent-aware tool output summarization.

When a tool output exceeds the observation threshold (12K chars), this module
uses Haiku to extract only the information relevant to the agent's current intent.
Falls back to deterministic head+tail truncation on any failure.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

MAX_HAIKU_INPUT = 400_000
CHUNK_SIZE = 350_000
CHUNK_OVERLAP = 5_000


def build_intent_string(
    thinking: Optional[str], tool_name: str, tool_input: Optional[dict]
) -> str:
    """Build a concise intent description from available context.

    Priority: thinking > tool_input > tool_name (fallback).
    """
    parts = []
    if thinking:
        # Last 500 chars of thinking captures the most recent reasoning
        snippet = thinking[-500:]
        parts.append(f"Agent's reasoning: {snippet}")
    if tool_input:
        input_str = json.dumps(tool_input, default=str)[:300]
        parts.append(f"Tool was called with: {input_str}")
    parts.append(f"Tool: {tool_name}")
    return "\n".join(parts)


def build_extraction_prompt(
    intent: str, content: str, target_chars: int
) -> str:
    """Build the Haiku prompt for extracting relevant information."""
    return (
        "You are extracting specific information from a large tool output.\n\n"
        f"CONTEXT — what the agent is looking for:\n{intent}\n\n"
        f"TOOL OUTPUT ({len(content):,} chars):\n{content}\n\n"
        "INSTRUCTIONS:\n"
        "- Extract ONLY the information relevant to what the agent is looking for\n"
        "- Preserve exact values (numbers, paths, names, versions) — never paraphrase data\n"
        "- If the output contains structured data (JSON, tables), keep the relevant rows/fields verbatim\n"
        "- If nothing relevant is found, say \"No relevant information found\"\n"
        f"- Keep your response under {target_chars} characters\n"
        "- Do NOT add commentary — just the extracted information"
    )


def split_with_overlap(content: str, chunk_size: int, overlap: int) -> list[str]:
    """Split content into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        chunks.append(content[start:end])
        start = end - overlap
        if start >= len(content):
            break
    return chunks


async def summarize_with_intent(
    content: str, intent: str, max_output: int = 11500, timeout: float = 30
) -> str:
    """Summarize tool output using Haiku, guided by agent intent.

    For outputs <= 400K: single Haiku pass.
    For outputs > 400K: chunked processing with merge.

    Raises on failure (caller handles fallback).
    """
    from core.sub_agent import _call_sub_agent_llm

    if len(content) <= MAX_HAIKU_INPUT:
        prompt = build_extraction_prompt(intent, content, max_output)
        result = await _call_sub_agent_llm(prompt, timeout=timeout)
        if result.startswith("[Error:"):
            raise RuntimeError(result)
        return result[:max_output]

    # Chunked processing for very large outputs
    chunks = split_with_overlap(content, CHUNK_SIZE, CHUNK_OVERLAP)
    per_chunk_budget = max(max_output // len(chunks), 2000)

    logger.info(
        f"[INTENT_SUMMARIZER] Chunked processing: {len(content):,} chars → "
        f"{len(chunks)} chunks of ~{CHUNK_SIZE:,}"
    )

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = build_extraction_prompt(intent, chunk, per_chunk_budget)
        summary = await _call_sub_agent_llm(prompt, timeout=timeout)
        if summary.startswith("[Error:"):
            raise RuntimeError(f"Chunk {i+1}/{len(chunks)} failed: {summary}")
        chunk_summaries.append(summary)

    combined = "\n---\n".join(chunk_summaries)

    if len(combined) <= max_output:
        return combined

    # Final merge pass
    merge_prompt = (
        f"Merge these {len(chunks)} extractions into one concise result.\n"
        f"Keep under {max_output} characters. Preserve exact values.\n\n"
        f"{combined}"
    )
    merged = await _call_sub_agent_llm(merge_prompt, timeout=timeout)
    if merged.startswith("[Error:"):
        raise RuntimeError(f"Merge failed: {merged}")
    return merged[:max_output]


def deterministic_truncate(
    observation: str, max_chars: int, archive_path: Optional[str] = None
) -> str:
    """Head(65%) + tail(30%) truncation with gap indicator."""
    head_size = int(max_chars * 0.65)
    tail_size = int(max_chars * 0.30)

    result = observation[:head_size]
    omitted = len(observation) - head_size - tail_size
    result += f"\n\n[... {omitted:,} chars omitted ...]\n\n"
    result += observation[-tail_size:]

    if archive_path:
        result += f"\n[Full content: {archive_path}]"
    return result
