"""
core/context_compactor.py — Intelligent Context Compaction

Replaces blind text[:N] truncation with LLM-aware compression that preserves
all facts, paths, values, and context. Uses Haiku for intelligent summarization
when text exceeds budget, with deterministic head+tail fallback on failure.

Usage:
    from core.context_compactor import smart_compact, async_smart_compact

    # Async (preferred in async contexts)
    result = await async_smart_compact(large_text, max_chars=12000, context_type="tool_output")

    # Sync (for synchronous functions like pipeline _run)
    result = smart_compact(large_text, max_chars=20000, context_type="pipeline_output")
"""

import asyncio
import logging
import time
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PRE_TRIM_MULTIPLIER = 3
HEAD_RATIO = 0.65
TAIL_RATIO = 0.35


# ── Metrics ──────────────────────────────────────────────────────────────────

class CompactionMetrics:
    """Thread-safe metrics for compaction operations."""
    _lock = threading.Lock()
    _total_calls = 0
    _haiku_calls = 0
    _fallback_calls = 0
    _passthrough_calls = 0
    _chars_saved = 0
    _total_latency_ms = 0.0

    @classmethod
    def record(cls, original_chars: int, result_chars: int, method: str, latency_ms: float):
        with cls._lock:
            cls._total_calls += 1
            if method == "haiku":
                cls._haiku_calls += 1
            elif method == "fallback":
                cls._fallback_calls += 1
            else:
                cls._passthrough_calls += 1
            cls._chars_saved += (original_chars - result_chars)
            cls._total_latency_ms += latency_ms

    @classmethod
    def summary(cls) -> dict:
        with cls._lock:
            return {
                "total_calls": cls._total_calls,
                "haiku_calls": cls._haiku_calls,
                "fallback_calls": cls._fallback_calls,
                "passthrough_calls": cls._passthrough_calls,
                "chars_saved": cls._chars_saved,
                "avg_latency_ms": round(cls._total_latency_ms / max(cls._total_calls, 1), 1),
            }

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._total_calls = 0
            cls._haiku_calls = 0
            cls._fallback_calls = 0
            cls._passthrough_calls = 0
            cls._chars_saved = 0
            cls._total_latency_ms = 0.0


# ── Deterministic Fallback ───────────────────────────────────────────────────

def _fallback_truncate(
    text: str,
    max_chars: int,
    head_ratio: float = HEAD_RATIO,
    tool_name: str = "",
) -> str:
    """Deterministic head+tail truncation — used when Haiku is unavailable.

    Preserves the beginning (context, setup) and end (results, conclusions)
    of the text with a clear marker showing what was cut.

    Args:
        text: Text to truncate.
        max_chars: Target character budget.
        head_ratio: Proportion allocated to head (default 0.65).
        tool_name: Source tool name for the metadata marker.

    Returns:
        Truncated text guaranteed <= max_chars (approximately).
    """
    if not text or len(text) <= max_chars:
        return text

    original_len = len(text)
    # Scale marker budget based on available space
    marker_budget = min(150, max_chars // 3)
    usable = max(max_chars - marker_budget, max_chars // 2)
    head_size = int(usable * head_ratio)
    tail_size = usable - head_size

    source = f"Tool: {tool_name}\n" if tool_name else ""
    marker = (
        f"\n[COMPACTED {original_len:,}→{max_chars:,} chars, "
        f"{source}middle removed]\n"
    )

    if tail_size > 0:
        return text[:head_size] + marker + text[-tail_size:]
    return text[:head_size] + marker


# ── Compression Prompts ──────────────────────────────────────────────────────

def _build_compaction_prompt(
    text: str,
    context_type: str,
    tool_name: str = "",
    tool_input_summary: str = "",
) -> str:
    """Build context-type-specific compression prompt for Haiku."""

    base_rules = """\
CRITICAL — PRESERVE ALL OF THESE EXACTLY:
1. ALL file paths, directory paths, URLs (verbatim, no shortening)
2. ALL error messages, warnings, failure reasons (exact text)
3. ALL numeric values: counts, sizes, line numbers, percentages, versions
4. ALL names: function/class/variable names, test names, package names
5. ALL status indicators: pass/fail, success/error, exit codes
6. ALL IDs, hashes, timestamps, job IDs
7. The overall outcome/conclusion

NEVER round numbers. Report 267 as 267, not ~265.
NEVER omit file paths — they are critical for the downstream agent.

You MAY remove:
- Repetitive lines (e.g., 100 lines of "test PASSED" → "100 tests passed")
- Progress indicators, loading spinners, redundant whitespace
- Decorative formatting that adds no information"""

    if context_type == "web_content":
        specific_rules = """
ADDITIONAL RULES FOR WEB CONTENT:
- Preserve document structure (headings, sections)
- Keep ALL code blocks and their language tags intact
- Preserve URLs and hyperlink targets
- Keep lists and their hierarchy
- Remove navigation elements, footers, ads, cookie notices"""
    elif context_type == "pipeline_output":
        specific_rules = """
ADDITIONAL RULES FOR PIPELINE OUTPUT:
- Preserve the execution flow (what ran in what order)
- Keep ALL stdout/stderr content that shows results or errors
- Preserve exit codes and success/failure status per stage
- Keep file paths of any created/modified artifacts
- Preserve timing information if present"""
    elif context_type == "agent_handoff":
        specific_rules = """
ADDITIONAL RULES FOR AGENT HANDOFF:
- Preserve ALL tool call results and their conclusions
- Keep the agent's final assessment/recommendation
- Preserve file paths discovered or modified
- Keep error states and what was tried
- Preserve any "next steps" or pending work items"""
    elif context_type == "conversation_context":
        specific_rules = """
ADDITIONAL RULES FOR CONVERSATION CONTEXT:
- Preserve ALL file paths and tool output locations
- Preserve ALL results, outcomes, and status (pass/fail, counts)
- Preserve ALL decisions made and their reasoning
- Preserve what was accomplished vs what remains
- Keep key tool names and what they returned
- Remove verbose formatting, decorative tables, repeated content"""
    else:
        specific_rules = ""

    source_info = ""
    if tool_name:
        source_info += f"\nSource tool: {tool_name}"
    if tool_input_summary:
        source_info += f"\nTool input: {tool_input_summary}"

    return f"""\
You are a context compression specialist. Compress the following text while \
preserving ALL factual information. The compressed version will be used by an \
AI agent that needs every fact, path, and value to do its work correctly.

{base_rules}
{specific_rules}
{source_info}

Output ONLY the compressed text — no preamble, no "Here is the compressed version".

TEXT TO COMPRESS ({len(text):,} chars):
{text}"""


# ── Core Async Compaction ────────────────────────────────────────────────────

async def async_smart_compact(
    text: str,
    max_chars: int,
    context_type: str = "tool_output",
    tool_name: str = "",
    tool_input_summary: str = "",
) -> str:
    """Intelligently compact text using Haiku LLM, preserving all facts.

    Logic:
        1. text <= max_chars → return as-is (zero cost, zero latency)
        2. text > 3× max_chars → pre-trim to 3×, then Haiku-compress
        3. text > max_chars but ≤ 3× → Haiku-compress the full text
        4. Haiku failure → deterministic head+tail fallback

    Args:
        text: Raw text to compact.
        max_chars: Target character budget for output.
        context_type: One of "tool_output", "web_content", "pipeline_output",
                      "agent_handoff". Adjusts the compression prompt.
        tool_name: Name of source tool (for logging and prompt context).
        tool_input_summary: Brief description of what produced this text.

    Returns:
        Compacted text, guaranteed approximately <= max_chars.
    """
    if not text or len(text) <= max_chars:
        CompactionMetrics.record(len(text or ""), len(text or ""), "passthrough", 0)
        return text

    original_len = len(text)
    start_time = time.time()

    # Pre-trim if massively over budget (>3× target)
    text_for_haiku = text
    if len(text) > max_chars * PRE_TRIM_MULTIPLIER:
        pre_trim_budget = max_chars * PRE_TRIM_MULTIPLIER
        head_size = int(pre_trim_budget * HEAD_RATIO)
        tail_size = pre_trim_budget - head_size - 100
        text_for_haiku = (
            text[:head_size]
            + f"\n[...{original_len - head_size - tail_size:,} chars omitted...]\n"
            + text[-tail_size:]
        )
        logger.info(
            f"[COMPACT] Pre-trimmed {tool_name or context_type}: "
            f"{original_len:,} → {len(text_for_haiku):,} chars before Haiku"
        )

    # Build prompt and call Haiku
    prompt = _build_compaction_prompt(
        text_for_haiku, context_type, tool_name, tool_input_summary
    )

    try:
        from core.sub_agent import _call_sub_agent_llm
        result = await _call_sub_agent_llm(prompt)
    except Exception as e:
        logger.warning(f"[COMPACT] Haiku call failed ({e}), using fallback")
        fallback = _fallback_truncate(text, max_chars, tool_name=tool_name)
        latency_ms = (time.time() - start_time) * 1000
        CompactionMetrics.record(original_len, len(fallback), "fallback", latency_ms)
        return fallback

    # Check for error responses from _call_sub_agent_llm
    if not result or result.startswith("[Error:"):
        logger.warning(f"[COMPACT] Haiku returned error: {result[:100] if result else 'empty'}")
        fallback = _fallback_truncate(text, max_chars, tool_name=tool_name)
        latency_ms = (time.time() - start_time) * 1000
        CompactionMetrics.record(original_len, len(fallback), "fallback", latency_ms)
        return fallback

    # If Haiku was too verbose, apply soft trim to its output
    if len(result) > max_chars:
        result = result[:max_chars]

    latency_ms = (time.time() - start_time) * 1000
    CompactionMetrics.record(original_len, len(result), "haiku", latency_ms)

    logger.info(
        f"[COMPACT] {context_type}/{tool_name}: "
        f"{original_len:,} → {len(result):,} chars "
        f"({len(result)/original_len*100:.0f}% kept, {latency_ms:.0f}ms)"
    )

    return result


# ── Sync Wrapper ─────────────────────────────────────────────────────────────

def smart_compact(
    text: str,
    max_chars: int,
    context_type: str = "tool_output",
    tool_name: str = "",
    tool_input_summary: str = "",
) -> str:
    """Synchronous wrapper for async_smart_compact.

    For use in sync contexts (e.g., pipeline tool _run method).
    If no event loop is running, uses asyncio.run().
    If already in an async context, falls back to deterministic truncation
    (the caller should use async_smart_compact directly instead).
    """
    if not text or len(text) <= max_chars:
        return text

    try:
        asyncio.get_running_loop()
        # Already in async context — can't nest asyncio.run()
        # Fall back to deterministic truncation
        logger.debug("[COMPACT] Sync wrapper in async context, using fallback")
        return _fallback_truncate(text, max_chars, tool_name=tool_name)
    except RuntimeError:
        # No running loop — safe to use asyncio.run()
        return asyncio.run(
            async_smart_compact(text, max_chars, context_type, tool_name, tool_input_summary)
        )
