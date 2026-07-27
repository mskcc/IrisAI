"""Sub-agent tools — isolated LLM calls for expensive multi-file operations.

Phase 3 implementation: Sub-agents for context isolation.
These tools internally make focused LLM calls with their own context,
do the expensive work (reading multiple files, analyzing output), and
return only a structured summary to the main agent.

This prevents file contents from accumulating in the main agent's context
window, saving significant token costs on subsequent iterations.

Architecture:
    - Each tool is a BaseTool subclass with kwargs-unwrapping logic
    - Internally reads files using standard Python (os, pathlib)
    - Makes a focused LLM call via ChatOpenAI (same LiteLLM proxy)
    - Returns a structured summary string
    - Main agent gets ~500 tokens of summary instead of ~10K of raw content

The tools use BaseTool (not @tool decorator) to handle the "kwargs wrapping"
pattern where the LLM sends {'kwargs': {'param': 'value'}} instead of
{'param': 'value'}. This matches the unwrapping logic in MCPTool (app.py).

No external dependencies beyond:
    - langchain_core.tools (for BaseTool — lazy import)
    - os, pathlib (for file reading)
    - aiohttp or langchain_openai (for LLM calls — lazy import)

These tools are registered in the single agent's tool pool and
available when relevant skills are active.
"""
import asyncio
import json
import os
import logging
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger("core.sub_agent")


# ── Configuration ───────────────────────────────────────────────────────

# LiteLLM proxy URL (same as main agent)
LITELLM_URL = os.environ.get("LITELLM_URL", "http://localhost:8080")

# Model for sub-agent calls — use a cheaper model for cost savings.
# Default to Haiku 3.5 (the model name registered in LiteLLM proxy).
# IMPORTANT: This must match a model allowed by the LiteLLM virtual key.
# The allowed models are configured in LiteLLM proxy — check there if
# you get 401 "key not allowed to access model" errors.
SUB_AGENT_MODEL = os.environ.get(
    "SUB_AGENT_MODEL", "anthropic.claude-haiku-4-5-20251001-v1"
)
# Worker agent uses Haiku by default — cost-efficient for tool orchestration
# within the constrained MAX_WORKER_TOOLS=8 tool pool.
# Override with WORKER_AGENT_MODEL env var if a task requires more capability.
WORKER_AGENT_MODEL = os.environ.get(
    "WORKER_AGENT_MODEL", "anthropic.claude-haiku-4-5-20251001-v1"
)

# Opus advisor model — only invoked when worker explicitly escalates.
# Gated by ENABLE_OPUS_ADVISOR env var (default: true).
OPUS_ADVISOR_MODEL = os.environ.get(
    "OPUS_ADVISOR_MODEL", "anthropic.claude-opus-4-6-v1"
)
ENABLE_OPUS_ADVISOR = os.environ.get("ENABLE_OPUS_ADVISOR", "true").lower() == "true"
MAX_ADVISOR_CALLS = int(os.environ.get("MAX_ADVISOR_CALLS", "3"))

# Worker agent constants (exported for testing)
ESSENTIAL_TOOLS = frozenset({
    "execute_dynamic_task", "batch_file_edit", "edit_file",
    "write_text_file", "read_text_file", "run_tests",
})
MAX_WORKER_TOOLS = 8
RESEARCH_ESSENTIAL_TOOLS = frozenset({
    "batch",
    "read_text_file",
    "read_memory", "query_software",
})
WORKER_AGENT_TIMEOUT = 600  # seconds (10 min — generous for multi-tool worker tasks)
WORKER_THINKING_BUDGET = int(os.environ.get("WORKER_THINKING_BUDGET", "1024"))

# Maximum file size to read (bytes) — prevent reading huge files
MAX_FILE_SIZE = 100_000  # 100KB per file

# Maximum total content to send to sub-agent LLM
MAX_TOTAL_CONTENT = 400_000  # ~100K tokens worth of content

# Maximum number of files to process in one call
MAX_FILES = 50


# ── kwargs unwrapping helper ────────────────────────────────────────────

def _unwrap_kwargs(kwargs: Dict[str, Any], tool_name: str = "unknown") -> Dict[str, Any]:
    """Unwrap kwargs that the LLM may have wrapped in various ways.

    The LLM sometimes sends tool arguments as:
        {'kwargs': {'param1': 'value1', 'param2': 'value2'}}
    instead of:
        {'param1': 'value1', 'param2': 'value2'}

    This function handles all known wrapping patterns:
        - Case 1: Normal — no wrapping needed
        - Case 2: {'kwargs': {actual_params}} — unwrap the dict
        - Case 3: {'kwargs': '{"param": "value"}'} — parse JSON string
        - Case 4: Single key with dict value — unwrap

    Args:
        kwargs: The raw kwargs dict from LangChain tool invocation.
        tool_name: Name of the tool (for logging context).

    Returns:
        Unwrapped kwargs dict with actual parameters at top level.
    """
    args = kwargs

    if "kwargs" in kwargs:
        val = kwargs["kwargs"]
        # Case 2: LLM wrapped everything in "kwargs" (as dict)
        if isinstance(val, dict):
            args = val
        # Case 3: LLM wrapped everything in "kwargs" (as JSON string)
        elif isinstance(val, str):
            try:
                parsed = json.loads(val, strict=False)
                if isinstance(parsed, dict):
                    args = parsed
                    logger.info(f"[KWARGS] Unwrapped '{tool_name}' using strict=False")
            except json.JSONDecodeError as e:
                if hasattr(e, 'pos') and e.pos > 2:
                    try:
                        parsed = json.loads(val[:e.pos], strict=False)
                        if isinstance(parsed, dict):
                            args = parsed
                            logger.info(f"[KWARGS] Recovered '{tool_name}' via prefix extraction at pos {e.pos}")
                    except (json.JSONDecodeError, ValueError):
                        pass
                if args is kwargs:
                    logger.warning(
                        f"[KWARGS] Failed to parse kwargs for '{tool_name}': "
                        f"{type(e).__name__}: {e} | preview={val[:200]!r}"
                    )
            except ValueError as e:
                logger.warning(
                    f"[KWARGS] Failed to parse kwargs for '{tool_name}': "
                    f"{type(e).__name__}: {e} | preview={val[:200]!r}"
                )
    # Case 4: LLM passed a single dict as first arg (only if no "kwargs" key)
    elif len(kwargs) == 1 and isinstance(next(iter(kwargs.values())), dict):
        args = next(iter(kwargs.values()))

    # Final safety net: if args still has a "kwargs" wrapper, unwrap it
    if "kwargs" in args and len(args) == 1:
        val = args["kwargs"]
        if isinstance(val, dict):
            args = val
        elif isinstance(val, str):
            try:
                parsed = json.loads(val, strict=False)
                if isinstance(parsed, dict):
                    args = parsed
                    logger.info(f"[KWARGS] Safety-net unwrap '{tool_name}' using strict=False")
            except json.JSONDecodeError as e:
                if hasattr(e, 'pos') and e.pos > 2:
                    try:
                        parsed = json.loads(val[:e.pos], strict=False)
                        if isinstance(parsed, dict):
                            args = parsed
                            logger.info(f"[KWARGS] Safety-net recovered '{tool_name}' via prefix at pos {e.pos}")
                    except (json.JSONDecodeError, ValueError):
                        pass
                if "kwargs" in args and len(args) == 1:
                    logger.warning(
                        f"[KWARGS] Safety-net parse failed for '{tool_name}': "
                        f"{type(e).__name__}: {e} | preview={val[:200]!r}"
                    )
            except ValueError as e:
                logger.warning(
                    f"[KWARGS] Safety-net parse failed for '{tool_name}': "
                    f"{type(e).__name__}: {e} | preview={val[:200]!r}"
                )

    # Strip framework-internal kwargs that callers may inject
    if isinstance(args, dict):
        for _internal_key in ("config", "run_manager", "callbacks"):
            args.pop(_internal_key, None)

    return args


# ── Pure helper functions (testable without LLM) ────────────────────────

def _read_file_safe(path: str, max_size: int = MAX_FILE_SIZE) -> str:
    """Read a file safely with size limits.

    Returns file content or an error message string.
    Never raises — always returns a string.

    Args:
        path: Absolute path to the file.
        max_size: Maximum bytes to read per file.

    Returns:
        File content string, or "[Error: ...]" message.
    """
    try:
        if not os.path.exists(path):
            return f"[Error: File not found: {path}]"
        if not os.path.isfile(path):
            return f"[Error: Not a file: {path}]"

        size = os.path.getsize(path)
        if size > max_size:
            # Read first portion + note about truncation
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_size)
            return (
                f"{content}\n\n[TRUNCATED — file is {size:,} bytes, "
                f"showing first {max_size:,} bytes]"
            )

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except PermissionError:
        return f"[Error: Permission denied: {path}]"
    except UnicodeDecodeError:
        return f"[Error: Binary file, cannot read as text: {path}]"
    except Exception as e:
        return f"[Error reading {path}: {e}]"


def _build_file_contents_block(
    paths: List[str],
    max_total: int = MAX_TOTAL_CONTENT,
) -> str:
    """Read multiple files and build a combined content block.

    Args:
        paths: List of file paths to read.
        max_total: Maximum total characters across all files.

    Returns:
        Combined content block with file headers.
    """
    parts = []
    total_chars = 0

    for path in paths[:MAX_FILES]:
        content = _read_file_safe(path)
        header = f"\n{'='*60}\nFILE: {path}\n{'='*60}\n"

        if total_chars + len(header) + len(content) > max_total:
            remaining = max_total - total_chars - len(header) - 100
            if remaining > 500:
                parts.append(header + content[:remaining] + "\n[TRUNCATED]")
            else:
                parts.append(
                    f"\n[SKIPPED remaining files — content limit reached. "
                    f"Processed {len(parts)} of {len(paths)} files.]"
                )
                break
            total_chars = max_total
            break

        parts.append(header + content)
        total_chars += len(header) + len(content)

    return "\n".join(parts)


def build_file_analysis_prompt(
    file_contents: str,
    question: str,
    file_count: int,
) -> str:
    """Build the prompt for the file analysis sub-agent.

    Pure function — testable without LLM.

    Args:
        file_contents: Combined file contents block.
        question: The user's question about the files.
        file_count: Number of files being analyzed.

    Returns:
        Complete prompt string for the sub-agent LLM call.
    """
    return f"""\
You are a file analysis specialist. You have been given {file_count} file(s) \
to analyze. Read them carefully and answer the question below.

Your response MUST follow this exact format:

Files analyzed: [list the file paths]
Key findings:
- [bullet point 1]
- [bullet point 2]
- [etc.]
Relevant paths: [any specific file paths, line numbers, or locations that matter]
Errors/warnings: [any errors encountered reading files, or "None"]
Answer: [direct, concise answer to the question]

Be specific — include exact paths, line numbers, function names, and values.
Do NOT be vague. If you don't know, say so.

QUESTION: {question}

FILE CONTENTS:
{file_contents}"""


def build_summary_prompt(output: str, focus: str) -> str:
    """Build the prompt for the command output summarization sub-agent.

    Pure function — testable without LLM.

    Args:
        output: The command/test/log output to summarize.
        focus: What aspect to focus the summary on.

    Returns:
        Complete prompt string for the sub-agent LLM call.
    """
    return f"""\
You are an output summarization specialist. Compress the following \
command/test/log output into a concise, structured summary.

Focus on: {focus}

Your response MUST follow this exact format:

Summary: [1-2 sentence overview]
Key results:
- [bullet point 1]
- [bullet point 2]
- [etc.]
Errors/failures: [list any errors, failures, or warnings — or "None"]
Important values: [any specific numbers, paths, IDs, or metrics that matter]
Status: [PASS/FAIL/PARTIAL/UNKNOWN]

Be specific — include exact error messages, line numbers, counts, and values.
Do NOT omit important details.

OUTPUT TO SUMMARIZE:
{output}"""


def build_review_prompt(
    file_contents: str,
    task: str,
    file_count: int,
) -> str:
    """Build the prompt for the codebase review sub-agent.

    Pure function — testable without LLM.

    Args:
        file_contents: Combined file contents block.
        task: The review task description.
        file_count: Number of files being reviewed.

    Returns:
        Complete prompt string for the sub-agent LLM call.
    """
    return f"""\
You are a code review specialist. You have been given {file_count} file(s) \
from a codebase to review. Analyze them according to the task below.

Your response MUST follow this exact format:

Files reviewed: [list the file paths]
Architecture overview: [brief description of how the code is organized]
Key findings:
- [bullet point 1]
- [bullet point 2]
- [etc.]
Issues found:
- [issue 1 with file path and line reference]
- [issue 2]
- [or "None"]
Recommendations:
- [recommendation 1]
- [recommendation 2]
- [or "None"]
Answer: [direct answer to the review task]

Be specific — include exact file paths, function names, line numbers, and code patterns.

REVIEW TASK: {task}

FILE CONTENTS:
{file_contents}"""


def build_escalation_handoff_prompt(raw_handoff: str) -> str:
    """Build the prompt for the escalation handoff summarization sub-agent.

    Pure function — testable without LLM.

    The prompt instructs Haiku to compress the raw handoff text into a
    concise summary that preserves ALL critical information:
    - Every tool that was called and its purpose
    - All file paths, IDs, names, and other concrete values produced
    - What the previous agent concluded or told the user
    - What remains to be done

    The key design principle: the summary must contain enough detail
    that the new agent can continue WITHOUT repeating any work.

    Args:
        raw_handoff: The full raw handoff text from
                     format_intermediate_steps_for_handoff().

    Returns:
        Complete prompt string for the Haiku sub-agent LLM call.
    """
    return f"""\
You are writing a handoff briefing. A colleague just finished part of a task \
and needs you to summarize their work so a fresh colleague can pick up exactly \
where they left off — with new tools they didn't have.

Write the briefing as NATURAL PROSE — like a senior engineer briefing the next \
shift. The reader should finish your briefing knowing:
- What was the goal
- What was tried, what worked, what was concluded
- What concrete artifacts exist now (file paths, job IDs, values)
- What specifically remains and WHY the new tools are needed for it

FORMAT — use these sections in this order:

## Situation
One sentence: what the user asked for and what skills were available.

## What I Did
Narrative of actions taken. Include ALL concrete values: file paths, job IDs, \
URLs found, error messages seen, search queries that worked vs didn't. \
Write this as "I searched for X and found Y" / "I edited /path/to/file to add Z" \
— first person, past tense. Group related actions logically, not chronologically.

## What I Concluded
The actual finding, decision, or solution — in enough detail that the reader \
could explain it to the user without looking anything up. If research was done, \
state the answer here, not just "research was done." If a fix was applied, \
describe the fix and why it works.

## Current State
What exists now that didn't before: files modified, values produced, \
current status of any in-progress work. Use exact paths and IDs.

## What Remains (for you)
Specific next steps that require the NEW tools. Be precise about what to do \
and what NOT to redo. If earlier work already answered a question, say so \
explicitly: "The answer to X is Y (already verified) — no need to re-check."

CRITICAL RULES:
- NEVER summarize away concrete values (paths, IDs, numbers, error messages). \
These are the handoff's payload — without them the next person can't continue.
- NEVER say "research was done" without stating what was found.
- NEVER say "files were edited" without stating which files and what changed.
- Keep it concise but COMPLETE — aim for 40-60% of the original length.
- Write for someone who will ACT on this immediately, not read it later.

RAW WORK LOG TO SUMMARIZE:
{raw_handoff}"""


# ── LLM call helper (lazy imports) ──────────────────────────────────────

async def _call_sub_agent_llm(prompt: str, model: str = "", system: str = "", timeout: float = 60) -> str:
    """Make a focused LLM call for sub-agent work.

    Uses the LiteLLM proxy. Model defaults to SUB_AGENT_MODEL (Haiku)
    for compression tasks, but callers can pass WORKER_AGENT_MODEL
    (Sonnet) for tasks requiring code reasoning.

    Prefers native Anthropic provider (prompt caching, lower overhead)
    when IRIS_USE_NATIVE_ANTHROPIC=1. Falls back to LangChain/ChatOpenAI
    if the native provider is unavailable.

    Args:
        prompt: The complete prompt to send as user message.
        model: Model to use. Defaults to SUB_AGENT_MODEL if empty.
        system: Optional system prompt. When provided, gets prompt caching
                (cache_control=ephemeral) so repeated calls with the same
                system prompt benefit from cache hits.
        timeout: Max seconds to wait for LLM response. Default 60.

    Returns:
        The LLM response text, or an error message.
    """
    import asyncio as _asyncio

    effective_model = model if model else SUB_AGENT_MODEL
    use_native = os.environ.get("IRIS_USE_NATIVE_ANTHROPIC", "1") == "1"

    if use_native:
        try:
            from core.llm_provider import get_provider

            provider = get_provider(
                "anthropic",
                model_id=effective_model,
                temperature=0,
                max_tokens=4096,
                timeout=int(timeout),
            )

            logger.info(
                f"[SUB-AGENT] Native Anthropic call using model={effective_model}"
            )

            _cache_system = bool(system)
            response = await _asyncio.wait_for(
                provider.create_message(
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                    cache_system=_cache_system,
                ),
                timeout=timeout,
            )
            return response.content

        except _asyncio.TimeoutError:
            logger.warning(f"[SUB-AGENT] Native provider timed out after {timeout}s")
            return f"[Error: Sub-agent LLM call timed out after {timeout}s]"
        except Exception as e:
            logger.warning(f"[SUB-AGENT] Native provider failed, falling back to LangChain: {e}")

    # LangChain fallback path
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=effective_model,
            openai_api_base=LITELLM_URL,
            openai_api_key=os.environ.get("LITELLM_VIRTUAL_KEY", "not-set"),
            temperature=0,
            max_tokens=4096,
            disable_streaming=True,
        )

        logger.info(
            f"[SUB-AGENT] LangChain call using model={effective_model}, "
            f"url={LITELLM_URL}"
        )

        response = await _asyncio.wait_for(
            llm.ainvoke(prompt),
            timeout=timeout,
        )
        return response.content

    except _asyncio.TimeoutError:
        logger.error(f"Sub-agent LLM call timed out after {timeout}s (LangChain)")
        return f"[Error: Sub-agent LLM call timed out after {timeout}s]"
    except ImportError:
        return (
            "[Error: langchain_openai not available. "
            "Sub-agent LLM calls require langchain_openai to be installed.]"
        )
    except Exception as e:
        logger.error(f"Sub-agent LLM call failed: {e}")
        return f"[Error: Sub-agent LLM call failed: {e}]"


# ── Slurm data extraction sub-agent ─────────────────────────────────────


def build_slurm_extraction_prompt(
    question: str,
    tool_outputs: str,
) -> str:
    """Build the prompt for the Slurm data extraction sub-agent.

    Pure function — testable without LLM.

    The prompt instructs Haiku to extract exact numeric answers from raw
    Slurm tool output. It enforces zero-rounding, zero-paraphrasing rules
    to ensure the main LLM receives perfectly accurate cluster data.

    Args:
        question: The user's original question about the cluster.
        tool_outputs: Concatenated raw output from one or more Slurm tools.

    Returns:
        Complete prompt string for the Haiku sub-agent LLM call.
    """
    return f"""\
You are a Slurm cluster data extraction specialist. Your ONLY job is to \
extract exact values from raw Slurm tool output and present them clearly.

ABSOLUTE RULES — VIOLATION MEANS FAILURE:
1. NEVER round numbers. 267 stays 267, not ~265 or "about 270".
2. NEVER perform arithmetic unless explicitly asked. Copy values verbatim.
3. ALWAYS pair field names with their exact values.
4. NEVER conflate different metrics:
   - "nodes_with_free_cpus" ≠ "total_nodes"
   - "used_cpus" ≠ "allocated_cpus"
   - "free_gpus" ≠ "idle_gpus"
5. NEVER invent data. If a value is not in the tool output, say "not available".
6. ALWAYS include units (GB, %, count) where applicable.
7. For GPU data, ALWAYS break down by type (A100, H100, L40S, etc.).
8. For partition data, list each partition separately — never merge them.

RESPONSE FORMAT:
- Lead with a direct answer to the question using exact values.
- Follow with a structured breakdown if the data supports it.
- Use plain text with clear labels — no markdown tables (they compress poorly).
- Keep it concise but complete — every relevant number must appear.

USER QUESTION:
{question}

RAW SLURM TOOL OUTPUT:
{tool_outputs}"""


async def run_slurm_extraction_subagent(
    question: str,
    tool_outputs: str,
) -> str:
    """Extract precise Slurm data from tool outputs using a focused Haiku call.

    Designed for complex multi-tool Slurm queries where the main LLM needs
    a clean, accurate summary of cluster data. The Haiku sub-agent reads
    the raw tool output and extracts exact values with zero rounding.

    Safety net: If the Haiku call fails, returns the raw tool outputs
    unchanged — ensuring zero data loss even on failure.

    Cost: ~$0.005-0.015 per call (Haiku pricing).

    Args:
        question: The user's original question about the cluster.
        tool_outputs: Concatenated raw output from one or more Slurm tools.

    Returns:
        Extracted data summary, or raw tool_outputs if extraction fails.
    """
    if not tool_outputs or not tool_outputs.strip():
        return tool_outputs

    prompt = build_slurm_extraction_prompt(question, tool_outputs)

    logger.info(
        f"[SUB-AGENT] Slurm extraction: {len(tool_outputs)} chars input, "
        f"question: {question[:100]}"
    )

    try:
        result = await _call_sub_agent_llm(prompt)

        # Safety check: if the LLM returned an error, fall back to raw
        if result.startswith("[Error:"):
            logger.warning(
                f"[SUB-AGENT] Slurm extraction failed, using raw output. "
                f"Error: {result[:200]}"
            )
            return tool_outputs

        logger.info(
            f"[SUB-AGENT] Slurm extraction complete: {len(tool_outputs)} chars -> "
            f"{len(result)} chars ({len(result)/max(len(tool_outputs),1)*100:.0f}%)"
        )
        return result

    except Exception as e:
        logger.error(
            f"[SUB-AGENT] Slurm extraction exception, using raw output: {e}"
        )
        return tool_outputs


# ── Escalation handoff summarization (async — uses Haiku) ──────────────

async def summarize_escalation_handoff(raw_handoff: str) -> str:
    """Summarize a raw escalation handoff using Haiku for cost efficiency.

    Takes the full raw handoff text (from format_intermediate_steps_for_handoff)
    and compresses it via a Haiku LLM call into a concise summary that
    preserves all critical information (file paths, IDs, results, next steps).

    Safety net: If the Haiku call fails for any reason, returns the raw
    handoff text unchanged — ensuring zero context loss even on failure.

    Cost: ~$0.01 per call (Haiku pricing), vs $0.50+ saved by preventing
    the new agent from repeating work.

    Args:
        raw_handoff: The full raw handoff text.

    Returns:
        Summarized handoff text, or the raw handoff if summarization fails.
    """
    if not raw_handoff or not raw_handoff.strip():
        return raw_handoff

    prompt = build_escalation_handoff_prompt(raw_handoff)

    logger.info(
        f"[SUB-AGENT] Summarizing escalation handoff: "
        f"{len(raw_handoff)} chars raw input"
    )

    try:
        summary = await _call_sub_agent_llm(prompt)

        # Safety check: if the LLM returned an error, fall back to raw
        if summary.startswith("[Error:"):
            logger.warning(
                f"[SUB-AGENT] Handoff summarization failed, using raw handoff. "
                f"Error: {summary[:200]}"
            )
            return raw_handoff

        logger.info(
            f"[SUB-AGENT] Handoff summarized: {len(raw_handoff)} chars -> "
            f"{len(summary)} chars ({len(summary)/max(len(raw_handoff),1)*100:.0f}%)"
        )
        return summary

    except Exception as e:
        logger.error(
            f"[SUB-AGENT] Handoff summarization exception, using raw handoff: {e}"
        )
        return raw_handoff


# Hard cap for tool observations — unified across all executors.
# Above 12K, outputs get intent-aware Haiku summarization (extracts relevant facts).
# Below 12K, outputs stay raw in context (focused enough to read directly).
MAX_TOOL_OBSERVATION_CHARS = 12_000  # ~3.5K tokens


def pre_truncate_large_observation(
    observation: str,
    tool_name: str = "",
    max_chars: int = MAX_TOOL_OBSERVATION_CHARS,
) -> str:
    """Deterministic head+tail truncation for oversized tool observations.

    Args:
        observation: Raw tool output string.
        tool_name: Name of the tool (for logging).
        max_chars: Maximum characters to keep.

    Returns:
        The observation, possibly truncated with head+tail preservation.
    """
    if not observation or len(observation) <= max_chars:
        return observation

    from core.context_compactor import _fallback_truncate
    original_len = len(observation)
    result = _fallback_truncate(observation, max_chars, tool_name=tool_name)
    logger.info(
        f"[PRE-TRUNCATE] {tool_name}: {original_len:,} → {max_chars:,} chars "
        f"({max_chars/original_len*100:.0f}% kept)"
    )
    return result


# ── Tool definitions (BaseTool subclasses with kwargs unwrapping) ───────
# These use lazy imports of langchain_core to allow the pure helper
# functions above to be imported and tested without langchain installed.


def _create_sub_agent_tools() -> list:
    """Create the sub-agent tool instances using BaseTool.

    Lazy-imports langchain_core.tools and pydantic to avoid import errors
    in test environments. The pure helper functions above can be imported
    and tested without this.

    Returns:
        List of [analyze_files, summarize_command_output, review_codebase_section]
        tool instances, or empty list if langchain_core is not available.
    """
    try:
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, Field
    except ImportError:
        logger.warning(
            "[SUB-AGENT] langchain_core not available — "
            "sub-agent tools will not be created"
        )
        return []

    # ── Pydantic input schemas ──────────────────────────────────────────

    class AnalyzeFilesInput(BaseModel):
        """Input schema for analyze_files tool."""
        paths: list = Field(description="List of absolute file paths to read and analyze. Each path must start with /. Maximum 50 files.")
        question: str = Field(description="The specific question to answer about the files.")

    class SummarizeCommandOutputInput(BaseModel):
        """Input schema for summarize_command_output tool."""
        output: str = Field(description="The command output text to summarize. Can be large (up to 400K chars).")
        focus: str = Field(default="", description="What to focus the summary on. Examples: 'errors and failures', 'test results', 'performance metrics'.")

    class ReviewCodebaseSectionInput(BaseModel):
        """Input schema for review_codebase_section tool."""
        directory: str = Field(description="Absolute path to the directory to review. Must start with /.")
        task: str = Field(description="The review task. Be specific.")

    class WorkerAgentInput(BaseModel):
        """Input schema for run_worker_agent tool."""
        task: str = Field(description="The full task description or approved plan to execute. Include all context the worker needs.")
        skill_instructions: str = Field(default="", description="Optional skill-specific instructions for the worker (e.g. 'Use conda env at /path/to/env for pytest').")
        mode: str = Field(default="execute", description="Worker mode: 'execute' for plan execution (default), 'research' for exploration/analysis tasks. Use 'research' when the task is to read, compare, or analyze code rather than modify it.")

    # ── Tool classes ────────────────────────────────────────────────

    class AnalyzeFilesTool(BaseTool):
        """Read multiple files and answer a question about them using an isolated LLM call."""
        name: str = "analyze_files"
        description: str = (
            "Read multiple files and answer a question about them using an isolated LLM call. "
            "Use this INSTEAD of reading files one-by-one when you need to analyze 3+ files. "
            "This tool reads all files in an isolated context and returns only a structured "
            "summary — keeping the main conversation context clean. "
            "Args: paths (list of absolute file paths), question (str)."
        )
        args_schema: Type[BaseModel] = AnalyzeFilesInput

        def _run(self, **kwargs) -> str:
            raise NotImplementedError("Use async _arun")

        async def _arun(self, **kwargs: Any) -> str:
            args = _unwrap_kwargs(kwargs, tool_name="analyze_files")

            paths = args.get("paths", [])
            question = args.get("question", "")

            if not paths:
                return "Error: No file paths provided."
            if not question or not question.strip():
                return "Error: No question provided."

            # Validate paths
            valid_paths = []
            for p in paths[:MAX_FILES]:
                if isinstance(p, str) and p.startswith("/"):
                    valid_paths.append(p)
                else:
                    logger.warning(f"Skipping invalid path: {p}")

            if not valid_paths:
                return "Error: No valid file paths provided. Paths must be absolute (start with /)."

            logger.info(
                f"[SUB-AGENT] analyze_files: {len(valid_paths)} files, "
                f"question: {question[:100]}..."
            )

            # Read files
            file_contents = _build_file_contents_block(valid_paths)

            # Build prompt
            prompt = build_file_analysis_prompt(
                file_contents=file_contents,
                question=question.strip(),
                file_count=len(valid_paths),
            )

            # Make isolated LLM call (Sonnet for code reasoning)
            result = await _call_sub_agent_llm(prompt, model=WORKER_AGENT_MODEL)

            logger.info(
                f"[SUB-AGENT] analyze_files complete: "
                f"{len(result)} chars returned"
            )

            return result

    class SummarizeCommandOutputTool(BaseTool):
        """Compress large command/test/log output into a concise structured summary."""
        name: str = "summarize_command_output"
        description: str = (
            "Compress large command/test/log output into a concise structured summary. "
            "Use this when you have large output from execute_dynamic_task, test runs, "
            "Slurm job logs, or build output that would bloat the conversation context. "
            "Args: output (str - the text to summarize), focus (str - what to focus on, optional)."
        )
        args_schema: Type[BaseModel] = SummarizeCommandOutputInput

        def _run(self, **kwargs) -> str:
            raise NotImplementedError("Use async _arun")

        async def _arun(self, **kwargs: Any) -> str:
            args = _unwrap_kwargs(kwargs, tool_name="summarize_command_output")

            output = args.get("output", "")
            focus = args.get("focus", "")

            if not output or not output.strip():
                return "Error: No output provided to summarize."

            if not focus or not focus.strip():
                focus = "key results, errors, and important values"

            # Truncate if too large
            if len(output) > MAX_TOTAL_CONTENT:
                output = (
                    output[:MAX_TOTAL_CONTENT]
                    + f"\n\n[TRUNCATED — output was {len(output):,} chars, "
                    f"showing first {MAX_TOTAL_CONTENT:,}]"
                )

            logger.info(
                f"[SUB-AGENT] summarize_command_output: {len(output)} chars, "
                f"focus: {focus[:100]}"
            )

            # Build prompt
            prompt = build_summary_prompt(output=output, focus=focus.strip())

            # Make isolated LLM call
            result = await _call_sub_agent_llm(prompt)

            logger.info(
                f"[SUB-AGENT] summarize_command_output complete: "
                f"{len(result)} chars returned"
            )

            return result

    class ReviewCodebaseSectionTool(BaseTool):
        """Read all code files in a directory and perform a review task using an isolated LLM call."""
        name: str = "review_codebase_section"
        description: str = (
            "Read all code files in a directory and perform a review task using an isolated LLM call. "
            "Use this for code review, architecture analysis, or codebase exploration. "
            "Reads .py, .md, .yaml, .yml, .json, .toml, .cfg, .ini, .sh files in the directory. "
            "Args: directory (str - absolute path to directory), task (str - the review task)."
        )
        args_schema: Type[BaseModel] = ReviewCodebaseSectionInput

        def _run(self, **kwargs) -> str:
            raise NotImplementedError("Use async _arun")

        async def _arun(self, **kwargs: Any) -> str:
            args = _unwrap_kwargs(kwargs, tool_name="review_codebase_section")

            directory = args.get("directory", "")
            task = args.get("task", "")

            if not directory or not directory.strip():
                return "Error: No directory path provided."
            if not task or not task.strip():
                return "Error: No review task provided."

            directory = directory.strip()

            if not os.path.isdir(directory):
                return f"Error: Directory not found: {directory}"

            # Discover code files
            code_extensions = {
                ".py", ".md", ".yaml", ".yml", ".json", ".toml",
                ".cfg", ".ini", ".sh", ".txt", ".conf",
            }

            try:
                all_files = sorted(os.listdir(directory))
            except PermissionError:
                return f"Error: Permission denied reading directory: {directory}"

            code_files = []
            for fname in all_files:
                fpath = os.path.join(directory, fname)
                if os.path.isfile(fpath):
                    _, ext = os.path.splitext(fname)
                    if ext.lower() in code_extensions:
                        code_files.append(fpath)

            if not code_files:
                return (
                    f"Error: No code files found in {directory}. "
                    f"Looked for extensions: {', '.join(sorted(code_extensions))}"
                )

            logger.info(
                f"[SUB-AGENT] review_codebase_section: {len(code_files)} files "
                f"in {directory}, task: {task[:100]}..."
            )

            # Read files
            file_contents = _build_file_contents_block(code_files)

            # Build prompt
            prompt = build_review_prompt(
                file_contents=file_contents,
                task=task.strip(),
                file_count=len(code_files),
            )

            # Make isolated LLM call (Sonnet for code reasoning)
            result = await _call_sub_agent_llm(prompt, model=WORKER_AGENT_MODEL)

            logger.info(
                f"[SUB-AGENT] review_codebase_section complete: "
                f"{len(result)} chars returned"
            )

            return result

    # ── WorkerAgentTool — Claude Code Task tool pattern ─────────────────
    # The main agent calls this tool to delegate multi-step execution to
    # a focused Sonnet worker. The worker gets a capped tool set (≤8 tools)
    # and returns a structured summary. tools list is injected by app.py
    # after the full tool pool is assembled (avoids circular dependency).

    class WorkerAgentTool(BaseTool):
        """Delegate a multi-step task to a focused Sonnet worker agent."""
        name: str = "run_worker_agent"
        description: str = (
            "Delegate a multi-step task to a focused worker agent that runs its own "
            "tool loop (up to 20 iterations) and returns a clean summary. Use this for tasks "
            "that require 3+ DIFFERENT tool calls in sequence where each step depends on the "
            "previous result. Examples: (1) create venv → install packages → run smoke test → "
            "report versions, (2) find files with grep → edit them → run tests → report, "
            "(3) download data → process it → generate plot → save image. Do NOT combine "
            "these into a single execute_dynamic_task bash script — delegate to this worker "
            "instead. The worker has its own context window so the main conversation stays "
            "clean and cheap. "
            "IMPORTANT LIMITATION: Do NOT delegate tasks that require writing large file "
            "content verbatim (scripts >100 lines, configs >5KB). The worker has a limited "
            "output window and cannot reliably write large files in one tool call. For large "
            "file writes, use write_text_file or execute_dynamic_task directly yourself. "
            "Only delegate tasks where the LOGIC is complex (multi-step), not where the "
            "CONTENT is large. "
            "Args: task (str — describe the full task clearly including all paths and context, "
            "the worker has NO conversation history), "
            "skill_instructions (str — optional skill context for the worker), "
            "mode (str — 'execute' for plan execution [default], 'research' for "
            "exploration/analysis tasks like comparing files, tracing code paths, "
            "or investigating behavior — research mode uses analyze_files and "
            "review_codebase_section as primary tools)."
        )
        args_schema: Type[BaseModel] = WorkerAgentInput
        # Injected by app.py after full tool pool is assembled
        worker_tools: list = Field(default_factory=list)
        # Injected by app.py — reference to the shared PolicyEnforcementLayer
        # singleton so the worker can check/reset budget (Fix 2 & 3)
        _pel_ref: Any = None
        # Injected per-turn by app.py — cost tracking callback for worker LLM calls
        _cost_tracker_ref: Any = None
        # Injected per-turn by app.py — Chainlit callback for per-tool UI steps
        _chainlit_callback_ref: Any = None

        def _run(self, **kwargs) -> str:
            raise NotImplementedError("Use async _arun")

        async def _arun(self, **kwargs: Any) -> str:
            args = _unwrap_kwargs(kwargs, tool_name="run_worker_agent")
            task = args.get("task", "")
            skill_instructions = args.get("skill_instructions", "")
            mode = args.get("mode", "execute")
            if not task or not task.strip():
                return "Error: No task provided to worker agent."
            if not self.worker_tools:
                return (
                    "[Error: WorkerAgentTool has no tools injected. "
                    "This is a configuration error — app.py must set worker_tools "
                    "after assembling the tool pool.]"
                )
            # Fix 3: Guard — refuse to spawn worker if main agent has nearly
            # exhausted the global budget. The worker needs headroom to operate;
            # delegating when <10 calls remain just causes immediate PEL blocks
            # inside the worker, making the cascade worse.
            if self._pel_ref is not None:
                remaining = self._pel_ref.get_remaining_budget()
                if remaining < 10:
                    return (
                        f"[run_worker_agent SKIPPED] Only {remaining} tool calls remain "
                        "in the global budget for this turn. Not enough headroom for the "
                        "worker to operate (needs ≥10). Please summarize what you have "
                        "so far and report to the user instead of delegating."
                    )
            # Guard: reject tasks with large verbatim content that will cause
            # output truncation loops. 10K chars of task text strongly suggests
            # the caller embedded full file content for the worker to write.
            WORKER_TASK_SIZE_WARN = 10_000
            if len(task) > WORKER_TASK_SIZE_WARN:
                logger.warning(
                    f"[WORKER-AGENT-TOOL] Task is very large ({len(task)} chars). "
                    "Large verbatim content may cause output truncation loops."
                )
                return (
                    f"[run_worker_agent REFUSED] Task is {len(task)} chars — too large. "
                    "The worker agent cannot reliably write large file content because "
                    "its output token limit causes truncation. Please write the file(s) "
                    "directly using write_text_file or execute_dynamic_task with a heredoc. "
                    "Only delegate tasks where the LOGIC is complex, not where the CONTENT "
                    "is large."
                )

            logger.info(
                f"[WORKER-AGENT-TOOL] Delegating task ({len(task)} chars) "
                f"with {len(self.worker_tools)} available tools"
            )
            # Workers share the main agent's PEL budget — no reset.
            # This ensures global limits (60/turn, 4 submit_slurm_job/turn)
            # apply across main agent + all workers combined, preventing the
            # cascade where 13 workers each get a fresh 4-call budget.
            # Get nested step callback (creates child steps under parent)
            _nested_step_cb = None
            if callable(self._chainlit_callback_ref):
                try:
                    _nested_step_cb = self._chainlit_callback_ref()
                except Exception:
                    pass
            try:
                result = await run_worker_agent(
                    plan=task,
                    tools=self.worker_tools,
                    user_context="",
                    max_iterations=20,
                    skill_instructions=skill_instructions,
                    callbacks=None,
                    mode=mode,
                    cost_tracker=self._cost_tracker_ref,
                    step_callback=_nested_step_cb,
                )
            finally:
                pass
            return result

    # Instantiate and return
    return [
        AnalyzeFilesTool(),
        SummarizeCommandOutputTool(),
        ReviewCodebaseSectionTool(),
        WorkerAgentTool(),
    ]


# ── Module-level tool instances ─────────────────────────────────────────
# Try to create the BaseTool instances. If langchain_core is not available
# (e.g. in test environments), fall back to simple async function stubs
# that can still be tested for their names and basic behavior.

_tools = _create_sub_agent_tools()

if _tools:
    # langchain_core available — use real BaseTool instances
    analyze_files = _tools[0]
    summarize_command_output = _tools[1]
    review_codebase_section = _tools[2]
    worker_agent = _tools[3]
    # Export WorkerAgentTool class at module level for test imports
    WorkerAgentTool = type(worker_agent)
else:
    # Fallback for test environments without langchain_core.
    # These are simple async functions (same as the old @tool approach)
    # that preserve the same interface for testing.
    async def analyze_files(paths: list, question: str) -> str:
        """Read multiple files and answer a question about them using an isolated LLM call."""
        if not paths:
            return "Error: No file paths provided."
        if not question or not question.strip():
            return "Error: No question provided."
        valid_paths = [p for p in paths[:MAX_FILES] if isinstance(p, str) and p.startswith("/")]
        if not valid_paths:
            return "Error: No valid file paths provided. Paths must be absolute (start with /)."
        file_contents = _build_file_contents_block(valid_paths)
        prompt = build_file_analysis_prompt(file_contents, question.strip(), len(valid_paths))
        return await _call_sub_agent_llm(prompt, model=WORKER_AGENT_MODEL)

    async def summarize_command_output(output: str, focus: str = "") -> str:
        """Compress large command/test/log output into a concise structured summary."""
        if not output or not output.strip():
            return "Error: No output provided to summarize."
        if not focus or not focus.strip():
            focus = "key results, errors, and important values"
        if len(output) > MAX_TOTAL_CONTENT:
            output = output[:MAX_TOTAL_CONTENT] + f"\n\n[TRUNCATED]"
        prompt = build_summary_prompt(output, focus.strip())
        return await _call_sub_agent_llm(prompt)

    async def review_codebase_section(directory: str, task: str) -> str:
        """Read all code files in a directory and perform a review task using an isolated LLM call."""
        if not directory or not directory.strip():
            return "Error: No directory path provided."
        if not task or not task.strip():
            return "Error: No review task provided."
        directory = directory.strip()
        if not os.path.isdir(directory):
            return f"Error: Directory not found: {directory}"
        code_extensions = {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".sh", ".txt", ".conf"}
        try:
            all_files = sorted(os.listdir(directory))
        except PermissionError:
            return f"Error: Permission denied reading directory: {directory}"
        code_files = [os.path.join(directory, f) for f in all_files
                      if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in code_extensions]
        if not code_files:
            return f"Error: No code files found in {directory}."
        file_contents = _build_file_contents_block(code_files)
        prompt = build_review_prompt(file_contents, task.strip(), len(code_files))
        return await _call_sub_agent_llm(prompt, model=WORKER_AGENT_MODEL)

    # Fallback WorkerAgentTool for test environments without langchain_core
    class WorkerAgentTool:
        """Fallback WorkerAgentTool stub for environments without langchain."""
        name: str = "run_worker_agent"
        description: str = (
            "Delegate a multi-step task to a focused worker agent. Use for tasks with "
            "3+ different tool calls in sequence. Returns a clean summary."
        )
        worker_tools: list = []

        def __init__(self):
            self.worker_tools = []

    worker_agent = WorkerAgentTool()


# ── Worker agent error handler factory (per-invocation, no globals) ───────

def _make_worker_error_handler():
    """Create a per-invocation error handler with stuck-loop detection.

    Each call returns a fresh closure with its own state, safe for
    concurrent worker agents.
    """
    state = {"consecutive": 0, "last_error": ""}

    def handler(error) -> str:
        error_str = str(error)
        # Normalize for comparison (strip dynamic input_value parts)
        normalized = error_str.split("input_value")[0] if "input_value" in error_str else error_str

        if normalized == state["last_error"]:
            state["consecutive"] += 1
        else:
            state["consecutive"] = 1
            state["last_error"] = normalized

        if state["consecutive"] >= 2:
            state["consecutive"] = 0
            state["last_error"] = ""
            return (
                "CRITICAL: You have repeated the same failed tool call multiple times. "
                "STOP calling tools immediately. Instead, produce your final "
                "Execution Summary with whatever work you have completed so far. "
                "Report the tool call failure as an issue encountered."
            )

        return f"Error: {error_str[:500]}"

    return handler


# ── Worker tool output truncation ─────────────────────────────────────────

WORKER_MAX_TOOL_OUTPUT_CHARS = 12000


def _truncate_tool_output(output: str) -> str:
    """Truncate a tool output for the worker agent (sync fallback)."""
    if not output or len(output) <= WORKER_MAX_TOOL_OUTPUT_CHARS:
        return output
    from core.context_compactor import _fallback_truncate
    return _fallback_truncate(output, WORKER_MAX_TOOL_OUTPUT_CHARS, tool_name="worker_tool")


def _wrap_tools_with_deterministic_truncation(tools: list) -> list:
    """Wrap each tool so its output is truncated deterministically (no LLM calls).

    Uses head+tail truncation when output exceeds WORKER_MAX_TOOL_OUTPUT_CHARS.
    Cannot hang, O(1) execution time. For the native executor path, this is a
    fallback — _trim_observation (file-reference) handles most large outputs first.

    Creates shallow copies to avoid mutating the original tool objects.
    """
    import copy
    import functools

    wrapped = []
    for tool in tools:
        t = copy.copy(tool)

        if hasattr(t, 'coroutine') and t.coroutine:
            orig_coro = t.coroutine

            @functools.wraps(orig_coro)
            async def _wrapped_coro(*args, _orig=orig_coro, **kwargs):
                result = await _orig(*args, **kwargs)
                if isinstance(result, str):
                    return _truncate_tool_output(result)
                return result

            t.coroutine = _wrapped_coro
        elif hasattr(t, 'func') and t.func:
            orig_func = t.func

            @functools.wraps(orig_func)
            def _wrapped_func(*args, _orig=orig_func, **kwargs):
                result = _orig(*args, **kwargs)
                return _truncate_tool_output(str(result)) if isinstance(result, str) else result

            t.func = _wrapped_func

        wrapped.append(t)
    return wrapped


# ── Opus Advisor Tool ────────────────────────────────────────────────────
# The worker agent (Haiku) calls this when it is genuinely stuck.
# Opus sees the full context and returns concise actionable guidance.
# Capped at MAX_ADVISOR_CALLS per worker run (enforced via _advisor_calls counter).


def _make_opus_advisor_tool(advisor_calls_ref: list, litellm_url: str, virtual_key: str) -> "BaseTool":
    """Factory that creates a RequestOpusAdvisor tool with a shared call counter.

    Args:
        advisor_calls_ref: A single-element list [int] used as a mutable counter.
                           Shared between the tool and run_worker_agent so the
                           cap is enforced without global state.
        litellm_url: LiteLLM proxy base URL.
        virtual_key: LiteLLM virtual API key.

    Returns:
        A BaseTool instance that calls Opus for guidance.
    """
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field

    class AdvisorInput(BaseModel):
        question: str = Field(
            description="The specific question or problem you are stuck on. Be precise."
        )
        context: str = Field(
            default="",
            description="Relevant context: what you have tried, what failed, what you need to decide.",
        )

    class RequestOpusAdvisorTool(BaseTool):
        """Escalate to Opus for guidance when genuinely stuck.

        Use this ONLY when:
        - You have tried at least 2 approaches and both failed
        - You need architectural guidance on a complex decision
        - A tool returned an unexpected error you cannot interpret
        - You are unsure which of 2+ valid approaches to take

        Do NOT use this for:
        - Simple lookups (use the appropriate tool instead)
        - Tasks you can complete with available tools
        - Curiosity — only escalate when genuinely blocked

        Capped at MAX_ADVISOR_CALLS per worker run.
        """

        name: str = "request_opus_advisor"
        description: str = (
            "Escalate to Opus for expert guidance when you are genuinely stuck. "
            "Provide a specific question and the context of what you have tried. "
            "Returns concise actionable guidance. "
            f"Limited to {MAX_ADVISOR_CALLS} calls per task — use sparingly."
        )
        args_schema: type = AdvisorInput

        def _run(self, question: str, context: str = "", **kwargs) -> str:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, self._async_run(question, context))
                        return future.result(timeout=60)
                else:
                    return loop.run_until_complete(self._async_run(question, context))
            except Exception as e:
                return f"[Advisor unavailable: {e}]"

        async def _arun(self, question: str, context: str = "", **kwargs) -> str:
            return await self._async_run(question, context)

        async def _async_run(self, question: str, context: str = "") -> str:
            # Enforce cap
            if advisor_calls_ref[0] >= MAX_ADVISOR_CALLS:
                return (
                    f"[Advisor cap reached: {MAX_ADVISOR_CALLS} calls used. "
                    "Proceed with your best judgment or report the blocker.]"
                )
            advisor_calls_ref[0] += 1
            call_num = advisor_calls_ref[0]

            if not ENABLE_OPUS_ADVISOR:
                return "[Opus advisor is disabled (ENABLE_OPUS_ADVISOR=false). Proceed with best judgment.]"

            system_text = (
                "You are an expert advisor. A worker agent is stuck and needs your guidance. "
                "Provide concise, actionable advice in 3-7 bullet points. "
                "Be specific — name exact functions, files, or approaches. "
                "Do NOT write code unless absolutely necessary. "
                "Do NOT repeat the question back. Just give the answer."
            )

            human_content = f"QUESTION:\n{question}"
            if context:
                human_content += f"\n\nCONTEXT (what was tried / what failed):\n{context}"

            logger.info(
                f"[OPUS-ADVISOR] Call {call_num}/{MAX_ADVISOR_CALLS}: "
                f"question length={len(question)}, context length={len(context)}"
            )

            use_native = os.environ.get("IRIS_USE_NATIVE_ANTHROPIC", "1") == "1"

            try:
                if use_native:
                    from core.llm_provider import get_provider

                    opus_provider = get_provider(
                        "anthropic",
                        model_id=OPUS_ADVISOR_MODEL,
                        base_url=litellm_url,
                        api_key=virtual_key,
                        temperature=0,
                        max_tokens=8192,
                        thinking_budget=5000,
                        timeout=120,
                    )
                    response = await opus_provider.create_message(
                        system=system_text,
                        messages=[{"role": "user", "content": human_content}],
                        cache_system=False,
                    )
                    guidance = response.content
                else:
                    from langchain_openai import ChatOpenAI
                    from langchain_core.messages import HumanMessage, SystemMessage

                    opus_llm = ChatOpenAI(
                        model=OPUS_ADVISOR_MODEL,
                        base_url=f"{litellm_url}/v1",
                        api_key=virtual_key,
                        temperature=0,
                        max_tokens=700,
                        streaming=False,
                    )
                    response = await opus_llm.ainvoke([
                        SystemMessage(content=system_text),
                        HumanMessage(content=human_content),
                    ])
                    guidance = response.content

                logger.info(
                    f"[OPUS-ADVISOR] Call {call_num} complete: "
                    f"guidance length={len(guidance)} chars"
                )

                return f"[Opus Advisor — Call {call_num}/{MAX_ADVISOR_CALLS}]\n\n{guidance}"

            except Exception as e:
                logger.error(f"[OPUS-ADVISOR] Call {call_num} failed: {e}")
                return f"[Advisor error: {e}. Proceed with best judgment.]"

    return RequestOpusAdvisorTool()


# ── Worker Sub-Agent (Sonnet-powered execution) ─────────────────────────
# This function runs a Sonnet-powered agent with tools to execute a plan.
# The expensive model (Opus/Sonnet) plans; the worker (Sonnet) executes.
# This prevents the quadratic cost growth of 45 Opus tool calls.
#
# Architecture:
#   1. Opus generates a plan (text-only, no tools)
#   2. run_worker_agent() executes the plan with Sonnet + tools
#   3. Returns a structured summary back to the main agent
#   4. Opus reviews the summary and responds to the user
#
# Cost: Worker isolation keeps results out of the main agent's context.


def _build_worker_system_prompt(mode: str, skill_instructions: str, user_context: str) -> str:
    """Build the worker agent system prompt (shared between native and LangChain paths)."""
    if mode == "research":
        worker_system = (
            "You are a research worker agent. Your job is to explore, analyze, "
            "and gather information — NOT to execute changes or follow a rigid plan.\n\n"
            "RULES:\n"
            "- Your PRIMARY tool is `batch`. Put ALL your operations in ONE call:\n"
            "  batch(operations=[\n"
            "    {'type': 'find', 'pattern': '*.py', 'start_path': '/work/src/'},\n"
            "    {'type': 'grep', 'path': '/work/src/main.py', 'pattern': 'class.*Error'},\n"
            "    {'type': 'read', 'path': '/work/README.md'},\n"
            "    {'type': 'shell', 'command': 'git log --oneline -20'},\n"
            "  ])\n\n"
            "EFFICIENCY RULES:\n"
            "- PLAN first, then batch. Before calling any tool, list ALL the commands/reads "
            "you need for this investigation step. Then put them ALL in one batch() call.\n"
            "- One batch call with 10 operations is better than 10 calls with 1 operation each.\n"
            "- If you need results from step 1 to decide step 2, that's OK — use 2 batch calls. "
            "But NEVER use more than 3 batch calls per task.\n"
            "- Use read_text_file ONLY for a single specific file section.\n"
            "- NEVER call batch multiple times for the same investigation step.\n"
            "- If batch output is truncated, read the full_output path for details.\n\n"
            "ADDITIONAL RULES:\n"
            "- Gather facts, trace code paths, compare implementations\n"
            "- Report findings clearly with file paths, line numbers, and key observations\n"
            "- IMPORTANT: Every tool call MUST have valid arguments. Never call "
            "a tool with empty {} arguments.\n"
            "- IMPORTANT: If a tool call returns a very large output, do NOT "
            "repeat the call. Extract only the information you need and move on.\n"
            f"- ADVISOR: If you are genuinely stuck, call request_opus_advisor "
            f"with a specific question. Limited to {MAX_ADVISOR_CALLS} uses.\n\n"
        )
    else:
        worker_system = (
            "You are a worker agent executing a pre-approved plan. "
            "Follow the plan EXACTLY. Do not deviate or add extra steps.\n\n"
            "RULES:\n"
            "- Your PRIMARY tool is `batch`. Put ALL steps into as few batch() calls as possible.\n"
            "- Edits + verification in one call:\n"
            "  batch(operations=[\n"
            "    {'type': 'edit', 'edits': [{'path': '...', 'find': '...', 'replace': '...'}]},\n"
            "    {'type': 'test', 'path': 'tests/', 'mode': 'failures_only'},\n"
            "  ])\n"
            "- Shell commands that don't depend on each other → same batch call.\n"
            "- LARGE FILE WRITES: Use batch shell with heredoc:\n"
            "  {'type': 'shell', 'command': \"cat << 'EOF' > /path/file.py\\ncontent\\nEOF\"}\n\n"
            "EXECUTION RULES:\n"
            "- Execute each step in the plan sequentially\n"
            "- Report results concisely: what succeeded, what failed\n"
            "- Do NOT explore beyond the plan scope\n"
            "- Do NOT re-read files unless the plan explicitly says to\n"
            "- If a step fails, report the failure and continue with remaining steps\n"
            "- When done, output a structured summary\n"
            "- IMPORTANT: If a tool call returns a very large output, do NOT "
            "repeat the call. Extract only the information you need and move on.\n"
            "- IMPORTANT: Every tool call MUST have valid arguments. Never call "
            "a tool with empty {} arguments. If you cannot determine the right "
            "arguments, skip that step and report it as incomplete.\n"
            f"- ADVISOR: If you are genuinely stuck after 2+ failed attempts, "
            f"call request_opus_advisor with a specific question. "
            f"Limited to {MAX_ADVISOR_CALLS} uses — do not call it speculatively.\n\n"
        )

    if skill_instructions:
        worker_system += f"SKILL CONTEXT:\n{skill_instructions}\n\n"

    if user_context:
        worker_system += f"USER ENVIRONMENT:\n{user_context}\n\n"

    if mode == "research":
        worker_system += (
            "OUTPUT FORMAT (when done):\n"
            "## Research Findings\n"
            "- Key observations: [what you discovered]\n"
            "- File paths: [relevant files with line numbers]\n"
            "- Analysis: [your interpretation of what the code does]\n"
            "- Open questions: [anything unresolved]\n"
        )
    else:
        worker_system += (
            "OUTPUT FORMAT (when done):\n"
            "## Execution Summary\n"
            "- Steps completed: N/M\n"
            "- Files modified: [list]\n"
            "- Tests: PASS/FAIL (details if failed)\n"
            "- Issues encountered: [list or 'none']\n"
        )

    return worker_system


def _filter_worker_tools(tools: list, plan: str, mode: str) -> list:
    """Filter tools for worker agent (max 8, mode-dependent essential set)."""
    if mode == "research":
        _ACTIVE_ESSENTIAL = set(RESEARCH_ESSENTIAL_TOOLS)
    else:
        _ACTIVE_ESSENTIAL = {
            "batch",
            "read_text_file", "submit_slurm_job",
            "slurm_monitor_job",
        }

    plan_lower = plan.lower()
    mentioned_tools = set()
    for tool in tools:
        tool_name = tool.name if hasattr(tool, 'name') else str(tool)
        if tool_name.lower() in plan_lower or tool_name in plan:
            mentioned_tools.add(tool_name)

    target_names = mentioned_tools | _ACTIVE_ESSENTIAL
    filtered_tools = [t for t in tools if (t.name if hasattr(t, 'name') else str(t)) in target_names]

    if len(filtered_tools) < 3:
        filtered_tools = [t for t in tools if (t.name if hasattr(t, 'name') else str(t)) in _ACTIVE_ESSENTIAL]
        logger.info(f"[WORKER-AGENT] Using essential tools only: {len(filtered_tools)} tools")
    elif len(filtered_tools) > 8:
        essential_filtered = [t for t in filtered_tools if (t.name if hasattr(t, 'name') else str(t)) in _ACTIVE_ESSENTIAL]
        non_essential = [t for t in filtered_tools if (t.name if hasattr(t, 'name') else str(t)) not in _ACTIVE_ESSENTIAL]
        filtered_tools = essential_filtered + non_essential[:8 - len(essential_filtered)]
        logger.info(f"[WORKER-AGENT] Capped tools to {len(filtered_tools)} (from {len(tools)} available)")
    else:
        logger.info(
            f"[WORKER-AGENT] Filtered tools: {len(filtered_tools)}/{len(tools)} "
            f"(mentioned: {mentioned_tools}, essential: {_ACTIVE_ESSENTIAL & target_names})"
        )

    return filtered_tools


async def _run_worker_agent_native(
    plan: str,
    tools: list,
    user_context: str = "",
    max_iterations: int = 20,
    skill_instructions: str = "",
    mode: str = "execute",
    cost_tracker=None,
    step_callback=None,
) -> str:
    """Native Anthropic path for worker agent execution.

    Uses NativeAgentExecutor with prompt caching + extended thinking.
    """
    from core.llm_provider import get_provider
    from core.cost_tracker import CostTracker as NativeCostTracker
    from core.native_executor import NativeAgentExecutor
    from core.stuck_detection_callback import StuckInterrupt
    from core.single_agent import SkillEscalationInterrupt

    worker_system = _build_worker_system_prompt(mode, skill_instructions, user_context)
    filtered_tools = _filter_worker_tools(tools, plan, mode)

    wrapped_tools = list(filtered_tools)

    # NOTE: batch tool (MCP) is already included via _filter_worker_tools
    # (it's in _ACTIVE_ESSENTIAL for both research and execute modes).
    # The old batch_tool_unified injection was removed to avoid name collision.

    _advisor_calls = [0]
    _advisor_tool = _make_opus_advisor_tool(
        _advisor_calls, LITELLM_URL, os.environ.get("LITELLM_VIRTUAL_KEY", "not-set")
    )
    wrapped_tools.append(_advisor_tool)

    provider = get_provider(
        "anthropic",
        model_id=WORKER_AGENT_MODEL,
        base_url=LITELLM_URL,
        api_key=os.environ.get("LITELLM_VIRTUAL_KEY", "not-set"),
        temperature=0,
        max_tokens=8192,
        thinking_budget=WORKER_THINKING_BUDGET,
        timeout=WORKER_AGENT_TIMEOUT,
    )

    worker_cost_tracker = NativeCostTracker()

    executor = NativeAgentExecutor(
        provider=provider,
        tools=wrapped_tools,
        system_prompt=worker_system,
        max_iterations=max_iterations,
        cost_tracker=worker_cost_tracker,
        step_callback=step_callback,
        max_observation_chars=WORKER_MAX_TOOL_OUTPUT_CHARS,
    )

    input_text = (
        f"Research this question:\n\n{plan}" if mode == "research"
        else f"Execute this plan:\n\n{plan}"
    )

    logger.info(
        f"[WORKER-AGENT] Starting NATIVE execution with "
        f"{len(filtered_tools)} tools (filtered from {len(tools)}), "
        f"max_iterations={max_iterations}, thinking_budget={WORKER_THINKING_BUDGET}"
    )

    try:
        result = await asyncio.wait_for(
            executor.ainvoke(
                {"input": input_text, "chat_history": [], "agent_scratchpad": []},
            ),
            timeout=WORKER_AGENT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(f"[WORKER-AGENT] Native execution timed out after {WORKER_AGENT_TIMEOUT}s")
        return (
            f"[Error: Worker agent timed out after {WORKER_AGENT_TIMEOUT} seconds. "
            "The main agent should retry with a simpler plan or execute directly.]"
        )
    except StuckInterrupt as stuck:
        logger.warning(f"[WORKER-AGENT] Stuck on '{stuck.tool_name}' ({stuck.failure_count}x)")
        return (
            f"[Worker agent stuck: repeated errors on '{stuck.tool_name}' "
            f"({stuck.failure_count} times). Error: {stuck.error_snippet}. "
            f"Suggestion: {stuck.suggested_query}]"
        )
    except SkillEscalationInterrupt as esc:
        return (
            f"[Worker agent requested skill '{esc.skill_name}' but cannot escalate. "
            "Returning partial results for main agent to handle.]"
        )

    if cost_tracker is not None:
        cost_tracker.merge(worker_cost_tracker)

    output = result.get("output", "Worker agent produced no output.")
    steps = result.get("intermediate_steps", [])

    logger.info(
        f"[WORKER-AGENT] Native execution complete: {len(steps)} tool calls, "
        f"output length: {len(output)} chars, cost: ${worker_cost_tracker.total_cost:.4f}"
    )

    report_type = "Research Report" if mode == "research" else "Execution Report"
    summary = (
        f"[Worker Agent {report_type}]\n"
        f"Tool calls made: {len(steps)}\n"
        f"Model: {WORKER_AGENT_MODEL}\n\n"
        f"{output}"
    )

    return summary


async def run_worker_agent(
    plan: str,
    tools: list,
    user_context: str = "",
    max_iterations: int = 20,
    skill_instructions: str = "",
    callbacks: list = None,
    mode: str = "execute",
    cost_tracker=None,
    step_callback=None,
) -> str:
    """Execute a plan or research task using a Haiku-powered agent with tools.

    This is the core of the "Orchestrator-Worker" pattern:
    - The main agent (Opus/Sonnet) generates a plan or research question
    - This function executes it cheaply with Haiku
    - Returns a structured summary of what was done

    Args:
        plan: The execution plan or research question.
        tools: List of LangChain tool objects the worker can use.
        user_context: User environment context (paths, settings).
        max_iterations: Maximum tool calls the worker can make (default 20).
        skill_instructions: Skill-specific instructions for the worker.
        callbacks: Optional list of LangChain callbacks (e.g. CostTrackingCallback).
        mode: "execute" (default) for rigid plan execution,
              "research" for exploration/analysis using batch analysis tools.
        cost_tracker: Optional NativeCostTracker to merge worker costs into.
        step_callback: Optional callback for Chainlit step rendering.

    Returns:
        Structured summary of execution or research results.
    """
    # Native Anthropic path (default) — prompt caching + thinking
    _use_native = os.environ.get("IRIS_USE_NATIVE_EXECUTOR", "1") == "1"
    if _use_native:
        return await _run_worker_agent_native(
            plan=plan,
            tools=tools,
            user_context=user_context,
            max_iterations=max_iterations,
            skill_instructions=skill_instructions,
            mode=mode,
            cost_tracker=cost_tracker,
            step_callback=step_callback,
        )

    # LangChain fallback path
    try:
        from langchain_openai import ChatOpenAI
        from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        # Create Haiku LLM for worker execution — cost-efficient within 8-tool pool.
        # Override WORKER_AGENT_MODEL env var to use Sonnet for complex tasks.
        # max_tokens=8192 is the model ceiling for Haiku 4.5 — allows large file
        # writes in a single tool call without output truncation causing retry loops.
        worker_llm = ChatOpenAI(
            model=WORKER_AGENT_MODEL,
            openai_api_base=LITELLM_URL,
            openai_api_key=os.environ.get("LITELLM_VIRTUAL_KEY", "not-set"),
            temperature=0,
            max_tokens=8192,
            disable_streaming=True,
        )

        # Opus advisor tool — shared call counter enforces MAX_ADVISOR_CALLS cap.
        # The counter is a single-element list so it can be mutated inside the closure.
        _advisor_calls = [0]
        _advisor_tool = _make_opus_advisor_tool(
            advisor_calls_ref=_advisor_calls,
            litellm_url=LITELLM_URL,
            virtual_key=os.environ.get("LITELLM_VIRTUAL_KEY", "not-set"),
        )

        # Build worker system prompt and filter tools (shared helpers)
        worker_system = _build_worker_system_prompt(mode, skill_instructions, user_context)
        filtered_tools = _filter_worker_tools(tools, plan, mode)

        wrapped_tools = _wrap_tools_with_deterministic_truncation(filtered_tools)
        wrapped_tools.append(_advisor_tool)

        # Create the worker agent
        # Escape { and } in system prompt so ChatPromptTemplate doesn't
        # parse them as variables (user_context/skill_instructions may contain braces)
        worker_system_escaped = worker_system.replace("{", "{{").replace("}", "}}")
        prompt = ChatPromptTemplate.from_messages([
            ("system", worker_system_escaped),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_tools_agent(worker_llm, wrapped_tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=wrapped_tools,
            max_iterations=max_iterations,
            handle_parsing_errors=_make_worker_error_handler(),
            return_intermediate_steps=True,
            early_stopping_method="force",
        )

        logger.info(
            f"[WORKER-AGENT] Starting worker execution with "
            f"{len(filtered_tools)} tools (filtered from {len(tools)}), "
            f"max_iterations={max_iterations}"
        )

        # Execute the plan or research question (with timeout to prevent hanging)
        if mode == "research":
            input_text = f"Research this question:\n\n{plan}"
        else:
            input_text = f"Execute this plan:\n\n{plan}"
        invoke_kwargs = {
            "input": input_text,
            "agent_scratchpad": [],
        }
        invoke_config = {}
        if callbacks:
            invoke_config["callbacks"] = callbacks

        try:
            result = await asyncio.wait_for(
                executor.ainvoke(invoke_kwargs, config=invoke_config if invoke_config else None),
                timeout=WORKER_AGENT_TIMEOUT,  # generous wall-clock time for multi-tool tasks
            )
        except asyncio.TimeoutError:
            logger.error(f"[WORKER-AGENT] Execution timed out after {WORKER_AGENT_TIMEOUT} seconds")
            return (
                f"[Error: Worker agent timed out after {WORKER_AGENT_TIMEOUT} seconds. "
                "The main agent should retry with a simpler plan or execute directly.]"
            )

        output = result.get("output", "Worker agent produced no output.")
        steps = result.get("intermediate_steps", [])

        logger.info(
            f"[WORKER-AGENT] Execution complete: {len(steps)} tool calls, "
            f"output length: {len(output)} chars"
        )

        # Add metadata to the output
        report_type = "Research Report" if mode == "research" else "Execution Report"
        summary = (
            f"[Worker Agent {report_type}]\n"
            f"Tool calls made: {len(steps)}\n"
            f"Model: {WORKER_AGENT_MODEL}\n\n"
            f"{output}"
        )

        return summary

    except ImportError as e:
        return (
            f"[Error: Worker agent dependencies not available: {e}. "
            f"Falling back to main agent execution.]"
        )
    except Exception as e:
        logger.error(f"[WORKER-AGENT] Execution failed: {e}")
        return (
            f"[Error: Worker agent execution failed: {str(e)[:500]}. "
            f"The main agent should retry the task directly.]"
        )


# ── Sub-agent tools list for registration in single agent's tool pool ───
# worker_agent is intentionally excluded from SUB_AGENT_TOOLS here.
# app.py injects worker_tools into it AFTER the full tool pool is assembled,
# then appends it separately. This avoids a circular dependency where the
# worker tool would be added to all_tools before its tools list is populated.
SUB_AGENT_TOOLS = [analyze_files, summarize_command_output, review_codebase_section]
