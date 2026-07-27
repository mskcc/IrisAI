"""History management utilities — extracted from app.py.

Pure functions for converting message history to text.
No Chainlit or LLM dependencies (LLM summarization is optional).
"""
from typing import List, Any, Tuple, Dict, Optional
from pathlib import Path
import json
import re
import logging

logger = logging.getLogger("core.history")


# Conservative chars-per-token ratio.
# Bedrock's actual tokenizer counts ~1 token per 3 chars on average.
# Using 3 (not 4) because our Phase 2 testing showed chars/4 underestimates
# by ~50%, causing 169K estimated to become 209K actual Bedrock tokens.
CHARS_PER_TOKEN = 3

# Default token budget for agent history.
# Bedrock limit is 200K tokens. After subtracting system prompt (~8K),
# tool definitions (~5K), safety margin (~17K), and accounting for
# duplication in agent_input (~1.3x), we get ~120K safe budget.
DEFAULT_AGENT_TOKEN_BUDGET = 120_000

# Hard ceiling for total tokens sent to Bedrock.
# Bedrock's absolute limit is 200K. We use 180K to leave room for
# system prompt (~8K), tool definitions (~5K), and safety margin (~7K).
# This is the FINAL guard — if total tokens exceed this after all other
# trimming, enforce_total_token_budget() will aggressively cut until safe.
BEDROCK_HARD_TOKEN_LIMIT = 180_000

# Maximum tokens allowed for a single message before truncation.
# If any individual message exceeds this, its content will be truncated.
# This prevents a single huge tool output or response from consuming
# the entire context window and causing infinite retry loops.
MAX_SINGLE_MESSAGE_TOKENS = 30_000

# ── Sliding window defaults ─────────────────────────────────────────────
# Minimum recent messages to keep raw (ensures 3 full Human+AI exchanges).
SLIDING_WINDOW_SIZE = 6

# Token budget for the "recent" portion kept raw after compaction.
# Messages are kept newest-first until this budget is exhausted.
# 40K tokens ≈ last 3-4 turns with hybrid history (~5-10K/turn).
# More raw recent context = higher reliability for follow-up questions.
SLIDING_WINDOW_TOKEN_BUDGET = 40_000

# Maximum number of fact entries to keep in the fact summary.
# Prevents the fact block from growing unbounded in very long sessions.
MAX_FACT_ENTRIES = 30

# ── LLM Summarization defaults ──────────────────────────────────────────
# Token budget for the older-message text sent to the main model for summarization.
# 150K tokens — effectively unlimited for typical sessions. The main model
# (Sonnet/Opus) can handle 200K context, so we send everything and let it
# produce a comprehensive summary. Only applies as a safety ceiling.
LLM_SUMMARY_TOKEN_BUDGET = 150_000

# ── History Compaction defaults ─────────────────────────────────────────
# Token threshold at which comprehensive compaction kicks in.
# With hybrid history (~5-10K/turn), 100K threshold means compaction fires
# every 12-15 turns. Most sessions finish before it fires at all.
# After compaction: summary (~15K) + raw recent (40K) = ~55K, leaving
# ~110K for current turn execution (Bedrock ceiling: 180K).
COMPACTION_TOKEN_THRESHOLD = 100_000

# DEPRECATED: No longer used. Single-step compaction replaces progressive cleanup.
# Kept for reference only.
PROGRESSIVE_CLEANUP_FORCE_COMPACTION_AFTER = 2

# Marker prefix used to detect messages that are already compacted summaries.
# After full compaction, history[0] starts with this marker. The sliding window
# can skip Haiku summarization and reuse the existing summary directly.
COMPACTED_SUMMARY_MARKER = "[CONTEXT FROM EARLIER IN THIS CONVERSATION]"

# Maximum number of new older messages before falling back to full re-summarization.
INCREMENTAL_SUMMARY_MAX_DELTA = 6

# System prompt for the Haiku summarization call.
# Designed to produce structured output useful for both the skill selector
# (needs to know what domain/topic the conversation is about) and the
# agent (needs file paths, decisions, errors, and task context).
SUMMARY_SYSTEM_PROMPT = """\
You are a conversation compactor for an AI assistant on an HPC cluster.
Your job is to produce a COMPREHENSIVE summary that preserves ALL important
information from the conversation so far.

This summary will REPLACE the original messages — anything you omit is LOST FOREVER.
Be thorough. The AI assistant reading this summary must be able to continue the
conversation as if it had seen every original message.

Your summary MUST include ALL of the following sections (skip a section ONLY if it
has zero relevant content):

## User Intent & Goals
- The user's original request and any refinements
- Implicit goals discovered during the conversation
- Priority ordering if multiple tasks are in flight

## Phase State
- Last phase completed: [research / planning / execution / none]
- Research findings confirmed by user: [yes / no / not yet asked]
- Plan approved by user: [yes / no / not yet asked]
- FINDINGS.md path (if written): [path or N/A]
- PLAN.md path (if written): [path or N/A]

## Decisions & Reasoning
For each significant decision:
- What was decided
- Why (the reasoning or user preference that drove it)
- Any alternatives that were explicitly rejected

## File Paths & Artifacts
- Every file path mentioned (FULL absolute paths, never abbreviated)
- Files created, modified, or deleted
- Configuration files and their key settings
- Output artifacts (logs, reports, plots)

## Code Artifacts (VERBATIM)
For any code that was discussed, written, debugged, or shown:
- Function/class definitions or signatures central to the task
- Code diffs or patches discussed (before/after)
- Configuration snippets (YAML, JSON, TOML, Slurm scripts) with specific values
- Import statements or dependency declarations that were added/changed
- Shell commands or one-liners that solved specific problems
- Preserve these VERBATIM — do not paraphrase code into natural language

## Architecture & Relationships
- How components interact (what imports what, what calls what, what triggers what)
- Design patterns in use and WHY they were chosen over alternatives
- Data flow: where input comes from, how it transforms, where output goes
- Constraints discovered: "X must happen before Y", "A cannot coexist with B"
- System boundaries: what talks to Bedrock, what talks to MCP, what talks to SLURM

## Tool Calls & Outcomes
For each tool invocation:
- Tool name and what it was called with (key arguments)
- What it returned (success/failure, key data points)
- Any side effects (files created, processes started, jobs submitted)

## Error -> Fix Chains
For each error encountered:
- The exact error message or symptom
- Root cause identified (if known)
- What was tried to fix it (in order)
- Whether the fix worked
- If unresolved, note it as still pending

## Environment & Configuration
- Working directory
- Software versions, modules loaded
- Environment variables set
- Cluster/node information (partitions, GPUs, etc.)
- Container paths (.sif files)

## Research Findings
Key facts discovered during research — data, benchmarks, documentation findings,
software capabilities, parameter values.

## What Was Completed
Bullet list of actions successfully finished, in chronological order.

## What Is Pending / In Progress
- Tasks discussed but not yet done
- Tasks started but not finished
- Next steps the user expects

## Current Working State
- What state is the system in RIGHT NOW?
- Any running processes (job IDs, PIDs)
- Partial implementations or open files
- Where exactly the user left off

## All User Messages (Intent Archaeology)
Preserve the key user messages verbatim (or near-verbatim) that express:
- Original requests
- Corrections or refinements
- Preferences or constraints
- Approvals or rejections

Rules:
- Be COMPREHENSIVE — aim for 2000-5000 words depending on conversation complexity.
- Preserve ALL file paths EXACTLY as they appear (full absolute paths, never shorten).
- Preserve ALL identifiers: job IDs, commit hashes, version numbers, PIDs, timestamps.
- Preserve ALL error messages verbatim (or their key diagnostic portions).
- Preserve ALL code blocks verbatim — function definitions, diffs, config snippets, shell commands.
  NEVER paraphrase code into prose. A 3-line fix is worth more than "fixed the function."
- Preserve ALL numeric values exactly: counts, sizes, line numbers, percentages.
- NEVER round numbers. Report 267 as 267, not ~265.
- Do NOT add information that isn't in the conversation.
- Do NOT include meta-commentary about the summarization process.
- Use markdown formatting for readability.
- Chronological order within sections where it matters.

ANCHOR BLOCK GENERATION:
You MUST produce a section titled EXACTLY:
## ANCHORED CONTEXT (DO NOT MODIFY — preserved verbatim)
This section MUST appear as the FIRST section of your output. It is the survival core —
everything in it is preserved VERBATIM across all future compactions.

The anchor MUST contain EXACTLY these 6 subsections (in this order). Include all that
apply; write "- none" if a section is empty — NEVER omit the heading:

### 1. User Constraints & Instructions
- Every instruction, constraint, requirement, or naming convention the user stated
- Every "always do X", "never do Y", "use X format", "prefix with Y" type statement
- Verbatim quotes where possible, tagged with turn number: [Turn N] "exact quote"
- These are NON-RECOVERABLE if lost — when in doubt, include it

### 2. Reasoning & Decisions
For each significant decision made during the conversation:
- Format: "[Turn N] Decided X because Y (rejected: Z)"
- Include the reasoning chain that led to the decision
- Include alternatives that were explicitly rejected and why

### 3. Lessons & Errors
- Error→fix chains: "[Turn N] Error: X → Fix: Y → Status: resolved/unresolved"
- Patterns to avoid: "AVOID: X because Y"
- Things that worked unexpectedly well (validated approaches)

### 4. Project Progress
- One line per completed action, in chronological order (append-only timeline)
- Format: "[Turn N] action_description"
- This is the reasoning timeline — shows how the session evolved

### 5. Working State
- Current phase (research / planning / execution / idle)
- Active jobs (IDs + state + partition)
- Pending items / next steps
- What was happening when compaction fired
- Blocked items and why

### 6. Key Facts & Values
- Numeric values that must not drift (counts, thresholds, IDs, version numbers)
- Absolute file paths (FULL, never abbreviated)
- Project name, working directory
- Container/module paths, partition names
- Any identifier the user explicitly stated (prefixes, suffixes, naming patterns)

Be thorough — 500-2000 words is appropriate for the anchor. Everything NOT in the anchor
is subject to lossy summarization on future compactions. When in doubt, anchor it.

CRITICAL ANCHOR RULE: If your input ALREADY contains "## ANCHORED CONTEXT (DO NOT MODIFY —
preserved verbatim)", you MUST reproduce that section VERBATIM as the FIRST section
of your output. You may APPEND new entries to any subsection (new constraints, new
decisions, new errors, new progress lines, updated working state, new facts) but NEVER
remove, modify, or rephrase existing entries. The anchor block is append-only.

KNOWLEDGE EXTRACTION (for disk persistence):
After your full summary (including anchor), produce a final section delimited EXACTLY as:
===KNOWLEDGE_EXTRACT===
Extract durable knowledge entries from the conversation that should persist permanently
in the project's knowledge file. Focus on: user constraints, validated approaches,
failed attempts, key reference paths, and configuration facts.

Format each entry as:
- [TYPE]: <exact fact with real values — paths, versions, IDs, commands>
  Why: <root cause or reasoning>
  Applies when: <trigger condition>

TYPE must be one of: CONSTRAINT, VALIDATED_APPROACH, FAILED_ATTEMPT, REFERENCE_PATH, CONFIGURATION, USER_PREFERENCE

Rules:
- Only include entries that are DURABLE (useful beyond this session)
- Include EXACT values — paths, IDs, prefixes, version numbers
- Maximum 10 entries per compaction
- Do NOT duplicate entries already present in the existing knowledge shown below
- If nothing is knowledge-worthy, output exactly: NOTHING_NEW
===END_KNOWLEDGE_EXTRACT===
{existing_knowledge_context}
"""

INCREMENTAL_SUMMARY_SYSTEM_PROMPT = """\
You are updating a conversation summary for an AI assistant on an HPC cluster.
You have a PREVIOUS SUMMARY and NEW MESSAGES that happened after that summary.

Produce an UPDATED summary that incorporates the new information into the
existing structure. The result will REPLACE both the previous summary and the
new messages — anything you omit is LOST FOREVER.

Rules:
- Keep ALL sections from the previous summary. Add new sections if needed.
- ADD new facts, decisions, tool calls, errors, and outcomes from the new messages.
- UPDATE state that changed (e.g., pending tasks now complete, errors now fixed).
- REMOVE information that is explicitly contradicted by new messages.
- Preserve ALL file paths, identifiers, and error messages exactly.
- The updated summary should be 2000-5000 words. Grow it as needed — never shrink
  it to fit an arbitrary word limit.
- Do NOT lose information from the previous summary unless explicitly contradicted.
- Do NOT add meta-commentary about the update process.
- Preserve the chronological order of events where it matters.
- Preserve ALL code blocks, function signatures, and diffs VERBATIM from the
  previous summary. Do NOT paraphrase code into natural language.
- Maintain the Architecture & Relationships section — if new messages reveal
  component interactions, ADD them without removing existing ones.

CRITICAL ANCHOR RULE: If your input contains "## ANCHORED CONTEXT (DO NOT MODIFY —
preserved verbatim)", you MUST reproduce that section VERBATIM as the FIRST section
of your output. Merge new critical facts into the correct numbered subsection:
- New user constraints → ### 1. User Constraints & Instructions
- New decisions → ### 2. Reasoning & Decisions
- New errors/lessons → ### 3. Lessons & Errors
- New progress → ### 4. Project Progress
- Updated state → ### 5. Working State (this section CAN be updated, not just appended)
- New facts/values → ### 6. Key Facts & Values
NEVER remove or rephrase existing entries in sections 1-4 and 6 — only APPEND.
Section 5 (Working State) may be updated to reflect current state.

KNOWLEDGE EXTRACTION (for disk persistence):
After your updated summary, produce a ===KNOWLEDGE_EXTRACT=== section (same rules as
initial compaction). Only extract NEW knowledge from the new messages — do not repeat
entries already in the anchor or in existing knowledge. If nothing new, write NOTHING_NEW.
===KNOWLEDGE_EXTRACT===
[entries or NOTHING_NEW]
===END_KNOWLEDGE_EXTRACT===
{existing_knowledge_context}
"""


def estimate_tokens(text: str) -> int:
    """Estimate token count from text using conservative ratio.
    
    Uses chars/3 based on empirical testing against Bedrock's tokenizer.
    This intentionally overestimates to prevent context overflow.
    
    Args:
        text: Input text string
    
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return len(text) // CHARS_PER_TOKEN + 1


def estimate_message_tokens(message: Any) -> int:
    """Estimate token count for a single message object.
    
    Args:
        message: Object with .content attribute (str)
    
    Returns:
        Estimated token count including role overhead
    """
    content = getattr(message, 'content', '')
    if not isinstance(content, str):
        content = str(content)
    # Add ~4 tokens overhead for role/formatting per message
    return estimate_tokens(content) + 4


def _extract_text_from_content(content: Any) -> str:
    """Extract plain text from message content, handling both string and list formats.
    
    LangChain messages can have content as:
    - str: "hello" — normal case
    - list: [{"type": "text", "text": "hello"}] — multi-block format from Claude/Bedrock
    - list: [] — empty content blocks
    
    This function normalizes all formats to a plain string.
    
    Args:
        content: Message content (str, list, or None)
    
    Returns:
        Extracted text as a string. Empty string if no text found.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Extract text from content blocks: [{"type": "text", "text": "..."}]
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "")
                if text and isinstance(text, str) and text.strip():
                    text_parts.append(text)
            elif isinstance(block, str):
                # Some LangChain versions use list of strings
                if block.strip():
                    text_parts.append(block)
        return "\n".join(text_parts)
    # Fallback: convert to string
    return str(content)


def _is_blank_message(msg: Any) -> bool:
    """Check if a message has blank/empty/None content.
    
    Used internally by sanitize_history and other functions to identify
    messages that would cause Bedrock 400 errors.
    
    Handles all content formats:
    - None → blank
    - "" → blank
    - "   " (whitespace only) → blank
    - [] (empty list) → blank
    - [{"type": "text", "text": ""}] (list with empty text blocks) → blank
    - [{"type": "text", "text": "hello"}] (list with real text) → NOT blank
    - "hello" → NOT blank
    
    Args:
        msg: Message object with .content attribute
    
    Returns:
        True if the message content is blank/empty/None
    """
    content = getattr(msg, 'content', None)
    if content is None:
        return True
    # Handle list content (LangChain multi-block format from Claude/Bedrock)
    if isinstance(content, list):
        extracted = _extract_text_from_content(content)
        return not extracted.strip()
    # Handle string content
    return not str(content).strip()


def _normalize_message_content(msg: Any) -> Any:
    """Normalize message content from list format to string format.
    
    Bedrock/Claude sometimes returns content as a list of content blocks:
        [{"type": "text", "text": "hello"}]
    
    LangChain's ChatOpenAI (via LiteLLM) may pass these through as-is.
    When these list-format messages end up in chat_history and get sent
    back to Bedrock, they can cause 400 errors if any block has blank text.
    
    This function converts list content to a plain string, which is the
    format Bedrock always accepts.
    
    Args:
        msg: Message object with .content attribute
    
    Returns:
        The message with content normalized to string (modified in place)
    """
    content = getattr(msg, 'content', None)
    if isinstance(content, list):
        extracted = _extract_text_from_content(content)
        msg.content = extracted
        if extracted:
            print(f"[SANITIZE] Normalized list content to string ({len(content)} blocks -> {len(extracted)} chars)")
        else:
            print(f"[SANITIZE] Normalized empty list content to empty string")
    return msg


def sanitize_history(messages: List[Any]) -> List[Any]:
    """Remove messages with empty/blank content that would cause Bedrock 400 errors.
    
    Bedrock strictly rejects messages where the text field in ContentBlock is blank.
    This can happen when an agent returns an empty response (timeout, context
    overflow, etc.) and the empty string gets saved to chat history.
    
    This function:
    1. Normalizes list-format content to strings (prevents Bedrock content block issues)
    2. Filters out any messages with blank content after normalization
    
    Handles all content formats:
    - None content → removed
    - Empty string content ("") → removed
    - Whitespace-only content ("   ", "\\n", etc.) → removed
    - Empty list content ([]) → removed
    - List with only blank text blocks ([{"type": "text", "text": ""}]) → removed
    - List with real text ([{"type": "text", "text": "hello"}]) → normalized to string, kept
    
    Args:
        messages: List of message objects with .content attribute
    
    Returns:
        Filtered list with blank messages removed and list content normalized
    """
    sanitized = []
    for msg in messages:
        # Step 1: Normalize list content to string
        _normalize_message_content(msg)
        # Step 2: Filter out blank messages
        if not _is_blank_message(msg):
            sanitized.append(msg)
        else:
            print(f"[SANITIZE] Dropping blank {type(msg).__name__} from history")
    return sanitized


def truncate_message_content(message: Any, max_tokens: int = MAX_SINGLE_MESSAGE_TOKENS) -> Any:
    """Truncate a single message's content if it exceeds max_tokens.

    Uses head+tail preservation (deterministic). For Haiku-based intelligent
    compaction, use async_truncate_message_content instead.

    Args:
        message: Object with .content attribute (str)
        max_tokens: Maximum tokens allowed for this message

    Returns:
        The message (possibly with truncated content)
    """
    content = getattr(message, 'content', '')
    if not isinstance(content, str):
        content = str(content)

    msg_tokens = estimate_tokens(content)
    if msg_tokens <= max_tokens:
        return message

    max_chars = max_tokens * CHARS_PER_TOKEN
    from core.context_compactor import _fallback_truncate
    message.content = _fallback_truncate(content, max_chars, tool_name="history_safety_net")
    return message


async def async_truncate_message_content(
    message: Any, max_tokens: int = MAX_SINGLE_MESSAGE_TOKENS
) -> Any:
    """Truncate a single message using Haiku smart compaction.

    Preferred over truncate_message_content — uses LLM to intelligently
    compress while preserving all facts, paths, and values.
    Falls back to head+tail if Haiku is unavailable.

    Args:
        message: Object with .content attribute (str)
        max_tokens: Maximum tokens allowed for this message

    Returns:
        The message (possibly with compacted content)
    """
    content = getattr(message, 'content', '')
    if not isinstance(content, str):
        content = str(content)

    msg_tokens = estimate_tokens(content)
    if msg_tokens <= max_tokens:
        return message

    max_chars = max_tokens * CHARS_PER_TOKEN
    from core.context_compactor import async_smart_compact
    message.content = await async_smart_compact(
        content, max_chars=max_chars, context_type="tool_output"
    )
    return message


def truncate_oversized_messages(
    history: List[Any],
    max_tokens_per_message: int = MAX_SINGLE_MESSAGE_TOKENS,
) -> List[Any]:
    """Sync fallback: truncate oversized messages using head+tail.

    For Haiku-based compaction, use async_truncate_oversized_messages.

    Args:
        history: List of message objects with .content attribute
        max_tokens_per_message: Maximum tokens per individual message

    Returns:
        The history list with oversized messages truncated in place
    """
    for msg in history:
        truncate_message_content(msg, max_tokens=max_tokens_per_message)
    return history


async def async_truncate_oversized_messages(
    history: List[Any],
    max_tokens_per_message: int = MAX_SINGLE_MESSAGE_TOKENS,
) -> List[Any]:
    """Truncate oversized messages using Haiku smart compaction.

    Compacts all oversized messages in parallel for efficiency.

    Args:
        history: List of message objects with .content attribute
        max_tokens_per_message: Maximum tokens per individual message

    Returns:
        The history list with oversized messages compacted in place
    """
    import asyncio
    tasks = []
    for msg in history:
        if estimate_message_tokens(msg) > max_tokens_per_message:
            tasks.append(async_truncate_message_content(msg, max_tokens=max_tokens_per_message))
    if tasks:
        await asyncio.gather(*tasks)
    return history


def enforce_total_token_budget(
    history: List[Any],
    max_total_tokens: int = BEDROCK_HARD_TOKEN_LIMIT,
) -> List[Any]:
    """GUARANTEE that total tokens across all messages is under the hard limit.
    
    This is the FINAL safety net — called after all other trimming/truncation.
    It ensures the total token count never exceeds Bedrock's limit, regardless
    of how many messages there are or how large each one is.
    
    Strategy (in order):
    1. Calculate total tokens
    2. If under budget → return as-is
    3. If over budget → drop oldest messages first (keep newest)
    4. If still over with just 1 message → truncate that message's content
    
    This function ALWAYS returns a list whose total tokens < max_total_tokens.
    
    Args:
        history: List of message objects with .content attribute
        max_total_tokens: Hard ceiling for total tokens (default: 180K)
    
    Returns:
        Trimmed/truncated history list guaranteed to be under budget
    """
    if not history:
        return []
    
    # Step 1: Calculate total
    def total_tokens(msgs):
        return sum(estimate_message_tokens(m) for m in msgs)
    
    current_total = total_tokens(history)
    if current_total <= max_total_tokens:
        return history  # Already under budget
    
    # Step 2: Drop oldest messages one at a time until under budget
    # Always try to keep at least the last message (the current user query)
    result = list(history)  # shallow copy
    while len(result) > 1 and total_tokens(result) > max_total_tokens:
        result.pop(0)  # Drop oldest message
    
    # Step 3: If still over budget with remaining messages, truncate them
    # Start from oldest remaining, truncate progressively
    current_total = total_tokens(result)
    if current_total > max_total_tokens:
        # Calculate how much we need to cut
        overage = current_total - max_total_tokens
        
        # Truncate each message proportionally, oldest first
        for msg in result:
            msg_tokens = estimate_message_tokens(msg)
            # Each message gets a fair share of the budget
            per_msg_budget = max_total_tokens // len(result)
            if msg_tokens > per_msg_budget:
                truncate_message_content(msg, max_tokens=per_msg_budget)
        
        # Final check — if STILL over (shouldn't happen but safety net)
        current_total = total_tokens(result)
        if current_total > max_total_tokens and len(result) == 1:
            # Last resort: hard truncate the single remaining message
            truncate_message_content(result[0], max_tokens=max_total_tokens - 10)
    
    return result


def trim_history_by_tokens(
    history: List[Any],
    max_tokens: int = DEFAULT_AGENT_TOKEN_BUDGET,
    keep_last_n: int = 2,
) -> Tuple[List[Any], int]:
    """Trim history to fit within a token budget.
    
    Keeps the most recent messages, dropping oldest first.
    Always keeps at least keep_last_n messages (the current exchange).
    
    Args:
        history: List of message objects with .content attribute
        max_tokens: Maximum total tokens allowed
        keep_last_n: Minimum messages to always keep (default 2 = last Q&A)
    
    Returns:
        Tuple of (trimmed_history, estimated_total_tokens)
    """
    if not history:
        return [], 0
    
    # Calculate tokens for each message (from newest to oldest)
    message_tokens = []
    for msg in reversed(history):
        tokens = estimate_message_tokens(msg)
        message_tokens.append(tokens)
    
    # Build trimmed list from newest, stopping when budget exceeded
    trimmed_indices = []  # indices from the original list
    running_total = 0
    
    for i, tokens in enumerate(message_tokens):
        original_idx = len(history) - 1 - i
        
        # Always keep the minimum required messages
        if i < keep_last_n:
            trimmed_indices.append(original_idx)
            running_total += tokens
            continue
        
        # Check if adding this message would exceed budget
        if running_total + tokens > max_tokens:
            break
        
        trimmed_indices.append(original_idx)
        running_total += tokens
    
    # Reverse to restore chronological order
    trimmed_indices.sort()
    trimmed = [history[i] for i in trimmed_indices]
    
    return trimmed, running_total


def history_to_text(
    history: List[Any],
    max_messages: int = 10,
    max_chars_per_message: int = None,
) -> str:
    """Convert message history to readable text.
    
    Takes the most recent max_messages and formats them.
    Optionally truncates individual message content to prevent
    bloated prompts (e.g. when used for supervisor routing context).
    
    Blank messages (empty/None/whitespace-only content) are automatically
    filtered out before conversion to prevent Bedrock 400 errors when
    the resulting text is used in prompts.
    
    Args:
        history: List of message objects with .type and .content attributes
        max_messages: Maximum number of recent messages to include
        max_chars_per_message: If set, truncate each message's content to
            this many characters. None means no truncation (full content).
            Recommended: 500 for supervisor context, None for agent context.
    
    Returns:
        Formatted string of conversation history
    """
    if not history:
        return "(No previous conversation)"
    recent = history[-max_messages:]
    lines = []
    for msg in recent:
        # Skip blank messages — they would produce "Assistant: " lines
        # with no content, and if this text is later parsed or sent to
        # Bedrock, it can trigger blank ContentBlock errors.
        if _is_blank_message(msg):
            continue
        role = "User" if msg.type == "human" else "Assistant"
        content = getattr(msg, 'content', '')
        # Normalize list content to string for text conversion
        if isinstance(content, list):
            content = _extract_text_from_content(content)
        if max_chars_per_message and len(content) > max_chars_per_message:
            content = content[:max_chars_per_message] + "... [truncated]"
        lines.append(f"{role}: {content}")
    if not lines:
        return "(No previous conversation)"
    return "\n".join(lines)


def history_to_text_with_budget(
    history: List[Any],
    max_tokens: int = DEFAULT_AGENT_TOKEN_BUDGET,
    max_messages: int = 30,
) -> str:
    """Convert message history to text, respecting a token budget.
    
    Combines message count limit with token budget limit.
    First caps by message count, then trims by tokens.
    
    Blank messages (empty/None/whitespace-only content) are automatically
    filtered out before conversion.
    
    Args:
        history: List of message objects with .type and .content attributes
        max_tokens: Maximum total tokens for the output text
        max_messages: Maximum number of messages to consider
    
    Returns:
        Formatted string of conversation history within token budget
    """
    if not history:
        return "(No previous conversation)"
    
    # First cap by message count
    recent = history[-max_messages:]
    
    # Then trim by token budget
    trimmed, total_tokens = trim_history_by_tokens(recent, max_tokens=max_tokens)
    
    if not trimmed:
        return "(No previous conversation)"
    
    # Filter blank messages during text conversion
    lines = []
    for msg in trimmed:
        if _is_blank_message(msg):
            continue
        content = getattr(msg, 'content', '')
        # Normalize list content to string for text conversion
        if isinstance(content, list):
            content = _extract_text_from_content(content)
        lines.append(
            f"{('User' if msg.type == 'human' else 'Assistant')}: {content}"
        )
    
    if not lines:
        return "(No previous conversation)"
    
    return "\n".join(lines)


# ── Sliding Window + Fact Extraction ────────────────────────────────────
# Priority 1: Keep last N turns raw, extract structured facts from older
# turns. This reduces token cost while preserving recent detail and
# long-term continuity.

# Patterns used to extract facts from message content.
# These are intentionally broad to catch common patterns in technical
# conversations without requiring NLP.
_FILE_PATH_PATTERN = re.compile(
    r'(?:^|\s)(/[a-zA-Z0-9_./-]{5,200})(?:\s|$|[,;:\)\]\}])',
    re.MULTILINE,
)


def extract_conversation_facts(
    history: List[Any],
    max_facts: int = MAX_FACT_ENTRIES,
) -> Dict[str, Any]:
    """Extract structured facts from conversation history.
    
    Scans all messages and extracts key factual information that the LLM
    needs for continuity, without keeping the full verbose conversation.
    
    This is NOT a summarization (which would require an LLM call). It is
    a deterministic, regex/heuristic-based extraction of structured data.
    
    Facts extracted:
    - files_mentioned: Unique file paths referenced in the conversation
    - errors_encountered: Error messages that appeared
    - tools_used: Tool names that were mentioned in assistant responses
    - decisions_made: Key decisions (detected by heuristic phrases)
    - current_task: The most recent user request (last human message)
    - topics: Unique topic keywords detected
    
    This is a pure function with no side effects.
    
    Args:
        history: List of message objects with .type and .content attributes.
                 Can be empty.
        max_facts: Maximum number of entries per fact category to prevent
                   unbounded growth. Default: MAX_FACT_ENTRIES (30).
    
    Returns:
        Dict with structured fact categories. All values are lists of
        strings or a single string. Never None.
    
    Raises:
        ValueError: If max_facts < 1.
    """
    if max_facts < 1:
        raise ValueError("max_facts must be >= 1")
    
    files_mentioned = set()
    errors_encountered = []
    tools_used = set()
    decisions_made = []
    current_task = ""
    topics = set()
    
    # Decision indicator phrases — when these appear in assistant messages,
    # the surrounding sentence is likely a decision or conclusion.
    decision_phrases = [
        "we decided", "decision:", "we'll use", "we will use",
        "the approach is", "going with", "chosen approach",
        "let's go with", "i recommend", "the plan is",
        "implemented", "committed", "completed",
    ]
    
    # Error indicator phrases
    error_phrases = [
        "error:", "failed:", "exception:", "traceback",
        "permission denied", "not found", "does not exist",
        "syntax error", "typeerror", "valueerror", "keyerror",
        "importerror", "modulenotfounderror",
    ]
    
    # Topic keywords — broad categories relevant to HPC/bio work
    topic_keywords = [
        "alphafold", "protein", "fasta", "slurm", "gpu", "conda",
        "docker", "singularity", "python", "git", "commit",
        "hallucination", "history", "agent", "tool", "prompt",
        "vcf", "mutation", "structure", "prediction",
    ]
    
    for msg in history:
        if _is_blank_message(msg):
            continue
        
        content = getattr(msg, 'content', '')
        if isinstance(content, list):
            content = _extract_text_from_content(content)
        if not isinstance(content, str):
            content = str(content)
        
        content_lower = content.lower()
        msg_type = getattr(msg, 'type', 'unknown')
        
        # Track the most recent user request
        if msg_type == 'human':
            current_task = content[:300]  # Cap at 300 chars
        
        # Extract file paths
        for match in _FILE_PATH_PATTERN.finditer(content):
            path = match.group(1).rstrip('.,;:)')
            # Filter out common false positives
            if not path.startswith('//'):
                files_mentioned.add(path)
        
        # Extract errors (from any message type)
        for phrase in error_phrases:
            if phrase in content_lower:
                # Extract the line containing the error
                for line in content.split('\n'):
                    if phrase in line.lower():
                        error_line = line.strip()[:200]
                        if error_line and error_line not in errors_encountered:
                            errors_encountered.append(error_line)
                        break
                break  # Only one error per message
        
        # Extract decisions (from assistant messages)
        if msg_type == 'ai':
            for phrase in decision_phrases:
                if phrase in content_lower:
                    # Find the sentence containing the decision phrase
                    idx = content_lower.find(phrase)
                    # Get surrounding context (up to 200 chars)
                    start = max(0, content.rfind('\n', 0, idx) + 1)
                    end = content.find('\n', idx)
                    if end == -1:
                        end = min(len(content), idx + 200)
                    decision_line = content[start:end].strip()[:200]
                    if decision_line and decision_line not in decisions_made:
                        decisions_made.append(decision_line)
                    break  # Only one decision per message
        
        # Extract tool names (from assistant messages mentioning tool calls)
        if msg_type == 'ai':
            # Common patterns: "called tool_name", "using tool_name", tool results
            tool_patterns = [
                r'(?:called|using|ran|executed|invoked)\s+([a-z_]{3,40})',
                r'tool[:\s]+([a-z_]{3,40})',
            ]
            for pattern in tool_patterns:
                for match in re.finditer(pattern, content_lower):
                    tool_name = match.group(1)
                    # Filter out common English words that match the pattern
                    if tool_name not in ('the', 'this', 'that', 'with', 'from', 'into', 'your'):
                        tools_used.add(tool_name)
        
        # Extract topics
        for keyword in topic_keywords:
            if keyword in content_lower:
                topics.add(keyword)
    
    # Cap all lists to max_facts
    return {
        "files_mentioned": sorted(files_mentioned)[:max_facts],
        "errors_encountered": errors_encountered[:max_facts],
        "tools_used": sorted(tools_used)[:max_facts],
        "decisions_made": decisions_made[:max_facts],
        "current_task": current_task,
        "topics": sorted(topics)[:max_facts],
    }


def format_facts_as_text(facts: Dict[str, Any]) -> str:
    """Format extracted facts into a concise text block for the LLM.
    
    Produces a structured, dense text block that gives the LLM continuity
    about the conversation without the full verbose history. This is
    injected before the recent raw messages in the sliding window.
    
    This is a pure function with no side effects.
    
    Args:
        facts: Dict returned by extract_conversation_facts().
               Must contain all expected keys.
    
    Returns:
        Formatted text string. Returns empty string if facts contain
        no meaningful content (all lists empty, no current_task).
    """
    sections = []
    
    if facts.get("current_task"):
        sections.append(f"Current task: {facts['current_task']}")
    
    if facts.get("topics"):
        sections.append(f"Topics discussed: {', '.join(facts['topics'])}")
    
    if facts.get("files_mentioned"):
        # Show at most 10 files to keep it concise
        files = facts["files_mentioned"][:10]
        sections.append(f"Files referenced: {', '.join(files)}")
    
    if facts.get("decisions_made"):
        decisions = facts["decisions_made"][:5]
        decision_lines = "\n".join(f"  - {d}" for d in decisions)
        sections.append(f"Key decisions:\n{decision_lines}")
    
    if facts.get("errors_encountered"):
        errors = facts["errors_encountered"][:5]
        error_lines = "\n".join(f"  - {e}" for e in errors)
        sections.append(f"Errors encountered:\n{error_lines}")
    
    if facts.get("tools_used"):
        sections.append(f"Tools used: {', '.join(facts['tools_used'][:10])}")
    
    if not sections:
        return ""
    
    return "CONVERSATION FACTS (from earlier messages):\n" + "\n".join(sections)


# ── LLM-based Conversation Compaction ──────────────────────────────────
# Uses the main model (Sonnet/Opus) to produce a comprehensive summary of
# older conversation messages. Claude Code style: send everything to the
# same model running the conversation, produce a 2000-5000 word summary
# that preserves all critical information.


def _prepare_older_messages_for_summary(
    older_messages: List[Any],
    max_tokens: int = LLM_SUMMARY_TOKEN_BUDGET,
) -> str:
    """Prepare older messages as text for LLM summarization.

    Converts ALL older messages to text for comprehensive summarization.
    The compaction model (Sonnet/Opus) can handle 200K context, so we send
    everything. Only applies a safety cap for extreme edge cases where
    messages would exceed the model's context window.

    Args:
        older_messages: List of message objects from before the sliding window.
        max_tokens: Safety ceiling (default 150K leaves room for system prompt + output).

    Returns:
        Formatted text of ALL older messages.
        Returns empty string if no messages.
    """
    if not older_messages:
        return ""

    text = history_to_text(older_messages, max_messages=len(older_messages))
    if text == "(No previous conversation)":
        return ""

    # Safety cap for extreme cases only (>150K tokens = ~450K chars)
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) > max_chars:
        # Keep BOTH beginning and end for maximum context preservation
        head_chars = int(max_chars * 0.3)
        tail_chars = int(max_chars * 0.65)
        omitted = len(text) - head_chars - tail_chars
        text = (
            text[:head_chars]
            + f"\n\n[...{omitted:,} characters from middle of conversation omitted "
            f"due to model context limits...]\n\n"
            + text[-tail_chars:]
        )
        logger.warning(
            f"[COMPACT] Older messages exceeded safety cap: {len(text):,} chars "
            f"trimmed with head(30%)+tail(65%) split"
        )

    return text


def summarize_facts_with_llm(
    older_messages: List[Any],
    llm: Any,
    max_input_tokens: int = LLM_SUMMARY_TOKEN_BUDGET,
) -> Optional[str]:
    """Summarize older conversation messages using the main LLM.

    Sends the ENTIRE older conversation to the main model (Sonnet/Opus)
    for comprehensive summarization. Produces a 2000-5000 word summary
    that preserves all critical information.

    Falls back to None on any error — caller should use format_facts_as_text
    as the fallback.

    This function makes a SYNCHRONOUS LLM call. For async contexts, use
    async_summarize_facts_with_llm() instead.

    Args:
        older_messages: List of message objects from before the sliding window.
        llm: Main LLM instance (Sonnet/Opus). Must support .invoke().
        max_input_tokens: Safety ceiling for input text (default 150K).

    Returns:
        Comprehensive summary string, or None if summarization failed.
    """
    if not older_messages or llm is None:
        return None

    older_text = _prepare_older_messages_for_summary(
        older_messages, max_tokens=max_input_tokens
    )
    if not older_text:
        return None

    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Summarize this entire conversation comprehensively. "
                f"This summary will REPLACE the original messages — preserve everything important.\n\n"
                f"CONVERSATION TO SUMMARIZE ({len(older_messages)} messages, "
                f"{len(older_text):,} chars):\n\n{older_text}"
            )),
        ]

        response = llm.invoke(messages)
        summary = getattr(response, 'content', '')
        if isinstance(summary, list):
            summary = _extract_text_from_content(summary)

        summary = summary.strip()
        if not summary:
            logger.warning("[HISTORY_SUMMARY] LLM returned empty summary, falling back to regex")
            return None

        logger.info(
            f"[HISTORY_SUMMARY] Comprehensive summary: {len(summary)} chars "
            f"({len(summary.split())} words) from {len(older_messages)} older messages"
        )
        return f"CONVERSATION SUMMARY (from earlier messages):\n{summary}"

    except Exception as e:
        logger.warning(f"[HISTORY_SUMMARY] LLM summarization failed: {e}, falling back to regex")
        return None


async def async_summarize_facts_with_llm(
    older_messages: List[Any],
    llm: Any,
    max_input_tokens: int = LLM_SUMMARY_TOKEN_BUDGET,
    existing_knowledge: str = "",
) -> Optional[str]:
    """Summarize older conversation messages using the main LLM (async).

    Sends the ENTIRE older conversation to the main model (Sonnet/Opus)
    for comprehensive summarization. Produces a 2000-5000 word summary
    that preserves all critical information.

    Falls back to None on any error — caller should use regex fallback.

    Args:
        older_messages: List of message objects from before the sliding window.
        llm: Main LLM instance (Sonnet/Opus). Must support .ainvoke().
        max_input_tokens: Safety ceiling for input text (default 150K).
        existing_knowledge: Current knowledge.md content for dedup in extraction.

    Returns:
        Comprehensive summary string, or None if summarization failed.
    """
    if not older_messages or llm is None:
        return None

    older_text = _prepare_older_messages_for_summary(
        older_messages, max_tokens=max_input_tokens
    )
    if not older_text:
        return None

    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        # Substitute existing knowledge context for dedup in KNOWLEDGE_EXTRACT
        system_prompt = SUMMARY_SYSTEM_PROMPT.replace(
            "{existing_knowledge_context}",
            (
                f"\nEXISTING KNOWLEDGE (already on disk — do NOT duplicate these):\n"
                f"{existing_knowledge[-4000:]}\n"
            ) if existing_knowledge else "\n(No existing knowledge on disk yet)\n"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Summarize this entire conversation comprehensively. "
                f"This summary will REPLACE the original messages — preserve everything important.\n\n"
                f"CONVERSATION TO SUMMARIZE ({len(older_messages)} messages, "
                f"{len(older_text):,} chars):\n\n{older_text}"
            )),
        ]

        response = await llm.ainvoke(messages)
        summary = getattr(response, 'content', '')
        if isinstance(summary, list):
            summary = _extract_text_from_content(summary)

        summary = summary.strip()
        if not summary:
            logger.warning("[HISTORY_SUMMARY] Async LLM returned empty summary, falling back to regex")
            return None

        logger.info(
            f"[HISTORY_SUMMARY] Comprehensive summary: {len(summary)} chars "
            f"({len(summary.split())} words) from {len(older_messages)} older messages"
        )
        return f"CONVERSATION SUMMARY (from earlier messages):\n{summary}"

    except Exception as e:
        logger.warning(f"[HISTORY_SUMMARY] Async LLM summarization failed: {e}, falling back to regex")
        return None


def build_sliding_window_context(
    history: List[Any],
    recent_window: int = SLIDING_WINDOW_SIZE,
    max_facts: int = MAX_FACT_ENTRIES,
    max_tokens: int = DEFAULT_AGENT_TOKEN_BUDGET // 3,
    llm: Any = None,
) -> Tuple[str, str]:
    """Build context using sliding window + fact extraction, returning parts separately.
    
    This is the main entry point for Priority 1 history management.
    It replaces the old approach of sending all history as text.
    
    Strategy:
    1. Split history into "older" (beyond window) and "recent" (within window)
    2. Extract structured facts from older messages:
       - If llm is provided: use LLM (Haiku) for rich structured summary
       - If llm is None or LLM fails: fall back to regex-based extraction
    3. Format recent messages as full text
    4. Return (summary_text, recent_text) as separate strings
    
    Returns a TUPLE of (summary_text, recent_text) so callers can use each
    part independently — e.g. the skill selector needs only the summary,
    while the agent needs both combined. This eliminates fragile string
    parsing to separate the parts after joining.
    
    This is a pure function with no side effects (except the optional LLM call).
    
    Args:
        history: Full conversation history (list of message objects).
                 Can be empty.
        recent_window: Number of recent messages to keep as full text.
                       Default: SLIDING_WINDOW_SIZE (6).
        max_facts: Maximum fact entries per category.
                   Default: MAX_FACT_ENTRIES (30).
        max_tokens: Token budget for the entire context string.
                    Default: DEFAULT_AGENT_TOKEN_BUDGET // 3.
        llm: Optional LLM instance for summarizing older messages.
             When provided, uses LLM-based summarization (e.g. Haiku)
             instead of regex-based fact extraction. Falls back to regex
             if the LLM call fails. Default: None (regex only).
    
    Returns:
        Tuple of (summary_text, recent_text).
        - summary_text: Structured facts/summary from older messages.
          Empty string if no older messages exist.
        - recent_text: Full text of recent messages within the sliding window.
          Empty string if no recent messages.
        Both are empty strings if history is empty.
    """
    if not history:
        return ("", "")

    # Token-based split: keep newest messages within budget, summarize rest
    split_idx = len(history)
    running_tokens = 0
    for i in range(len(history) - 1, -1, -1):
        msg_tokens = estimate_message_tokens(history[i])
        if running_tokens + msg_tokens > SLIDING_WINDOW_TOKEN_BUDGET and split_idx < len(history):
            break
        running_tokens += msg_tokens
        split_idx = i

    split_idx = min(split_idx, len(history) - recent_window)
    # Fallback: if all messages fit in token budget but there are enough
    # messages to split by count, use message-count split
    if split_idx <= 0 and len(history) > recent_window:
        split_idx = len(history) - recent_window
    if split_idx <= 0:
        older_messages = []
        recent_messages = history
    else:
        older_messages = history[:split_idx]
        recent_messages = history[split_idx:]

    # Part 1: Fact summary from older messages (if any)
    summary_text = ""
    if older_messages:
        facts_text = None

        # Try LLM-based summarization first (if LLM provided)
        if llm is not None:
            facts_text = summarize_facts_with_llm(
                older_messages, llm=llm,
                max_input_tokens=LLM_SUMMARY_TOKEN_BUDGET,
            )

        # Fall back to regex-based extraction if LLM not available or failed
        if facts_text is None:
            facts = extract_conversation_facts(older_messages, max_facts=max_facts)
            facts_text = format_facts_as_text(facts)

        if facts_text:
            summary_text = facts_text

    # Part 2: Recent messages as full text
    recent_text = history_to_text(recent_messages, max_messages=len(recent_messages))
    if recent_text == "(No previous conversation)":
        recent_text = ""

    # Enforce token budget on the combined text
    combined = ""
    parts = [p for p in [summary_text, recent_text] if p]
    if parts:
        combined = "\n\n".join(parts)
    
    if not combined:
        return ("", "")
    
    combined_tokens = estimate_tokens(combined)
    if combined_tokens > max_tokens:
        # Truncate from the beginning (facts are less important than recent)
        max_chars = max_tokens * CHARS_PER_TOKEN
        truncation_notice = "[Earlier context truncated for token budget]\n\n"
        available_chars = max_chars - len(truncation_notice)
        if available_chars > 100:
            combined = truncation_notice + combined[-available_chars:]
        else:
            combined = combined[-max_chars:]
        # After truncation, we can't cleanly separate summary from recent,
        # so put everything in recent_text and clear summary_text
        return ("", combined)
    
    return (summary_text, recent_text)


def _detect_compacted_summary(history: List[Any]) -> Optional[str]:
    """Check if history[0] is an already-compacted summary message.

    After full compaction, the first message is a HumanMessage starting with
    the COMPACTED_SUMMARY_MARKER. If present, returns its content (minus the
    marker) so the sliding window can skip Haiku summarization.
    """
    if not history:
        return None

    first_msg = history[0]
    content = getattr(first_msg, 'content', '')
    if not isinstance(content, str):
        return None

    if content.startswith(COMPACTED_SUMMARY_MARKER):
        summary = content[len(COMPACTED_SUMMARY_MARKER):].lstrip('\n')
        if summary:
            return summary

    return None


async def async_incremental_summary(
    previous_summary: str,
    new_messages: List[Any],
    llm: Any,
    existing_knowledge: str = "",
) -> Optional[str]:
    """Incrementally update a conversation summary with new messages.

    Instead of re-summarizing ALL older messages, takes the previous summary
    and only processes messages that rotated out of the sliding window.
    Falls back to None on error — caller should do full re-summarization.
    """
    if not previous_summary or not new_messages or llm is None:
        return None

    new_text = history_to_text(new_messages, max_messages=len(new_messages))
    if new_text == "(No previous conversation)":
        return None

    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        system_prompt = INCREMENTAL_SUMMARY_SYSTEM_PROMPT.replace(
            "{existing_knowledge_context}",
            (
                f"\nEXISTING KNOWLEDGE (already on disk — do NOT duplicate these):\n"
                f"{existing_knowledge[-4000:]}\n"
            ) if existing_knowledge else "\n(No existing knowledge on disk yet)\n"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"PREVIOUS SUMMARY:\n{previous_summary}\n\n"
                f"NEW MESSAGES:\n{new_text}\n\n"
                f"Produce the UPDATED summary:"
            )),
        ]

        response = await llm.ainvoke(messages)
        summary = getattr(response, 'content', '')
        if isinstance(summary, list):
            summary = _extract_text_from_content(summary)

        summary = summary.strip()
        if not summary:
            logger.warning("[INCREMENTAL_SUMMARY] LLM returned empty — falling back to full")
            return None

        logger.info(
            f"[INCREMENTAL_SUMMARY] Updated summary: {len(summary)} chars "
            f"(+{len(new_messages)} new msgs)"
        )
        return f"CONVERSATION SUMMARY (from earlier messages):\n{summary}"

    except Exception as e:
        logger.warning(f"[INCREMENTAL_SUMMARY] Failed: {e} — falling back to full")
        return None


async def async_build_sliding_window_context(
    history: List[Any],
    recent_window: int = SLIDING_WINDOW_SIZE,
    max_facts: int = MAX_FACT_ENTRIES,
    max_tokens: int = DEFAULT_AGENT_TOKEN_BUDGET // 3,
    llm: Any = None,
    cached_summary: Optional[str] = None,
    cached_summary_msg_count: Optional[int] = None,
) -> Tuple[str, str]:
    """Async version of build_sliding_window_context.

    Uses async LLM summarization for non-blocking operation in async
    contexts (e.g. Chainlit message handlers). Falls back to regex-based
    extraction if LLM is not provided or fails.

    Optimizations:
    - Detects already-compacted summaries and reuses them (no Haiku call)
    - Supports incremental summarization via cached_summary/cached_summary_msg_count

    Args:
        history: Full conversation history (list of message objects).
        recent_window: Number of recent messages to keep as full text.
        max_facts: Maximum fact entries per category.
        max_tokens: Token budget for the entire context string.
        llm: Optional LLM instance for async summarization.
        cached_summary: Previous turn's conversation_summary (for incremental updates).
        cached_summary_msg_count: Count of older messages when cached_summary was built.

    Returns:
        Tuple of (summary_text, recent_text).
        - summary_text: Structured facts/summary from older messages.
          Empty string if no older messages exist.
        - recent_text: Full text of recent messages within the sliding window.
          Empty string if no recent messages.
    """
    if not history:
        return ("", "")

    # ── OPTIMIZATION: Detect existing compacted summary ──────────────────
    existing_summary = _detect_compacted_summary(history)
    if existing_summary:
        if len(history) <= recent_window:
            recent_messages = history
        else:
            recent_messages = history[-recent_window:]

        recent_text = history_to_text(recent_messages, max_messages=len(recent_messages))
        if recent_text == "(No previous conversation)":
            recent_text = ""

        logger.debug(
            f"[SLIDING_WINDOW] Reusing compacted summary ({len(existing_summary)} chars), "
            f"skipping Haiku summarization"
        )

        combined_tokens = estimate_tokens(existing_summary) + estimate_tokens(recent_text)
        if combined_tokens > max_tokens:
            max_summary_chars = (max_tokens - estimate_tokens(recent_text)) * CHARS_PER_TOKEN
            if max_summary_chars > 200:
                existing_summary = existing_summary[:max_summary_chars] + "\n[...summary truncated for budget]"
            else:
                existing_summary = ""

        return (existing_summary, recent_text)

    # ── Token-based split into older and recent ─────────────────────────────
    # Keep newest messages within SLIDING_WINDOW_TOKEN_BUDGET, summarize rest
    split_idx = len(history)
    running_tokens = 0
    for i in range(len(history) - 1, -1, -1):
        msg_tokens = estimate_message_tokens(history[i])
        if running_tokens + msg_tokens > SLIDING_WINDOW_TOKEN_BUDGET and split_idx < len(history):
            break
        running_tokens += msg_tokens
        split_idx = i

    split_idx = min(split_idx, len(history) - recent_window)
    # Fallback: if all messages fit in token budget but enough to split by count
    if split_idx <= 0 and len(history) > recent_window:
        split_idx = len(history) - recent_window
    if split_idx <= 0:
        older_messages = []
        recent_messages = history
    else:
        older_messages = history[:split_idx]
        recent_messages = history[split_idx:]

    # Part 1: Fact summary from older messages (if any)
    summary_text = ""
    if older_messages:
        facts_text = None
        older_count = len(older_messages)

        # ── INCREMENTAL SUMMARY: reuse cache if delta is small ────────────
        if (
            cached_summary
            and cached_summary_msg_count is not None
            and older_count > cached_summary_msg_count
            and (older_count - cached_summary_msg_count) <= INCREMENTAL_SUMMARY_MAX_DELTA
            and llm is not None
        ):
            delta = older_count - cached_summary_msg_count
            new_older_messages = older_messages[-delta:]
            facts_text = await async_incremental_summary(
                previous_summary=cached_summary,
                new_messages=new_older_messages,
                llm=llm,
            )
            if facts_text:
                logger.debug(
                    f"[SLIDING_WINDOW] Incremental summary: +{delta} msgs, "
                    f"reused cache from {cached_summary_msg_count} msgs"
                )

        # ── FULL SUMMARY: fallback or first time ─────────────────────────
        if facts_text is None:
            older_tokens = sum(estimate_message_tokens(m) for m in older_messages)
            if llm is not None and older_tokens > 2000:
                facts_text = await async_summarize_facts_with_llm(
                    older_messages, llm=llm,
                    max_input_tokens=LLM_SUMMARY_TOKEN_BUDGET,
                )

            # Fall back to regex-based extraction
            if facts_text is None:
                facts = extract_conversation_facts(older_messages, max_facts=max_facts)
                facts_text = format_facts_as_text(facts)

        if facts_text:
            summary_text = facts_text

    # Part 2: Recent messages as full text
    recent_text = history_to_text(recent_messages, max_messages=len(recent_messages))
    if recent_text == "(No previous conversation)":
        recent_text = ""

    # Enforce token budget on the combined text
    combined = ""
    parts = [p for p in [summary_text, recent_text] if p]
    if parts:
        combined = "\n\n".join(parts)

    if not combined:
        return ("", "")

    combined_tokens = estimate_tokens(combined)
    if combined_tokens > max_tokens:
        max_chars = max_tokens * CHARS_PER_TOKEN
        truncation_notice = "[Earlier context truncated for token budget]\n\n"
        available_chars = max_chars - len(truncation_notice)
        if available_chars > 100:
            combined = truncation_notice + combined[-available_chars:]
        else:
            combined = combined[-max_chars:]
        return ("", combined)

    return (summary_text, recent_text)


# ── Phase 5: True History Compaction ────────────────────────────────────
# Replaces older messages in the chat_history message list with a single
# summary message. This bounds the actual message count sent to the LLM,
# preventing unbounded growth of the {chat_history} placeholder.
#
# Key difference from build_sliding_window_context():
# - build_sliding_window_context() produces TEXT strings for the {input} field
# - compact_history() produces a MESSAGE LIST for the {chat_history} field
#
# The compacted list is: [HumanMessage(summary), AIMessage(ack)] + recent_messages
# This maintains the human/AI alternation pattern that Bedrock requires.


def _estimate_history_tokens(history: List[Any]) -> int:
    """Estimate total tokens across all messages in a history list.
    
    Args:
        history: List of message objects with .content attribute.
    
    Returns:
        Total estimated tokens across all messages.
    """
    return sum(estimate_message_tokens(m) for m in history)


def compact_history(
    history: List[Any],
    token_threshold: int = COMPACTION_TOKEN_THRESHOLD,
    recent_window: int = SLIDING_WINDOW_SIZE,
    recent_token_budget: int = SLIDING_WINDOW_TOKEN_BUDGET,
    max_facts: int = MAX_FACT_ENTRIES,
    llm: Any = None,
) -> List[Any]:
    """Compact chat history by replacing older messages with a summary message.

    Split strategy: keep the most recent messages that fit within
    recent_token_budget (default 30K tokens). Summarize everything older
    using the main model (Sonnet/Opus) for high-quality preservation.

    Safety guarantees:
    - Short conversations (under threshold) are NEVER modified
    - If summarization fails completely, returns history unchanged
    - Recent messages are always preserved in full
    - Idempotent — calling on already-compacted history under threshold is a no-op

    Args:
        history: Full chat history as a list of message objects.
        token_threshold: Token count above which compaction triggers.
        recent_window: Minimum recent messages to keep (floor).
        recent_token_budget: Token budget for the recent raw portion.
        max_facts: Maximum fact entries for regex fallback.
        llm: Optional LLM instance for summarization.

    Returns:
        Compacted message list. Either the original list (if under threshold)
        or [summary_msg, ack_msg] + recent_messages.
    """
    if not history:
        return []

    # Step 1: Check if compaction is needed
    total_tokens = _estimate_history_tokens(history)
    if total_tokens <= token_threshold:
        logger.debug(
            f"[COMPACT] History {len(history)} msgs, ~{total_tokens} tokens "
            f"— under threshold ({token_threshold}), no compaction needed"
        )
        return history

    # Step 2: Token-based split — keep newest messages within budget
    if len(history) <= 2:
        return history

    split_idx = len(history)
    running_tokens = 0
    for i in range(len(history) - 1, -1, -1):
        msg_tokens = estimate_message_tokens(history[i])
        if running_tokens + msg_tokens > recent_token_budget and split_idx < len(history):
            break
        running_tokens += msg_tokens
        split_idx = i

    # Ensure at least recent_window messages kept
    split_idx = min(split_idx, len(history) - recent_window)
    # Fallback: if token budget didn't split but we exceed threshold,
    # use message-count split (many small messages scenario)
    if split_idx <= 0 and len(history) > recent_window:
        split_idx = len(history) - recent_window
    if split_idx <= 0:
        return history

    older_messages = history[:split_idx]
    recent_messages = history[split_idx:]

    older_tokens = _estimate_history_tokens(older_messages)
    recent_tokens = _estimate_history_tokens(recent_messages)

    logger.info(
        f"[COMPACT] Compacting: {len(history)} msgs (~{total_tokens} tokens) → "
        f"summarizing {len(older_messages)} older msgs (~{older_tokens} tokens), "
        f"keeping {len(recent_messages)} recent msgs (~{recent_tokens} tokens)"
    )
    
    # Step 3: Generate summary of older messages
    summary_text = None
    
    # Try LLM-based summarization first
    if llm is not None:
        summary_text = summarize_facts_with_llm(
            older_messages, llm=llm,
            max_input_tokens=LLM_SUMMARY_TOKEN_BUDGET,
        )
    
    # Fall back to regex-based extraction
    if summary_text is None:
        facts = extract_conversation_facts(older_messages, max_facts=max_facts)
        summary_text = format_facts_as_text(facts)
    
    # If we got no summary at all (very unlikely — would mean no facts at all),
    # return history unchanged rather than losing context
    if not summary_text:
        logger.warning(
            "[COMPACT] Could not generate any summary — returning history unchanged"
        )
        return history
    
    # Step 4: Build compacted message list
    # Lazy import to avoid requiring langchain at module level
    from langchain_core.messages import HumanMessage, AIMessage
    
    # The summary goes in a HumanMessage so the LLM sees it as provided context.
    # The AIMessage acknowledgment maintains the required alternation pattern.
    summary_msg = HumanMessage(
        content=f"[CONTEXT FROM EARLIER IN THIS CONVERSATION]\n{summary_text}"
    )
    ack_msg = AIMessage(
        content="Understood. I have the context from our earlier conversation. How can I help you continue?"
    )
    
    compacted = [summary_msg, ack_msg] + list(recent_messages)
    
    compacted_tokens = _estimate_history_tokens(compacted)
    logger.info(
        f"[COMPACT] Compaction complete: {len(history)} msgs (~{total_tokens} tokens) → "
        f"{len(compacted)} msgs (~{compacted_tokens} tokens). "
        f"Saved ~{total_tokens - compacted_tokens} tokens."
    )
    
    return compacted


async def async_compact_history(
    history: List[Any],
    token_threshold: int = COMPACTION_TOKEN_THRESHOLD,
    recent_window: int = SLIDING_WINDOW_SIZE,
    recent_token_budget: int = SLIDING_WINDOW_TOKEN_BUDGET,
    max_facts: int = MAX_FACT_ENTRIES,
    llm: Any = None,
    existing_knowledge: str = "",
) -> List[Any]:
    """Async version of compact_history.

    Uses async LLM summarization for non-blocking operation in async
    contexts (e.g. Chainlit message handlers). Falls back to regex-based
    extraction if LLM is not provided or fails.

    Split strategy: keep the most recent messages that fit within
    recent_token_budget (default 30K tokens). Summarize everything older.
    This handles IrisAI's pattern of few but large messages (tool outputs).

    Args:
        history: Full chat history as a list of message objects.
        token_threshold: Token count above which compaction triggers.
        recent_window: Minimum recent messages to keep (fallback floor).
        recent_token_budget: Token budget for the recent raw portion.
        max_facts: Maximum fact entries for regex fallback.
        llm: Optional LLM instance for async summarization.
        existing_knowledge: Current knowledge.md content for dedup in extraction.

    Returns:
        Compacted message list. Either the original list (if under threshold)
        or [summary_msg, ack_msg] + recent_messages.
    """
    if not history:
        return []

    # Step 1: Check if compaction is needed
    total_tokens = _estimate_history_tokens(history)
    if total_tokens <= token_threshold:
        logger.debug(
            f"[COMPACT] History {len(history)} msgs, ~{total_tokens} tokens "
            f"— under threshold ({token_threshold}), no compaction needed"
        )
        return history

    # Step 2: Token-based split — keep newest messages within budget
    if len(history) <= 2:
        return history

    # Walk backwards from newest, accumulating tokens until budget exhausted
    split_idx = len(history)
    running_tokens = 0
    for i in range(len(history) - 1, -1, -1):
        msg_tokens = estimate_message_tokens(history[i])
        if running_tokens + msg_tokens > recent_token_budget and split_idx < len(history):
            break
        running_tokens += msg_tokens
        split_idx = i

    # Ensure minimum recent messages (at least recent_window)
    split_idx = min(split_idx, len(history) - recent_window)
    # Fallback: if token budget didn't split but we exceed threshold,
    # use message-count split (many small messages scenario)
    if split_idx <= 0 and len(history) > recent_window:
        split_idx = len(history) - recent_window
    if split_idx <= 0:
        return history

    older_messages = history[:split_idx]
    recent_messages = history[split_idx:]

    older_tokens = _estimate_history_tokens(older_messages)
    recent_tokens = _estimate_history_tokens(recent_messages)
    
    logger.info(
        f"[COMPACT] Async compacting: {len(history)} msgs (~{total_tokens} tokens) → "
        f"summarizing {len(older_messages)} older msgs (~{older_tokens} tokens), "
        f"keeping {len(recent_messages)} recent msgs (~{recent_tokens} tokens)"
    )
    
    # Step 3: Extract existing anchor block (if this is a re-compaction)
    existing_anchor = ""
    if older_messages:
        first_content = getattr(older_messages[0], "content", "")
        if COMPACTED_SUMMARY_MARKER in first_content:
            existing_anchor = extract_anchor_block(first_content)
            if existing_anchor:
                logger.info(
                    f"[COMPACT] Found existing anchor block ({len(existing_anchor)} chars)"
                )

    # Step 4: Generate summary of older messages
    summary_text = None

    # Try async LLM-based summarization first
    if llm is not None:
        summary_text = await async_summarize_facts_with_llm(
            older_messages, llm=llm,
            max_input_tokens=LLM_SUMMARY_TOKEN_BUDGET,
            existing_knowledge=existing_knowledge,
        )

    # Fall back to regex-based extraction
    if summary_text is None:
        facts = extract_conversation_facts(older_messages, max_facts=max_facts)
        summary_text = format_facts_as_text(facts)

    # If we got no summary at all, return history unchanged
    if not summary_text:
        logger.warning(
            "[COMPACT] Could not generate any summary — returning history unchanged"
        )
        return history

    # Step 5: Anchor block management
    # The LLM is now instructed to produce the anchor block as part of its summary.
    # We verify it's present; if not (LLM failed to follow instructions), fall back
    # to the old keyword extraction.
    new_anchor = extract_anchor_block(summary_text)

    if existing_anchor:
        if new_anchor:
            # LLM produced an anchor — verify old entries survived
            if not verify_anchor_preserved(summary_text, existing_anchor):
                logger.warning(
                    "[COMPACT] Anchor verification failed — re-injecting old anchor"
                )
                summary_text = inject_anchor_block(summary_text, existing_anchor)
            else:
                logger.info(
                    f"[COMPACT] LLM-generated anchor verified — "
                    f"all old entries preserved ({len(new_anchor)} chars)"
                )
        else:
            # LLM didn't produce anchor — inject the old one
            logger.warning(
                "[COMPACT] LLM did not generate anchor block — re-injecting existing"
            )
            summary_text = inject_anchor_block(summary_text, existing_anchor)
    else:
        # First compaction
        if new_anchor and len(new_anchor) > 100:
            # LLM produced a good anchor — use it as-is
            logger.info(
                f"[COMPACT] LLM-generated initial anchor ({len(new_anchor)} chars)"
            )
        else:
            # Fallback: build anchor via keyword extraction (legacy path)
            initial_anchor = build_initial_anchor(summary_text)
            if initial_anchor and len(initial_anchor) > 50:
                summary_text = inject_anchor_block(summary_text, initial_anchor)
                logger.info(
                    f"[COMPACT] Fallback: built keyword-extracted anchor "
                    f"({len(initial_anchor)} chars)"
                )

    # Step 6: Build compacted message list
    from langchain_core.messages import HumanMessage, AIMessage

    summary_msg = HumanMessage(
        content=f"[CONTEXT FROM EARLIER IN THIS CONVERSATION]\n{summary_text}"
    )
    ack_msg = AIMessage(
        content="Understood. I have the context from our earlier conversation. How can I help you continue?"
    )

    compacted = [summary_msg, ack_msg] + list(recent_messages)

    compacted_tokens = _estimate_history_tokens(compacted)
    logger.info(
        f"[COMPACT] Async compaction complete: {len(history)} msgs (~{total_tokens} tokens) → "
        f"{len(compacted)} msgs (~{compacted_tokens} tokens). "
        f"Saved ~{total_tokens - compacted_tokens} tokens."
    )

    return compacted


# ── Anchored Compaction (Project-Scoped) ───────────────────────────────────

ANCHOR_HEADER = "## ANCHORED CONTEXT (DO NOT MODIFY — preserved verbatim)"
ANCHOR_PROJECT_PREFIX = "### Project:"
ANCHOR_SESSION_WIDE = "### Session-wide"


def extract_anchor_block(summary_text: str) -> str:
    """Extract the anchored context block from a compaction summary.

    Returns the full anchor block text (from ANCHOR_HEADER to the next
    non-anchor section), or empty string if no anchor block exists.
    """
    if ANCHOR_HEADER not in summary_text:
        return ""

    lines = summary_text.split("\n")
    anchor_lines = []
    in_anchor = False

    for line in lines:
        if line.strip() == ANCHOR_HEADER:
            in_anchor = True
            anchor_lines.append(line)
            continue

        if in_anchor:
            # Anchor block ends at the next ## heading that isn't part of anchor
            if line.startswith("## ") and ANCHOR_HEADER not in line:
                break
            anchor_lines.append(line)

    return "\n".join(anchor_lines) if anchor_lines else ""


def inject_anchor_block(summary_text: str, anchor_block: str) -> str:
    """Inject or replace the anchor block at the top of a summary.

    If the summary already has an anchor block, replaces it.
    Otherwise, prepends the anchor block.
    """
    if not anchor_block:
        return summary_text

    existing = extract_anchor_block(summary_text)
    if existing:
        return summary_text.replace(existing, anchor_block, 1)

    return anchor_block + "\n\n" + summary_text


def verify_anchor_preserved(new_summary: str, anchor_block: str) -> bool:
    """Verify that all critical anchor entries survived compaction.

    Checks that key identifier lines (those with concrete values like paths,
    IDs, decisions) are still present in the new summary.
    """
    if not anchor_block:
        return True

    for line in anchor_block.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Check lines that contain concrete values (have a colon with content)
        if line.startswith("- ") and ":" in line:
            # Extract the value portion after the colon
            value_part = line.split(":", 1)[1].strip()
            if value_part and len(value_part) > 3 and value_part not in ("none", "N/A", "..."):
                if value_part not in new_summary:
                    logger.warning(
                        f"[ANCHOR] Verification failed — missing: {value_part[:80]}"
                    )
                    return False

    return True


def build_initial_anchor(summary_text: str, active_project: str = "") -> str:
    """Build initial anchor block from first compaction summary.

    Extracts key facts (paths, IDs, decisions) from the summary sections
    and structures them into the anchor format.
    """
    lines = [ANCHOR_HEADER, ""]

    # Extract project-specific facts if we know the active project
    if active_project:
        lines.append(f"{ANCHOR_PROJECT_PREFIX} {active_project}")
    else:
        lines.append(f"{ANCHOR_PROJECT_PREFIX} (unknown)")

    # Extract working directory
    work_dir = _extract_fact(summary_text, "working directory", "Working directory")
    lines.append(f"- Working directory: {work_dir or 'N/A'}")

    # Extract key file paths (from File Paths & Artifacts section)
    artifacts = _extract_paths(summary_text)
    if artifacts:
        lines.append(f"- Key artifacts: {', '.join(artifacts[:5])}")
    else:
        lines.append("- Key artifacts: none yet")

    # Extract active jobs
    jobs = _extract_fact(summary_text, "job", "JOBID", "job_id", "Job ID")
    lines.append(f"- Active jobs: {jobs or 'none'}")

    # Extract decisions
    decisions = _extract_decisions(summary_text)
    if decisions:
        for i, d in enumerate(decisions[:5], 1):
            lines.append(f"- Decisions: ({i}) {d}")
    else:
        lines.append("- Decisions: none yet")

    # Extract errors
    errors = _extract_fact(summary_text, "unresolved", "pending", "error")
    lines.append(f"- Unresolved errors: {errors or 'none'}")

    # Session-wide
    lines.extend(["", ANCHOR_SESSION_WIDE])
    prefs = _extract_fact(summary_text, "preference", "user want", "user prefer")
    lines.append(f"- User preferences: {prefs or 'none stated'}")

    return "\n".join(lines)


def _extract_fact(text: str, *keywords: str) -> str:
    """Extract a fact from summary by finding lines containing keywords."""
    for line in text.split("\n"):
        line_lower = line.lower()
        for kw in keywords:
            if kw.lower() in line_lower and line.strip().startswith("- "):
                value = line.strip().lstrip("- ").strip()
                if ":" in value:
                    value = value.split(":", 1)[1].strip()
                return value[:200] if value else ""
    return ""


def _extract_paths(text: str) -> list:
    """Extract absolute file paths from summary text."""
    import re
    paths = re.findall(r'(/(?:home|scratch|data1|usersoftware|tmp)/\S+)', text)
    seen = set()
    unique = []
    for p in paths:
        p_clean = p.rstrip(".,;:)]}")
        if p_clean not in seen and len(p_clean) > 5:
            seen.add(p_clean)
            unique.append(p_clean)
    return unique[:10]


def _extract_decisions(text: str) -> list:
    """Extract decision statements from the Decisions section."""
    decisions = []
    in_decisions = False
    for line in text.split("\n"):
        if "decision" in line.lower() and line.startswith("#"):
            in_decisions = True
            continue
        if in_decisions:
            if line.startswith("#"):
                break
            stripped = line.strip()
            if stripped.startswith("- ") and len(stripped) > 10:
                decisions.append(stripped.lstrip("- ")[:150])
    return decisions


# ── Per-Turn Transcript Writer ─────────────────────────────────────────────

def write_turn_transcript(
    work_dir: str,
    turn_number: int,
    user_input: str,
    intermediate_steps: list,
    ai_response: str,
) -> str:
    """Write a per-turn markdown transcript to disk.

    Creates a human-readable .md file with full tool outputs — the LLM
    can read_text_file() on this path if it needs details from old turns
    after compaction. This is the insurance layer.

    Returns the path to the written file, or empty string on failure.
    """
    transcript_dir = Path(work_dir) / "dynamic_tasks" / ".turn_transcripts"
    try:
        transcript_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"[TRANSCRIPT] Failed to create directory: {e}")
        return ""

    filename = f"turn_{turn_number:03d}.md"
    filepath = transcript_dir / filename

    from datetime import datetime as _dt
    timestamp = _dt.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [f"# Turn {turn_number} — {timestamp}", ""]

    # User input
    lines.extend(["## User Input", user_input or "(empty)", ""])

    # Tool calls with full outputs
    if intermediate_steps:
        lines.append("## Tool Calls")
        for i, step in enumerate(intermediate_steps, 1):
            try:
                action = step[0]
                observation = str(step[1]) if len(step) > 1 else ""
                tool_name = getattr(action, "tool", "unknown")
                tool_input = getattr(action, "tool_input", {})

                if isinstance(tool_input, dict):
                    args_str = json.dumps(tool_input, ensure_ascii=False, default=str)
                    if len(args_str) > 500:
                        args_str = args_str[:500] + "..."
                else:
                    args_str = str(tool_input)[:500]

                lines.extend([
                    f"### [{i}] {tool_name}",
                    f"**Args:** `{args_str}`",
                    f"**Output** ({len(observation):,} chars):",
                    "```",
                    observation[:50000] if observation else "(no output)",
                    "```",
                    "",
                ])
            except (IndexError, TypeError, AttributeError):
                continue
    else:
        lines.extend(["## Tool Calls", "None", ""])

    # AI response
    lines.extend(["## AI Response", ai_response or "(empty)", ""])

    try:
        filepath.write_text("\n".join(lines), encoding="utf-8")
        logger.info(
            f"[TRANSCRIPT] Turn {turn_number} written: {filepath} "
            f"({len(intermediate_steps)} tools, {len(ai_response)} chars response)"
        )
        return str(filepath)
    except Exception as e:
        logger.warning(f"[TRANSCRIPT] Failed to write turn {turn_number}: {e}")
        return ""


# ── Tool Result Serialization ──────────────────────────────────────────────

TOOL_RESULTS_HEADER = "\n\n[TOOL CALLS THIS TURN]"
TOOL_RESULT_MAX_CHARS = 10000  # DEPRECATED — used only by serialize_tool_results (legacy)

# Archived output marker — used to detect disk-referenced tool outputs
ARCHIVED_MARKER = "┌─ ARCHIVED:"


TURN_TOOL_BUDGET = 40_000
MAX_OUTPUT_PER_TOOL = 12_000


def format_tool_call_record(
    intermediate_steps: list,
    max_output_per_tool: int = MAX_OUTPUT_PER_TOOL,
    total_budget: int = TURN_TOOL_BUDGET,
) -> str:
    """Format tool calls with full outputs (capped) for cross-turn history.

    Preserves tool output content so the LLM can reference prior-turn results
    without re-reading. Each output is capped at max_output_per_tool chars.
    If the total exceeds total_budget, oldest outputs are truncated first.
    """
    if not intermediate_steps:
        return ""

    records = []
    for step in intermediate_steps:
        try:
            action = step[0]
            observation = str(step[1]) if len(step) > 1 else ""

            tool_name = getattr(action, "tool", "unknown")
            tool_input = getattr(action, "tool_input", {})

            if isinstance(tool_input, dict):
                key_args = []
                for k, v in list(tool_input.items())[:3]:
                    v_str = str(v)
                    if len(v_str) > 60:
                        v_str = v_str[:60] + "..."
                    key_args.append(f'{k}="{v_str}"')
                args_str = ", ".join(key_args)
                if len(args_str) > 150:
                    args_str = args_str[:150] + "..."
            else:
                args_str = str(tool_input)[:150]

            obs_len = len(observation)
            is_error = (
                observation.lstrip().lower().startswith("error")
                or "traceback" in observation[:200].lower()
            )
            status = "error" if is_error else "success"

            capped_output = observation[:max_output_per_tool]
            if len(observation) > max_output_per_tool:
                capped_output += f"\n[...truncated, {obs_len:,} total chars]"

            records.append({
                "header": f"─ {tool_name}({args_str}) → {status} | {obs_len:,} chars",
                "output": capped_output,
            })
        except (IndexError, TypeError, AttributeError):
            continue

    if not records:
        return ""

    # Budget enforcement: if total exceeds budget, truncate oldest outputs
    total_size = sum(len(r["header"]) + len(r["output"]) + 2 for r in records)
    if total_size > total_budget:
        # Truncate from oldest first, keeping newest intact
        for i in range(len(records)):
            if total_size <= total_budget:
                break
            old_output_len = len(records[i]["output"])
            # Reduce to just first line + status
            first_line = ""
            for line in records[i]["output"].split("\n"):
                if line.strip():
                    first_line = line.strip()[:200]
                    break
            records[i]["output"] = first_line or "(output trimmed for budget)"
            total_size -= (old_output_len - len(records[i]["output"]))

    lines = [TOOL_RESULTS_HEADER]
    for r in records:
        lines.append(r["header"])
        lines.append(r["output"])
        lines.append("")  # blank line separator

    return "\n".join(lines)


def serialize_tool_results(intermediate_steps: list, max_chars_per_output: int = TOOL_RESULT_MAX_CHARS) -> str:
    """DEPRECATED — use format_tool_call_record() instead.

    Kept for backwards compatibility with tests that reference this function.
    Now delegates to format_tool_call_record() which produces compact records.
    """
    return format_tool_call_record(intermediate_steps)


# ── Progressive Context Cleanup (DEPRECATED) ──────────────────────────────
# DEPRECATED: Replaced by single-step comprehensive compaction using the main
# model. Kept for rollback safety — not called in production.
# Previously: multi-layer approach (truncate tool outputs -> remove tool sections).
# Now: single high-quality compaction when threshold is hit.

PROGRESSIVE_CLEANUP_TRIGGER = 80_000  # DEPRECATED
TOOL_OUTPUT_TRUNCATE_CHARS = 500  # DEPRECATED


def progressive_context_cleanup(
    history: List[Any],
    token_budget: int = DEFAULT_AGENT_TOKEN_BUDGET,
    llm: Any = None,
) -> List[Any]:
    """DEPRECATED: Not called in production. Kept for rollback safety.

    Previously: Progressive cleanup when history approaches token budget.
    Layer 1: Truncate tool outputs in oldest messages (keep tool name + first 500 chars)
    Layer 2: Remove [TOOL CALLS THIS TURN] sections from oldest messages entirely

    Replaced by single-step comprehensive compaction using the main model.
    """
    total_tokens = _estimate_history_tokens(history)

    if total_tokens <= token_budget:
        return history

    logger.info(
        f"[PROGRESSIVE_CLEANUP] History ~{total_tokens} tokens exceeds "
        f"budget {token_budget} — starting progressive cleanup"
    )

    # Work on a copy
    from langchain_core.messages import AIMessage
    cleaned = list(history)

    # Layer 1: Truncate tool outputs in older messages (oldest first)
    for i in range(len(cleaned)):
        if _estimate_history_tokens(cleaned) <= token_budget:
            break
        msg = cleaned[i]
        if not isinstance(msg, AIMessage):
            continue
        content = getattr(msg, 'content', '')
        if TOOL_RESULTS_HEADER not in content:
            continue

        # Split into response text and tool section
        parts = content.split(TOOL_RESULTS_HEADER, 1)
        if len(parts) != 2:
            continue

        response_text = parts[0]
        tool_section = parts[1]

        # Truncate each tool output to TOOL_OUTPUT_TRUNCATE_CHARS
        truncated_lines = []
        for line in tool_section.split("\n"):
            if line.startswith("  → ") and len(line) > TOOL_OUTPUT_TRUNCATE_CHARS:
                truncated_lines.append(line[:TOOL_OUTPUT_TRUNCATE_CHARS] + " [truncated]")
            else:
                truncated_lines.append(line)

        new_content = response_text + TOOL_RESULTS_HEADER + "\n".join(truncated_lines)
        cleaned[i] = AIMessage(content=new_content)

    tokens_after_l1 = _estimate_history_tokens(cleaned)
    if tokens_after_l1 <= token_budget:
        logger.info(f"[PROGRESSIVE_CLEANUP] Layer 1 sufficient: ~{tokens_after_l1} tokens")
        return cleaned

    # Layer 2: Remove tool sections entirely from oldest messages
    for i in range(len(cleaned)):
        if _estimate_history_tokens(cleaned) <= token_budget:
            break
        msg = cleaned[i]
        if not isinstance(msg, AIMessage):
            continue
        content = getattr(msg, 'content', '')
        if TOOL_RESULTS_HEADER not in content:
            continue

        # Keep only the response text
        response_text = content.split(TOOL_RESULTS_HEADER, 1)[0]
        cleaned[i] = AIMessage(content=response_text)

    tokens_after_l2 = _estimate_history_tokens(cleaned)
    logger.info(
        f"[PROGRESSIVE_CLEANUP] After Layer 2: ~{tokens_after_l2} tokens "
        f"(budget: {token_budget})"
    )

    return cleaned
