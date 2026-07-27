"""
core/session_facts.py — Session Facts: cumulative fact extraction from agent turns.

After each agent turn, Haiku extracts key facts from:
  1. intermediate_steps (tool outputs — source of truth for paths, numbers, errors)
  2. AI response text (decisions, conclusions, interpretations)

Facts accumulate across turns. When they grow past MAX_SESSION_FACTS_CHARS,
Haiku consolidates (deduplicates, supersedes old facts with newer ones).
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

MAX_SESSION_FACTS_CHARS = 8000
CONSOLIDATION_THRESHOLD = 10000
MAX_TURN_DATA_CHARS = 12000
MAX_AI_RESPONSE_CHARS = 3000
MAX_OBSERVATION_CHARS = 3000

FACT_EXTRACTION_SYSTEM = """\
You are a session facts extractor. Extract ALL key facts from agent turns. Output as a bullet list.

MUST PRESERVE:
- ALL file paths, directory paths (verbatim, full paths)
- ALL results: pass/fail counts, status, outcomes
- ALL commands executed and their results
- ALL error messages and their causes
- ALL decisions made and their reasons
- ALL names: tools used, files created/modified, services called
- ALL numeric values: counts, sizes, IDs, job numbers

Output ONLY bullet points (one fact per line, starting with •).
Keep each bullet to one line. No headers, no grouping.
If a fact from "Previous facts" is superseded by new information, output only the new version.

IMPORTANCE FLAG: If this turn contains a critical discovery, project milestone, significant error,
or important state change that should be saved to permanent memory immediately, append [PROMOTE_NOW]
on its own line at the very end of your response. Only use this for genuinely important facts
(new project created, job succeeded/failed, architecture decision, critical error diagnosed)."""

FACT_EXTRACTION_USER = """\
{previous_facts_section}

AGENT TURN DATA:
{turn_data}
"""

CONSOLIDATION_SYSTEM = """\
You are a session facts consolidator. Rules:
- Remove duplicates (keep the most recent/complete version)
- If two facts contradict, keep only the newer one (later in the list)
- Preserve ALL file paths, numeric values, and error messages exactly
- Output ONLY bullet points (one fact per line, starting with •)"""

CONSOLIDATION_USER = """\
Consolidate these session facts into a shorter list. Keep under {max_chars} characters total.

FACTS TO CONSOLIDATE:
{facts}
"""


def _build_turn_data(intermediate_steps: list, ai_response: str) -> str:
    """Build turn data string from intermediate steps and AI response."""
    parts = []
    total_chars = 0

    for step in intermediate_steps:
        if total_chars >= MAX_TURN_DATA_CHARS:
            break
        try:
            action = step[0]
            observation = step[1] if len(step) > 1 else ""

            tool_name = getattr(action, "tool", "unknown")
            tool_input = getattr(action, "tool_input", {})
            if isinstance(tool_input, dict):
                input_summary = str({
                    k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v)
                    for k, v in list(tool_input.items())[:5]
                })
            else:
                input_summary = str(tool_input)[:150]

            obs_str = str(observation)[:MAX_OBSERVATION_CHARS]
            entry = f"Tool: {tool_name} | Input: {input_summary} | Result: {obs_str}"
            parts.append(entry)
            total_chars += len(entry)
        except (IndexError, TypeError, AttributeError):
            continue

    if ai_response:
        ai_part = f"\nAI Response: {ai_response[:MAX_AI_RESPONSE_CHARS]}"
        parts.append(ai_part)

    return "\n".join(parts)


async def extract_session_facts(
    intermediate_steps: list,
    ai_response: str,
    existing_facts: str = "",
) -> str:
    """Extract facts from a completed agent turn.

    Args:
        intermediate_steps: The (action, observation) tuples from the turn.
        ai_response: The agent's final text response.
        existing_facts: Current session facts (for dedup/supersede).

    Returns:
        Updated session facts string (bullet list).
    """
    from core.sub_agent import _call_sub_agent_llm

    turn_data = _build_turn_data(intermediate_steps, ai_response)
    if not turn_data.strip():
        return existing_facts

    previous_facts_section = ""
    if existing_facts and existing_facts.strip():
        previous_facts_section = f"Previous facts (supersede if outdated):\n{existing_facts}"

    user_msg = FACT_EXTRACTION_USER.format(
        previous_facts_section=previous_facts_section,
        turn_data=turn_data,
    )

    result = await _call_sub_agent_llm(user_msg, system=FACT_EXTRACTION_SYSTEM)

    if not result or result.startswith("[Error:"):
        logger.warning(f"[SESSION_FACTS] Extraction failed: {result[:200] if result else 'empty'}")
        return existing_facts

    # Merge: new facts replace existing (Haiku already handles superseding)
    if existing_facts and existing_facts.strip():
        merged = existing_facts.rstrip() + "\n" + result.strip()
    else:
        merged = result.strip()

    # Consolidate if over threshold
    if len(merged) > CONSOLIDATION_THRESHOLD:
        merged = await consolidate_facts(merged)

    return merged


async def consolidate_facts(facts: str) -> str:
    """Consolidate session facts when they grow too large.

    Haiku merges duplicates, removes outdated facts superseded by newer ones,
    and keeps the list under MAX_SESSION_FACTS_CHARS.
    """
    from core.sub_agent import _call_sub_agent_llm

    user_msg = CONSOLIDATION_USER.format(
        max_chars=MAX_SESSION_FACTS_CHARS,
        facts=facts,
    )

    result = await _call_sub_agent_llm(user_msg, system=CONSOLIDATION_SYSTEM)

    if not result or result.startswith("[Error:"):
        logger.warning(f"[SESSION_FACTS] Consolidation failed, using head+tail trim")
        # Fallback: keep head (foundational facts) + tail (recent facts)
        if len(facts) > MAX_SESSION_FACTS_CHARS:
            lines = facts.split("\n")
            head_budget = int(MAX_SESSION_FACTS_CHARS * 0.6)
            tail_budget = MAX_SESSION_FACTS_CHARS - head_budget
            # Head: oldest facts (paths, env setup, foundational decisions)
            head_lines = []
            head_total = 0
            for line in lines:
                if head_total + len(line) + 1 > head_budget:
                    break
                head_lines.append(line)
                head_total += len(line) + 1
            # Tail: most recent facts
            tail_lines = []
            tail_total = 0
            for line in reversed(lines):
                if tail_total + len(line) + 1 > tail_budget:
                    break
                tail_lines.insert(0, line)
                tail_total += len(line) + 1
            return "\n".join(head_lines) + "\n[...older facts omitted...]\n" + "\n".join(tail_lines)
        return facts

    # Enforce hard limit on result
    if len(result) > MAX_SESSION_FACTS_CHARS:
        result = result[:MAX_SESSION_FACTS_CHARS]

    return result.strip()
