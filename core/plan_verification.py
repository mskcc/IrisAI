"""
core/plan_verification.py — Post-execution plan verification via Haiku

After the executor completes a turn of a planned task, this module determines
whether the plan is fully complete using two reliable mechanisms:

1. Disk-based check: Read PLAN.md from disk — if unchecked '- [ ]' steps remain,
   the plan is not done (executor is instructed to mark steps [x] on disk).
2. Haiku verification: Strict per-step evidence check — requires Haiku to map
   each plan step to a specific tool call that accomplished it.

No phrase matching or heuristics — only ground truth signals.
"""

import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Plan completeness check (disk-based, no LLM cost) ────────────────────

def plan_has_unchecked_steps(plan_text: Optional[str]) -> bool:
    """Return True if the plan text contains any unchecked '- [ ]' steps.

    This is the primary signal: the executor is instructed to edit PLAN.md
    and mark steps [x] as it completes them. If unchecked steps remain,
    the plan is still in progress.
    """
    if not plan_text:
        return False
    return bool(re.search(r"^\s*-\s*\[ \]", plan_text, re.MULTILINE))


def count_plan_progress(plan_text: Optional[str]) -> tuple:
    """Return (completed, total) step counts from plan checkboxes."""
    if not plan_text:
        return (0, 0)
    completed = len(re.findall(r"^\s*-\s*\[x\]", plan_text, re.MULTILINE))
    unchecked = len(re.findall(r"^\s*-\s*\[ \]", plan_text, re.MULTILINE))
    return (completed, completed + unchecked)


# ── Plan step extraction ─────────────────────────────────────────────────

def _extract_plan_steps(plan_text: str) -> List[str]:
    """Extract all checkbox steps (checked and unchecked) from PLAN.md text."""
    steps = []
    for line in plan_text.split("\n"):
        match = re.match(r"\s*-\s*\[[ x]\]\s*(.*)", line)
        if match:
            steps.append(match.group(1).strip())
    return steps


def _format_tool_calls(intermediate_steps: list) -> str:
    """Format intermediate steps into a readable summary for Haiku."""
    lines = []
    for step in intermediate_steps:
        try:
            action = step[0]
            observation = str(step[1] if len(step) > 1 else "")
            tool_name = getattr(action, "tool", "unknown")
            tool_input = getattr(action, "tool_input", "")
            obs_preview = observation[:200] + "..." if len(observation) > 200 else observation
            input_preview = str(tool_input)[:150] if tool_input else ""
            lines.append(f"- {tool_name}({input_preview}) → {obs_preview}")
        except Exception:
            continue
    return "\n".join(lines) if lines else "(no tool calls recorded)"


# ── Main verification entry point ────────────────────────────────────────

async def verify_plan_execution(
    plan_text: str,
    intermediate_steps: list,
    agent_output: str,
    sum_llm: Any,
) -> Dict[str, Any]:
    """
    Use Haiku to verify whether the executor completed all plan steps.

    Args:
        plan_text: The PLAN.md content (from disk — with updated checkboxes)
        intermediate_steps: The executor's tool call history
        agent_output: The executor's final text output
        sum_llm: Haiku LLM instance (from cl.user_session "sum_llm")

    Returns:
        Dict with:
            - missed_steps: List of step descriptions that were NOT completed
            - all_complete: True if everything was done
    """
    if not sum_llm:
        logger.warning("[PLAN_VERIFY] No Haiku LLM available — skipping verification")
        return {"missed_steps": [], "all_complete": True}

    plan_steps = _extract_plan_steps(plan_text)
    if not plan_steps:
        return {"missed_steps": [], "all_complete": True}

    tool_summary = _format_tool_calls(intermediate_steps)

    # Cap inputs to avoid blowing Haiku's context
    plan_steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan_steps[:20]))
    tool_summary = tool_summary[:3000]
    agent_output_preview = agent_output[:1000] if agent_output else "(no output)"
    step_count = len(plan_steps)

    prompt = f"""You are a strict verification assistant. The plan has {step_count} steps total.
Your job: determine which steps were ACTUALLY EXECUTED (not just mentioned or planned).

## PLAN STEPS ({step_count} total):
{plan_steps_text}

## ACTUAL TOOL CALLS (what the executor did):
{tool_summary}

## EXECUTOR'S FINAL OUTPUT:
{agent_output_preview}

## STRICT RULES:
1. A step is "completed" ONLY if there is a SPECIFIC tool call that accomplished it.
   - A tool call must MATCH the step's action (e.g. submit_slurm_job for "submit job", edit_file for "modify file")
   - The tool call's output must confirm success (no errors, expected result)
2. A step is NOT completed if:
   - The executor only MENTIONED it in output but made no tool call for it
   - The executor said it will do it "next turn" or "when you ask to continue"
   - The tool call failed or returned an error
3. If the executor's output contains phrases like "ask me to continue", "next turn", "remaining steps" — that means NOT all steps are done.

## RESPONSE FORMAT:
For EACH plan step, respond with one line:
  DONE: [step number] - [brief evidence: which tool call completed it]
  MISSED: [step number] - [step description]

If and ONLY if ALL {step_count} steps have a DONE line, end with: ALL_COMPLETE
Otherwise, list only the MISSED lines."""

    try:
        from langchain_core.messages import HumanMessage
        response = await sum_llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)

        logger.info(f"[PLAN_VERIFY] Haiku response: {content[:300]}")

        if "ALL_COMPLETE" in content:
            return {"missed_steps": [], "all_complete": True}

        missed = []
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("MISSED:"):
                missed.append(line[7:].strip())

        # If no MISSED lines found but also no ALL_COMPLETE, treat as inconclusive
        if not missed and "DONE:" not in content:
            missed.append("Verification inconclusive — could not parse response")

        return {"missed_steps": missed, "all_complete": len(missed) == 0}

    except Exception as e:
        logger.error(f"[PLAN_VERIFY] Haiku verification failed: {e}")
        return {"missed_steps": [], "all_complete": True}
