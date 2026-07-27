"""
core/opus_planner.py — Plan file utilities

Provides the write_plan tool and utilities for managing plan files.
Plans are stored under the project memory path:
  /home/{USER}/{APP}/memory/projects/{project}/plans/{date}_{slug}.md

This ensures plans survive work_dir changes and are accessible to the
curation system for cross-session persistence.
"""

import re
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_PLAN_INJECT_CHARS = 100000


# ── Plan directory & naming ───────────────────────────────────────────────────

def get_plans_dir(project_name: str) -> Path:
    """Return the plans directory under project memory, creating it if needed.

    Stored at: /home/{USER}/{APP}/memory/projects/{project}/plans/
    Falls back to _no_project/ if project_name is empty.
    """
    from core.memory_state import get_memory_root
    safe_name = project_name.strip() if project_name else "_no_project"
    plans_dir = get_memory_root() / "projects" / safe_name / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    return plans_dir


def generate_plan_name(task: str) -> str:
    """Generate a plan filename from date + task keywords.

    Example: '2026-06-06_scipy_gpu_test.md'
    """
    date_prefix = time.strftime("%Y-%m-%d")
    words = re.sub(r"[^a-z0-9\s]", "", task.lower()).split()
    stop_words = {"the", "a", "an", "is", "to", "and", "or", "for", "in", "on", "of", "i", "my", "me", "do"}
    keywords = [w for w in words if w not in stop_words and len(w) > 2][:4]
    slug = "_".join(keywords) if keywords else "plan"
    return f"{date_prefix}_{slug}.md"


# ── Utility functions ─────────────────────────────────────────────────────────

def read_plan_file(plan_path: str) -> Optional[str]:
    """Read a plan file from an explicit path. Returns None if missing."""
    p = Path(plan_path)
    if p.exists():
        try:
            content = p.read_text(encoding="utf-8")
            return content if content.strip() else None
        except Exception as e:
            logger.warning(f"[PLANNER] Could not read {p}: {e}")
    return None


def get_plan_for_prompt(plan_path: str, max_chars: int = MAX_PLAN_INJECT_CHARS) -> str:
    """Return plan content formatted for injection into the executor prompt.

    Args:
        plan_path: Explicit path to the plan file.

    Returns empty string if plan doesn't exist or path is empty.
    """
    if not plan_path:
        return ""
    content = read_plan_file(plan_path)
    if not content:
        return ""
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n... [PLAN truncated — see full file at {plan_path}]"
    return (
        f"\n\n═══ ACTIVE PLAN (follow this) ═══\n"
        f"{content}\n"
        f"═══ END PLAN — update checkboxes as you complete steps ═══\n"
    )


def _write_plan(content: str, plans_dir: str, plan_name: str) -> str:
    """Write plan to the plans directory. Returns plan content for user presentation."""
    try:
        plans_path = Path(plans_dir)
        plans_path.mkdir(parents=True, exist_ok=True)
        plan_path = plans_path / plan_name
        plan_path.write_text(content, encoding="utf-8")
        return (
            f"Here's my plan:\n\n"
            f"{content}\n\n"
            f"Would you like me to go ahead with this?"
        )
    except Exception as e:
        return f"Failed to write plan: {e}"


# ── write_plan tool factory ───────────────────────────────────────────────────

def create_write_plan_tool(plans_dir: str, plan_name: str):
    """Create the write_plan LangChain tool that terminates the planning phase."""
    from langchain_core.tools import StructuredTool

    def _write_plan_impl(content: str) -> str:
        """Write the final plan to end the planning phase.

        Args:
            content: The complete plan in markdown format with Find/Replace blocks
                     for code changes and exact commands for shell steps.
        """
        if not content.strip():
            return "ERROR: Plan content cannot be empty. Pass your full plan as the content argument."
        return _write_plan(content, plans_dir, plan_name)

    return StructuredTool.from_function(
        func=_write_plan_impl,
        name="write_plan",
        description=(
            "Write the final plan to disk. This ENDS the planning phase. "
            "You MUST pass content='<your full plan>' as an argument. "
            "Call this when you are done exploring and ready to hand off to the executor. "
            "The plan MUST include exact Find/Replace blocks for code changes "
            "and exact commands for shell steps. "
            "After calling this, show the FULL plan content to the user (not just a summary) and ask for approval."
        ),
    )


# ── edit_plan tool factory ───────────────────────────────────────────────────

def _edit_plan(content: str, plan_path: str) -> str:
    """Update the active plan file (mark steps complete, add notes)."""
    try:
        p = Path(plan_path)
        if not p.exists():
            return f"ERROR: Plan file does not exist: {plan_path}"
        p.write_text(content, encoding="utf-8")
        done = content.count("- [x]") + content.count("- [X]")
        pending = content.count("- [ ]")
        return f"Plan updated ({done} done, {pending} pending). Path: {plan_path}"
    except Exception as e:
        return f"Failed to update plan: {e}"


def create_edit_plan_tool(plan_path: str):
    """Create the edit_plan tool for marking steps complete during execution."""
    from langchain_core.tools import StructuredTool

    def _edit_plan_impl(content: str) -> str:
        """Update the active plan file to mark steps as completed.

        Args:
            content: The full updated plan content (with checkboxes toggled).
        """
        if not content.strip():
            return "ERROR: content cannot be empty."
        return _edit_plan(content, plan_path)

    return StructuredTool.from_function(
        func=_edit_plan_impl,
        name="edit_plan",
        description=(
            "Update the active plan file to mark steps as completed (change '- [ ]' to '- [x]'). "
            "This is a full-file replacement — pass the COMPLETE updated plan text as the single "
            "'content' parameter (string). This is NOT a patch tool and does not accept "
            "path/old_text/new_text parameters."
        ),
    )
