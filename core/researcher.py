"""
core/researcher.py — Research findings utilities

Provides the write_findings tool and utilities for managing findings files.
Findings are stored under the project memory path:
  /home/{USER}/{APP}/memory/projects/{project}/findings/{date}_{slug}.md

This ensures findings survive work_dir changes and are accessible to the
curation system for cross-session persistence.
"""

import re
import time
import uuid
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_FINDINGS_INJECT_CHARS = 100000


# ── Findings directory & naming ──────────────────────────────────────────────

def get_findings_dir(project_name: str) -> Path:
    """Return the findings directory under project memory, creating it if needed.

    Stored at: /home/{USER}/{APP}/memory/projects/{project}/findings/
    Falls back to _no_project/ if project_name is empty.
    """
    from core.memory_state import get_memory_root
    safe_name = project_name.strip() if project_name else "_no_project"
    findings_dir = get_memory_root() / "projects" / safe_name / "findings"
    findings_dir.mkdir(parents=True, exist_ok=True)
    return findings_dir


def generate_findings_name(task: str) -> str:
    """Generate a unique findings filename from date + task keywords + short UUID."""
    date_prefix = time.strftime("%Y-%m-%d")
    words = re.sub(r"[^a-z0-9\s]", "", task.lower()).split()
    stop_words = {"the", "a", "an", "is", "to", "and", "or", "for", "in", "on", "of", "i", "my", "me", "do"}
    keywords = [w for w in words if w not in stop_words and len(w) > 2][:4]
    slug = "_".join(keywords) if keywords else "findings"
    short_id = uuid.uuid4().hex[:6]
    return f"{date_prefix}_{slug}_{short_id}.md"


# ── Utility functions ────────────────────────────────────────────────────────

def read_findings_file(findings_path: str) -> Optional[str]:
    """Read a findings file from an explicit path. Returns None if missing."""
    p = Path(findings_path)
    if p.exists():
        try:
            content = p.read_text(encoding="utf-8")
            return content if content.strip() else None
        except Exception as e:
            logger.warning(f"[RESEARCHER] Could not read {p}: {e}")
    return None


def get_findings_for_prompt(findings_path, max_chars: int = MAX_FINDINGS_INJECT_CHARS) -> str:
    """Return findings content formatted for injection into planner/executor prompt.

    Args:
        findings_path: A single path (str) or list of paths to findings files.
        max_chars: Safety limit for total findings size (default 100K).
    """
    if not findings_path:
        return ""

    paths = findings_path if isinstance(findings_path, list) else [findings_path]
    all_content = []
    for p in paths:
        if not p:
            continue
        content = read_findings_file(p)
        if content:
            all_content.append(content)

    if not all_content:
        return ""

    combined = "\n\n---\n\n".join(all_content)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n... [FINDINGS truncated — see full files on disk]"
    return (
        f"\n\n═══ RESEARCH FINDINGS (use these facts) ═══\n"
        f"{combined}\n"
        f"═══ END FINDINGS ═══\n"
    )


def _write_findings(content: str, findings_dir: str, findings_name: str) -> str:
    """Write findings to the findings directory. Returns confirmation or error."""
    try:
        findings_path = Path(findings_dir)
        findings_path.mkdir(parents=True, exist_ok=True)
        fpath = findings_path / findings_name
        fpath.write_text(content, encoding="utf-8")
        return (
            f"Here's what I found:\n\n"
            f"{content}"
        )
    except Exception as e:
        return f"Failed to write findings: {e}"


# ── write_findings tool factory ──────────────────────────────────────────────

def create_write_findings_tool(findings_dir: str, findings_name: str):
    """Create the write_findings LangChain tool that terminates the research phase."""
    from langchain_core.tools import StructuredTool

    def _write_findings_impl(content: str) -> str:
        """Write the research findings to end the research phase.

        Args:
            content: The complete findings in markdown format with discovered
                     facts, file paths, configurations, and relevant context.
        """
        return _write_findings(content, findings_dir, findings_name)

    return StructuredTool.from_function(
        func=_write_findings_impl,
        name="write_findings",
        description=(
            "Write your research findings to disk. This ENDS the research phase. "
            "Call this with your complete findings when you have explored enough "
            "to understand the problem. Include: what you found, relevant file paths, "
            "key configurations, current state, and any constraints discovered. "
            "Do NOT include plans or implementation steps — just facts. "
            "After calling this, present the findings to the user and ask what they'd like to do next."
        ),
    )
