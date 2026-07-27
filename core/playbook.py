"""Playbook Memory System — persistent record of tool call outcomes.

Records successes and failures so the agent can learn from past experience.
Consulted BEFORE tool calls to provide "what worked" and "what to avoid" context.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Tools worth recording (high-value decisions, not routine reads)
RECORDABLE_TOOLS = frozenset({
    "execute_dynamic_task",
    "submit_slurm_job",
    "slurm_monitor_job",
    "find_files",
})

# Params worth recording per tool (the "decision" parameters)
RECORDABLE_PARAMS = {
    "submit_slurm_job": ["container_image", "partition", "gres"],
    "execute_dynamic_task": ["command", "task_name"],
    "find_files": ["pattern", "path"],
}

DEFAULT_TRANSIENT_MAX_AGE_DAYS = 30
DEFAULT_MAX_ENTRIES = 500


@dataclass
class PlaybookEntry:
    tool: str
    param: str
    value: str
    outcome: str  # "success" or "failure"
    reason: Optional[str] = None  # why it failed (e.g. "file_not_found", "policy_blocked")
    category: Optional[str] = None  # "permanent", "transient", or None (for successes)
    context: Optional[str] = None  # what the user was trying to do
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class PlaybookMemory:
    """Append-only log of tool outcomes, queryable for pre-call context."""

    def __init__(self, playbook_path: str = "logs/playbook.jsonl"):
        self.playbook_path = Path(playbook_path)
        self._entries: list[PlaybookEntry] = []
        self._load()

    def _load(self):
        """Load existing playbook entries from disk."""
        if not self.playbook_path.exists():
            return
        try:
            with open(self.playbook_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        self._entries.append(PlaybookEntry(**data))
                    except (json.JSONDecodeError, TypeError):
                        continue
        except Exception as e:
            logger.warning(f"[Playbook] Failed to load {self.playbook_path}: {e}")

    def record(
        self,
        tool: str,
        param: str,
        value: str,
        outcome: str,
        reason: Optional[str] = None,
        category: Optional[str] = None,
        context: Optional[str] = None,
    ) -> None:
        """Record a tool call outcome."""
        if not tool or not param or not value:
            return

        entry = PlaybookEntry(
            tool=tool,
            param=param,
            value=value,
            outcome=outcome,
            reason=reason,
            category=category,
            context=context,
        )
        self._entries.append(entry)
        self._persist(entry)

    def record_success(
        self, tool: str, tool_input: dict, context: Optional[str] = None
    ) -> None:
        """Record a successful tool call, extracting recordable params."""
        if tool not in RECORDABLE_TOOLS:
            return
        params_to_record = RECORDABLE_PARAMS.get(tool, [])
        for param in params_to_record:
            value = tool_input.get(param)
            if value and isinstance(value, str) and len(value) < 500:
                self.record(tool, param, value, "success", context=context)

    def record_failure(
        self,
        tool: str,
        tool_input: dict,
        reason: str,
        category: str = "transient",
        context: Optional[str] = None,
    ) -> None:
        """Record a failed tool call, extracting recordable params."""
        if tool not in RECORDABLE_TOOLS:
            return
        params_to_record = RECORDABLE_PARAMS.get(tool, [])
        for param in params_to_record:
            value = tool_input.get(param)
            if value and isinstance(value, str) and len(value) < 500:
                self.record(
                    tool, param, value, "failure",
                    reason=reason, category=category, context=context,
                )

    def lookup(
        self, tool: str, param: Optional[str] = None
    ) -> list[PlaybookEntry]:
        """Find past experiences for a tool (optionally filtered by param)."""
        results = []
        for entry in self._entries:
            if entry.tool != tool:
                continue
            if param and entry.param != param:
                continue
            results.append(entry)
        return results

    def get_context_for_tool(self, tool: str, tool_input: dict) -> str:
        """Format relevant playbook history as context for injection.

        Returns empty string if no relevant history exists.
        """
        if tool not in RECORDABLE_TOOLS:
            return ""

        relevant = self.lookup(tool)
        if not relevant:
            return ""

        # Deduplicate: keep latest entry per (param, value, outcome)
        seen: dict[tuple, PlaybookEntry] = {}
        for entry in relevant:
            key = (entry.param, entry.value, entry.outcome)
            seen[key] = entry  # last wins (most recent)

        successes = []
        failures_permanent = []
        failures_transient = []

        for entry in seen.values():
            if entry.outcome == "success":
                successes.append(entry)
            elif entry.category == "permanent":
                failures_permanent.append(entry)
            else:
                # Skip transient failures older than max_age
                try:
                    ts = datetime.fromisoformat(entry.timestamp)
                    if datetime.now(timezone.utc) - ts > timedelta(
                        days=DEFAULT_TRANSIENT_MAX_AGE_DAYS
                    ):
                        continue
                except (ValueError, TypeError):
                    pass
                failures_transient.append(entry)

        if not successes and not failures_permanent and not failures_transient:
            return ""

        lines = [f"📋 PLAYBOOK (past experience with {tool}):"]

        if successes:
            lines.append("  ✅ Previously worked:")
            for e in successes[-5:]:  # last 5 successes
                ctx = f" ({e.context})" if e.context else ""
                lines.append(f"    - {e.param}='{e.value}'{ctx}")

        if failures_permanent:
            lines.append("  ⛔ NEVER works (permanent failures):")
            for e in failures_permanent[-5:]:
                reason = f" — {e.reason}" if e.reason else ""
                lines.append(f"    - {e.param}='{e.value}'{reason}")

        if failures_transient:
            lines.append("  ⚠️ Previously failed (may have changed):")
            for e in failures_transient[-3:]:
                reason = f" — {e.reason}" if e.reason else ""
                lines.append(f"    - {e.param}='{e.value}'{reason}")

        return "\n".join(lines)

    def prune(
        self,
        max_age_days: int = DEFAULT_TRANSIENT_MAX_AGE_DAYS,
        keep_permanent: bool = True,
    ) -> int:
        """Remove stale entries. Returns count of pruned entries."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        original_count = len(self._entries)

        kept = []
        for entry in self._entries:
            if keep_permanent and entry.category == "permanent":
                kept.append(entry)
                continue
            try:
                ts = datetime.fromisoformat(entry.timestamp)
                if ts >= cutoff:
                    kept.append(entry)
            except (ValueError, TypeError):
                kept.append(entry)  # keep unparseable entries

        # Also enforce max entries (keep most recent)
        if len(kept) > DEFAULT_MAX_ENTRIES:
            kept = kept[-DEFAULT_MAX_ENTRIES:]

        self._entries = kept
        pruned = original_count - len(kept)

        if pruned > 0:
            self._rewrite()
            logger.info(f"[Playbook] Pruned {pruned} stale entries")

        return pruned

    def _persist(self, entry: PlaybookEntry) -> None:
        """Append a single entry to the playbook file."""
        try:
            self.playbook_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.playbook_path, 'a') as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except Exception as e:
            logger.debug(f"[Playbook] Write failed: {e}")

    def _rewrite(self) -> None:
        """Rewrite the entire playbook (after prune)."""
        try:
            self.playbook_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.playbook_path, 'w') as f:
                for entry in self._entries:
                    f.write(json.dumps(asdict(entry)) + "\n")
        except Exception as e:
            logger.warning(f"[Playbook] Rewrite failed: {e}")

    @property
    def entry_count(self) -> int:
        return len(self._entries)
