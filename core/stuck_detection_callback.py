"""Mid-loop stuck detection callback for LangChain AgentExecutor.

This module provides a LangChain AsyncCallbackHandler that detects when the
agent is stuck in a repeated-error loop and interrupts the agent mid-loop
(not after it finishes) to ask the user for web search approval.

Design:
  - StuckDetectionCallback.on_tool_end() fires after EVERY tool call.
  - It tracks (tool_name, error_fingerprint) counts using the same
    normalization logic as detect_stuck_needs_websearch() in single_agent.py.
  - When the same error is seen `threshold` times, it raises StuckInterrupt.
  - app.py wraps ainvoke() in try/except StuckInterrupt and handles it
    using the same _approval_gate() path that normal web_search uses —
    so the user sees the same familiar approval dialog, mid-loop.

This replaces the post-loop Step 5b stuck detection in app.py.
"""

import hashlib
import re as _re
from collections import defaultdict
from typing import Any, Dict, Optional

from langchain_core.callbacks import AsyncCallbackHandler


# ── Fingerprint normalization (mirrors detect_stuck_needs_websearch) ──────────
_NORMALIZE_PATTERNS = [
    (_re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"), "<TS>"),
    (_re.compile(r"\b\d{10,13}\b"), "<EPOCH>"),
    (_re.compile(r"\b[0-9a-f]{8,}\b"), "<HEX>"),
    (_re.compile(r"line \d+"), "line <N>"),
    (_re.compile(r"job \d+"), "job <N>"),
    (_re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"), "<UUID>"),
]

_SUCCESS_SIGNALS = ('"success":true', '"success": true', '"iserror":false', '"iserror": false')


def _normalize_observation(obs: str) -> str:
    """Strip volatile tokens so the same error always produces the same fingerprint."""
    # Apply pattern substitutions BEFORE lowercasing so case-sensitive patterns
    # (e.g. ISO timestamps with uppercase T) match correctly.
    result = obs
    for pattern, replacement in _NORMALIZE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result.lower()


def _is_error_observation(obs: str) -> bool:
    """Return True if the observation looks like an error (not a success)."""
    obs_lower = obs.lower()
    # Skip if tool explicitly reported success
    for signal in _SUCCESS_SIGNALS:
        if signal in obs_lower:
            return False
    # Must contain an error-like keyword
    error_keywords = ("error", "exception", "traceback", "failed", "failure",
                      "not found", "no such", "fatal", "permission denied",
                      "not readable", "not writable", "timeout", "invalid")
    return any(kw in obs_lower for kw in error_keywords)


def _make_fingerprint(tool_name: str, obs: str, window: int = 200) -> str:
    """Create a stable fingerprint for a (tool, error) pair."""
    normalized = _normalize_observation(obs)[:window]
    raw = f"{tool_name}::{normalized}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _is_local_error(obs: str) -> bool:
    """Return True if the error is a local/OS issue unsolvable by web search.

    Permission errors, missing files, disk space, and OS-level failures are
    never helped by web search — they require local admin action.
    """
    obs_lower = obs.lower()
    local_patterns = (
        "permission denied",
        "is not readable",
        "is not writable",
        "operation not permitted",
        "no such file or directory",
        "not a directory",
        "is a directory",
        "disk quota exceeded",
        "no space left on device",
        "read-only file system",
        "too many open files",
        "connection refused",
        "could not open image",
        "check permissions",
        "eacces",
        "eperm",
        "enoent",
        "enospc",
    )
    return any(p in obs_lower for p in local_patterns)


def _suggest_query(tool_name: str, error_snippet: str) -> str:
    """Derive a web search query from the tool name and error."""
    # Strip noise from error snippet
    clean = _re.sub(r"[<>{}()\[\]\"']", " ", error_snippet)
    clean = _re.sub(r"\s+", " ", clean).strip()[:120]
    return f"{tool_name} error: {clean}"


SCHEMA_ERROR_PREFIX = "[SCHEMA_ERROR] "

# ── Exception raised mid-loop when stuck ─────────────────────────────────────

class StuckInterrupt(Exception):
    """Raised by StuckDetectionCallback when the agent is stuck mid-loop.

    Attributes:
        tool_name: The tool that kept failing.
        error_fingerprint: Normalized fingerprint of the repeated error.
        failure_count: How many times the same error occurred.
        suggested_query: A suggested web search query.
        error_snippet: The raw error text (first 200 chars).
        is_internal: True if the error is a schema/parameter error.
        is_local: True if the error is a local OS/permission issue (web search won't help).
    """

    def __init__(
        self,
        tool_name: str,
        error_fingerprint: str,
        failure_count: int,
        suggested_query: str,
        error_snippet: str,
        is_internal: bool = False,
        is_local: bool = False,
    ) -> None:
        self.tool_name = tool_name
        self.error_fingerprint = error_fingerprint
        self.failure_count = failure_count
        self.suggested_query = suggested_query
        self.error_snippet = error_snippet
        self.is_internal = is_internal
        self.is_local = is_local
        super().__init__(
            f"Agent stuck: {tool_name} failed {failure_count}x "
            f"with fingerprint={error_fingerprint}"
        )


# ── The callback ──────────────────────────────────────────────────────────────

class StuckDetectionCallback(AsyncCallbackHandler):
    """LangChain async callback that detects repeated errors mid-loop.

    Attach this to AgentExecutor.callbacks so it fires after every tool call.
    When the same (tool, error) pair is seen `threshold` times, it raises
    StuckInterrupt — which app.py catches to show the user an approval dialog
    (same UX as normal web_search approval) without waiting for the full loop
    to finish.

    Usage in create_skill_based_agent():
        from core.stuck_detection_callback import StuckDetectionCallback
        stuck_cb = StuckDetectionCallback(threshold=3)
        executor = AgentExecutor(..., callbacks=[stuck_cb])

    Usage in app.py:
        try:
            result = await executor.ainvoke(...)
        except StuckInterrupt as e:
            # handle mid-loop stuck — show approval dialog
    """

    def __init__(self, threshold: int = 3) -> None:
        super().__init__()
        self.threshold = threshold
        # fingerprint -> count
        self._counts: Dict[str, int] = defaultdict(int)
        # fingerprint -> (tool_name, error_snippet)
        self._details: Dict[str, tuple] = {}

    def reset(self) -> None:
        """Reset state — call between turns if reusing the callback."""
        self._counts.clear()
        self._details.clear()

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        """Called after every tool invocation. Checks for repeated errors."""
        # Get tool name from kwargs (LangChain passes it as 'name')
        tool_name: str = kwargs.get("name", "") or ""
        obs: str = str(output) if output is not None else ""

        # Skip "invalid tool" observations — these are handled by the
        # escalation system, not the web search stuck detection.
        if "is not a valid tool" in obs:
            return

        if not _is_error_observation(obs):
            return  # Not an error — nothing to track

        fingerprint = _make_fingerprint(tool_name, obs)
        self._counts[fingerprint] += 1

        if fingerprint not in self._details:
            # Store the raw snippet for the suggested query
            error_snippet = obs[:200]
            self._details[fingerprint] = (tool_name, error_snippet)

        count = self._counts[fingerprint]
        if count >= self.threshold:
            tool_name_stored, error_snippet = self._details[fingerprint]
            suggested_query = _suggest_query(tool_name_stored, error_snippet)
            raise StuckInterrupt(
                tool_name=tool_name_stored,
                error_fingerprint=fingerprint,
                failure_count=count,
                suggested_query=suggested_query,
                error_snippet=error_snippet,
                is_internal=obs.startswith(SCHEMA_ERROR_PREFIX),
                is_local=_is_local_error(obs),
            )
