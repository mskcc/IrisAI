"""Checkpointing stubs — no-ops retained for API compatibility.

Checkpointing was removed as redundant (session_logs provide crash safety).
These stubs prevent ImportError in app.py without any disk I/O.
"""
from typing import Dict, List, Any


def checkpoint_tool_call(username: str, tool_name: str, args: dict) -> None:
    pass


def checkpoint_tool_result(username: str, tool_name: str, args: dict, result: Any) -> None:
    pass


def checkpoint_tool_error(username: str, tool_name: str, args: dict, error: str) -> None:
    pass


def clear_session_checkpoints(username: str) -> int:
    return 0


def get_completed_tool_calls(username: str) -> List[Dict]:
    return []
