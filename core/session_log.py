"""Incremental session log — crash-safe JSONL append per message.

Appends each human/AI message to a per-session JSONL file as it arrives.
If the session crashes mid-turn, messages up to the crash point are preserved.

No Chainlit or LLM dependencies. Pure file I/O.
"""
import json
import datetime
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import core.persistence as persistence


def get_session_log_dir(username: str) -> Path:
    """Get (and create) the session logs directory for a user."""
    log_dir = persistence.get_user_data_dir(username) / "session_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def init_session_log(
    username: str,
    session_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Create a new session log file and write the header line.

    Args:
        username: OS username.
        session_id: Chainlit session ID.
        metadata: Optional dict (work_dir, project_name, etc.).

    Returns:
        Path to the created JSONL file.
    """
    log_dir = get_session_log_dir(username)
    log_path = log_dir / f"{session_id}.jsonl"

    header = {
        "type": "session_start",
        "ts": datetime.datetime.now().isoformat(),
        "username": username,
        "session_id": session_id,
    }
    if metadata:
        header.update(metadata)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

    return log_path


def append_message(
    log_path: Path,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a single message to the session log.

    Uses open-append-flush-fsync for crash safety.

    Args:
        log_path: Path to the session JSONL file.
        role: "user" or "assistant".
        content: Message content text.
        metadata: Optional extra fields to include.
    """
    if not content or not content.strip():
        return

    entry = {
        "type": "message",
        "ts": datetime.datetime.now().isoformat(),
        "role": role,
        "content": content,
    }
    if metadata:
        entry.update(metadata)

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"[SESSION_LOG] Failed to append message: {e}")


def append_phase_marker(
    log_path: Path,
    phase: str,
    event: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write an explicit phase state marker to the session log.

    These are the authoritative record of what phase the session reached.
    Curation and cross-session resumption both read these.

    Args:
        log_path: Path to the session JSONL file.
        phase: Phase name ("research", "plan", "execute").
        event: One of "started", "completed", "awaiting_approval", "abandoned".
        metadata: Optional dict (artifact paths, phases_planned, etc.).
    """
    entry = {
        "type": "phase_marker",
        "ts": datetime.datetime.now().isoformat(),
        "phase": phase,
        "event": event,
    }
    if metadata:
        entry["metadata"] = metadata
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"[SESSION_LOG] Failed to append phase marker: {e}")


def load_history_from_session_log(
    username: str,
    session_id: str,
    message_factory: dict = None,
) -> list:
    """Load conversation messages from an existing session log (reconnect recovery).

    Parses the JSONL file for this session_id and returns all message entries
    as LangChain message objects (if factory provided) or raw dicts.

    Args:
        username: OS username.
        session_id: Chainlit session ID (full UUID).
        message_factory: Optional dict mapping role names to constructors.
            e.g. {"human": HumanMessage, "ai": AIMessage}

    Returns:
        List of message objects (if factory provided) or raw dicts.
        Empty list if file doesn't exist or has no messages.
    """
    log_dir = get_session_log_dir(username)
    log_path = log_dir / f"{session_id}.jsonl"

    if not log_path.exists():
        return []

    messages = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") != "message":
                    continue
                content = entry.get("content", "")
                if not content or not isinstance(content, str) or not content.strip():
                    continue
                role = entry.get("role", "")
                if role == "tool":
                    continue
                if message_factory:
                    factory_key = "human" if role == "user" else "ai"
                    if factory_key in message_factory:
                        messages.append(message_factory[factory_key](content=content))
                else:
                    messages.append({"type": "human" if role == "user" else "ai", "content": content})
    except Exception as e:
        print(f"[SESSION_LOG] Failed to load history from session log: {e}")
        return []

    if messages:
        print(f"[SESSION_LOG] Loaded {len(messages)} messages from {log_path.name}")
    return messages


def cleanup_old_session_logs(username: str, keep_days: int = 30) -> int:
    """Delete session logs older than keep_days.

    Args:
        username: OS username.
        keep_days: Number of days to retain.

    Returns:
        Number of files deleted.
    """
    log_dir = persistence.get_user_data_dir(username) / "session_logs"
    if not log_dir.exists():
        return 0

    cutoff = datetime.datetime.now() - datetime.timedelta(days=keep_days)
    cutoff_ts = cutoff.timestamp()
    deleted = 0

    for f in log_dir.glob("*.jsonl"):
        try:
            if f.stat().st_mtime < cutoff_ts:
                f.unlink()
                deleted += 1
        except OSError:
            pass

    return deleted


def list_session_logs(username: str) -> List[Dict[str, Any]]:
    """List available session log files with metadata.

    Returns:
        List of dicts with session_id, path, size_bytes, modified timestamp.
    """
    log_dir = persistence.get_user_data_dir(username) / "session_logs"
    if not log_dir.exists():
        return []

    results = []
    for f in sorted(log_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = f.stat()
        results.append({
            "session_id": f.stem,
            "path": str(f),
            "size_bytes": stat.st_size,
            "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })

    return results


def search_session_logs(
    username: str,
    query: str,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """Search across all session logs for messages matching a query.

    Simple case-insensitive substring search across message content.

    Args:
        username: OS username.
        query: Search string.
        max_results: Maximum number of matching messages to return.

    Returns:
        List of matching message dicts with session_id added.
    """
    log_dir = persistence.get_user_data_dir(username) / "session_logs"
    if not log_dir.exists():
        return []

    query_lower = query.lower()
    matches = []

    for f in sorted(log_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("type") != "message":
                        continue
                    content = entry.get("content", "")
                    if query_lower in content.lower():
                        entry["session_id"] = f.stem
                        matches.append(entry)
                        if len(matches) >= max_results:
                            return matches
        except OSError:
            continue

    return matches
