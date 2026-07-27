"""Protocol Recorder — toggle-based capture of research steps for reproducibility.

Records tool calls as protocol steps when the user enables recording via the
Protocol button. Writes incrementally to a JSONL file (crash-safe via fsync).
On stop, compiles the raw log into a versioned protocol.yaml + protocol.md.

No Chainlit or LLM dependencies. Pure file I/O.
"""

import hashlib
import json
import os
import platform
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import core.persistence as persistence

RESULT_PREVIEW_MAX = 500
PROCEDURAL_TOOLS = frozenset({
    "submit_slurm_job",
    "submit_slurm_job",
    "run_pipeline_script",
    "execute_dynamic_task",
})


class DiskFullError(OSError):
    """Raised when a protocol write fails due to disk space."""
    pass


@dataclass
class ProtocolRecording:
    """In-memory state for an active recording session."""

    name: str
    recording_id: str
    session_id: str
    username: str
    started_at: str
    status: str  # "in_progress" | "complete" | "draft"
    step_counter: int = 0
    version: str = "1.0.0"
    log_path: Path = field(default_factory=lambda: Path())
    references: List[Dict[str, Any]] = field(default_factory=list)
    file_references: Dict[str, str] = field(default_factory=dict)  # path → "input"/"output"
    paused: bool = False


class ProtocolRecorder:
    """Manages the lifecycle of a protocol recording.

    One instance per user session (stored in cl.user_session).
    State is reconstructed from disk on recovery.
    """

    def __init__(self, protocols_dir: Path):
        self.protocols_dir = protocols_dir
        self.active: Optional[ProtocolRecording] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self, name: str, username: str, session_id: str) -> ProtocolRecording:
        """Start a new protocol recording. Creates JSONL with header."""
        if self.active and self.active.status == "in_progress":
            raise RuntimeError("A recording is already active. Stop it first.")

        recording_id = uuid.uuid4().hex[:12]
        active_dir = self.protocols_dir / ".active"
        active_dir.mkdir(parents=True, exist_ok=True)
        log_path = active_dir / f"{recording_id}.jsonl"

        version = self._next_version(name)
        started_at = datetime.now(timezone.utc).isoformat()

        self.active = ProtocolRecording(
            name=name,
            recording_id=recording_id,
            session_id=session_id,
            username=username,
            started_at=started_at,
            status="in_progress",
            version=version,
            log_path=log_path,
        )

        header = {
            "type": "header",
            "name": name,
            "recording_id": recording_id,
            "session_id": session_id,
            "username": username,
            "started_at": started_at,
            "status": "in_progress",
            "version": version,
        }
        self._write_entry(header)
        return self.active

    def stop(self) -> Path:
        """Stop recording, write footer, compile output. Returns output directory."""
        if not self.active:
            raise RuntimeError("No active recording to stop.")

        footer = {
            "type": "footer",
            "ts": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "total_steps": self.active.step_counter,
            "total_references": len(self.active.references),
        }
        self._write_entry(footer)
        self.active.status = "complete"

        output_dir = self.compile()
        self.active = None
        return output_dir

    def save_draft(self, reason: str = "fresh_start") -> Path:
        """Save current recording as draft without compiling. Returns log path."""
        if not self.active:
            raise RuntimeError("No active recording to save as draft.")

        footer = {
            "type": "footer",
            "ts": datetime.now(timezone.utc).isoformat(),
            "status": "draft",
            "reason": reason,
            "total_steps": self.active.step_counter,
            "total_references": len(self.active.references),
        }
        self._write_entry(footer)
        self.active.status = "draft"
        log_path = self.active.log_path
        self.active = None
        return log_path

    def pause(self, reason: str = "disk_full") -> None:
        """Pause recording (e.g., disk full). Steps are silently dropped until resume."""
        if self.active:
            self.active.paused = True
            try:
                self._write_entry({
                    "type": "event",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "paused",
                    "reason": reason,
                })
            except DiskFullError:
                pass  # Can't even write the pause event — that's fine

    def resume(self) -> None:
        """Resume a paused recording."""
        if self.active:
            self.active.paused = False
            self._write_entry({
                "type": "event",
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "resumed",
            })

    # ── Step Capture ──────────────────────────────────────────────────────

    def record_step(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        duration_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Record a tool call as a numbered protocol step."""
        if not self.active or self.active.paused:
            return

        self.active.step_counter += 1
        result_preview = _truncate_result(result)

        entry = {
            "type": "step",
            "step_number": self.active.step_counter,
            "ts": datetime.now(timezone.utc).isoformat(),
            "tool_name": tool_name,
            "args": _sanitize_args(args),
            "result_preview": result_preview,
            "duration_ms": round(duration_ms, 1),
            "error": error,
        }
        self._write_entry(entry)
        self._detect_file_references(tool_name, args, result)

    def record_reference(self, query: str, sources: List[Dict[str, str]]) -> None:
        """Record a web search as a citation/reference (not a numbered step)."""
        if not self.active or self.active.paused:
            return

        ref = {"query": query, "sources": sources}
        self.active.references.append(ref)

        entry = {
            "type": "reference",
            "ts": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "sources": sources[:5],  # Cap at 5 sources per search
        }
        self._write_entry(entry)

    # ── File Reference Detection ────────────────────────────────────────

    def _detect_file_references(self, tool_name: str, args: Dict[str, Any], result: Any) -> None:
        """Scan args and result for file paths, classify as input/output."""
        if not self.active:
            return
        INPUT_TOOLS = {"read_text_file", "grep_file", "find_files", "list_directory"}
        OUTPUT_TOOLS = {"write_text_file", "submit_slurm_job"}

        for key, value in args.items():
            if isinstance(value, str) and _looks_like_path(value):
                if tool_name in INPUT_TOOLS or key in ("path", "input_file", "reference", "file_path"):
                    self.active.file_references.setdefault(value, "input")
                elif tool_name in OUTPUT_TOOLS or key in ("output", "output_path", "output_dir"):
                    self.active.file_references.setdefault(value, "output")

        if isinstance(result, dict):
            for key in ("job_dir", "output_path", "script_path", "file_path"):
                val = result.get(key)
                if isinstance(val, str) and _looks_like_path(val):
                    self.active.file_references.setdefault(val, "output")
            content = result.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        try:
                            data = json.loads(item.get("text", ""))
                            if isinstance(data, dict):
                                for k in ("job_dir", "output_path", "script_path"):
                                    v = data.get(k)
                                    if isinstance(v, str) and _looks_like_path(v):
                                        self.active.file_references.setdefault(v, "output")
                        except (json.JSONDecodeError, TypeError):
                            pass

    # ── Queries ───────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        """True if recording is on and not paused."""
        return self.active is not None and self.active.status == "in_progress" and not self.active.paused

    @property
    def step_count(self) -> int:
        return self.active.step_counter if self.active else 0

    def get_status_line(self) -> str:
        """Short string for UI injection."""
        if not self.active:
            return ""
        state = "PAUSED" if self.active.paused else "Recording"
        return f"{self.active.name} ({state} | {self.active.step_counter} steps)"

    # ── Compilation ───────────────────────────────────────────────────────

    def compile(self) -> Path:
        """Compile JSONL into protocol.yaml + protocol.md in versioned directory."""
        if not self.active:
            raise RuntimeError("No active recording to compile.")

        name = self.active.name
        version = self.active.version
        dir_name = f"{name}_v{version}"
        output_dir = self.protocols_dir / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        steps, references, header = self._parse_jsonl(self.active.log_path)
        environment = _collect_environment()

        protocol_data = {
            "protocol": {
                "name": name,
                "version": version,
                "recording_id": self.active.recording_id,
                "recorded_by": self.active.username,
                "recorded_at": self.active.started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "total_steps": len(steps),
            },
            "environment": environment,
            "steps": steps,
            "references": references,
        }

        # Build dataset manifest with checksums for reproducibility
        dataset_manifest = self._build_dataset_manifest()

        yaml_path = output_dir / "protocol.yaml"
        md_path = output_dir / "protocol.md"

        _write_yaml(yaml_path, protocol_data)
        _write_markdown(md_path, protocol_data)

        # Write dataset manifest as machine-readable JSON (used by play mode)
        if dataset_manifest:
            manifest_path = output_dir / "dataset_manifest.json"
            manifest_path.write_text(
                json.dumps(dataset_manifest, indent=2, default=str), encoding="utf-8"
            )

        # Move raw log to output directory
        dest_log = output_dir / "recording.jsonl"
        shutil.copy2(self.active.log_path, dest_log)
        try:
            self.active.log_path.unlink()
        except OSError:
            pass

        return output_dir

    def _build_dataset_manifest(self) -> Dict[str, Any]:
        """Compute checksums for all referenced files that still exist."""
        if not self.active or not self.active.file_references:
            return {}

        manifest: Dict[str, List] = {"inputs": [], "outputs": []}
        for path, role in self.active.file_references.items():
            entry: Dict[str, Any] = {"path": path}
            if os.path.isfile(path):
                try:
                    entry["sha256"] = _chunked_hash(path)
                    entry["size_bytes"] = os.path.getsize(path)
                except (OSError, PermissionError):
                    entry["sha256"] = "UNAVAILABLE"
                    entry["size_bytes"] = 0
            else:
                entry["sha256"] = "FILE_NOT_FOUND_AT_COMPILE"
                entry["size_bytes"] = 0

            if role == "output":
                entry["checksum_relevant"] = not any(
                    path.endswith(ext) for ext in (".log", ".out", ".err")
                )

            key = f"{role}s"
            if key in manifest:
                manifest[key].append(entry)

        return manifest if (manifest["inputs"] or manifest["outputs"]) else {}

    # ── Recovery ──────────────────────────────────────────────────────────

    @classmethod
    def find_incomplete(cls, protocols_dir: Path) -> List[Path]:
        """Find JSONL files with status=in_progress that have no footer (crash recovery)."""
        active_dir = protocols_dir / ".active"
        if not active_dir.exists():
            return []

        incomplete = []
        for jsonl in active_dir.glob("*.jsonl"):
            try:
                with open(jsonl, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        continue
                    header = json.loads(first_line)
                    if header.get("type") != "header":
                        continue
                    if header.get("status") != "in_progress":
                        continue
                    # Check if there's a footer (last non-empty line)
                    last_line = ""
                    f.seek(0)
                    for line in f:
                        if line.strip():
                            last_line = line.strip()
                    try:
                        last_entry = json.loads(last_line)
                        if last_entry.get("type") == "footer":
                            continue  # Has footer — already closed
                    except json.JSONDecodeError:
                        pass
                    incomplete.append(jsonl)
            except (OSError, json.JSONDecodeError):
                continue
        return incomplete

    @classmethod
    def recover(cls, jsonl_path: Path) -> "ProtocolRecorder":
        """Reconstruct recorder state from an incomplete JSONL file."""
        protocols_dir = jsonl_path.parent.parent  # .active/ is one level down
        recorder = cls(protocols_dir)

        max_step = 0
        references = []
        recording = None

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if entry.get("type") == "header":
                    recording = ProtocolRecording(
                        name=entry["name"],
                        recording_id=entry["recording_id"],
                        session_id=entry.get("session_id", ""),
                        username=entry.get("username", ""),
                        started_at=entry.get("started_at", ""),
                        status="in_progress",
                        version=entry.get("version", "1.0.0"),
                        log_path=jsonl_path,
                    )
                elif entry.get("type") == "step":
                    max_step = max(max_step, entry.get("step_number", 0))
                elif entry.get("type") == "reference":
                    references.append(entry)

        if recording:
            recording.step_counter = max_step
            recording.references = references
            recorder.active = recording

        return recorder

    # ── Stale Cleanup ────────────────────────────────────────────────────

    @staticmethod
    def read_header(jsonl_path: Path) -> Dict[str, Any]:
        """Read just the header line from a JSONL recording file."""
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    header = json.loads(first_line)
                    if header.get("type") == "header":
                        return header
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    @classmethod
    def cleanup_stale(
        cls, protocols_dir: Path, current_session_id: str
    ) -> Dict[str, List[str]]:
        """Auto-cleanup .active/ recordings from dead sessions.

        Recordings from a different session_id with >0 steps: move to _drafts/.
        Recordings from a different session_id with 0 steps: delete.
        Recordings matching current_session_id: leave untouched.

        Returns: {"drafted": [...], "deleted": [...], "kept": [...]}
        """
        active_dir = protocols_dir / ".active"
        result: Dict[str, List[str]] = {"drafted": [], "deleted": [], "kept": []}

        if not active_dir.exists():
            return result

        for jsonl_path in list(active_dir.glob("*.jsonl")):
            header = cls.read_header(jsonl_path)
            if not header:
                continue

            session_id = header.get("session_id", "")
            if session_id == current_session_id:
                result["kept"].append(str(jsonl_path))
                continue

            step_count = 0
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            if entry.get("type") == "step":
                                step_count += 1
                        except json.JSONDecodeError:
                            continue
            except OSError:
                continue

            if step_count == 0:
                try:
                    jsonl_path.unlink()
                    result["deleted"].append(header.get("name", str(jsonl_path.name)))
                except OSError:
                    pass
            else:
                drafts_dir = protocols_dir / "_drafts"
                drafts_dir.mkdir(parents=True, exist_ok=True)
                footer = {
                    "type": "footer",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "status": "draft",
                    "reason": "stale_auto_cleanup",
                    "total_steps": step_count,
                    "original_session_id": session_id,
                }
                try:
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(footer, ensure_ascii=False, default=str) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                    recording_id = header.get("recording_id", jsonl_path.stem)
                    dest = drafts_dir / f"{recording_id}_draft.jsonl"
                    shutil.move(str(jsonl_path), str(dest))
                    result["drafted"].append(header.get("name", recording_id))
                except OSError:
                    pass

        return result

    # ── Version Management ────────────────────────────────────────────────

    def _next_version(self, name: str) -> str:
        """Determine next semantic version for this protocol name."""
        if not self.protocols_dir.exists():
            return "1.0.0"

        versions = []
        for d in self.protocols_dir.iterdir():
            if not d.is_dir():
                continue
            match = re.match(rf"^{re.escape(name)}_v(\d+)\.(\d+)\.(\d+)$", d.name)
            if match:
                versions.append(tuple(int(x) for x in match.groups()))

        if not versions:
            return "1.0.0"

        latest = max(versions)
        return f"{latest[0]}.{latest[1]}.{latest[2] + 1}"

    # ── Internal ──────────────────────────────────────────────────────────

    def _write_entry(self, entry: dict) -> None:
        """Append entry to JSONL with fsync. Raises DiskFullError on failure."""
        if not self.active:
            return
        try:
            with open(self.active.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            if self.active:
                self.active.paused = True
            raise DiskFullError(f"Protocol write failed: {e}") from e

    def _parse_jsonl(self, log_path: Path) -> Tuple[List[dict], List[dict], dict]:
        """Parse JSONL into (steps, references, header)."""
        steps = []
        references = []
        header = {}

        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") == "header":
                    header = entry
                elif entry.get("type") == "step":
                    steps.append({
                        "step": entry["step_number"],
                        "tool": entry["tool_name"],
                        "parameters": entry.get("args", {}),
                        "result": entry.get("result_preview", ""),
                        "duration_ms": entry.get("duration_ms", 0),
                        "error": entry.get("error"),
                        "timestamp": entry.get("ts", ""),
                    })
                elif entry.get("type") == "reference":
                    references.append({
                        "query": entry.get("query", ""),
                        "sources": entry.get("sources", []),
                    })
        return steps, references, header


# ── Nudge Helper ──────────────────────────────────────────────────────────


def _looks_like_path(value: str) -> bool:
    """Heuristic: is this string likely a file path on the cluster?"""
    if not value or len(value) > 500 or "\n" in value:
        return False
    if value.startswith("/") and not value.startswith("//"):
        parts = value.split("/")
        return len(parts) >= 3 and all(len(p) < 256 for p in parts)
    return False


def _chunked_hash(path: str) -> str:
    """Compute SHA-256 using chunked reads for large files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def should_nudge(tool_name: str, recording_active: bool, already_nudged: bool) -> bool:
    """Return True if we should suggest the user start recording."""
    if recording_active:
        return False
    if already_nudged:
        return False
    return tool_name in PROCEDURAL_TOOLS


# ── Private Helpers ───────────────────────────────────────────────────────


def _truncate_result(result: Any) -> str:
    """Extract and truncate tool result to a preview string."""
    if result is None:
        return ""
    if isinstance(result, dict):
        content = result.get("content", [])
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            text = "\n".join(texts) if texts else json.dumps(result, default=str)
        else:
            text = str(content)
    elif isinstance(result, str):
        text = result
    else:
        text = str(result)
    if len(text) > RESULT_PREVIEW_MAX:
        return text[:RESULT_PREVIEW_MAX] + f"... [{len(text)} chars total]"
    return text


def _sanitize_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize tool arguments for safe storage (truncate large values)."""
    sanitized = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 2000:
            sanitized[k] = v[:2000] + f"... [{len(v)} chars]"
        else:
            sanitized[k] = v
    return sanitized


def _collect_environment() -> dict:
    """Gather system environment info for protocol metadata."""
    env = {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    # Capture loaded modules (HPC)
    loaded_modules = os.environ.get("LOADEDMODULES", "")
    if loaded_modules:
        env["modules_loaded"] = loaded_modules.split(":")
    return env


def _write_yaml(path: Path, data: dict) -> None:
    """Write protocol data as YAML-like format (without requiring PyYAML)."""
    lines = ["# Research Protocol — Auto-generated by IrisAI Protocol Recorder", ""]

    def _dump(obj, indent=0):
        prefix = "  " * indent
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{prefix}{k}:")
                    _dump(v, indent + 1)
                else:
                    lines.append(f"{prefix}{k}: {json.dumps(v, default=str)}")
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    lines.append(f"{prefix}-")
                    _dump(item, indent + 1)
                else:
                    lines.append(f"{prefix}- {json.dumps(item, default=str)}")

    _dump(data)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown(path: Path, data: dict) -> None:
    """Write protocol as human-readable markdown."""
    proto = data["protocol"]
    env = data["environment"]
    steps = data["steps"]
    references = data["references"]

    lines = [
        f"# Protocol: {proto['name']} (v{proto['version']})",
        "",
        f"**Recorded by:** {proto['recorded_by']}  ",
        f"**Date:** {proto['recorded_at'][:10]}  ",
        f"**Total steps:** {proto['total_steps']}  ",
        "",
        "## Environment",
        "",
        "| Component | Value |",
        "|-----------|-------|",
        f"| Hostname | {env.get('hostname', 'N/A')} |",
        f"| OS | {env.get('os', 'N/A')} |",
        f"| Python | {env.get('python', 'N/A')} |",
    ]

    modules = env.get("modules_loaded", [])
    if modules:
        lines.append(f"| Modules | {', '.join(modules)} |")

    lines.extend(["", "## Procedure", ""])

    for step in steps:
        lines.append(f"### Step {step['step']}: `{step['tool']}`")
        if step.get("parameters"):
            params_str = json.dumps(step["parameters"], indent=2, default=str)
            lines.append(f"**Parameters:**\n```json\n{params_str}\n```")
        if step.get("result"):
            lines.append(f"**Result:** {step['result']}")
        if step.get("error"):
            lines.append(f"**Error:** {step['error']}")
        lines.append(f"**Duration:** {step.get('duration_ms', 0):.0f}ms")
        lines.append("")

    if references:
        lines.extend(["## References", ""])
        for i, ref in enumerate(references, 1):
            lines.append(f"{i}. **Query:** \"{ref['query']}\"")
            for src in ref.get("sources", []):
                title = src.get("title", "Untitled")
                url = src.get("url", "")
                if url:
                    lines.append(f"   - [{title}]({url})")
                else:
                    lines.append(f"   - {title}")
            lines.append("")

    lines.extend([
        "---",
        "*Generated by IrisAI Protocol Recorder*",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
