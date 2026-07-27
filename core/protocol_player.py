"""Protocol Player — replay recorded protocols for reproducibility verification.

Two modes:
  Reproduce: verify same results with identical inputs/environment (hard checks)
  Transfer: apply same method to new data (soft checks, variable substitution)

No Chainlit or LLM dependencies. Pure file I/O + async execution.
"""

import asyncio
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class PlayMode(Enum):
    REPRODUCE = "reproduce"
    TRANSFER = "transfer"
    GOLDEN = "golden"


class PlayState(Enum):
    IDLE = "idle"
    PREFLIGHT = "preflight"
    EXECUTING = "executing"
    WAITING_SLURM = "waiting_slurm"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class StepResult:
    step_number: int
    tool_name: str
    parameters: Dict[str, Any]
    original_result_preview: str
    actual_result_preview: str
    duration_ms: float
    status: str  # "success" | "deviation" | "error" | "skipped"
    deviation_detail: str = ""
    timestamp: str = ""


@dataclass
class VersionCheck:
    software: str
    expected: str
    actual: str
    match: bool
    overridden: bool = False


@dataclass
class PlaySession:
    protocol_dir: Path
    mode: PlayMode
    state: PlayState = PlayState.IDLE
    current_step_index: int = 0
    completed_steps: List[StepResult] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)
    version_checks: List[VersionCheck] = field(default_factory=list)
    version_overrides: List[str] = field(default_factory=list)
    slurm_job_id: Optional[str] = None
    slurm_job_dir: Optional[str] = None
    started_at: str = ""
    finished_at: str = ""
    error_message: str = ""


SLURM_TOOLS = frozenset({"submit_slurm_job"})
CHECKSUM_CHUNK_SIZE = 65536
VARIABLE_PATTERN = re.compile(r"\{\{(\w+)\}\}")


# ── Utilities ────────────────────────────────────────────────────────────────


def compute_file_checksum(path: str, algorithm: str = "sha256") -> str:
    """Compute file hash using chunked reads (handles multi-GB files)."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHECKSUM_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def list_available_protocols(protocols_dir: Path) -> List[Dict[str, str]]:
    """List all compiled protocol directories with name, version, date."""
    if not protocols_dir.exists():
        return []

    protocols = []
    for d in sorted(protocols_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        match = re.match(r"^(.+)_v(\d+\.\d+\.\d+)$", d.name)
        if not match:
            continue
        jsonl_path = d / "recording.jsonl"
        if not jsonl_path.exists():
            continue
        name = match.group(1)
        version = match.group(2)
        mtime = datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        protocols.append({
            "name": name,
            "version": version,
            "dir_name": d.name,
            "path": str(d),
            "modified": mtime,
        })
    return protocols


def find_protocol_by_name(protocols_dir: Path, name: str, version: str = "") -> Optional[Path]:
    """Find protocol dir by name. Returns latest version if version not specified."""
    if not protocols_dir.exists():
        return None

    candidates = []
    for d in protocols_dir.iterdir():
        if not d.is_dir():
            continue
        if version:
            if d.name == f"{name}_v{version}":
                return d
        else:
            match = re.match(rf"^{re.escape(name)}_v(\d+)\.(\d+)\.(\d+)$", d.name)
            if match:
                ver_tuple = tuple(int(x) for x in match.groups())
                candidates.append((ver_tuple, d))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None


def detect_slurm_checkpoints(user_data_dir: Path) -> List[Path]:
    """Find saved play-mode SLURM checkpoint files."""
    checkpoint_dir = user_data_dir / "play_checkpoints"
    if not checkpoint_dir.exists():
        return []
    return sorted(checkpoint_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def _collect_current_environment() -> Dict[str, Any]:
    """Gather current system environment for comparison."""
    env = {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }
    loaded_modules = os.environ.get("LOADEDMODULES", "")
    if loaded_modules:
        env["modules_loaded"] = loaded_modules.split(":")
    else:
        env["modules_loaded"] = []
    return env


def _truncate(text: Any, max_len: int = 500) -> str:
    """Truncate value to a preview string."""
    if text is None:
        return ""
    s = str(text) if not isinstance(text, str) else text
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


# ── Protocol Player ──────────────────────────────────────────────────────────


class ProtocolPlayer:
    """Drives protocol replay in Reproduce or Transfer mode.

    One instance per play session (stored in cl.user_session).
    No Chainlit deps — app.py injects call_tool_fn for MCP access.
    """

    def __init__(self, protocol_dir: Path, mode: PlayMode):
        self.session = PlaySession(protocol_dir=protocol_dir, mode=mode)
        self._protocol_data: Optional[Dict[str, Any]] = None
        self._steps: List[Dict[str, Any]] = []
        self._environment: Dict[str, Any] = {}
        self._manifest: Dict[str, Any] = {}

    # ── Loading ──────────────────────────────────────────────────────────

    def load(self) -> Dict[str, Any]:
        """Load protocol from recording.jsonl in protocol_dir."""
        jsonl_path = self.session.protocol_dir / "recording.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"No recording.jsonl in {self.session.protocol_dir}")

        steps = []
        references = []
        header = {}

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
                    header = entry
                elif entry.get("type") == "step":
                    steps.append({
                        "step": entry["step_number"],
                        "tool": entry["tool_name"],
                        "parameters": entry.get("args", {}),
                        "result_preview": entry.get("result_preview", ""),
                        "duration_ms": entry.get("duration_ms", 0),
                        "error": entry.get("error"),
                        "timestamp": entry.get("ts", ""),
                    })
                elif entry.get("type") == "reference":
                    references.append(entry)

        # Try to load dataset manifest from protocol.yaml-adjacent JSON
        manifest_path = self.session.protocol_dir / "dataset_manifest.json"
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        self._protocol_data = {
            "protocol": header,
            "steps": steps,
            "references": references,
            "manifest": manifest,
        }
        self._steps = steps
        self._environment = header.get("environment", {})
        self._manifest = manifest
        return self._protocol_data

    def load_golden(self) -> Dict[str, Any]:
        """Load from golden_protocol.json for strict execution mode.

        In golden mode:
        - Steps are followed exactly as written
        - Variables are substituted before execution
        - On failure: only variable adjustment allowed (retry same step)
        """
        golden_path = self.session.protocol_dir / "golden_protocol.json"
        if not golden_path.exists():
            raise FileNotFoundError(f"No golden_protocol.json in {self.session.protocol_dir}")

        golden_doc = json.loads(golden_path.read_text(encoding="utf-8"))

        steps = []
        for s in golden_doc.get("steps", []):
            steps.append({
                "step": s["step"],
                "tool": s["tool"],
                "parameters": s.get("args", {}),
                "result_preview": "",
                "duration_ms": 0,
                "error": None,
                "timestamp": "",
                "description": s.get("description", ""),
            })

        self._protocol_data = {
            "protocol": {
                "name": golden_doc.get("name", "unknown"),
                "version": golden_doc.get("version", "1.0.0"),
                "golden": True,
                "execution_mode": golden_doc.get("execution_mode", "strict"),
                "on_failure": golden_doc.get("on_failure", "adjust_variables_only"),
            },
            "steps": steps,
            "references": [],
            "manifest": {},
            "variables": golden_doc.get("variables", []),
        }
        self._steps = steps
        self._environment = {}
        self._manifest = {}
        return self._protocol_data

    @property
    def is_golden(self) -> bool:
        """True if this is a golden protocol in strict execution mode."""
        if self._protocol_data and self._protocol_data.get("protocol"):
            return self._protocol_data["protocol"].get("golden", False)
        return False

    @property
    def golden_variables(self) -> List[Dict[str, Any]]:
        """Return the variable definitions from a golden protocol."""
        if self._protocol_data:
            return self._protocol_data.get("variables", [])
        return []

    @property
    def total_steps(self) -> int:
        return len(self._steps)

    @property
    def protocol_name(self) -> str:
        if self._protocol_data and self._protocol_data.get("protocol"):
            return self._protocol_data["protocol"].get("name", "unknown")
        return self.session.protocol_dir.name

    @property
    def protocol_version(self) -> str:
        if self._protocol_data and self._protocol_data.get("protocol"):
            return self._protocol_data["protocol"].get("version", "0.0.0")
        return "0.0.0"

    # ── Pre-flight ───────────────────────────────────────────────────────

    def run_preflight(self, variables: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Run pre-flight checks based on mode.

        Returns:
            {
                "passed": bool,
                "blockers": List[str],      # hard failures (Reproduce) or missing inputs
                "warnings": List[str],      # soft issues
                "input_checksums": List[Dict],  # {path, expected, actual, match}
                "version_checks": List[VersionCheck],
            }
        """
        self.session.state = PlayState.PREFLIGHT
        blockers = []
        warnings = []
        input_checksums = []
        version_checks = []

        if variables:
            self.session.variables = variables

        # 1. Input file verification
        if self._manifest.get("inputs"):
            for inp in self._manifest["inputs"]:
                path = inp["path"]
                # Apply variable substitution in Transfer mode
                if self.session.mode == PlayMode.TRANSFER:
                    path = self._substitute_in_string(path)

                expected_hash = inp.get("sha256", "")
                result = {"path": path, "expected": expected_hash, "actual": "", "match": False}

                if not os.path.isfile(path):
                    result["actual"] = "FILE_NOT_FOUND"
                    if self.session.mode == PlayMode.REPRODUCE:
                        blockers.append(f"Input file missing: {path}")
                    else:
                        blockers.append(f"Substituted input file missing: {path}")
                elif expected_hash and self.session.mode == PlayMode.REPRODUCE:
                    actual_hash = compute_file_checksum(path)
                    result["actual"] = actual_hash
                    result["match"] = actual_hash == expected_hash
                    if not result["match"]:
                        blockers.append(
                            f"Checksum mismatch for {path}: "
                            f"expected {expected_hash[:12]}... got {actual_hash[:12]}..."
                        )
                else:
                    result["actual"] = "(not checked in Transfer mode)"
                    result["match"] = True

                input_checksums.append(result)

        # 2. Software version verification
        current_env = _collect_current_environment()
        recorded_modules = self._environment.get("modules_loaded", [])
        current_modules = current_env.get("modules_loaded", [])

        # Compare module lists
        recorded_set = set(recorded_modules) if isinstance(recorded_modules, list) else set()
        current_set = set(current_modules) if isinstance(current_modules, list) else set()

        for mod in recorded_set:
            vc = VersionCheck(
                software=mod,
                expected=mod,
                actual=mod if mod in current_set else "(not loaded)",
                match=mod in current_set,
            )
            version_checks.append(vc)
            if not vc.match:
                if self.session.mode == PlayMode.REPRODUCE:
                    blockers.append(f"Module not loaded: {mod}")
                else:
                    warnings.append(f"Module differs: expected {mod}, not currently loaded")

        # Python version check
        recorded_python = self._environment.get("python", "")
        if recorded_python:
            current_python = current_env["python"]
            py_match = current_python == recorded_python
            version_checks.append(VersionCheck(
                software="python",
                expected=recorded_python,
                actual=current_python,
                match=py_match,
            ))
            if not py_match:
                if self.session.mode == PlayMode.REPRODUCE:
                    blockers.append(f"Python version mismatch: expected {recorded_python}, got {current_python}")
                else:
                    warnings.append(f"Python version differs: {recorded_python} → {current_python}")

        # 3. Hostname (soft warning only)
        recorded_host = self._environment.get("hostname", "")
        if recorded_host and recorded_host != current_env["hostname"]:
            warnings.append(f"Different host: recorded on {recorded_host}, running on {current_env['hostname']}")

        self.session.version_checks = version_checks
        passed = len(blockers) == 0
        if passed:
            self.session.state = PlayState.EXECUTING
            self.session.started_at = datetime.now(timezone.utc).isoformat()
        else:
            self.session.state = PlayState.FAILED

        return {
            "passed": passed,
            "blockers": blockers,
            "warnings": warnings,
            "input_checksums": input_checksums,
            "version_checks": version_checks,
        }

    def override_blocker(self, description: str) -> None:
        """User acknowledges a blocker, allowing execution to proceed (logged in report)."""
        self.session.version_overrides.append(description)
        self.session.state = PlayState.EXECUTING
        if not self.session.started_at:
            self.session.started_at = datetime.now(timezone.utc).isoformat()

    # ── Variable Substitution ────────────────────────────────────────────

    def detect_variables(self) -> List[str]:
        """Scan all step parameters for {{variable}} placeholders."""
        found = set()
        for step in self._steps:
            params = step.get("parameters", {})
            self._scan_for_variables(params, found)
        return sorted(found)

    def _scan_for_variables(self, obj: Any, found: set) -> None:
        if isinstance(obj, str):
            for match in VARIABLE_PATTERN.finditer(obj):
                found.add(match.group(1))
        elif isinstance(obj, dict):
            for v in obj.values():
                self._scan_for_variables(v, found)
        elif isinstance(obj, list):
            for item in obj:
                self._scan_for_variables(item, found)

    def _substitute_in_string(self, s: str) -> str:
        """Replace {{var}} in a string with session variables."""
        def replacer(m):
            var_name = m.group(1)
            return self.session.variables.get(var_name, m.group(0))
        return VARIABLE_PATTERN.sub(replacer, s)

    def _substitute_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-substitute {{var}} in a parameters dict."""
        return self._deep_substitute(params)

    def _deep_substitute(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self._substitute_in_string(obj)
        elif isinstance(obj, dict):
            return {k: self._deep_substitute(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_substitute(item) for item in obj]
        return obj

    # ── Execution ────────────────────────────────────────────────────────

    async def execute_next_step(self, call_tool_fn: Callable) -> StepResult:
        """Execute the current step using the injected call_tool_fn.

        call_tool_fn signature: async (tool_name: str, args: Dict) -> Any
        Returns StepResult. Advances internal state.
        """
        if self.session.state not in (PlayState.EXECUTING,):
            raise RuntimeError(f"Cannot execute in state {self.session.state.value}")

        if self.session.current_step_index >= len(self._steps):
            self.session.state = PlayState.COMPLETED
            self.session.finished_at = datetime.now(timezone.utc).isoformat()
            raise RuntimeError("All steps already executed")

        step = self._steps[self.session.current_step_index]
        tool_name = step["tool"]
        params = dict(step.get("parameters", {}))

        # Apply variable substitution in Transfer and Golden modes
        if self.session.mode in (PlayMode.TRANSFER, PlayMode.GOLDEN) and self.session.variables:
            params = self._substitute_params(params)

        start_time = time.time()
        try:
            result = await call_tool_fn(tool_name, params)
            duration_ms = (time.time() - start_time) * 1000
            result_preview = _truncate(result)

            step_result = StepResult(
                step_number=step["step"],
                tool_name=tool_name,
                parameters=params,
                original_result_preview=step.get("result_preview", ""),
                actual_result_preview=result_preview,
                duration_ms=duration_ms,
                status="success",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            step_result = StepResult(
                step_number=step["step"],
                tool_name=tool_name,
                parameters=params,
                original_result_preview=step.get("result_preview", ""),
                actual_result_preview="",
                duration_ms=duration_ms,
                status="error",
                deviation_detail=str(e),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        self.session.completed_steps.append(step_result)

        # In GOLDEN mode: don't advance on failure (allow retry with variable fix)
        if step_result.status == "error" and self.session.mode == PlayMode.GOLDEN:
            self.session.state = PlayState.PAUSED
            self.session.error_message = (
                f"Step {step_result.step_number} ({tool_name}) failed: "
                f"{step_result.deviation_detail}\n"
                f"Adjust variables and retry this step — do not skip or modify the procedure."
            )
            return step_result

        self.session.current_step_index += 1

        # Check if this was a SLURM submission
        if step_result.status == "success" and tool_name in SLURM_TOOLS:
            job_id = self._extract_job_id(result)
            if job_id:
                self.session.slurm_job_id = job_id
                self.session.slurm_job_dir = self._extract_job_dir(result)
                self.session.state = PlayState.WAITING_SLURM

        # Handle errors/deviations
        if step_result.status == "error":
            if self.session.mode == PlayMode.REPRODUCE:
                self.session.state = PlayState.FAILED
                self.session.error_message = f"Step {step_result.step_number} failed: {step_result.deviation_detail}"
            else:
                self.session.state = PlayState.PAUSED
                self.session.error_message = f"Step {step_result.step_number} failed: {step_result.deviation_detail}"

        # Check if all steps done
        if self.session.current_step_index >= len(self._steps) and self.session.state == PlayState.EXECUTING:
            self.session.state = PlayState.COMPLETED
            self.session.finished_at = datetime.now(timezone.utc).isoformat()

        return step_result

    def _extract_job_id(self, result: Any) -> Optional[str]:
        """Extract SLURM job ID from tool result."""
        if isinstance(result, dict):
            # Direct dict result
            jid = result.get("job_id")
            if jid:
                return str(jid)
            # MCP result format
            content = result.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        try:
                            data = json.loads(text)
                            if "job_id" in data:
                                return str(data["job_id"])
                        except (json.JSONDecodeError, TypeError):
                            match = re.search(r"job_id[\"']?\s*[:=]\s*[\"']?(\d+)", text)
                            if match:
                                return match.group(1)
        return None

    def _extract_job_dir(self, result: Any) -> Optional[str]:
        """Extract job directory from result."""
        if isinstance(result, dict):
            jd = result.get("job_dir")
            if jd:
                return str(jd)
            content = result.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        try:
                            data = json.loads(item.get("text", ""))
                            if "job_dir" in data:
                                return str(data["job_dir"])
                        except (json.JSONDecodeError, TypeError):
                            pass
        return None

    def resume_from_pause(self) -> None:
        """Resume execution after a pause (Transfer mode user intervention)."""
        if self.session.state == PlayState.PAUSED:
            self.session.state = PlayState.EXECUTING

    def retry_with_variables(self, updated_variables: Dict[str, str]) -> None:
        """Update variables and allow retrying the failed step (GOLDEN mode).

        Does not advance step index — the next execute_next_step() call
        will retry the same step with new variable substitutions.
        """
        if self.session.state != PlayState.PAUSED:
            raise RuntimeError("Can only retry from PAUSED state")
        if self.session.mode != PlayMode.GOLDEN:
            raise RuntimeError("Variable retry only available in GOLDEN mode")
        self.session.variables.update(updated_variables)
        self.session.state = PlayState.EXECUTING
        self.session.error_message = ""

    def resume_from_slurm(self, job_result: Dict[str, Any]) -> StepResult:
        """Process SLURM job completion and advance past the wait."""
        if self.session.state != PlayState.WAITING_SLURM:
            raise RuntimeError("Not in WAITING_SLURM state")

        finished = job_result.get("finished", False)
        status = job_result.get("status", "UNKNOWN")
        exit_code = job_result.get("exit_code", -1)

        if finished and status == "COMPLETED" and exit_code == 0:
            self.session.state = PlayState.EXECUTING
            self.session.slurm_job_id = None
            self.session.slurm_job_dir = None
            return StepResult(
                step_number=self.session.current_step_index,
                tool_name="slurm_monitor_job",
                parameters={"job_id": self.session.slurm_job_id or ""},
                original_result_preview="",
                actual_result_preview=f"Job completed: {status}, exit_code={exit_code}",
                duration_ms=0,
                status="success",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        else:
            detail = f"SLURM job {status}, exit_code={exit_code}"
            if self.session.mode == PlayMode.REPRODUCE:
                self.session.state = PlayState.FAILED
                self.session.error_message = detail
            else:
                self.session.state = PlayState.PAUSED
                self.session.error_message = detail
            return StepResult(
                step_number=self.session.current_step_index,
                tool_name="slurm_monitor_job",
                parameters={"job_id": self.session.slurm_job_id or ""},
                original_result_preview="",
                actual_result_preview=detail,
                duration_ms=0,
                status="deviation",
                deviation_detail=detail,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    # ── SLURM Checkpoint ─────────────────────────────────────────────────

    def save_checkpoint(self, user_data_dir: Path) -> Path:
        """Save play state to JSON for SLURM wait recovery."""
        checkpoint_dir = user_data_dir / "play_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "version": "1.0.0",
            "protocol_dir": str(self.session.protocol_dir),
            "mode": self.session.mode.value,
            "current_step_index": self.session.current_step_index,
            "total_steps": len(self._steps),
            "slurm_job_id": self.session.slurm_job_id,
            "slurm_job_dir": self.session.slurm_job_dir,
            "completed_steps": [
                {
                    "step_number": s.step_number,
                    "tool_name": s.tool_name,
                    "status": s.status,
                    "duration_ms": s.duration_ms,
                }
                for s in self.session.completed_steps
            ],
            "variables": self.session.variables,
            "version_overrides": self.session.version_overrides,
            "started_at": self.session.started_at,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        filename = f"{self.protocol_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = checkpoint_dir / filename
        path.write_text(json.dumps(checkpoint, indent=2, default=str), encoding="utf-8")
        return path

    @classmethod
    def resume_from_checkpoint(cls, checkpoint_path: Path) -> "ProtocolPlayer":
        """Reconstruct player from a saved checkpoint."""
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        protocol_dir = Path(data["protocol_dir"])
        mode = PlayMode(data["mode"])
        player = cls(protocol_dir, mode)
        player.load()

        player.session.state = PlayState.WAITING_SLURM
        player.session.current_step_index = data["current_step_index"]
        player.session.slurm_job_id = data.get("slurm_job_id")
        player.session.slurm_job_dir = data.get("slurm_job_dir")
        player.session.variables = data.get("variables", {})
        player.session.version_overrides = data.get("version_overrides", [])
        player.session.started_at = data.get("started_at", "")

        # Reconstruct completed steps (minimal info)
        for s in data.get("completed_steps", []):
            player.session.completed_steps.append(StepResult(
                step_number=s["step_number"],
                tool_name=s["tool_name"],
                parameters={},
                original_result_preview="",
                actual_result_preview="(restored from checkpoint)",
                duration_ms=s.get("duration_ms", 0),
                status=s["status"],
                timestamp="",
            ))

        # Remove checkpoint file after successful resume
        try:
            checkpoint_path.unlink()
        except OSError:
            pass

        return player

    # ── Output Validation (Reproduce only) ───────────────────────────────

    def validate_outputs(self) -> Dict[str, Any]:
        """Compare output file checksums against dataset manifest."""
        if not self._manifest.get("outputs"):
            return {"all_match": True, "results": [], "note": "No outputs in manifest"}

        results = []
        all_match = True

        for out in self._manifest["outputs"]:
            path = out["path"]
            expected = out.get("sha256", "")
            relevant = out.get("checksum_relevant", True)

            if not relevant:
                results.append({
                    "path": path,
                    "expected": "(not relevant)",
                    "actual": "(skipped)",
                    "match": True,
                })
                continue

            if not os.path.isfile(path):
                results.append({
                    "path": path,
                    "expected": expected[:16] + "..." if expected else "",
                    "actual": "FILE_NOT_FOUND",
                    "match": False,
                })
                all_match = False
                continue

            if expected:
                actual = compute_file_checksum(path)
                match = actual == expected
                results.append({
                    "path": path,
                    "expected": expected[:16] + "...",
                    "actual": actual[:16] + "...",
                    "match": match,
                })
                if not match:
                    all_match = False
            else:
                results.append({
                    "path": path,
                    "expected": "(none recorded)",
                    "actual": "(not compared)",
                    "match": True,
                })

        return {"all_match": all_match, "results": results}

    # ── Report Generation ────────────────────────────────────────────────

    def generate_report(self) -> str:
        """Generate Markdown reproduction/transfer report."""
        if self.session.mode == PlayMode.REPRODUCE:
            return self._generate_reproduction_report()
        return self._generate_transfer_report()

    def save_report(self) -> Path:
        """Save report to protocol directory and return path."""
        report = self.generate_report()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{'reproduction' if self.session.mode == PlayMode.REPRODUCE else 'transfer'}_report_{timestamp}.md"
        report_path = self.session.protocol_dir / filename
        report_path.write_text(report, encoding="utf-8")
        return report_path

    def _generate_reproduction_report(self) -> str:
        """Markdown report for Reproduce mode."""
        output_validation = self.validate_outputs()
        all_steps_ok = all(s.status == "success" for s in self.session.completed_steps)
        outputs_ok = output_validation["all_match"]
        verdict = "REPRODUCED" if (all_steps_ok and outputs_ok) else "DIVERGED"

        duration = ""
        if self.session.started_at and self.session.finished_at:
            try:
                start = datetime.fromisoformat(self.session.started_at)
                end = datetime.fromisoformat(self.session.finished_at)
                delta = end - start
                mins = int(delta.total_seconds() // 60)
                secs = int(delta.total_seconds() % 60)
                duration = f"{mins}m {secs}s"
            except (ValueError, TypeError):
                duration = "unknown"

        lines = [
            "# Reproduction Report",
            "",
            f"**Protocol:** {self.protocol_name} v{self.protocol_version}",
            f"**Mode:** Reproduce",
            f"**Date:** {self.session.finished_at[:10] if self.session.finished_at else 'in progress'}",
            f"**Duration:** {duration}",
            "",
            f"## Verdict: {verdict}",
            "",
        ]

        # Environment comparison
        current_env = _collect_current_environment()
        lines.extend([
            "## Environment Comparison",
            "",
            "| Component | Original | Replay | Match |",
            "|-----------|----------|--------|-------|",
            f"| Hostname | {self._environment.get('hostname', 'N/A')} | {current_env['hostname']} | {'YES' if self._environment.get('hostname') == current_env['hostname'] else 'NO'} |",
            f"| Python | {self._environment.get('python', 'N/A')} | {current_env['python']} | {'YES' if self._environment.get('python') == current_env['python'] else 'NO'} |",
        ])
        recorded_mods = ", ".join(self._environment.get("modules_loaded", [])[:5]) or "none"
        current_mods = ", ".join(current_env.get("modules_loaded", [])[:5]) or "none"
        lines.append(f"| Modules | {recorded_mods} | {current_mods} | {'YES' if recorded_mods == current_mods else 'NO'} |")
        lines.extend(["", ""])

        # Step results
        lines.extend([
            "## Step Results",
            "",
            "| Step | Tool | Status | Duration (orig/replay) |",
            "|------|------|--------|------------------------|",
        ])
        for i, sr in enumerate(self.session.completed_steps):
            orig_step = self._steps[i] if i < len(self._steps) else {}
            orig_dur = f"{orig_step.get('duration_ms', 0):.0f}ms"
            replay_dur = f"{sr.duration_ms:.0f}ms"
            status_icon = "SUCCESS" if sr.status == "success" else sr.status.upper()
            lines.append(f"| {sr.step_number} | {sr.tool_name} | {status_icon} | {orig_dur} / {replay_dur} |")
        lines.extend(["", ""])

        # Output verification
        if output_validation["results"]:
            lines.extend([
                "## Output Verification",
                "",
                "| File | Original | Replay | Match |",
                "|------|----------|--------|-------|",
            ])
            for r in output_validation["results"]:
                match_str = "YES" if r["match"] else "NO"
                lines.append(f"| {r['path']} | {r['expected']} | {r['actual']} | {match_str} |")
            lines.extend(["", ""])

        # Version overrides
        lines.append("## Version Overrides")
        lines.append("")
        if self.session.version_overrides:
            for override in self.session.version_overrides:
                lines.append(f"- {override}")
        else:
            lines.append("(none)")
        lines.extend(["", ""])

        # Deviations
        lines.append("## Deviations")
        lines.append("")
        deviations = [s for s in self.session.completed_steps if s.status != "success"]
        if deviations:
            for d in deviations:
                lines.append(f"- Step {d.step_number} ({d.tool_name}): {d.deviation_detail}")
        else:
            lines.append("(none)")
        lines.extend(["", "---", "*Generated by IrisAI Play Mode*"])

        return "\n".join(lines) + "\n"

    def _generate_transfer_report(self) -> str:
        """Markdown report for Transfer mode."""
        duration = ""
        if self.session.started_at and self.session.finished_at:
            try:
                start = datetime.fromisoformat(self.session.started_at)
                end = datetime.fromisoformat(self.session.finished_at)
                delta = end - start
                mins = int(delta.total_seconds() // 60)
                secs = int(delta.total_seconds() % 60)
                duration = f"{mins}m {secs}s"
            except (ValueError, TypeError):
                duration = "unknown"

        all_ok = all(s.status == "success" for s in self.session.completed_steps)
        verdict = "COMPLETED" if all_ok else "PARTIAL"

        lines = [
            "# Transfer Report",
            "",
            f"**Protocol:** {self.protocol_name} v{self.protocol_version}",
            f"**Mode:** Transfer",
            f"**Date:** {self.session.finished_at[:10] if self.session.finished_at else 'in progress'}",
            f"**Duration:** {duration}",
            "",
            f"## Verdict: {verdict}",
            "",
        ]

        # Variable substitutions
        lines.extend(["## Variable Substitutions", ""])
        if self.session.variables:
            lines.append("| Variable | Value |")
            lines.append("|----------|-------|")
            for var, val in self.session.variables.items():
                lines.append(f"| `{{{{{var}}}}}` | `{val}` |")
        else:
            lines.append("(no variables)")
        lines.extend(["", ""])

        # Step results
        lines.extend([
            "## Step Results",
            "",
            "| Step | Tool | Status | Duration |",
            "|------|------|--------|----------|",
        ])
        for sr in self.session.completed_steps:
            status_icon = "SUCCESS" if sr.status == "success" else sr.status.upper()
            lines.append(f"| {sr.step_number} | {sr.tool_name} | {status_icon} | {sr.duration_ms:.0f}ms |")
        lines.extend(["", ""])

        # Errors
        errors = [s for s in self.session.completed_steps if s.status != "success"]
        if errors:
            lines.extend(["## Issues", ""])
            for e in errors:
                lines.append(f"- Step {e.step_number} ({e.tool_name}): {e.deviation_detail}")
            lines.extend(["", ""])

        lines.extend(["---", "*Generated by IrisAI Play Mode*"])
        return "\n".join(lines) + "\n"

    # ── Queries ──────────────────────────────────────────────────────────

    @property
    def is_complete(self) -> bool:
        return self.session.state == PlayState.COMPLETED

    @property
    def is_failed(self) -> bool:
        return self.session.state == PlayState.FAILED

    @property
    def is_waiting_slurm(self) -> bool:
        return self.session.state == PlayState.WAITING_SLURM

    @property
    def is_paused(self) -> bool:
        return self.session.state == PlayState.PAUSED

    @property
    def can_execute(self) -> bool:
        return self.session.state == PlayState.EXECUTING

    @property
    def progress_summary(self) -> str:
        done = self.session.current_step_index
        total = len(self._steps)
        if self.is_complete:
            return f"Complete ({total}/{total} steps)"
        if self.is_waiting_slurm:
            return f"Step {done}/{total}: waiting for SLURM job {self.session.slurm_job_id}"
        if self.is_paused:
            return f"Step {done}/{total}: paused — {self.session.error_message}"
        if self.is_failed:
            return f"Failed at step {done}/{total}: {self.session.error_message}"
        if done < total:
            next_step = self._steps[done]
            return f"Step {done + 1}/{total}: {next_step['tool']}"
        return f"{done}/{total} steps"

    def abort(self) -> None:
        """User-initiated abort."""
        self.session.state = PlayState.ABORTED
        self.session.finished_at = datetime.now(timezone.utc).isoformat()
