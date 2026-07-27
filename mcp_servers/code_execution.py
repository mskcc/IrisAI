# Copyright 2026 Lohit Valleru and contributors at
# Memorial Sloan Kettering Cancer Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fastmcp import FastMCP
from shared_auth import StaticBearerProvider
import asyncio
import json
import re
import subprocess
import os
import signal
import uuid
from pathlib import Path
from typing import Annotated, Dict

from pydantic import Field

import pwd
import httpx

mcp = FastMCP("Dynamic Code Execution Server", auth=StaticBearerProvider())


def _get_work_dir() -> str:
    """Get authoritative work_dir. Checks env var first, then settings file.

    Env var takes precedence: set at container launch, updated by
    set_user_work_directory() in the file_ops MCP server. Settings file
    is the cross-session/cross-container fallback.
    """
    env_wd = os.environ.get("WORK_DIR", "")
    if env_wd:
        return env_wd
    try:
        username = pwd.getpwuid(os.getuid()).pw_name
        app_name = os.environ.get("IRISAI_APP_NAME", "IrisAIdev")
        settings_path = Path(f"/home/{username}/{app_name}/usersettings.json")
        if settings_path.exists():
            with open(settings_path, "r") as f:
                import json as _json
                settings = _json.load(f)
            work_dir = settings.get("work_dir", "")
            if work_dir:
                return work_dir
    except Exception:
        pass
    return ""


# ── HARD CEILING for tool output size ──
# Prevents huge stdout/stderr from overflowing the LLM context window.
# 30,000 chars ≈ 10K tokens — safe for a single tool output.
TOOL_OUTPUT_MAX_CHARS = 30_000

# ── Running task registry (P1: enables cancel_task + P3: cleanup on shutdown) ──
_running_tasks: Dict[str, asyncio.subprocess.Process] = {}


def _truncate_output(text: str, max_chars: int = TOOL_OUTPUT_MAX_CHARS) -> str:
    """Truncate tool output to prevent context window overflow.
    Keeps the beginning and end of the output (most useful parts)
    with a truncation notice in the middle.
    """
    if not text or len(text) <= max_chars:
        return text

    # Keep 60% from the beginning, 30% from the end, 10% for the notice
    head_chars = int(max_chars * 0.6)
    tail_chars = int(max_chars * 0.3)

    truncated_chars = len(text) - max_chars
    notice = (
        f"\n\n... [OUTPUT TRUNCATED: {truncated_chars:,} characters removed "
        f"({len(text):,} total chars, showing first {head_chars:,} + last {tail_chars:,})] ...\n\n"
    )

    return text[:head_chars] + notice + text[-tail_chars:]


_DIRECT_EXEC_MAX_TIMEOUT = 300  # 5-minute hard cap for direct execution
_BLOCKED_SLURM_PATTERN = re.compile(r'\b(sbatch|srun|salloc)\b')


async def _classify_timeout(command: str) -> dict:
    """Call Haiku to classify why a command timed out.

    Returns {"classification": "heavy_compute"|"bad_command"|"unknown", "reason": "..."}
    """

    prompt = (
        "A shell command was killed after exceeding a 5-minute timeout on an HPC system.\n"
        "Classify it as ONE of:\n"
        '- "heavy_compute": Legitimate long-running task (encoding, training, simulation, '
        "alignment, compilation, large data processing). Should be resubmitted as a Slurm batch job.\n"
        '- "bad_command": Overly broad or buggy operation (scanning entire filesystem, infinite loop, '
        "unbounded recursion, reading device files). Should be fixed, NOT resubmitted.\n"
        '- "unknown": Cannot determine. Advise user to check.\n\n'
        f"Command that timed out:\n```\n{command[:2000]}\n```\n\n"
        'Respond with ONLY a JSON object: {"classification": "...", "reason": "one sentence why"}'
    )

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                os.environ.get("LITELLM_BASE_URL", "http://localhost:4000") + "/v1/chat/completions",
                json={
                    "model": os.environ.get("HAIKU_MODEL_ID", "haiku"),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0,
                },
                headers={"Authorization": f"Bearer {os.environ.get('LITELLM_API_KEY', 'sk-local')}"},
                timeout=10.0,
            )
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
    except Exception:
        return {"classification": "unknown", "reason": "Classification unavailable"}


@mcp.tool
async def execute_dynamic_task(
    command: Annotated[str, Field(description="Shell command to execute via bash. Multiple commands separated by newlines or &&. Example: 'squeue -u $USER' or 'cd /work && python train.py'.", min_length=1)],
    work_dir: Annotated[str, Field(description="Working directory base. Leave empty to use default WORK_DIR. You do NOT need to pass this.", default="")],
    timeout: Annotated[int, Field(description="Max execution time in seconds. HARD LIMIT: 300 (5 min). For longer tasks use submit_slurm_job.", default=300, ge=1, le=300)],
    task_name: Annotated[str, Field(description="Name for the task directory. Alphanumeric and underscores only.", default="dynamic_task", pattern=r"^[a-zA-Z0-9_]+$")],
    run_in_directory: Annotated[str, Field(description="Directory to cd into before running the command. Use this when your command needs to access files created by previous tasks. Must be within WORK_DIR.", default="")] = "",
) -> dict:
    """Execute a SINGLE shell command. For one-off tasks: running a script, checking status, installing a package. IMPORTANT: If you need to run 2 or more commands, use the 'batch' tool instead — it handles multiple operations in ONE call and is far more efficient. Do NOT call this tool multiple times in sequence; use batch. HARD LIMIT: 5 minutes (300s). For GPU/long tasks use submit_slurm_job. Returns dict with stdout, stderr, return_code, task_dir, and task_id."""

    if timeout > _DIRECT_EXEC_MAX_TIMEOUT:
        return {
            "error": (
                f"REJECTED: requested timeout ({timeout}s) exceeds 5-minute limit for direct execution. "
                "Use submit_slurm_job for tasks needing more than 5 minutes."
            ),
            "suggestion": "submit_slurm_job",
            "requested_timeout": timeout,
            "max_allowed": _DIRECT_EXEC_MAX_TIMEOUT,
        }

    blocked_match = _BLOCKED_SLURM_PATTERN.search(command)
    if blocked_match:
        blocked_cmd = blocked_match.group(1)
        return {
            "error": (
                f"REJECTED: Direct use of '{blocked_cmd}' is forbidden in execute_dynamic_task. "
                "All Slurm jobs MUST run inside containers. "
                "Use submit_slurm_job instead — it enforces container wrapping automatically."
            ),
            "suggestion": "submit_slurm_job",
            "blocked_command": blocked_cmd,
        }

    timeout = min(timeout, _DIRECT_EXEC_MAX_TIMEOUT)

    if not work_dir:
        work_dir = _get_work_dir()
        if not work_dir:
            return {"error": "No work directory configured. Set one with set_user_work_directory()."}

    # Constrain work_dir to MCP WORK_DIR
    mcp_work_dir = os.environ.get("WORK_DIR", "")
    if mcp_work_dir:
        resolved_wd = str(Path(work_dir).resolve())
        resolved_mcp = str(Path(mcp_work_dir).resolve())
        if not resolved_wd.startswith(resolved_mcp + os.sep) and resolved_wd != resolved_mcp:
            work_dir = mcp_work_dir

    # Validate run_in_directory if provided — must be within work_dir
    effective_run_dir = ""
    if run_in_directory:
        resolved_run = str(Path(run_in_directory).resolve())
        resolved_wd = str(Path(work_dir).resolve())
        if resolved_run.startswith(resolved_wd + os.sep) or resolved_run == resolved_wd:
            effective_run_dir = run_in_directory
            Path(effective_run_dir).mkdir(parents=True, exist_ok=True)
        else:
            return {"error": f"run_in_directory must be within work_dir ({work_dir})"}

    try:
        # Create unique task directory
        task_id = str(uuid.uuid4())[:8]
        task_dir = Path(work_dir) / "dynamic_tasks" / task_name / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # ── Detect if command needs full environment setup ──
        # Only set up conda/mamba/spack if the command actually references them.
        # For simple scripts (Python, shell, data analysis), skip the heavy setup.
        needs_full_env = any(
            keyword in command.lower()
            for keyword in [
                "conda", "mamba", "spack", "create_conda_env",
                "install_with_package_manager", "download_data",
                "pip install", "apt install", "dnf install",
            ]
        )
        
        # ── Build the script ──
        # Lightweight preamble for all tasks (no set -x, minimal output)
        script_parts = [f"""#!/bin/bash
set -e

# ── Slurm submission guard ──
# Exported functions override sbatch/srun/salloc in this process AND all child scripts.
# All Slurm jobs MUST go through submit_slurm_job (container-enforced).
sbatch() {{ echo "BLOCKED: Direct sbatch is forbidden. Use the submit_slurm_job tool." >&2; return 1; }}
srun() {{ echo "BLOCKED: Direct srun is forbidden. Use the submit_slurm_job tool." >&2; return 1; }}
salloc() {{ echo "BLOCKED: Direct salloc is forbidden. Use the submit_slurm_job tool." >&2; return 1; }}
export -f sbatch srun salloc

export PATH=${SINGULARITY_DIR:-/usr/local/bin}:$PATH
export TASK_DIR={task_dir}
export TASK_NAME={task_name}
export TASK_ID={task_id}
cd "$TASK_DIR"
"""]
        
        # Only add heavy environment setup if command needs it
        if needs_full_env:
            script_parts.append("""
# =============================================================================
# PACKAGE MANAGER SETUP (only loaded because command references conda/pip/etc.)
# =============================================================================

# Initialize conda/mamba if available (path-agnostic)
if command -v conda >/dev/null 2>&1; then
    export CONDA_ENVS_PATH="$TASK_DIR/conda_envs"
    export CONDA_PKGS_DIRS="$TASK_DIR/conda_pkgs"
    eval "$(conda shell.bash hook 2>/dev/null)" || true
fi

if command -v mamba >/dev/null 2>&1; then
    eval "$(mamba shell hook --shell bash 2>/dev/null)" || true
fi

# Initialize spack if available (path-agnostic)
if command -v spack >/dev/null 2>&1; then
    export SPACK_USER_CONFIG_PATH="$TASK_DIR/spack_config"
    export SPACK_USER_CACHE_PATH="$TASK_DIR/spack_cache"
    eval "$(spack env activate --sh 2>/dev/null)" || true
fi

# Local installation prefix
export LOCAL_PREFIX="$TASK_DIR/local"
export PATH="$LOCAL_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$LOCAL_PREFIX/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$LOCAL_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
export CMAKE_PREFIX_PATH="$LOCAL_PREFIX:$CMAKE_PREFIX_PATH"

# Create standard directories
mkdir -p {software,data,scripts,results,notebooks,docs,tmp}
mkdir -p conda_envs conda_pkgs spack_config spack_cache
mkdir -p local/{bin,lib,include,share}

# Helper functions
create_conda_env() {
    local env_name=${1:-"default"}
    local python_version=${2:-"3.11"}
    if command -v mamba >/dev/null 2>&1; then
        echo "Creating conda environment with mamba: $env_name"
        mamba create -p "$CONDA_ENVS_PATH/$env_name" python=$python_version -y
    elif command -v conda >/dev/null 2>&1; then
        echo "Creating conda environment: $env_name"
        conda create -p "$CONDA_ENVS_PATH/$env_name" python=$python_version -y
    else
        echo "Error: conda/mamba not available"
        return 1
    fi
}

install_with_package_manager() {
    local packages="$1"
    if command -v mamba >/dev/null 2>&1; then
        mamba install -c conda-forge -c bioconda $packages -y
    elif command -v conda >/dev/null 2>&1; then
        conda install -c conda-forge -c bioconda $packages -y
    elif command -v dnf >/dev/null 2>&1; then
        dnf install -y $packages
    elif command -v apt >/dev/null 2>&1; then
        apt update && apt install -y $packages
    else
        echo "No package manager available"
        return 1
    fi
}

download_data() {
    local url=$1
    local filename=${2:-$(basename "$url")}
    echo "Downloading data: $filename"
    cd "$TASK_DIR/data"
    wget "$url" -O "$filename"
    echo "✓ Downloaded: $TASK_DIR/data/$filename"
}

test_command() {
    local cmd=$1
    if command -v "$cmd" >/dev/null 2>&1; then
        echo "✓ $cmd is available: $(which $cmd)"
        $cmd --version 2>/dev/null || $cmd -v 2>/dev/null || echo "  (version info not available)"
    else
        echo "✗ $cmd not found"
        return 1
    fi
}

export -f create_conda_env install_with_package_manager download_data test_command

echo "Full environment loaded (conda/mamba/spack available)"
echo ""
""")
        
        # Add the user's command
        run_dir_cd = f'cd "{effective_run_dir}"\n' if effective_run_dir else ""
        script_parts.append(f"""
# =============================================================================
# USER TASK EXECUTION
# =============================================================================

{run_dir_cd}{command}

echo ""
echo "Task completed: $(date)"
""")
        
        script_content = "\n".join(script_parts)
        
        script_file = task_dir / "execute_task.sh"
        script_file.write_text(script_content)
        os.chmod(script_file, 0o755)
        
        # ── P0 FIX: Non-blocking subprocess execution ──
        # Uses asyncio.create_subprocess_exec() instead of subprocess.run().
        # This keeps the event loop responsive — the MCP server can handle
        # heartbeats, new requests, and HTTP stream keepalive while the
        # subprocess runs. Without this, tasks >2-3 minutes cause the HTTP
        # stream to timeout and disconnect the MCP session.
        proc = await asyncio.create_subprocess_exec(
            "bash", str(script_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(task_dir),
            # Start new process group so we can kill the entire tree on cancel/timeout
            preexec_fn=os.setsid
        )
        
        # Register in running tasks (P1: enables cancel_task)
        _running_tasks[task_id] = proc
        
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            result_stdout = stdout_bytes.decode('utf-8', errors='replace')
            result_stderr = stderr_bytes.decode('utf-8', errors='replace')
            returncode = proc.returncode
        except asyncio.TimeoutError:
            # Kill the entire process group (not just the shell, but all children)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                proc.kill()
            await proc.wait()

            # Classify the timeout via Haiku
            classification = await _classify_timeout(command)
            cls = classification.get("classification", "unknown")
            reason = classification.get("reason", "")

            if cls == "heavy_compute":
                msg = (
                    f"TIMEOUT ({timeout}s): {reason} "
                    "This task requires dedicated HPC resources. "
                    "Resubmit using submit_slurm_job with appropriate cores/memory/time_limit."
                )
            elif cls == "bad_command":
                msg = (
                    f"TIMEOUT ({timeout}s): {reason} "
                    "This appears to be an overly broad or buggy operation. "
                    "Fix the command and narrow its scope — do NOT resubmit as a Slurm job."
                )
            else:
                msg = (
                    f"TIMEOUT ({timeout}s): Task exceeded 5-minute direct execution limit. {reason} "
                    "If this is a legitimate heavy workload, resubmit via submit_slurm_job. "
                    "Otherwise, check for unintended recursion or large input."
                )

            return {"error": msg, "task_id": task_id, "task_dir": str(task_dir), "classification": cls}
        finally:
            _running_tasks.pop(task_id, None)
        
        # ── TRUNCATE OUTPUT: Prevent context window overflow ──
        # stdout and stderr go into tool result → into history → into LLM context.
        # Without truncation, a command that prints 500KB would overflow the context.
        raw_stdout = result_stdout
        raw_stderr = result_stderr
        stdout_truncated = len(raw_stdout) > TOOL_OUTPUT_MAX_CHARS
        stderr_truncated = len(raw_stderr) > TOOL_OUTPUT_MAX_CHARS
        
        # Collect results — lean output (like Claude Code's Bash tool)
        task_results = {
            "success": returncode == 0,
            "stdout": _truncate_output(raw_stdout),
            "stderr": _truncate_output(raw_stderr) if result_stderr.strip() else "",
            "return_code": returncode,
            "task_id": task_id,
            "task_dir": str(task_dir),
        }

        if stdout_truncated or stderr_truncated:
            task_results["output_truncated"] = True

        # List files created (lightweight — just names, no content)
        try:
            files = [str(p.relative_to(task_dir)) for p in task_dir.rglob("*") if p.is_file() and p.name != "execute_task.sh"]
            if files:
                task_results["files_created"] = files[:30]
        except Exception:
            pass

        return task_results
        
    except asyncio.TimeoutError:
        # This shouldn't be reached (handled above), but kept as safety net
        return {"error": f"Task timed out after {timeout} seconds", "task_id": task_id, "task_dir": str(task_dir)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
async def cancel_task(
    task_id: Annotated[str, Field(description="The task_id returned by execute_dynamic_task. Must be a valid UUID prefix (8 chars).")],
) -> dict:
    """Cancel a running task by its task_id. Sends SIGTERM then SIGKILL if needed. Returns dict with cancellation status."""
    proc = _running_tasks.get(task_id)
    if proc is None:
        return {"status": "not_found", "task_id": task_id, "message": "No running task with this ID (may have already completed)"}
    
    try:
        # Send SIGTERM to the entire process group
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        
        # Wait up to 5 seconds for graceful shutdown
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            # Force kill if SIGTERM didn't work
            os.killpg(pgid, signal.SIGKILL)
            await proc.wait()
        
        _running_tasks.pop(task_id, None)
        return {"status": "cancelled", "task_id": task_id, "message": "Task cancelled successfully"}
    except ProcessLookupError:
        _running_tasks.pop(task_id, None)
        return {"status": "already_exited", "task_id": task_id, "message": "Process already exited"}
    except Exception as e:
        return {"status": "error", "task_id": task_id, "message": str(e)}


@mcp.tool
async def list_running_tasks() -> dict:
    """List all currently running tasks.

        Returns dict with list of running task IDs and their PIDs."""
    tasks = []
    for task_id, proc in list(_running_tasks.items()):
        if proc.returncode is None:  # Still running
            tasks.append({"task_id": task_id, "pid": proc.pid})
        else:
            # Process finished but wasn't cleaned up yet
            _running_tasks.pop(task_id, None)
    return {"running_tasks": tasks, "count": len(tasks)}


# ══════════════════════════════════════════════════════════════════════════════
# BATCH TOOL — Execute multiple operations in ONE tool call
# ══════════════════════════════════════════════════════════════════════════════

_BATCH_MAX_OPS = 20
_BATCH_MAX_FILE_SIZE = 200_000  # 200KB per file for edits
_BATCH_MAX_EDITS = 30  # max edits per edit operation
_BATCH_PER_OP_TIMEOUT = 120  # default per-op timeout


async def _batch_run_shell(command: str, work_dir: str, timeout: int) -> dict:
    """Run a shell command asynchronously with process group isolation.

    Same infrastructure as execute_dynamic_task but lightweight (no script file,
    no task directory, no conda setup). For quick shell commands.
    """
    if not command or not command.strip():
        return {"ok": False, "output": "Error: empty command"}

    blocked = _BLOCKED_SLURM_PATTERN.search(command)
    if blocked:
        return {
            "ok": False,
            "output": f"BLOCKED: '{blocked.group(1)}' forbidden. Use submit_slurm_job.",
        }

    cwd = work_dir if work_dir and os.path.isdir(work_dir) else None

    try:
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            preexec_fn=os.setsid,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            ok = proc.returncode == 0

            output = stdout
            if stderr.strip():
                output += f"\nSTDERR: {stderr}" if output.strip() else stderr
            if not ok and not output.strip():
                output = f"Command failed (exit {proc.returncode})"

            return {"ok": ok, "output": output}

        except asyncio.TimeoutError:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                proc.kill()
            await proc.wait()
            return {"ok": False, "output": f"TIMEOUT after {timeout}s. Command killed."}

    except Exception as e:
        return {"ok": False, "output": f"Error: {str(e)[:300]}"}


def _batch_run_read(path: str, work_dir: str) -> dict:
    """Read a text file (up to 200KB). Returns content or error."""
    try:
        p = Path(path).resolve()
        if not p.is_file():
            return {"ok": False, "output": f"Error: not a file: {path}"}
        size = p.stat().st_size
        if size > _BATCH_MAX_FILE_SIZE:
            return {"ok": False, "output": f"Error: file too large ({size} bytes, max {_BATCH_MAX_FILE_SIZE}). Use grep or read_file_lines."}
        content = p.read_text(encoding="utf-8", errors="replace")
        return {"ok": True, "output": content}
    except Exception as e:
        return {"ok": False, "output": f"Error: {str(e)[:300]}"}


def _batch_run_grep(path: str, pattern: str, max_matches: int = 30) -> dict:
    """Grep a file for a regex pattern. Returns matching lines with numbers."""
    import re as _re
    try:
        p = Path(path).resolve()
        if not p.is_file():
            return {"ok": False, "output": f"Error: not a file: {path}"}
        regex = _re.compile(pattern)
        matches = []
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                if regex.search(line):
                    matches.append(f"{lineno}: {line.rstrip()}")
                    if len(matches) >= max_matches:
                        break
        if not matches:
            return {"ok": True, "output": f"No matches for '{pattern}' in {path}"}
        return {"ok": True, "output": "\n".join(matches)}
    except _re.error as e:
        return {"ok": False, "output": f"Error: invalid regex '{pattern}': {e}"}
    except Exception as e:
        return {"ok": False, "output": f"Error: {str(e)[:300]}"}


def _batch_run_find(pattern: str, start_path: str, max_results: int = 50) -> dict:
    """Find files matching a glob pattern. Returns list of paths."""
    try:
        start = Path(start_path).resolve()
        if not start.is_dir():
            return {"ok": False, "output": f"Error: not a directory: {start_path}"}
        results = []
        for p in start.rglob(pattern):
            results.append(str(p))
            if len(results) >= max_results:
                break
        if not results:
            return {"ok": True, "output": f"No files matching '{pattern}' in {start_path}"}
        return {"ok": True, "output": "\n".join(results)}
    except Exception as e:
        return {"ok": False, "output": f"Error: {str(e)[:300]}"}


def _batch_run_list(path: str, show_hidden: bool = False) -> dict:
    """List directory contents. Returns formatted listing."""
    try:
        p = Path(path).resolve()
        if not p.is_dir():
            return {"ok": False, "output": f"Error: not a directory: {path}"}
        items = []
        for item in sorted(p.iterdir()):
            if not show_hidden and item.name.startswith('.'):
                continue
            suffix = "/" if item.is_dir() else ""
            try:
                size = item.stat().st_size
                items.append(f"{item.name}{suffix}  ({size} bytes)")
            except OSError:
                items.append(f"{item.name}{suffix}")
            if len(items) >= 200:
                items.append(f"... (truncated)")
                break
        return {"ok": True, "output": "\n".join(items) if items else "(empty directory)"}
    except Exception as e:
        return {"ok": False, "output": f"Error: {str(e)[:300]}"}


def _batch_run_edit(edits: list, work_dir: str) -> dict:
    """Apply find/replace edits atomically. Sync file I/O (fast, non-blocking in practice)."""
    import tempfile as _tempfile

    if not edits:
        return {"ok": False, "output": "Error: empty edits list"}
    if len(edits) > _BATCH_MAX_EDITS:
        return {"ok": False, "output": f"Error: too many edits ({len(edits)}). Max {_BATCH_MAX_EDITS}."}

    work_dir_resolved = str(Path(work_dir).resolve()) if work_dir else ""
    results = []
    ok_count = 0

    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            results.append(f"  #{i+1} ERROR: not a dict")
            continue
        path = edit.get("path", "")
        find_text = edit.get("find", "")
        replace_text = edit.get("replace", "")
        if not path or not find_text:
            results.append(f"  #{i+1} ERROR: missing 'path' or 'find'")
            continue

        try:
            # Check symlink BEFORE resolving
            if os.path.islink(path):
                results.append(f"  #{i+1} BLOCKED: symlinks not allowed")
                continue

            resolved = str(Path(path).resolve())

            # Work_dir confinement
            if work_dir_resolved and not resolved.startswith(work_dir_resolved + os.sep) and resolved != work_dir_resolved:
                results.append(f"  #{i+1} BLOCKED: path outside WORK_DIR")
                continue
            if not os.path.isfile(resolved):
                results.append(f"  #{i+1} ERROR: file not found: {os.path.basename(path)}")
                continue
            if os.path.getsize(resolved) > _BATCH_MAX_FILE_SIZE:
                results.append(f"  #{i+1} BLOCKED: file too large (>{_BATCH_MAX_FILE_SIZE//1000}KB)")
                continue

            content = Path(resolved).read_text(encoding="utf-8")
            count = content.count(find_text)
            if count == 0:
                results.append(f"  #{i+1} ERROR: pattern not found in {os.path.basename(path)}")
                continue
            elif count > 1:
                results.append(f"  #{i+1} ERROR: pattern not unique ({count} matches) in {os.path.basename(path)}")
                continue

            new_content = content.replace(find_text, replace_text, 1)
            dir_name = os.path.dirname(resolved)
            fd, tmp_path = _tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                    tmp_f.write(new_content)
                os.replace(tmp_path, resolved)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

            results.append(f"  #{i+1} OK: {os.path.basename(path)}")
            ok_count += 1
        except Exception as e:
            results.append(f"  #{i+1} ERROR: {str(e)[:100]}")

    output = f"Edits: {ok_count}/{len(edits)} applied\n" + "\n".join(results)
    return {"ok": ok_count > 0, "output": output}


async def _batch_run_test(test_path: str, mode: str, work_dir: str, timeout: int) -> dict:
    """Run pytest asynchronously with concise output."""
    if not test_path:
        return {"ok": False, "output": "Error: no test path"}

    # Find pytest — check work_dir conda envs first, fall back to system
    python_exec = "python3"
    if work_dir:
        conda_dir = Path(work_dir) / "conda_envs"
        if conda_dir.exists():
            for env_dir in sorted(conda_dir.iterdir()):
                candidate = env_dir / "bin" / "python"
                if candidate.exists():
                    python_exec = str(candidate)
                    break

    cmd = [python_exec, "-m", "pytest", test_path, "--tb=short", "-q"]
    cwd = work_dir if work_dir and os.path.isdir(work_dir) else None

    # If test_path is absolute, use its project root as cwd
    if os.path.isabs(test_path):
        search = os.path.dirname(test_path)
        while search and search != os.path.dirname(search):
            if any(os.path.isfile(os.path.join(search, f)) for f in ("pyproject.toml", "setup.py", "setup.cfg")):
                cwd = search
                break
            search = os.path.dirname(search)
        if not cwd:
            cwd = os.path.dirname(test_path)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            preexec_fn=os.setsid,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                proc.kill()
            await proc.wait()
            return {"ok": False, "output": f"TIMEOUT: pytest exceeded {timeout}s"}

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        passed = proc.returncode == 0

        # Extract summary line
        summary_line = ""
        for line in reversed(stdout.split("\n")):
            line_s = line.strip()
            if line_s and ("passed" in line_s or "failed" in line_s or "error" in line_s):
                summary_line = line_s
                break

        if mode == "summary":
            return {"ok": passed, "output": f"{'PASS' if passed else 'FAIL'}: {summary_line}"}

        if passed:
            return {"ok": True, "output": f"PASS: {summary_line}"}

        # Failure details
        lines = stdout.split("\n")
        failure_lines = [l for l in lines if "FAILED" in l or l.startswith("E ") or l.startswith("> ")]
        output_parts = [f"FAIL: {summary_line}"]
        if failure_lines:
            output_parts.extend(failure_lines[:30])
        else:
            output_parts.extend(lines[-20:])
        if stderr.strip():
            output_parts.append(f"STDERR: {stderr[:500]}")

        output = "\n".join(output_parts)
        return {"ok": False, "output": output[:5000]}

    except FileNotFoundError:
        return {"ok": False, "output": f"ERROR: {python_exec} not found or pytest not installed"}
    except Exception as e:
        return {"ok": False, "output": f"ERROR: {str(e)[:300]}"}


@mcp.tool
async def batch(
    operations: Annotated[list, Field(
        description=(
            "List of operations to execute sequentially. Each operation is an object with a 'type' field.\n\n"
            "OPERATION TYPES:\n"
            "  shell — Run a shell command:\n"
            '    {"type": "shell", "command": "git log --oneline -20"}\n'
            "  read — Read a text file (up to 200KB):\n"
            '    {"type": "read", "path": "/abs/path/file.py"}\n'
            "  grep — Search file for regex pattern:\n"
            '    {"type": "grep", "path": "/abs/path/file.py", "pattern": "def main"}\n'
            "  find — Find files by glob pattern:\n"
            '    {"type": "find", "pattern": "*.py", "start_path": "/abs/path/"}\n'
            "  list — List directory contents:\n"
            '    {"type": "list", "path": "/abs/path/"}\n'
            "  edit — Find/replace in files (unique match required):\n"
            '    {"type": "edit", "edits": [{"path": "/abs/path.py", "find": "old", "replace": "new"}]}\n'
            "  test — Run pytest:\n"
            '    {"type": "test", "path": "tests/test_unit.py", "mode": "failures_only"}\n\n'
            "RULE: When you need 2+ reads/greps/finds/commands, put them ALL in one batch call.\n"
            "One batch(10 ops) saves 9 round-trips vs calling tools individually.\n\n"
            "EXAMPLE — git investigation:\n"
            '  [{"type": "shell", "command": "git log --oneline -30"},\n'
            '   {"type": "shell", "command": "git shortlog -sn --all"},\n'
            '   {"type": "shell", "command": "git tag -l"},\n'
            '   {"type": "shell", "command": "git diff --stat HEAD~10"}]\n\n'
            "Max 20 operations per call. Later operations can depend on earlier ones."
        ),
        min_length=1,
    )],
    timeout: Annotated[int, Field(
        description="Max seconds for EACH shell/test operation. Default 120, max 300.",
        default=120, ge=10, le=300,
    )] = 120,
) -> dict:
    """Execute multiple operations (shell, edit, test) in ONE tool call — the PRIMARY tool for all multi-step work. ALWAYS use this when you need to edit files, run tests, or combine reads+writes. One batch call with 5 ops saves 4 round-trips. Operations execute sequentially — later ops see effects of earlier ones. Returns per-operation results with ok/fail status."""

    # Accept both list (native from Claude/GPT-OSS after sanitization) and string (backward compat)
    if isinstance(operations, str):
        try:
            operations = json.loads(operations)
        except (json.JSONDecodeError, TypeError) as e:
            return {"error": f"Invalid JSON in operations: {str(e)[:200]}. Must be a JSON array of objects."}
        if not isinstance(operations, list):
            return {"error": "operations must be a list of operation objects"}

    ops = operations
    if not ops:
        return {"error": "Empty operations list. Provide at least one operation."}
    if len(ops) > _BATCH_MAX_OPS:
        return {"error": f"Too many operations ({len(ops)}). Maximum is {_BATCH_MAX_OPS}."}

    work_dir = _get_work_dir()
    if not work_dir:
        return {"error": "No work directory configured. Set one with set_user_work_directory()."}

    # Constrain to MCP WORK_DIR
    mcp_work_dir = os.environ.get("WORK_DIR", "")
    if mcp_work_dir:
        resolved_wd = str(Path(work_dir).resolve())
        resolved_mcp = str(Path(mcp_work_dir).resolve())
        if not resolved_wd.startswith(resolved_mcp + os.sep) and resolved_wd != resolved_mcp:
            work_dir = mcp_work_dir

    results = []
    ok_count = 0
    fail_count = 0

    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            results.append({"idx": i, "type": "?", "ok": False, "output": "Error: operation must be an object"})
            fail_count += 1
            continue

        op_type = op.get("type", "")
        if not op_type:
            results.append({"idx": i, "type": "?", "ok": False, "output": "Error: missing 'type' field"})
            fail_count += 1
            continue

        if op_type == "shell":
            command = op.get("command", "")
            if not command:
                results.append({"idx": i, "type": "shell", "ok": False, "output": "Error: missing 'command'"})
                fail_count += 1
                continue
            r = await _batch_run_shell(command, work_dir, timeout)
            r["output"] = _truncate_output(r["output"], TOOL_OUTPUT_MAX_CHARS // max(1, len(ops)))
            results.append({"idx": i, "type": "shell", "ok": r["ok"], "output": r["output"]})
            if r["ok"]:
                ok_count += 1
            else:
                fail_count += 1

        elif op_type == "edit":
            edits = op.get("edits", [])
            if not edits:
                results.append({"idx": i, "type": "edit", "ok": False, "output": "Error: missing 'edits' list"})
                fail_count += 1
                continue
            r = _batch_run_edit(edits, work_dir)
            results.append({"idx": i, "type": "edit", "ok": r["ok"], "output": r["output"]})
            if r["ok"]:
                ok_count += 1
            else:
                fail_count += 1

        elif op_type == "test":
            test_path = op.get("path", "")
            test_mode = op.get("mode", "failures_only")
            if not test_path:
                results.append({"idx": i, "type": "test", "ok": False, "output": "Error: missing 'path'"})
                fail_count += 1
                continue
            r = await _batch_run_test(test_path, test_mode, work_dir, timeout)
            r["output"] = _truncate_output(r["output"], TOOL_OUTPUT_MAX_CHARS // max(1, len(ops)))
            results.append({"idx": i, "type": "test", "ok": r["ok"], "output": r["output"]})
            if r["ok"]:
                ok_count += 1
            else:
                fail_count += 1

        elif op_type == "read":
            read_path = op.get("path", "")
            if not read_path:
                results.append({"idx": i, "type": "read", "ok": False, "output": "Error: missing 'path'"})
                fail_count += 1
                continue
            r = _batch_run_read(read_path, work_dir)
            r["output"] = _truncate_output(r["output"], TOOL_OUTPUT_MAX_CHARS // max(1, len(ops)))
            results.append({"idx": i, "type": "read", "ok": r["ok"], "output": r["output"]})
            if r["ok"]:
                ok_count += 1
            else:
                fail_count += 1

        elif op_type == "grep":
            grep_path = op.get("path", "")
            grep_pattern = op.get("pattern", "")
            if not grep_path or not grep_pattern:
                results.append({"idx": i, "type": "grep", "ok": False, "output": "Error: missing 'path' or 'pattern'"})
                fail_count += 1
                continue
            max_m = op.get("max_matches", 30)
            r = _batch_run_grep(grep_path, grep_pattern, max_m)
            r["output"] = _truncate_output(r["output"], TOOL_OUTPUT_MAX_CHARS // max(1, len(ops)))
            results.append({"idx": i, "type": "grep", "ok": r["ok"], "output": r["output"]})
            if r["ok"]:
                ok_count += 1
            else:
                fail_count += 1

        elif op_type == "find":
            find_pattern = op.get("pattern", "")
            find_start = op.get("start_path", work_dir)
            if not find_pattern:
                results.append({"idx": i, "type": "find", "ok": False, "output": "Error: missing 'pattern'"})
                fail_count += 1
                continue
            max_r = op.get("max_results", 50)
            r = _batch_run_find(find_pattern, find_start, max_r)
            results.append({"idx": i, "type": "find", "ok": r["ok"], "output": r["output"]})
            if r["ok"]:
                ok_count += 1
            else:
                fail_count += 1

        elif op_type == "list":
            list_path = op.get("path", work_dir)
            show_hidden = op.get("show_hidden", False)
            r = _batch_run_list(list_path, show_hidden)
            results.append({"idx": i, "type": "list", "ok": r["ok"], "output": r["output"]})
            if r["ok"]:
                ok_count += 1
            else:
                fail_count += 1

        else:
            results.append({
                "idx": i, "type": op_type, "ok": False,
                "output": f"Error: unknown type '{op_type}'. Valid: shell, edit, test, read, grep, find, list"
            })
            fail_count += 1

    return {
        "summary": {"ok": ok_count, "failed": fail_count, "total": len(results)},
        "results": results,
    }


# ── Read-only command validation for batch_readonly ──

_WRITE_INDICATORS = [
    re.compile(r'[^<\d]>(?![&>])'),
    re.compile(r'>>'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?rm\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?rmdir\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?mv\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?cp\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?chmod\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?chown\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?truncate\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?tee\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?dd\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?install\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?mkdir\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?touch\b'),
    re.compile(r'\bsed\b[^|;]*-i'),
    re.compile(r'\bgit\s+(?:reset|push|rebase|merge|commit|stash\s+drop|branch\s+-[dD]|checkout\s+--)'),
    re.compile(r'(?:os\.remove|os\.unlink|shutil\.rmtree|shutil\.move)\b'),
    re.compile(r'open\s*\([^)]*["\']w'),
    re.compile(r'\bpip\s+install\b'),
    re.compile(r'\bconda\s+install\b'),
    re.compile(r'\bnpm\s+install\b'),
]
_SAFE_REDIRECT_PATTERN = re.compile(r'\d*>{1,2}\s*/dev/null')
_SAFE_FD_DUP_PATTERN = re.compile(r'\d*>&\d+')

_READONLY_BATCH_TYPES = {"shell", "read", "grep", "find", "list"}


def _is_readonly_command(commands: str) -> bool:
    """Check whether a shell command string contains only read-only operations."""
    if not commands:
        return True
    cleaned = _SAFE_REDIRECT_PATTERN.sub('', commands)
    cleaned = _SAFE_FD_DUP_PATTERN.sub('', cleaned)
    for pattern in _WRITE_INDICATORS:
        if pattern.search(cleaned):
            return False
    return True


@mcp.tool
async def batch_readonly(
    operations: Annotated[list, Field(
        description=(
            "List of READ-ONLY operations. Each must have a 'type' field.\n\n"
            "OPERATION TYPES (read-only only):\n"
            "  shell — Run a read-only shell command:\n"
            '    {"type": "shell", "command": "grep -rn pattern src/"}\n'
            "  read — Read a text file (up to 200KB):\n"
            '    {"type": "read", "path": "/abs/path/file.py"}\n'
            "  grep — Search file for regex pattern:\n"
            '    {"type": "grep", "path": "/abs/path/file.py", "pattern": "def main"}\n'
            "  find — Find files by glob pattern:\n"
            '    {"type": "find", "pattern": "*.py", "start_path": "/abs/path/"}\n'
            "  list — List directory contents:\n"
            '    {"type": "list", "path": "/abs/path/"}\n\n'
            "NOT supported: edit, test, or commands that write/modify files.\n"
            "Allowed shell commands: grep, find, cat, head, tail, ls, wc, file, stat, du, df,\n"
            "  squeue, sacct, sinfo, git log/status/diff/show, python -c (read-only), tree\n\n"
            "RULE: In research/planning phases, put 2+ read operations in one batch_readonly call.\n"
            "NOTE: If you need to EDIT files or run TESTS, use 'batch' (not batch_readonly).\n\n"
            "EXAMPLE — investigate a project:\n"
            '  [{"type": "shell", "command": "find . -name \'*.py\' | head -20"},\n'
            '   {"type": "shell", "command": "grep -rn \'class.*Error\' src/"},\n'
            '   {"type": "read", "path": "/abs/path/src/config.py"}]\n\n'
            "Max 20 operations per call."
        ),
        min_length=1,
    )],
    timeout: Annotated[int, Field(
        description="Max seconds for EACH shell operation. Default 120, max 300.",
        default=120, ge=10, le=300,
    )] = 120,
) -> dict:
    """Execute multiple READ-ONLY operations in ONE tool call — for RESEARCH and PLANNING phases only. If you need to edit files, write code, or run tests, use 'batch' instead (not this tool). This tool ONLY allows reading: shell (grep, find, cat, ls, git log), read, grep, find, list. It BLOCKS edits and tests."""

    # Accept both list (native from Claude/GPT-OSS after sanitization) and string (backward compat)
    if isinstance(operations, str):
        try:
            operations = json.loads(operations)
        except (json.JSONDecodeError, TypeError) as e:
            return {"error": f"Invalid JSON in operations: {str(e)[:200]}. Must be a JSON array of objects."}
        if not isinstance(operations, list):
            return {"error": "operations must be a list of operation objects"}

    ops = operations
    if not ops:
        return {"error": "Empty operations list. Provide at least one operation."}
    if len(ops) > _BATCH_MAX_OPS:
        return {"error": f"Too many operations ({len(ops)}). Maximum is {_BATCH_MAX_OPS}."}

    work_dir = _get_work_dir()
    if not work_dir:
        return {"error": "No work directory configured. Set one with set_user_work_directory()."}

    mcp_work_dir = os.environ.get("WORK_DIR", "")
    if mcp_work_dir:
        resolved_wd = str(Path(work_dir).resolve())
        resolved_mcp = str(Path(mcp_work_dir).resolve())
        if not resolved_wd.startswith(resolved_mcp + os.sep) and resolved_wd != resolved_mcp:
            work_dir = mcp_work_dir

    results = []
    ok_count = 0
    fail_count = 0

    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            results.append({"idx": i, "type": "?", "ok": False, "output": "Error: operation must be an object"})
            fail_count += 1
            continue

        op_type = op.get("type", "")
        if not op_type:
            results.append({"idx": i, "type": "?", "ok": False, "output": "Error: missing 'type' field"})
            fail_count += 1
            continue

        if op_type == "edit":
            results.append({"idx": i, "type": "edit", "ok": False,
                            "output": "BLOCKED: 'edit' operations are NOT available in research/planning phase."})
            fail_count += 1
            continue

        if op_type == "test":
            results.append({"idx": i, "type": "test", "ok": False,
                            "output": "BLOCKED: 'test' operations are NOT available in research/planning phase."})
            fail_count += 1
            continue

        if op_type not in _READONLY_BATCH_TYPES:
            results.append({"idx": i, "type": op_type, "ok": False,
                            "output": f"Error: type '{op_type}' not allowed in read-only mode. Valid: {', '.join(sorted(_READONLY_BATCH_TYPES))}"})
            fail_count += 1
            continue

        if op_type == "shell":
            command = op.get("command", "")
            if not command:
                results.append({"idx": i, "type": "shell", "ok": False, "output": "Error: missing 'command'"})
                fail_count += 1
                continue
            if not _is_readonly_command(command):
                results.append({"idx": i, "type": "shell", "ok": False,
                                "output": f"BLOCKED: write operation detected in '{command[:80]}'. Only read-only commands allowed."})
                fail_count += 1
                continue
            r = await _batch_run_shell(command, work_dir, timeout)
            r["output"] = _truncate_output(r["output"], TOOL_OUTPUT_MAX_CHARS // max(1, len(ops)))
            results.append({"idx": i, "type": "shell", "ok": r["ok"], "output": r["output"]})

        elif op_type == "read":
            read_path = op.get("path", "")
            if not read_path:
                results.append({"idx": i, "type": "read", "ok": False, "output": "Error: missing 'path'"})
                fail_count += 1
                continue
            r = _batch_run_read(read_path, work_dir)
            r["output"] = _truncate_output(r["output"], TOOL_OUTPUT_MAX_CHARS // max(1, len(ops)))
            results.append({"idx": i, "type": "read", "ok": r["ok"], "output": r["output"]})

        elif op_type == "grep":
            grep_path = op.get("path", "")
            grep_pattern = op.get("pattern", "")
            if not grep_path or not grep_pattern:
                results.append({"idx": i, "type": "grep", "ok": False, "output": "Error: missing 'path' or 'pattern'"})
                fail_count += 1
                continue
            max_m = op.get("max_matches", 30)
            r = _batch_run_grep(grep_path, grep_pattern, max_m)
            r["output"] = _truncate_output(r["output"], TOOL_OUTPUT_MAX_CHARS // max(1, len(ops)))
            results.append({"idx": i, "type": "grep", "ok": r["ok"], "output": r["output"]})

        elif op_type == "find":
            find_pattern = op.get("pattern", "")
            find_start = op.get("start_path", work_dir)
            if not find_pattern:
                results.append({"idx": i, "type": "find", "ok": False, "output": "Error: missing 'pattern'"})
                fail_count += 1
                continue
            max_r = op.get("max_results", 50)
            r = _batch_run_find(find_pattern, find_start, max_r)
            results.append({"idx": i, "type": "find", "ok": r["ok"], "output": r["output"]})

        elif op_type == "list":
            list_path = op.get("path", work_dir)
            show_hidden = op.get("show_hidden", False)
            r = _batch_run_list(list_path, show_hidden)
            results.append({"idx": i, "type": "list", "ok": r["ok"], "output": r["output"]})

        if results[-1]["ok"]:
            ok_count += 1
        else:
            fail_count += 1

    return {
        "summary": {"ok": ok_count, "failed": fail_count, "total": len(results)},
        "results": results,
    }


# ── P3: Cleanup running tasks on server shutdown ──
async def _cleanup_running_tasks():
    """Kill all running subprocesses on server shutdown.
    Prevents orphan processes from continuing after the session ends."""
    for task_id, proc in list(_running_tasks.items()):
        if proc.returncode is None:  # Still running
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
    _running_tasks.clear()


# Register cleanup via asyncio shutdown hook
import atexit

def _sync_cleanup():
    """Synchronous cleanup wrapper for atexit."""
    if _running_tasks:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule cleanup in the running loop
                loop.create_task(_cleanup_running_tasks())
            else:
                loop.run_until_complete(_cleanup_running_tasks())
        except RuntimeError:
            # No event loop available — do synchronous kill
            for task_id, proc in list(_running_tasks.items()):
                if proc.returncode is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
            _running_tasks.clear()

atexit.register(_sync_cleanup)


if __name__ == "__main__":
    port = int(os.environ.get("MCP_CODE_EXECUTION_PORT", 8005))
    mcp.run(transport="streamable-http", host="127.0.0.1", port=port, path="/")
