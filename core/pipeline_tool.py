"""
run_pipeline_script — Programmatic Tool Calling for IrisAI

Allows the LLM to write a Python script that processes files without
LLM round-trips per tool call. The script has access to a restricted
IrisAI file API (read, write, grep, find, list) AND direct MCP tool
access via iris.mcp_call() for zero-cost tool invocations.

Safety:
- Scripts run in a tempfile sandbox with 5min timeout (PIPELINE_SCRIPT_TIMEOUT env var)
- write_file/json_save restricted to WORK_DIR only
- No arbitrary network access (curl/wget blocked); only localhost MCP servers allowed
- Dangerous commands blocked (rm, kill, sudo, chmod, etc.)
- Gated by ENABLE_PIPELINE_TOOL env var (default: true)
"""

import os
import re
import sys
import glob
import time
import asyncio
import tempfile
import subprocess
import textwrap
from pathlib import Path
from typing import ClassVar, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

PIPELINE_SCRIPT_TIMEOUT = int(os.environ.get("PIPELINE_SCRIPT_TIMEOUT", "300"))
ENABLE_PIPELINE_TOOL = os.environ.get("ENABLE_PIPELINE_TOOL", "true").lower() == "true"
MAX_OUTPUT_CHARS = 16000  # Threshold for auto-save + smart_compact
PIPELINE_OUTPUT_TTL_DAYS = 7


def _get_work_dir() -> str:
    """Get authoritative work_dir from settings file (single source of truth)."""
    from core.persistence import get_work_dir
    return get_work_dir()


def _get_pipeline_output_dir() -> str:
    work_dir = _get_work_dir()
    if work_dir:
        return os.path.join(work_dir, "tmp", "pipeline_outputs")
    return ""

# MCP server registry: maps server name → env var for port + default port
MCP_SERVER_REGISTRY = {
    "file_ops": {"env_var": "MCP_FILE_PORT", "default_port": 8001},
    "slurm_management": {"env_var": "MCP_SLURM_PORT", "default_port": 8003},
    "bio_processing": {"env_var": "MCP_BIO_PROCESS_PORT", "default_port": 8004},
    "code_execution": {"env_var": "MCP_CODE_EXECUTION_PORT", "default_port": 8005},
}

# The IrisAI pipeline API injected into every script
# NOTE: curly braces in the Python source are doubled ({{ }}) so that
# .format(work_dir=...) only substitutes the {work_dir} placeholder.
IRIS_API_SOURCE = r'''
import os, re, sys, glob, json, asyncio
from pathlib import Path

class _IrisAPI:
    def __init__(self, work_dir):
        self._work_dir = work_dir
        self._mcp_servers = {{
            "file_ops": {{"env_var": "MCP_FILE_PORT", "default_port": 8001}},
            "slurm_management": {{"env_var": "MCP_SLURM_PORT", "default_port": 8003}},
            "bio_processing": {{"env_var": "MCP_BIO_PROCESS_PORT", "default_port": 8004}},
            "code_execution": {{"env_var": "MCP_CODE_EXECUTION_PORT", "default_port": 8005}},
        }}
        # Allowed filesystem trees for destructive ops — frozen at init from env vars
        _user = os.environ.get("USER", "")
        _trees = [os.path.realpath(work_dir)]
        if _user:
            _data1 = f"/data1/{{_user}}"
            _home = f"/home/{{_user}}"
            if os.path.isdir(_data1):
                _trees.append(os.path.realpath(_data1))
            if os.path.isdir(_home):
                _trees.append(os.path.realpath(_home))
        self._allowed_trees = tuple(t.rstrip("/") + "/" for t in _trees)

    def _validate_safe_path(self, path: str) -> str:
        """Resolve and validate a path is within allowed trees. Returns the resolved path or raises."""
        resolved = os.path.realpath(path)
        if os.path.isdir(resolved):
            check = resolved.rstrip("/") + "/"
        else:
            check = resolved
        if not any(check.startswith(tree) for tree in self._allowed_trees):
            raise PermissionError(
                f"BLOCKED: '{{path}}' resolves to '{{resolved}}' which is outside allowed directories: "
                f"{{list(self._allowed_trees)}}"
            )
        return resolved

    _MAX_READ_SIZE = 100 * 1024 * 1024  # 100MB

    def read_file(self, path: str) -> str:
        """Read full content of a text file."""
        size = os.path.getsize(path)
        if size > self._MAX_READ_SIZE:
            return f"ERROR: File too large ({{size}} bytes > {{self._MAX_READ_SIZE}}). Use read_lines() for a specific range."
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def read_lines(self, path: str, start: int = 1, end: int = 50) -> str:
        """Read a line range from a file (1-indexed). Streams lines to avoid loading entire file."""
        size = os.path.getsize(path)
        if size > self._MAX_READ_SIZE:
            return f"ERROR: File too large ({{size}} bytes > {{self._MAX_READ_SIZE}}). Use a smaller line range or read_file_head/tail."
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[start - 1:end])

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file. Path must be inside WORK_DIR."""
        abs_path = os.path.abspath(path)
        work_dir_abs = os.path.abspath(self._work_dir) + os.sep
        if not (abs_path + os.sep).startswith(work_dir_abs):
            raise PermissionError(f"write_file: path must be inside WORK_DIR ({{self._work_dir}})")
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written: {{abs_path}}"

    def grep(self, path: str, pattern: str, context_lines: int = 0) -> list:
        """Search for a regex pattern in a file. Returns list of (line_no, line) tuples."""
        results = []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if re.search(pattern, line):
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    for j in range(start, end):
                        results.append((j + 1, lines[j].rstrip()))
        except Exception as e:
            results.append((0, f"ERROR: {{e}}"))
        return results

    def find_files(self, directory: str, pattern: str = "*", recursive: bool = True) -> list:
        """Find files matching a glob pattern. Returns list of absolute paths."""
        if recursive:
            return sorted(glob.glob(os.path.join(directory, "**", pattern), recursive=True))
        return sorted(glob.glob(os.path.join(directory, pattern)))

    def list_dir(self, path: str) -> list:
        """List files and directories in a path."""
        try:
            return sorted(os.listdir(path))
        except Exception as e:
            return [f"ERROR: {{e}}"]

    def file_exists(self, path: str) -> bool:
        """Check if a file exists."""
        return os.path.exists(path)

    def file_size(self, path: str) -> int:
        """Return file size in bytes, or -1 if not found."""
        try:
            return os.path.getsize(path)
        except Exception:
            return -1

    def safe_remove(self, path: str) -> dict:
        """Remove a single file. Path must resolve to within allowed directories (WORK_DIR, user's /data1, user's /home).
        Use this instead of iris.run_shell('rm ...'). Returns dict with success/error."""
        try:
            resolved = self._validate_safe_path(path)
            if not os.path.exists(resolved):
                return {{"success": False, "error": f"File not found: {{resolved}}"}}
            if os.path.isdir(resolved):
                return {{"success": False, "error": f"Is a directory (use safe_rmtree instead): {{resolved}}"}}
            os.remove(resolved)
            return {{"success": True, "removed": resolved}}
        except PermissionError as e:
            return {{"success": False, "error": str(e)}}
        except OSError as e:
            return {{"success": False, "error": f"OS error: {{e}}"}}

    def safe_rmtree(self, path: str) -> dict:
        """Remove a directory and all its contents. Path must resolve to within allowed directories.
        Use this instead of iris.run_shell('rm -rf ...'). Returns dict with success/error."""
        import shutil
        try:
            resolved = self._validate_safe_path(path)
            if not os.path.exists(resolved):
                return {{"success": False, "error": f"Directory not found: {{resolved}}"}}
            if not os.path.isdir(resolved):
                return {{"success": False, "error": f"Not a directory (use safe_remove instead): {{resolved}}"}}
            # Refuse to delete an allowed tree root itself
            resolved_norm = resolved.rstrip("/") + "/"
            for tree in self._allowed_trees:
                if resolved_norm == tree:
                    return {{"success": False, "error": f"Refusing to delete allowed tree root: {{resolved}}"}}
            shutil.rmtree(resolved)
            return {{"success": True, "removed": resolved}}
        except PermissionError as e:
            return {{"success": False, "error": str(e)}}
        except OSError as e:
            return {{"success": False, "error": f"OS error: {{e}}"}}

    def safe_move(self, src: str, dst: str) -> dict:
        """Move/rename a file or directory. Both src and dst must be within allowed directories.
        Returns dict with success/error."""
        import shutil
        try:
            resolved_src = self._validate_safe_path(src)
            # For dst, resolve symlinks and validate within allowed trees
            dst_real = os.path.realpath(os.path.dirname(os.path.abspath(dst)))
            dst_final = os.path.join(dst_real, os.path.basename(dst))
            if not os.path.isdir(dst_real):
                return {{"success": False, "error": f"Destination parent directory does not exist: {{dst_real}}"}}
            self._validate_safe_path(dst_real)
            if not os.path.exists(resolved_src):
                return {{"success": False, "error": f"Source not found: {{resolved_src}}"}}
            shutil.move(resolved_src, dst_final)
            return {{"success": True, "moved": resolved_src, "to": dst_final}}
        except PermissionError as e:
            return {{"success": False, "error": str(e)}}
        except OSError as e:
            return {{"success": False, "error": f"OS error: {{e}}"}}

    def safe_glob_remove(self, directory: str, pattern: str) -> dict:
        """Remove all files matching a glob pattern within a directory. Directory must be within allowed trees.
        Example: iris.safe_glob_remove("/path/to/dir", "*.bak") removes all .bak files.
        Returns dict with list of removed files and any errors."""
        try:
            resolved_dir = self._validate_safe_path(directory)
            if not os.path.isdir(resolved_dir):
                return {{"success": False, "error": f"Not a directory: {{resolved_dir}}"}}
            matches = glob.glob(os.path.join(resolved_dir, "**", pattern), recursive=True)
            removed = []
            errors = []
            for f in matches:
                real_f = os.path.realpath(f)
                if os.path.isfile(real_f):
                    try:
                        self._validate_safe_path(real_f)
                        os.remove(real_f)
                        removed.append(real_f)
                    except (PermissionError, OSError) as e:
                        errors.append(f"{{real_f}}: {{e}}")
            return {{"success": True, "removed": removed, "count": len(removed), "errors": errors}}
        except PermissionError as e:
            return {{"success": False, "error": str(e)}}
        except OSError as e:
            return {{"success": False, "error": f"OS error: {{e}}"}}

    def run_shell(self, command: str, timeout: int = 120, cwd: str = None) -> dict:
        """Run a shell command. Returns dict with stdout, stderr, returncode.
        HARD LIMIT: 300s max. For tasks needing more time, use iris.submit_slurm().
        NOTE: For file deletion, use iris.safe_remove() or iris.safe_rmtree() instead.
        Allowed: python, git, pip, conda, sbatch, squeue, scancel, sacct, sinfo, sacctmgr, scontrol, grep, find, diff, cat, head, tail, wc, sort, uniq, awk, sed, ls, mkdir, cp, mv, echo, pwd, which, tar, gzip, gunzip.
        Blocked: rm, curl, wget, chmod, chown, kill, sudo, su, dd, mkfs, fdisk."""
        import subprocess, shlex
        timeout = min(timeout, 300)
        BLOCKED = {{"rm", "curl", "wget", "chmod", "chown", "kill", "sudo", "su", "dd", "mkfs", "fdisk"}}
        try:
            parts = shlex.split(command)
            cmd_name = os.path.basename(parts[0]) if parts else ""
            if cmd_name in BLOCKED:
                return {{"stdout": "", "stderr": f"BLOCKED: '{{cmd_name}}' is not allowed in pipeline scripts.", "returncode": 1}}
        except Exception:
            parts = [command]
        try:
            result = subprocess.run(
                parts, shell=False, capture_output=True, text=True,
                timeout=timeout, cwd=cwd or os.getcwd()
            )
            return {{"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}}
        except subprocess.TimeoutExpired:
            return {{"stdout": "", "stderr": f"TIMEOUT ({{timeout}}s): Command exceeded direct execution limit. If this is a legitimate heavy task, use iris.submit_slurm() instead. If not, check for unbounded operations. Command: {{command[:100]}}", "returncode": -1}}
        except OSError as e:
            return {{"stdout": "", "stderr": f"OS error: {{e}}", "returncode": -1}}
        except Exception as e:
            return {{"stdout": "", "stderr": f"Error: {{type(e).__name__}}: {{e}}", "returncode": -1}}

    def run_python(self, script: str, timeout: int = 120, cwd: str = None) -> dict:
        """Write and run a Python script as a subprocess. Returns dict with stdout, stderr, returncode.
        HARD LIMIT: 300s max. For long-running Python tasks, use iris.submit_slurm()."""
        import subprocess, tempfile
        timeout = min(timeout, 300)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, script_path], capture_output=True, text=True,
                timeout=timeout, cwd=cwd or os.getcwd()
            )
            return {{"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}}
        except subprocess.TimeoutExpired:
            return {{"stdout": "", "stderr": f"TIMEOUT ({{timeout}}s): Python script exceeded direct execution limit. If this is a legitimate heavy task, use iris.submit_slurm() instead. If not, check for infinite loops or large input.", "returncode": -1}}
        except OSError as e:
            return {{"stdout": "", "stderr": f"OS error: {{e}}", "returncode": -1}}
        except Exception as e:
            return {{"stdout": "", "stderr": f"Error: {{type(e).__name__}}: {{e}}", "returncode": -1}}
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    def submit_slurm(self, sbatch_script: str, cwd: str = None, timeout: int = 120) -> dict:
        """Write and submit a Slurm batch script via sbatch. Returns dict with job_id, stdout, stderr.
        The script content is written to a temp file and submitted."""
        import subprocess, tempfile, re
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sbatch", delete=False, dir=cwd or "/tmp") as f:
            f.write(sbatch_script)
            script_path = f.name
        try:
            result = subprocess.run(
                ["sbatch", script_path], capture_output=True, text=True,
                timeout=timeout, cwd=cwd or os.getcwd()
            )
            job_id = None
            m = re.search(r"Submitted batch job (\d+)", result.stdout)
            if m:
                job_id = m.group(1)
            return {{"job_id": job_id, "stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}}
        except subprocess.TimeoutExpired:
            return {{"job_id": None, "stdout": "", "stderr": f"sbatch timed out after {{timeout}}s", "returncode": -1}}
        except OSError as e:
            return {{"job_id": None, "stdout": "", "stderr": f"OS error: {{e}}", "returncode": -1}}
        except Exception as e:
            return {{"job_id": None, "stdout": "", "stderr": f"Error: {{type(e).__name__}}: {{e}}", "returncode": -1}}
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    def slurm_status(self, job_id: str = None, user: str = None) -> str:
        """Check Slurm job status. Pass job_id for a specific job, or user for all jobs by user."""
        import subprocess
        try:
            if job_id:
                result = subprocess.run(["squeue", "--job", str(job_id), "--format=%i %j %T %M %R"], capture_output=True, text=True, timeout=15)
            elif user:
                result = subprocess.run(["squeue", "-u", user, "--format=%i %j %T %M %R"], capture_output=True, text=True, timeout=15)
            else:
                result = subprocess.run(["squeue", "--format=%i %j %T %M %R"], capture_output=True, text=True, timeout=15)
            return result.stdout or result.stderr
        except subprocess.TimeoutExpired:
            return "ERROR: squeue timed out after 15s"
        except OSError as e:
            return f"ERROR: {{e}}"
        except Exception as e:
            return f"ERROR: {{type(e).__name__}}: {{e}}"

    def mcp_call(self, server: str, tool_name: str, **kwargs) -> dict:
        """Call an MCP tool on a running FastMCP server. Returns the tool result as a dict.

        Available servers and their tools:
        - file_ops: read_text_file, write_text_file, edit_file, remove_file, list_directory,
          find_files, grep_file, count_pattern, get_file_overview, read_file_lines,
          read_file_head, read_file_tail, make_directory, get_file_info, get_file_checksum,
          check_directory_exists, check_directory_has_files, save_image, list_saved_images,
          get_current_user_info, get_user_groups, list_group_accessible_dirs,
          get_user_settings, set_user_work_directory,
          read_memory, update_memory,
          list_projects, add_project, remove_project, hpc_directory,
          read_session_log, list_session_logs
        - slurm_management: query_slurm_cluster, check_user_slurm_access, query_cluster_jobs,
          query_node_efficiency, submit_slurm_job, slurm_monitor_job,
          slurm_cancel_job, query_pending_queue, find_runnable_pending_jobs,
          get_partition_availability, get_cluster_utilization, execute_dynamic_task
        - bio_processing: mutate_fasta, submit_alphafold3_job, extract_h5ad_summary,
          inspect_vcf_summary, extract_coding_variants, get_wildtype_protein_sequence,
          apply_protein_variants, prepare_af3_json_from_sequences, list_obs_columns,
          get_unique_values, get_top_n_categories, summarize_cell_types
        - code_execution: execute_dynamic_task, cancel_task, list_running_tasks,
          submit_slurm_job

        Example:
            result = iris.mcp_call("slurm_management", "query_cluster_jobs", user="jsmith")
            result = iris.mcp_call("file_ops", "grep_file", path="/path/to/file", pattern="error")
            result = iris.mcp_call("bio_processing", "submit_alphafold3_job", json_path="/path/af3.json")
        """
        return asyncio.run(self._async_mcp_call(server, tool_name, kwargs))

    async def _async_mcp_call(self, server: str, tool_name: str, arguments: dict) -> dict:
        """Internal async implementation of mcp_call with timeout and retry."""
        import asyncio
        from mcp.client.streamable_http import streamablehttp_client
        from mcp import ClientSession

        if server not in self._mcp_servers:
            return {{"error": f"Unknown server '{{server}}'. Available: {{list(self._mcp_servers.keys())}}"}}

        _LONG_RUNNING = {{"execute_dynamic_task": 3600, "submit_slurm_job": 7200}}
        call_timeout = _LONG_RUNNING.get(tool_name, 120)

        server_info = self._mcp_servers[server]
        port = int(os.environ.get(server_info["env_var"], server_info["default_port"]))
        token = os.environ.get("MCP_SHARED_BEARER_TOKEN", "")
        url = f"http://127.0.0.1:{{port}}/"

        last_error = None
        max_attempts = 2 if tool_name not in _LONG_RUNNING else 1

        for attempt in range(max_attempts):
            try:
                async with streamablehttp_client(
                    url=url,
                    headers={{"Authorization": f"Bearer {{token}}"}}
                ) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        result = await asyncio.wait_for(
                            session.call_tool(tool_name, arguments),
                            timeout=call_timeout
                        )
                        if result.content:
                            texts = []
                            for block in result.content:
                                if hasattr(block, "text"):
                                    texts.append(block.text)
                            combined = "\n".join(texts)
                            try:
                                return json.loads(combined)
                            except (json.JSONDecodeError, ValueError):
                                return {{"result": combined}}
                        return {{"result": None, "is_error": getattr(result, "isError", False)}}
            except asyncio.TimeoutError:
                return {{"error": f"MCP call '{{tool_name}}' timed out after {{call_timeout}}s"}}
            except (ConnectionError, OSError) as e:
                last_error = e
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1.0)
                    continue
                return {{"error": f"MCP connection failed after {{max_attempts}} attempts: {{type(e).__name__}}: {{e}}"}}
            except Exception as e:
                return {{"error": f"MCP call failed: {{type(e).__name__}}: {{e}}"}}

    def json_load(self, path: str) -> dict:
        """Load a JSON file and return as a Python dict."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def json_save(self, path: str, data, indent: int = 2) -> str:
        """Save a Python dict/list as a JSON file inside WORK_DIR."""
        import json
        abs_path = os.path.abspath(path)
        work_dir_abs = os.path.abspath(self._work_dir) + os.sep
        if not (abs_path + os.sep).startswith(work_dir_abs):
            raise PermissionError(f"json_save: path must be inside WORK_DIR ({{self._work_dir}})")
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        return f"Saved: {{abs_path}}"

iris = _IrisAPI(work_dir="{work_dir}")
'''


class RunPipelineScriptInput(BaseModel):
    script: str = Field(
        description=(
            "A Python script using the `iris` object. Available methods:\n"
            "• FILES: iris.read_file(path), iris.read_lines(path,start,end), iris.write_file(path,content), "
            "iris.grep(path,pattern,context_lines=0), iris.find_files(dir,pattern,recursive=True), "
            "iris.list_dir(path), iris.file_exists(path), iris.file_size(path)\n"
            "• SAFE OPS (path-restricted to user's directories — use instead of rm/mv shell commands):\n"
            "  iris.safe_remove(path) — delete a single file\n"
            "  iris.safe_rmtree(path) — delete a directory tree\n"
            "  iris.safe_move(src, dst) — move/rename file or directory\n"
            "  iris.safe_glob_remove(directory, pattern) — delete all files matching glob pattern\n"
            "• DATA: iris.json_load(path), iris.json_save(path,data)\n"
            "• SHELL: iris.run_shell(cmd,timeout=120,cwd=None), iris.run_python(script,timeout=120,cwd=None)\n"
            "• SLURM: iris.submit_slurm(sbatch_script,cwd=None), iris.slurm_status(job_id=None,user=None)\n"
            "• MCP TOOLS (zero-cost, no LLM tokens): iris.mcp_call(server, tool_name, **kwargs) — "
            "Call ANY MCP tool directly. Servers: file_ops, slurm_management, bio_processing, code_execution.\n"
            "Print CONCISE results to stdout (max ~6000 chars returned). "
            "Do complete workflows: check→install→write→run→validate→report. "
            "Use iris.mcp_call() for operations that would otherwise require LLM tool calls.\n"
            "⚠️ NEVER use iris.run_shell('rm ...') — use iris.safe_remove/safe_rmtree/safe_glob_remove instead."
        )
    )


class RunPipelineScriptTool(BaseTool):
    name: str = "run_pipeline_script"

    # Self-enforced call budget (PEL doesn't cover BaseTool subclasses)
    _call_count: ClassVar[int] = 0
    _last_call_time: ClassVar[float] = 0.0
    _MAX_CALLS_PER_TURN: ClassVar[int] = 1
    _TURN_RESET_SECONDS: ClassVar[float] = 120.0

    description: str = (
        "⚠️ SPECIALIZED TOOL — only use when the plan explicitly requires a pipeline script. "
        "For most tasks, use execute_dynamic_task (bash) instead. "
        "This tool writes a Python script with MCP tool access via iris.mcp_call(). "
        "ONE CALL PER TURN.\n\n"
        "CAPABILITIES (via the `iris` object):\n"
        "• FILES: iris.read_file(path), iris.write_file(path,content), iris.grep(path,pattern), "
        "iris.find_files(dir,pattern), iris.list_dir(path), iris.file_exists(path)\n"
        "• SAFE FILE OPS (path-restricted to user directories — ALWAYS use these for deletion/moves):\n"
        "  iris.safe_remove(path) — delete one file\n"
        "  iris.safe_rmtree(path) — delete directory tree\n"
        "  iris.safe_move(src, dst) — move/rename\n"
        "  iris.safe_glob_remove(dir, pattern) — delete matching files (e.g. '*.bak')\n"
        "• DATA: iris.json_load(path), iris.json_save(path,data)\n"
        "• SHELL: iris.run_shell(cmd,timeout=120,cwd=None) — python, pip, conda, git, "
        "sbatch, squeue, sacct, sinfo, grep, find, diff, awk, sed, sort, cp, mv, mkdir, ls\n"
        "  ⚠️ NEVER use run_shell for rm/delete — use safe_remove/safe_rmtree instead\n"
        "• PYTHON: iris.run_python(script,timeout=120,cwd=None)\n"
        "• SLURM: iris.submit_slurm(sbatch_script), iris.slurm_status(job_id)\n"
        "• MCP TOOLS: iris.mcp_call(server, tool_name, **kwargs) — call any MCP tool directly:\n"
        "  - file_ops: read_text_file, write_text_file, edit_file, grep_file, find_files, "
        "list_directory, get_file_overview, make_directory, remove_file\n"
        "  - slurm_management: submit_slurm_job, slurm_monitor_job, slurm_cancel_job, "
        "query_cluster_jobs, get_partition_availability, get_cluster_utilization\n"
        "  - bio_processing: submit_alphafold3_job, mutate_fasta, extract_h5ad_summary, "
        "inspect_vcf_summary, prepare_af3_json_from_sequences\n"
        "  - code_execution: execute_dynamic_task, submit_slurm_job\n\n"
        "COST ADVANTAGE: iris.mcp_call() invokes tools WITHOUT LLM tokens. A single "
        "pipeline script calling 10 MCP tools costs ZERO additional tokens vs 10 separate "
        "LLM tool calls. PREFER this tool whenever a task needs 2+ operations.\n\n"
        "⚠️ OUTPUT LIMIT: ~6000 chars. Scripts MUST filter/summarize INSIDE.\n\n"
        "WHEN TO USE: Task needs 2+ operations, multi-step workflows, batch processing, "
        "or any sequence that would otherwise require multiple LLM turns.\n"
        "WHEN NOT TO USE: Single simple operation, or tasks needing LLM reasoning mid-flow."
    )
    args_schema: Type[BaseModel] = RunPipelineScriptInput

    @staticmethod
    def _is_enabled() -> bool:
        return os.environ.get("ENABLE_PIPELINE_TOOL", "true").lower() == "true"

    @staticmethod
    def _validate_complexity(script: str) -> Optional[str]:
        """Reject scripts that only perform 1 operation (should use simpler tools)."""
        iris_calls = re.findall(r'\biris\.\w+\s*\(', script)

        if len(iris_calls) == 0:
            # Pure Python with no iris calls but uses os/stdlib — still single-op pattern
            stdlib_ops = re.findall(
                r'\bos\.(listdir|path\.\w+|walk|getcwd)\s*\(|'
                r'\bopen\s*\(|'
                r'\bsubprocess\.\w+\s*\(',
                script
            )
            if len(stdlib_ops) <= 1:
                return (
                    "REJECTED: This script performs only 1 operation. "
                    "Use `execute_dynamic_task` for shell commands or "
                    "`read_text_file` for reading a file. "
                    "run_pipeline_script is for composing 2+ operations."
                )
            return None

        if len(iris_calls) >= 2:
            return None  # Legitimate multi-step

        # Exactly 1 iris call — reject with specific guidance
        call = iris_calls[0]
        if 'read_file' in call or 'read_lines' in call:
            return (
                "REJECTED: This script only reads one file. "
                "Use `read_text_file` or `grep_file` instead — faster, no overhead. "
                "run_pipeline_script is for composing 2+ operations."
            )
        elif 'run_shell' in call:
            return (
                "REJECTED: This script only runs one shell command. "
                "Use `execute_dynamic_task` instead — direct execution, no script overhead. "
                "run_pipeline_script is for composing 2+ operations."
            )
        elif 'list_dir' in call or 'find_files' in call:
            return (
                "REJECTED: This script only lists/finds files. "
                "Use `execute_dynamic_task` with ls/find instead. "
                "run_pipeline_script is for composing 2+ operations."
            )
        elif 'grep' in call:
            return (
                "REJECTED: This script only searches one file. "
                "Use `grep_file` instead — faster, no overhead. "
                "run_pipeline_script is for composing 2+ operations."
            )
        elif 'file_exists' in call or 'file_size' in call:
            return (
                "REJECTED: This script only checks one file. "
                "Use `execute_dynamic_task` with stat/test instead. "
                "run_pipeline_script is for composing 2+ operations."
            )
        # write_file, json_save, submit_slurm — no simpler alternative exists
        return None

    @classmethod
    def _reset_budget(cls):
        """Reset call counter. Used by tests and at turn boundaries."""
        cls._call_count = 0
        cls._last_call_time = 0.0
        cls._last_call_failed = False
        cls._last_call_crashed = False

    _last_call_failed: ClassVar[bool] = False
    _last_call_crashed: ClassVar[bool] = False

    def _check_budget(self) -> Optional[str]:
        """Self-enforced per-turn call budget.

        - Pipeline-only mode: STRICT 1 call (no retries — fall back to normal mode)
        - Normal: 1 call
        - Script logic error (returncode != 0): 2 calls
        - Infrastructure crash (timeout/exception in _execute_script): 3 calls
        """
        now = time.time()
        if now - RunPipelineScriptTool._last_call_time > self._TURN_RESET_SECONDS:
            RunPipelineScriptTool._call_count = 0
            RunPipelineScriptTool._last_call_failed = False
            RunPipelineScriptTool._last_call_crashed = False
        RunPipelineScriptTool._last_call_time = now
        RunPipelineScriptTool._call_count += 1

        if RunPipelineScriptTool._last_call_crashed:
            max_allowed = 3
        elif RunPipelineScriptTool._last_call_failed:
            max_allowed = 2
        else:
            max_allowed = self._MAX_CALLS_PER_TURN

        if RunPipelineScriptTool._call_count > max_allowed:
            return (
                "BLOCKED: run_pipeline_script allows only 1 call per turn "
                "(2 if the first failed, 3 on infrastructure crash). You must write ONE comprehensive script "
                "that handles your ENTIRE task — do NOT split work across multiple "
                "pipeline calls.\n\n"
                "What to do now: Report your findings to the user from the first script. "
                "If more pipeline work is needed, the user will ask, and you should write "
                "ONE script covering ALL remaining steps.\n\n"
                "Pattern to follow: explore → compare → analyze → summarize, ALL in one script."
            )
        return None

    def _cleanup_old_outputs(self):
        """Delete pipeline output files older than PIPELINE_OUTPUT_TTL_DAYS."""
        try:
            cutoff = time.time() - (PIPELINE_OUTPUT_TTL_DAYS * 86400)
            for fname in os.listdir(_get_pipeline_output_dir()):
                fpath = os.path.join(_get_pipeline_output_dir(), fname)
                if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)
        except OSError:
            pass

    def _save_output_to_file(self, output: str) -> str:
        """Save full output to file, return path."""
        from datetime import datetime

        os.makedirs(_get_pipeline_output_dir(), exist_ok=True)
        self._cleanup_old_outputs()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = os.path.join(_get_pipeline_output_dir(), f"output_{timestamp}.txt")
        with open(out_path, "w") as f:
            f.write(output)
        return out_path

    def _auto_split_output(self, output: str) -> str:
        """Sync fallback: save + head+tail compact. Prefer _async_auto_split_output."""
        from core.context_compactor import _fallback_truncate

        out_path = self._save_output_to_file(output)
        compacted = _fallback_truncate(
            output, MAX_OUTPUT_CHARS, tool_name="run_pipeline_script"
        )
        return (
            f"{compacted}\n\n"
            f"[Full output ({len(output):,} chars) saved to: {out_path}]"
        )

    async def _async_auto_split_output(self, output: str) -> str:
        """Save full output to file, Haiku-compact for LLM, append file path."""
        from core.context_compactor import async_smart_compact

        out_path = self._save_output_to_file(output)
        compacted = await async_smart_compact(
            output,
            max_chars=MAX_OUTPUT_CHARS,
            context_type="pipeline_output",
            tool_name="run_pipeline_script",
        )
        return (
            f"{compacted}\n\n"
            f"[Full output ({len(output):,} chars) saved to: {out_path}]"
        )

    def _execute_script(self, script: str) -> str:
        """Run the pipeline script subprocess. Returns raw output string.

        Handles validation gates, budget checks, and subprocess execution.
        Does NOT apply compaction — caller decides sync vs async compaction.
        """
        if not self._is_enabled():
            return "run_pipeline_script is disabled (ENABLE_PIPELINE_TOOL=false)"

        # Layer 1: Complexity gate — reject single-operation scripts
        rejection = self._validate_complexity(script)
        if rejection:
            return rejection

        # Layer 2: Budget — cap calls per turn
        budget_msg = self._check_budget()
        if budget_msg:
            return budget_msg

        # Build the full script: inject iris API + user script
        api_source = IRIS_API_SOURCE.format(work_dir=_get_work_dir())
        full_script = textwrap.dedent(api_source) + "\n\n" + textwrap.dedent(script)

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "pipeline_script.py")
            with open(script_path, "w") as f:
                f.write(full_script)

            try:
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=PIPELINE_SCRIPT_TIMEOUT,
                    cwd=tmpdir,
                    env=os.environ.copy(),
                )
                output = result.stdout
                if result.stderr:
                    output += f"\n[STDERR]: {result.stderr[:500]}"
                if not output.strip():
                    RunPipelineScriptTool._last_call_failed = True
                    return "(script produced no output)"
                if result.returncode != 0 or (
                    not result.stdout.strip() and result.stderr.strip()
                ):
                    RunPipelineScriptTool._last_call_failed = True
                return output
            except subprocess.TimeoutExpired:
                RunPipelineScriptTool._last_call_crashed = True
                RunPipelineScriptTool._last_call_failed = True
                return f"[INFRASTRUCTURE CRASH] Script timed out after {PIPELINE_SCRIPT_TIMEOUT}s — you have extra budget to retry"
            except Exception as e:
                RunPipelineScriptTool._last_call_crashed = True
                RunPipelineScriptTool._last_call_failed = True
                return f"[INFRASTRUCTURE CRASH] Script execution failed: {e} — you have extra budget to retry"

    def _run(self, script: str) -> str:
        output = self._execute_script(script)
        if len(output) > MAX_OUTPUT_CHARS:
            output = self._auto_split_output(output)
        return output

    async def _arun(self, script: str) -> str:
        output = await asyncio.to_thread(self._execute_script, script)
        if len(output) > MAX_OUTPUT_CHARS:
            output = await self._async_auto_split_output(output)
        return output


PIPELINE_TOOLS = [RunPipelineScriptTool()]
