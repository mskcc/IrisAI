"""Unified batch tool for worker agents.

Replaces scattered tools (execute_dynamic_task, batch_file_edit, run_tests,
analyze_files) with a single declarative interface. Phase-aware enforcement
ensures research workers can't write files.

Worker-only: the main agent never sees this tool.
"""
import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Type

logger = logging.getLogger(__name__)

MAX_BATCH_OUTPUT_CHARS = 12000
MAX_OPERATIONS = 20
SHELL_TIMEOUT = 120  # seconds per command
MAX_FILE_SIZE = 200_000  # 200KB
MAX_READ_FILES = 20


def _run_shell(command: str, work_dir: str = "", timeout: int = SHELL_TIMEOUT) -> dict:
    """Run a shell command via subprocess. Returns {ok, stdout, stderr, exit_code}."""
    cwd = work_dir if work_dir and os.path.isdir(work_dir) else None
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": f"TIMEOUT after {timeout}s", "exit_code": -1}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": str(e), "exit_code": -1}


def _run_edit(edits: list, work_dir: str = "") -> dict:
    """Apply find/replace edits. Returns {ok, results: list[str]}."""
    if not edits:
        return {"ok": False, "results": ["Error: empty edits list"]}

    from core.persistence import get_work_dir
    _security_work_dir = get_work_dir()
    work_dir_resolved = str(Path(_security_work_dir).resolve()) if _security_work_dir else ""

    results = []
    ok_count = 0

    for i, edit in enumerate(edits[:30]):
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
            resolved_path = str(Path(path).resolve())
            if work_dir_resolved and not resolved_path.startswith(work_dir_resolved + os.sep) and resolved_path != work_dir_resolved:
                results.append(f"  #{i+1} SKIP: {os.path.basename(path)} — path outside WORK_DIR")
                continue
            if not os.path.isfile(resolved_path):
                results.append(f"  #{i+1} ERROR: file not found: {path}")
                continue
            if os.path.islink(resolved_path):
                results.append(f"  #{i+1} SKIP: {os.path.basename(path)} — symlinks not allowed")
                continue
            if os.path.getsize(resolved_path) > MAX_FILE_SIZE:
                results.append(f"  #{i+1} SKIP: {os.path.basename(path)} — file too large")
                continue

            content = Path(resolved_path).read_text(encoding="utf-8")
            count = content.count(find_text)
            if count == 0:
                results.append(f"  #{i+1} ERROR: pattern not found in {os.path.basename(path)}")
                continue
            elif count > 1:
                results.append(f"  #{i+1} ERROR: pattern not unique in {os.path.basename(path)} (found {count} times)")
                continue

            new_content = content.replace(find_text, replace_text, 1)
            dir_name = os.path.dirname(resolved_path)
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                    tmp_f.write(new_content)
                os.replace(tmp_path, resolved_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

            results.append(f"  #{i+1} OK: {os.path.basename(path)}")
            ok_count += 1
        except Exception as e:
            results.append(f"  #{i+1} ERROR: {os.path.basename(path)} — {str(e)[:100]}")

    return {"ok": ok_count > 0, "results": results, "ok_count": ok_count, "total": len(edits)}


def _run_test(test_path: str, mode: str = "failures_only", work_dir: str = "") -> dict:
    """Run pytest. Returns {ok, output}."""
    from core.batch_tools import _find_pytest_executable

    pytest_python = _find_pytest_executable()
    cmd = [pytest_python, "-m", "pytest", test_path, "--tb=short", "-q"]

    cwd = None
    if os.path.isabs(test_path):
        search = os.path.dirname(test_path)
        while search and search != os.path.dirname(search):
            if any(os.path.isfile(os.path.join(search, f)) for f in ("pyproject.toml", "setup.py", "setup.cfg")):
                cwd = search
                break
            search = os.path.dirname(search)
        if not cwd:
            cwd = os.path.dirname(test_path)
    elif work_dir and os.path.isdir(work_dir):
        cwd = work_dir

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "TIMEOUT: pytest exceeded 300s"}
    except FileNotFoundError:
        return {"ok": False, "output": "ERROR: pytest not found"}
    except Exception as e:
        return {"ok": False, "output": f"ERROR: {str(e)[:300]}"}

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    passed = result.returncode == 0

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

    # Extract failure info
    lines = stdout.split("\n")
    failure_lines = []
    for line in lines:
        if "FAILED" in line or line.startswith("E ") or line.startswith("> "):
            failure_lines.append(line)

    output_parts = [f"FAIL: {summary_line}"]
    if failure_lines:
        output_parts.extend(failure_lines[:30])
    else:
        output_parts.extend(lines[-20:])
    if stderr:
        output_parts.append(f"STDERR: {stderr[:500]}")

    output = "\n".join(output_parts)
    if len(output) > 5000:
        output = output[:5000] + "\n[TRUNCATED]"
    return {"ok": False, "output": output}


async def _run_read(paths: list, question: str, work_dir: str = "") -> dict:
    """Read files and answer a question using Haiku LLM call. Returns {ok, output}."""
    from core.sub_agent import (
        _build_file_contents_block, build_file_analysis_prompt, _call_sub_agent_llm,
        WORKER_AGENT_MODEL, MAX_TOTAL_CONTENT,
    )

    if not paths:
        return {"ok": False, "output": "Error: no file paths provided"}
    if not question or not question.strip():
        return {"ok": False, "output": "Error: no question provided"}

    valid_paths = [p for p in paths[:MAX_READ_FILES] if isinstance(p, str) and p.strip()]
    # Resolve relative paths against work_dir
    resolved_paths = []
    for p in valid_paths:
        if os.path.isabs(p):
            resolved_paths.append(p)
        elif work_dir:
            resolved_paths.append(os.path.join(work_dir, p))
        else:
            resolved_paths.append(p)

    if not resolved_paths:
        return {"ok": False, "output": "Error: no valid file paths"}

    file_contents = _build_file_contents_block(resolved_paths)
    if not file_contents.strip():
        return {"ok": False, "output": "Error: could not read any of the specified files"}

    prompt = build_file_analysis_prompt(file_contents, question.strip(), len(resolved_paths))
    try:
        result = await _call_sub_agent_llm(prompt, model=WORKER_AGENT_MODEL)
        return {"ok": True, "output": result}
    except Exception as e:
        return {"ok": False, "output": f"Error in LLM analysis: {str(e)[:200]}"}


def _save_overflow(results_text: str, work_dir: str) -> str:
    """Save full batch results to filesystem when they exceed budget."""
    overflow_dir = os.path.join(work_dir, "tmp", "batch_results") if work_dir else "/tmp/batch_results"
    os.makedirs(overflow_dir, exist_ok=True)
    filename = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.txt"
    filepath = os.path.join(overflow_dir, filename)
    Path(filepath).write_text(results_text, encoding="utf-8")
    return filepath


def _truncate_per_op(output: str, budget: int) -> str:
    """Truncate an operation's output to fit within budget."""
    if len(output) <= budget:
        return output
    half = budget // 2
    return output[:half] + f"\n... [{len(output)} chars total, truncated] ...\n" + output[-half:]


async def execute_batch(
    operations: list,
    mode: str = "execute",
    work_dir: str = "",
) -> str:
    """Execute a batch of operations sequentially, respecting phase constraints.

    Args:
        operations: List of operation dicts with 'type' field.
        mode: "research" (read-only) or "execute" (all types).
        work_dir: Working directory for commands.

    Returns:
        Formatted results string within MAX_BATCH_OUTPUT_CHARS budget.
    """
    if not operations:
        return "Error: empty operations list. Provide at least one operation."

    if len(operations) > MAX_OPERATIONS:
        return f"Error: too many operations ({len(operations)}). Maximum is {MAX_OPERATIONS}."

    results = []
    raw_outputs = []

    for i, op in enumerate(operations):
        if not isinstance(op, dict):
            results.append({"idx": i, "ok": False, "output": "Error: operation must be a dict"})
            raw_outputs.append("Error: operation must be a dict")
            continue

        op_type = op.get("type", "")
        if not op_type:
            results.append({"idx": i, "ok": False, "output": "Error: missing 'type' field"})
            raw_outputs.append("Error: missing 'type' field")
            continue

        if op_type == "edit" and mode == "research":
            results.append({
                "idx": i, "ok": False,
                "output": "REJECTED: 'edit' operations are not allowed in research mode (read-only). Use shell/read/test only."
            })
            raw_outputs.append("REJECTED: edit not allowed in research mode")
            continue

        if op_type == "shell":
            command = op.get("command", "")
            if not command:
                results.append({"idx": i, "ok": False, "output": "Error: shell operation missing 'command'"})
                raw_outputs.append("Error: missing command")
                continue
            r = _run_shell(command, work_dir=work_dir)
            output = r["stdout"]
            if r["stderr"]:
                output += f"\nSTDERR: {r['stderr']}" if output else r["stderr"]
            if not r["ok"] and not output.strip():
                output = f"Command failed (exit {r['exit_code']})"
            results.append({"idx": i, "ok": r["ok"], "output": output, "command": command})
            raw_outputs.append(output)

        elif op_type == "read":
            paths = op.get("paths", [])
            question = op.get("question", "")
            if not paths:
                results.append({"idx": i, "ok": False, "output": "Error: read operation missing 'paths'"})
                raw_outputs.append("Error: missing paths")
                continue
            if not question:
                results.append({"idx": i, "ok": False, "output": "Error: read operation missing 'question'"})
                raw_outputs.append("Error: missing question")
                continue
            r = await _run_read(paths, question, work_dir=work_dir)
            results.append({"idx": i, "ok": r["ok"], "output": r["output"]})
            raw_outputs.append(r["output"])

        elif op_type == "edit":
            edits = op.get("edits", [])
            if not edits:
                results.append({"idx": i, "ok": False, "output": "Error: edit operation missing 'edits'"})
                raw_outputs.append("Error: missing edits")
                continue
            r = _run_edit(edits, work_dir=work_dir)
            output = f"Edits: {r.get('ok_count', 0)}/{r.get('total', 0)} applied\n" + "\n".join(r["results"])
            results.append({"idx": i, "ok": r["ok"], "output": output})
            raw_outputs.append(output)

        elif op_type == "test":
            test_path = op.get("path", "")
            test_mode = op.get("mode", "failures_only")
            if not test_path:
                results.append({"idx": i, "ok": False, "output": "Error: test operation missing 'path'"})
                raw_outputs.append("Error: missing path")
                continue
            r = _run_test(test_path, mode=test_mode, work_dir=work_dir)
            results.append({"idx": i, "ok": r["ok"], "output": r["output"]})
            raw_outputs.append(r["output"])

        else:
            results.append({"idx": i, "ok": False, "output": f"Error: unknown operation type '{op_type}'. Valid: shell, read, edit, test"})
            raw_outputs.append(f"Error: unknown type '{op_type}'")

    # Format output with budget management
    total_raw = sum(len(o) for o in raw_outputs)
    ok_count = sum(1 for r in results if r["ok"])
    fail_count = len(results) - ok_count

    header = f"## Batch Results: {ok_count} ok, {fail_count} failed ({len(results)} total)\n\n"
    per_op_budget = max(500, (MAX_BATCH_OUTPUT_CHARS - len(header) - 200) // max(1, len(results)))

    formatted_parts = [header]
    for r in results:
        idx = r["idx"]
        op = operations[idx] if idx < len(operations) else {}
        op_type = op.get("type", "?") if isinstance(op, dict) else "?"
        status = "OK" if r["ok"] else "FAIL"

        label = f"### [{idx+1}] {op_type.upper()} — {status}"
        if op_type == "shell" and "command" in r:
            label += f"\n`{r['command'][:80]}`"

        # Failed ops get priority (full error)
        if not r["ok"]:
            body = r["output"][:per_op_budget * 2]
        else:
            body = _truncate_per_op(r["output"], per_op_budget)

        formatted_parts.append(f"{label}\n{body}\n")

    full_output = "\n".join(formatted_parts)

    # Overflow to filesystem if too large
    if len(full_output) > MAX_BATCH_OUTPUT_CHARS:
        overflow_path = _save_overflow(full_output, work_dir)
        # Build truncated version
        truncated_parts = [header]
        remaining_budget = MAX_BATCH_OUTPUT_CHARS - len(header) - 200
        small_per_op = max(200, remaining_budget // max(1, len(results)))

        for r in results:
            idx = r["idx"]
            op = operations[idx] if idx < len(operations) else {}
            op_type = op.get("type", "?") if isinstance(op, dict) else "?"
            status = "OK" if r["ok"] else "FAIL"
            truncated_parts.append(f"[{idx+1}] {op_type.upper()} {status}: {_truncate_per_op(r['output'], small_per_op)}")

        truncated_parts.append(f"\n⚠️ Output truncated ({len(full_output)} chars). Full results: {overflow_path}")
        return "\n".join(truncated_parts)

    return full_output


def create_batch_tool(mode: str = "execute", work_dir: str = ""):
    """Create a BatchTool instance for worker agents.

    This creates a LangChain-compatible BaseTool with the batch interface.
    Must be called lazily (not at import time) due to langchain_core dependency.

    Args:
        mode: "research" or "execute" — controls which operation types are allowed.
        work_dir: Working directory for shell commands and overflow files.

    Returns:
        A BaseTool instance named 'batch'.
    """
    try:
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError("langchain_core and pydantic required for BatchTool")

    class BatchInput(BaseModel):
        operations: list = Field(
            description=(
                "List of operations to execute. Each must have a 'type' field.\n"
                "Types: shell, read, edit, test.\n"
                "Examples:\n"
                '  {"type": "shell", "command": "git log --oneline -20"}\n'
                '  {"type": "read", "paths": ["/path/file.py"], "question": "what does this do?"}\n'
                '  {"type": "edit", "edits": [{"path": "/f.py", "find": "old", "replace": "new"}]}\n'
                '  {"type": "test", "path": "tests/", "mode": "failures_only"}\n'
            )
        )

    class BatchTool(BaseTool):
        name: str = "batch"
        description: str = (
            "Execute multiple operations in ONE call. This is your PRIMARY tool — "
            "use it for ALL shell commands, file reads, edits, and tests.\n\n"
            "ALWAYS batch related operations together. One call with 10 operations "
            "is far better than 10 separate calls.\n\n"
            "Operations:\n"
            "  SHELL — run a command:\n"
            "    {'type': 'shell', 'command': 'git log --oneline -20'}\n"
            "    {'type': 'shell', 'command': 'grep -rn \"pattern\" src/'}\n"
            "  READ — analyze files with a question:\n"
            "    {'type': 'read', 'paths': ['/path/a.py', '/path/b.py'], 'question': 'compare'}\n"
            "  EDIT — find/replace in files (execute mode only):\n"
            "    {'type': 'edit', 'edits': [{'path': '/f.py', 'find': 'old', 'replace': 'new'}]}\n"
            "  TEST — run pytest:\n"
            "    {'type': 'test', 'path': 'tests/test_unit.py', 'mode': 'failures_only'}\n\n"
            "Output: Per-operation results. Large outputs saved to disk (path in response).\n"
            "Max operations: 20 per call."
        )
        args_schema: Type[BaseModel] = BatchInput
        _mode: str = "execute"
        _work_dir: str = ""

        model_config = {"arbitrary_types_allowed": True}

        def _run(self, operations: list, **kwargs) -> str:
            """Sync wrapper — runs the async batch execution."""
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, execute_batch(operations, self._mode, self._work_dir))
                        return future.result(timeout=600)
                else:
                    return loop.run_until_complete(execute_batch(operations, self._mode, self._work_dir))
            except RuntimeError:
                return asyncio.run(execute_batch(operations, self._mode, self._work_dir))

        async def _arun(self, operations: list, **kwargs) -> str:
            """Async execution — preferred path."""
            return await execute_batch(operations, self._mode, self._work_dir)

    tool = BatchTool()
    tool._mode = mode
    tool._work_dir = work_dir
    return tool
