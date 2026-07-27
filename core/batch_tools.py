"""Batch tools — composite operations that replace multi-step tool call patterns.

These tools encapsulate expensive read→edit→verify cycles into single tool calls,
preventing the quadratic cost growth that occurs when the LLM makes 30+ individual
tool calls (each subsequent call includes ALL prior results in context).

Architecture:
    - batch_file_edit: Applies multiple find/replace edits across files in one call,
      optionally runs a verification command, returns only a concise summary.
    - run_tests: Runs pytest with concise output (failures only by default).

Cost savings:
    - Old pattern: 7 tool calls × growing context = ~$5-10 per edit cycle
    - New pattern: 1 tool call × fixed context = ~$0.15 per edit cycle

These tools use the @tool decorator (like spend_tools.py) and export a
BATCH_TOOLS list for registration in the agent's tool pool.
"""
import json
import os
import shlex
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from core.pipeline_tool import PIPELINE_TOOLS


# ── Configuration ───────────────────────────────────────────────────────

# Maximum file size to process (prevent reading huge files)
MAX_FILE_SIZE = 200_000  # 200KB per file

# Maximum number of edits in one batch
MAX_EDITS = 30

# Maximum test output to return (prevent context bloat)
MAX_TEST_OUTPUT = 3000  # chars


@tool
def batch_file_edit(
    edits: str,
    test_command: str = "",
    work_dir: str = "",
    **_,
) -> str:
    """Apply multiple file edits in one operation and optionally verify with a test command.

    USE THIS instead of calling read_text_file + edit_file + execute_dynamic_task
    separately for each file. This tool:
    1. Applies all find/replace edits atomically
    2. Optionally runs a verification command (e.g. pytest)
    3. Returns ONLY a concise summary (~500 chars instead of 200K+ chars)

    Args:
        edits: JSON string — a list of edit objects, each with:
            - "path": absolute file path to edit
            - "find": exact text to find (must appear exactly once in file)
            - "replace": replacement text
            Example: [{"path": "/path/to/file.py", "find": "old_code", "replace": "new_code"}]
        test_command: Optional shell command to run after edits (e.g. "pytest tests/ --tb=line -q").
            If empty, no verification is run.
        work_dir: Working directory for the test command. Defaults to parent of first edited file.

    Returns:
        Concise summary: which edits succeeded/failed, test results (pass/fail + failures only).

    WHEN TO USE:
        - You need to make the same or similar changes across multiple files
        - You have a clear find/replace pattern for each edit
        - You want to verify all edits pass tests in one shot

    WHEN NOT TO USE:
        - You need to read a file first to understand its structure (use analyze_files)
        - The edit requires complex logic (use execute_dynamic_task with a Python script)
        - You're making a single edit to one file (use edit_file directly)
    """
    if not edits:
        return "Error: 'edits' is required. Provide a JSON list of edit objects."

    try:
        # Parse edits JSON
        if isinstance(edits, str):
            edit_list = json.loads(edits)
        else:
            edit_list = edits

        if not isinstance(edit_list, list):
            return "Error: 'edits' must be a JSON list of edit objects."

        if len(edit_list) > MAX_EDITS:
            return f"Error: Too many edits ({len(edit_list)}). Maximum is {MAX_EDITS}."

        if not edit_list:
            return "Error: 'edits' list is empty."

        # Validate all edits have required fields
        for i, edit in enumerate(edit_list):
            if not isinstance(edit, dict):
                return f"Error: Edit #{i+1} is not a dict."
            if "path" not in edit or "find" not in edit or "replace" not in edit:
                return f"Error: Edit #{i+1} missing required fields (path, find, replace)."

        # Validate work_dir for path containment checks (settings file is source of truth)
        from core.persistence import get_work_dir
        _security_work_dir = get_work_dir()
        work_dir_resolved = str(Path(_security_work_dir).resolve()) if _security_work_dir else ""

        # Apply edits
        results = []
        files_modified = set()

        for i, edit in enumerate(edit_list):
            path = edit["path"]
            find_text = edit["find"]
            replace_text = edit["replace"]

            try:
                # Security: resolve path and check containment within WORK_DIR
                resolved_path = str(Path(path).resolve())
                if work_dir_resolved and not resolved_path.startswith(work_dir_resolved + os.sep) and resolved_path != work_dir_resolved:
                    results.append(f"  #{i+1} SKIP: {os.path.basename(path)} — path outside WORK_DIR")
                    continue

                # Check file exists and size
                if not os.path.isfile(resolved_path):
                    results.append(f"  #{i+1} SKIP: {os.path.basename(path)} — file not found")
                    continue

                # Reject symlinks to prevent symlink attacks
                if os.path.islink(resolved_path):
                    results.append(f"  #{i+1} SKIP: {os.path.basename(path)} — symlinks not allowed")
                    continue

                file_size = os.path.getsize(resolved_path)
                if file_size > MAX_FILE_SIZE:
                    results.append(f"  #{i+1} SKIP: {os.path.basename(path)} — too large ({file_size} bytes)")
                    continue

                # Read file
                with open(resolved_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check find_text exists exactly once
                count = content.count(find_text)
                if count == 0:
                    results.append(f"  #{i+1} SKIP: {os.path.basename(path)} — pattern not found")
                    continue
                elif count > 1:
                    results.append(f"  #{i+1} SKIP: {os.path.basename(path)} — pattern found {count} times (must be unique)")
                    continue

                # Apply replacement atomically: write to temp file then rename
                new_content = content.replace(find_text, replace_text, 1)
                dir_name = os.path.dirname(resolved_path)
                fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                        tmp_f.write(new_content)
                    os.replace(tmp_path, resolved_path)
                except Exception:
                    # Clean up temp file on failure
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    raise

                files_modified.add(resolved_path)
                results.append(f"  #{i+1} OK: {os.path.basename(path)}")

            except Exception as e:
                results.append(f"  #{i+1} ERROR: {os.path.basename(path)} — {str(e)[:100]}")

        # Build summary
        ok_count = sum(1 for r in results if " OK:" in r)
        skip_count = sum(1 for r in results if " SKIP:" in r)
        error_count = sum(1 for r in results if " ERROR:" in r)

        summary_lines = [
            f"## Batch Edit Results",
            f"Applied: {ok_count}/{len(edit_list)} | Skipped: {skip_count} | Errors: {error_count}",
            "",
            "### Details:",
        ]
        summary_lines.extend(results)

        # Run test command if provided
        if test_command and ok_count > 0:
            # Determine working directory
            if not work_dir:
                work_dir = os.path.dirname(list(files_modified)[0]) if files_modified else "/tmp"

            summary_lines.append("")
            summary_lines.append(f"### Verification: `{test_command}`")

            try:
                # Use shell=False with shlex.split for safety
                cmd_args = shlex.split(test_command)
                proc = subprocess.run(
                    cmd_args,
                    shell=False,
                    cwd=work_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    timeout=120,
                )

                if proc.returncode == 0:
                    summary_lines.append("✅ PASSED")
                    # Show last few lines of output for context
                    output_lines = proc.stdout.strip().split("\n")
                    last_lines = output_lines[-5:] if len(output_lines) > 5 else output_lines
                    summary_lines.append("```")
                    summary_lines.extend(last_lines)
                    summary_lines.append("```")
                else:
                    summary_lines.append(f"❌ FAILED (exit code {proc.returncode})")
                    # Extract only failure lines
                    all_output = proc.stdout + proc.stderr
                    failure_lines = [
                        line for line in all_output.split("\n")
                        if any(kw in line for kw in ["FAILED", "ERROR", "Error", "assert", "raise"])
                    ]
                    if failure_lines:
                        summary_lines.append("Failures:")
                        summary_lines.append("```")
                        for line in failure_lines[:15]:  # Max 15 failure lines
                            summary_lines.append(line[:200])  # Truncate long lines
                        summary_lines.append("```")
                    else:
                        # Show last N chars of output
                        tail = all_output[-MAX_TEST_OUTPUT:]
                        summary_lines.append("```")
                        summary_lines.append(tail)
                        summary_lines.append("```")

            except subprocess.TimeoutExpired:
                summary_lines.append("⏰ TIMEOUT (120s limit)")
            except subprocess.CalledProcessError as e:
                summary_lines.append(f"⚠️ Test process error (exit {e.returncode}): {str(e)[:200]}")
            except (FileNotFoundError, OSError) as e:
                summary_lines.append(f"⚠️ Test command not found or OS error: {str(e)[:200]}")
            except Exception as e:
                summary_lines.append(f"⚠️ Test execution error: {str(e)[:200]}")

        return "\n".join(summary_lines)

    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in 'edits' parameter: {e}"
    except Exception as e:
        return f"Error in batch_file_edit: {str(e)[:300]}\n{traceback.format_exc()[-500:]}"


# ── run_tests: Run pytest with concise output ───────────────────────────


def _find_pytest_executable() -> str:
    """Find a Python interpreter that has pytest installed.

    Search order:
    1. PYTEST_PYTHON environment variable (explicit override)
    2. WORK_DIR/conda_envs/*/bin/python (scan for working pytest)
    3. sys.executable (fallback — may not have pytest)
    """
    # 1. Explicit override via environment variable
    pytest_python = os.environ.get("PYTEST_PYTHON", "")
    if pytest_python and os.path.isfile(pytest_python):
        return pytest_python

    # 2. Scan conda envs in work_dir for a python with pytest
    from core.persistence import get_work_dir
    work_dir = get_work_dir()
    if work_dir:
        conda_envs_dir = os.path.join(work_dir, "conda_envs")
        if os.path.isdir(conda_envs_dir):
            # Prefer irisai_dev env first, then any other
            candidates = []
            irisai_dev = os.path.join(conda_envs_dir, "irisai_dev", "bin", "python")
            if os.path.isfile(irisai_dev):
                candidates.append(irisai_dev)
            # Add other envs
            try:
                for entry in os.listdir(conda_envs_dir):
                    candidate = os.path.join(conda_envs_dir, entry, "bin", "python")
                    if candidate != irisai_dev and os.path.isfile(candidate):
                        candidates.append(candidate)
            except OSError:
                pass

            for candidate in candidates:
                try:
                    result = subprocess.run(
                        [candidate, "-m", "pytest", "--version"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        return candidate
                except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                    continue

    # 3. Fallback to sys.executable
    return sys.executable


@tool
def run_tests(test_path: str, mode: Optional[str] = "failures_only", extra_args: Optional[str] = "", **_) -> str:
    """Run pytest and return CONCISE results. Use this INSTEAD of execute_dynamic_task("pytest ...").

    This tool runs pytest and returns only what matters:
    - In 'failures_only' mode (default): pass/fail counts + only the FAILED test details
    - In 'summary' mode: just pass/fail/skip counts, zero test output
    - In 'full' mode: complete pytest output (use sparingly)

    This prevents the 60K+ chars of full pytest output from flooding your context.

    Args:
        test_path: Path to test file or directory. Examples:
            - "tests/" (run all tests)
            - "tests/test_unit.py" (specific file)
            - "tests/test_unit.py::TestClass::test_method" (specific test)
        mode: One of 'failures_only' (default), 'summary', 'full'
        extra_args: Additional pytest arguments (e.g. '-x' to stop on first failure,
            '-k pattern' to filter tests)

    Returns:
        Concise test results based on mode.
    """
    if not test_path:
        return "Error: 'test_path' is required. Provide a path to test file or directory."

    if mode not in ("failures_only", "summary", "full"):
        return f"Error: mode must be 'failures_only', 'summary', or 'full'. Got: {mode}"

    # Find a Python interpreter that has pytest installed
    pytest_python = _find_pytest_executable()
    cmd = [pytest_python, "-m", "pytest", test_path, "--tb=short", "-q"]
    if extra_args:
        cmd.extend(extra_args.split())

    # Resolve working directory for pytest:
    # - Absolute test_path: walk up from its dir to find project root (pyproject.toml/setup.py)
    # - Relative test_path: search WORK_DIR subdirs and cwd for a directory that contains it
    work_dir = None
    if os.path.isabs(test_path):
        # Absolute path — walk up to find project root marker
        search = os.path.dirname(test_path)
        while search and search != os.path.dirname(search):
            if any(os.path.isfile(os.path.join(search, f)) for f in ("pyproject.toml", "setup.py", "setup.cfg")):
                work_dir = search
                break
            search = os.path.dirname(search)
        if not work_dir:
            work_dir = os.path.dirname(test_path)
    else:
        # Relative path — search candidate directories for one that contains test_path
        candidates = []
        from core.persistence import get_work_dir
        _env_work_dir = get_work_dir()
        if _env_work_dir and os.path.isdir(_env_work_dir):
            # Check work_dir itself and its immediate subdirs (e.g. IrisAIdev_code_review/)
            candidates.append(_env_work_dir)
            try:
                for entry in os.listdir(_env_work_dir):
                    sub = os.path.join(_env_work_dir, entry)
                    if os.path.isdir(sub):
                        candidates.append(sub)
            except OSError:
                pass
        candidates.append(os.getcwd())
        for candidate in candidates:
            if os.path.exists(os.path.join(candidate, test_path)):
                work_dir = candidate
                break

    # Set up PYTHONPATH to include project's .pytest_lib if it exists
    env = os.environ.copy()
    if work_dir:
        pytest_lib = os.path.join(work_dir, ".pytest_lib")
        if os.path.isdir(pytest_lib):
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{pytest_lib}:{existing}" if existing else pytest_lib

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=work_dir,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return "ERROR: pytest timed out after 300 seconds"
    except FileNotFoundError:
        return "ERROR: pytest not found. Is it installed in the current environment?"
    except Exception as e:
        return f"ERROR running pytest: {str(e)[:300]}"

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    returncode = result.returncode

    # Parse summary line (e.g. "5 passed, 2 failed, 1 error in 3.45s")
    summary_line = ""
    for line in reversed(stdout.split("\n")):
        line = line.strip()
        if line and ("passed" in line or "failed" in line or "error" in line):
            summary_line = line
            break

    if not summary_line and stderr:
        # Check stderr for collection errors
        summary_line = f"COLLECTION ERROR: {stderr[:200]}"

    if mode == "summary":
        status = "PASS ✅" if returncode == 0 else "FAIL ❌"
        return f"Test Result: {status}\n{summary_line}"

    if mode == "full":
        # Cap at 10K chars to prevent context flooding even in full mode
        full_output = stdout + (f"\nSTDERR:\n{stderr}" if stderr else "")
        if len(full_output) > 10000:
            full_output = full_output[:10000] + f"\n... [TRUNCATED — {len(stdout)} total chars]"
        return f"Test Result: {'PASS ✅' if returncode == 0 else 'FAIL ❌'}\n{summary_line}\n\n{full_output}"

    # mode == "failures_only" (default)
    if returncode == 0:
        return f"Test Result: PASS ✅\n{summary_line}\nAll tests passed — no failures to report."

    # Extract only FAILED test sections
    lines = stdout.split("\n")
    failure_lines = []
    in_failure = False
    failure_count = 0
    max_failures_shown = 10

    for line in lines:
        if line.startswith("FAILED ") or "FAILED" in line and "::" in line:
            failure_lines.append(line)
            failure_count += 1
            in_failure = False  # FAILED lines are one-liners in -q mode
        elif line.startswith("E ") or line.startswith(">  "):
            # Error/assertion lines from traceback
            if failure_count <= max_failures_shown:
                failure_lines.append(line)
        elif line.startswith("_____") or line.startswith("====="):
            in_failure = False

    # Also grab short traceback sections
    short_failures = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("FAILED "):
            # Grab context: up to 5 lines before (traceback) + the FAILED line
            start = max(0, i - 5)
            short_failures.extend(lines[start:i + 1])
            short_failures.append("")
        i += 1

    output_parts = [
        f"Test Result: FAIL ❌",
        summary_line,
        f"\nShowing {min(failure_count, max_failures_shown)} of {failure_count} failures:",
        "",
    ]

    if short_failures:
        output_parts.extend(short_failures[:200])  # Cap lines
    elif failure_lines:
        output_parts.extend(failure_lines[:100])
    else:
        # Fallback: last 30 lines of output (usually contains the failures)
        output_parts.append("--- Last 30 lines of output ---")
        output_parts.extend(lines[-30:])

    result_text = "\n".join(output_parts)
    # Hard cap at 5K chars
    if len(result_text) > 5000:
        result_text = result_text[:5000] + "\n... [TRUNCATED]"

    return result_text


# ── Batch tools list for registration in single agent's tool pool ───────
BATCH_TOOLS = [batch_file_edit, run_tests] + PIPELINE_TOOLS
