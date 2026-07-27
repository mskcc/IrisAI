"""Read-only shell tool for research and planning phases.

Wraps the MCP execute_dynamic_task tool with read-only command validation.
The tool NAME (execute_shell_readonly) communicates the constraint to the LLM
upfront, and the description lists what's allowed. This is preventive enforcement
via naming + description, with a validation backstop.

Large output handling: The NativeAgentExecutor's _trim_observation method handles
writing large outputs to disk (harness-level write). The read-only constraint
applies only to what the LLM's command does, not what the harness does with output.
"""

import logging
import re
from typing import Any, Type

logger = logging.getLogger(__name__)

# Write indicators — commands that modify the filesystem.
# Reuses the same patterns from policy_enforcement.py but consolidated here
# as the single source of truth for the read-only shell.
_WRITE_INDICATORS = [
    re.compile(r'[^<\d]>(?![&>])'),          # redirect to file (not >> or >&)
    re.compile(r'>>'),                         # append redirect
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
    re.compile(r'\bsed\b[^|;]*-i'),           # in-place sed
    re.compile(r'\bgit\s+(?:reset|push|rebase|merge|commit|stash\s+drop|branch\s+-[dD]|checkout\s+--)'),
    re.compile(r'(?:os\.remove|os\.unlink|shutil\.rmtree|shutil\.move)\b'),
    re.compile(r'open\s*\([^)]*["\']w'),       # Python open with write mode
    re.compile(r'\bpip\s+install\b'),          # package installation
    re.compile(r'\bconda\s+install\b'),
    re.compile(r'\bnpm\s+install\b'),
]

# Safe patterns that look like writes but aren't (stripped before checking)
_SAFE_REDIRECT_PATTERN = re.compile(r'\d*>{1,2}\s*/dev/null')
_SAFE_FD_DUP_PATTERN = re.compile(r'\d*>&\d+')


def is_readonly_command(commands: str) -> bool:
    """Check whether a shell command string contains only read-only operations.

    Returns True if no write patterns are detected.
    """
    if not commands:
        return True

    # Strip safe redirects (e.g., 2>/dev/null, 2>&1)
    cleaned = _SAFE_REDIRECT_PATTERN.sub('', commands)
    cleaned = _SAFE_FD_DUP_PATTERN.sub('', cleaned)

    for pattern in _WRITE_INDICATORS:
        if pattern.search(cleaned):
            return False
    return True


def create_readonly_shell_tool(shell_tool):
    """Create a read-only shell tool wrapping an existing execute_dynamic_task tool.

    Args:
        shell_tool: The original execute_dynamic_task MCP tool object.
                   Must have an _arun method that accepts command=str.

    Returns:
        A new tool object with name="execute_shell_readonly" that validates
        commands before execution.
    """
    try:
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, Field
    except ImportError:
        logger.warning("langchain_core not available — returning stub readonly shell")
        return _create_stub_readonly_shell(shell_tool)

    class ReadOnlyShellInput(BaseModel):
        command: str = Field(
            description="Shell command to execute (read-only only)"
        )
        work_dir: str = Field(
            default="",
            description="Working directory for command execution (optional)"
        )

    class ReadOnlyShellTool(BaseTool):
        """Execute read-only shell commands for research and planning phases."""
        name: str = "execute_shell_readonly"
        description: str = (
            "Execute read-only shell commands for investigation and exploration. "
            "Supported: ls, cat, head, tail, grep, find, wc, file, stat, du, df, "
            "squeue, sacct, sinfo, scontrol show, git log, git status, git diff, "
            "git show, python -c (read-only), module list, env, echo, which, "
            "tree, less, more. "
            "NOT supported: writing files (>), rm, mv, cp, mkdir, touch, chmod, "
            "sed -i, tee, pip install, git push/commit/reset, or any command that "
            "modifies the filesystem. Use this tool to explore and gather information."
        )
        args_schema: Type[BaseModel] = ReadOnlyShellInput
        _shell_tool: Any = None

        class Config:
            arbitrary_types_allowed = True

        def _run(self, **kwargs) -> str:
            raise NotImplementedError("Use async _arun")

        async def _arun(self, **kwargs: Any) -> str:
            command = kwargs.get("command", "") or kwargs.get("commands", "")
            if not command or not command.strip():
                return "Error: No command provided."

            if not is_readonly_command(command):
                logger.info(
                    f"[READONLY_SHELL] Blocked write command: {command[:100]}"
                )
                return (
                    "BLOCKED: This command contains write operations which are not "
                    "available in research/planning phase. Only read-only commands "
                    "are supported. Remove any file writes, redirects (>), "
                    "destructive operations (rm, mv, cp), or package installs. "
                    "Use ls, cat, grep, find, git log, squeue, etc. instead."
                )

            # Pass through to the real shell tool
            try:
                result = await self._shell_tool._arun(**kwargs)
                return result
            except Exception as e:
                return f"Error executing command: {type(e).__name__}: {str(e)}"

    tool = ReadOnlyShellTool()
    tool._shell_tool = shell_tool
    return tool


def _create_stub_readonly_shell(shell_tool):
    """Fallback for environments without langchain_core."""

    class StubReadOnlyShell:
        name = "execute_shell_readonly"
        description = (
            "Execute read-only shell commands. "
            "Supports: ls, cat, grep, find, squeue, git log/status/diff. "
            "Does NOT support: writing, rm, mv, cp, mkdir, sed -i, redirects."
        )

        def __init__(self, inner_tool):
            self._shell_tool = inner_tool

        async def _arun(self, **kwargs):
            command = kwargs.get("command", "") or kwargs.get("commands", "")
            if not is_readonly_command(command):
                return (
                    "BLOCKED: Write operations not available in this phase. "
                    "Use read-only commands only."
                )
            return await self._shell_tool._arun(**kwargs)

    return StubReadOnlyShell(shell_tool)


# ══════════════════════════════════════════════════════════════════════════════
# BATCH READONLY — research/plan phase wrapper for the batch MCP tool
# ══════════════════════════════════════════════════════════════════════════════

def create_readonly_batch_tool(batch_tool):
    """Create a read-only batch tool wrapping the full-power batch MCP tool.

    Follows the same pattern as create_readonly_shell_tool:
    - Blocks 'edit' operations entirely
    - Blocks 'test' operations (may have side effects)
    - Validates 'shell' commands are read-only using is_readonly_command()
    - Passes validated operations through to the real batch tool

    Args:
        batch_tool: The original batch MCP tool object.
                   Must have an _arun method that accepts operations=str.

    Returns:
        A new tool with name="batch_readonly" for research/plan phases.
    """
    try:
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, Field
    except ImportError:
        logger.warning("langchain_core not available — returning stub batch_readonly")
        return _create_stub_readonly_batch(batch_tool)

    import json as _json

    class BatchReadonlyInput(BaseModel):
        operations: list = Field(
            description=(
                "List of READ-ONLY operations. Each must have a 'type' field.\n"
                "Allowed types: shell (read-only commands only)\n"
                "Examples:\n"
                '  [{"type": "shell", "command": "grep -rn pattern src/"},\n'
                '   {"type": "shell", "command": "find . -name \'*.py\' | wc -l"},\n'
                '   {"type": "shell", "command": "cat src/main.py"},\n'
                '   {"type": "shell", "command": "ls -la data/"}]\n\n'
                "NOT allowed: edit, test, or commands that write/modify files.\n"
                "Max 20 operations per call."
            )
        )
        timeout: int = Field(
            default=120,
            description="Max seconds per shell operation. Default 120, max 300.",
        )

    class BatchReadonlyTool(BaseTool):
        """Execute multiple read-only operations in one call for research/planning."""
        name: str = "batch_readonly"
        description: str = (
            "Execute multiple READ-ONLY shell operations in ONE call for efficient investigation. "
            "PREFER THIS over calling execute_shell_readonly multiple times when you need to run "
            "2+ commands (grep multiple patterns, find + cat, list + inspect).\n\n"
            "Supported operations:\n"
            "  SHELL (read-only): grep, find, cat, head, tail, ls, wc, file, stat, du, df,\n"
            "    squeue, sacct, sinfo, git log/status/diff/show, python -c (read-only), tree\n\n"
            "NOT supported: edit, test, writing files, rm, mv, cp, mkdir, redirects (>), "
            "pip/conda install, or any filesystem-modifying command.\n\n"
            "Example: investigate a project in one call:\n"
            '  [{"type": "shell", "command": "find . -name \'*.py\' | head -20"},\n'
            '   {"type": "shell", "command": "grep -rn \'class.*Error\' src/"},\n'
            '   {"type": "shell", "command": "cat src/config.py"}]'
        )
        args_schema: Type[BaseModel] = BatchReadonlyInput
        _batch_tool: Any = None

        model_config = {"arbitrary_types_allowed": True}

        def _run(self, **kwargs) -> str:
            raise NotImplementedError("Use async _arun")

        async def _arun(self, **kwargs: Any) -> str:
            operations_raw = kwargs.get("operations", "")
            timeout = kwargs.get("timeout", 120)

            # Accept both list (native provider) and JSON string (MCP/LiteLLM)
            if isinstance(operations_raw, list):
                ops = operations_raw
            elif isinstance(operations_raw, str) and operations_raw.strip():
                try:
                    ops = _json.loads(operations_raw)
                except (ValueError, TypeError) as e:
                    return f"Error: Invalid JSON: {str(e)[:200]}"
            else:
                return "Error: No operations provided."

            if not isinstance(ops, list):
                return "Error: operations must be a JSON array"
            if not ops:
                return "Error: empty operations list"

            # Validate each operation
            _READONLY_BATCH_TYPES = {"shell", "read", "grep", "find", "list"}
            validated_ops = []
            for i, op in enumerate(ops):
                if not isinstance(op, dict):
                    return f"Error: operation #{i+1} must be an object"

                op_type = op.get("type", "")

                if op_type == "edit":
                    return (
                        "BLOCKED: 'edit' operations are NOT available in research/planning phase. "
                        "Only read-only operations are supported. "
                        "You cannot modify files in this phase."
                    )

                if op_type == "test":
                    return (
                        "BLOCKED: 'test' operations are NOT available in research/planning phase. "
                        "Tests may have side effects. Use shell commands to inspect test files instead."
                    )

                if op_type not in _READONLY_BATCH_TYPES:
                    return (
                        f"Error: type '{op_type}' not allowed in read-only mode. "
                        f"Valid: {', '.join(sorted(_READONLY_BATCH_TYPES))}"
                    )

                if op_type == "shell":
                    command = op.get("command", "")
                    if not command:
                        return f"Error: operation #{i+1} missing 'command'"
                    if not is_readonly_command(command):
                        return (
                            f"BLOCKED: operation #{i+1} contains write operations: "
                            f"'{command[:80]}'. Only read-only commands allowed. "
                            "Remove file writes, redirects (>), rm, mv, cp, mkdir, "
                            "sed -i, pip install, etc."
                        )

                validated_ops.append(op)

            # All validated — pass through to real batch tool
            try:
                result = await self._batch_tool._arun(
                    operations=validated_ops,
                    timeout=timeout,
                )
                return result if isinstance(result, str) else _json.dumps(result)
            except Exception as e:
                return f"Error executing batch: {type(e).__name__}: {str(e)[:200]}"

    tool = BatchReadonlyTool()
    tool._batch_tool = batch_tool
    return tool


def _create_stub_readonly_batch(batch_tool):
    """Fallback for environments without langchain_core."""
    import json as _json

    class StubBatchReadonly:
        name = "batch_readonly"
        description = (
            "Execute multiple read-only operations in ONE call. "
            "Supports: shell (grep, find, cat, ls, git), read, grep, find, list. "
            "Does NOT support: edit, test, writes, rm, mv, redirects."
        )

        def __init__(self, inner_tool):
            self._batch_tool = inner_tool

        async def _arun(self, **kwargs):
            _READONLY_TYPES = {"shell", "read", "grep", "find", "list"}
            operations_raw = kwargs.get("operations", "")

            if isinstance(operations_raw, list):
                ops = operations_raw
            elif isinstance(operations_raw, str) and operations_raw.strip():
                try:
                    ops = _json.loads(operations_raw)
                except (ValueError, TypeError):
                    return "Error: Invalid JSON"
            else:
                return "Error: No operations provided."

            if not isinstance(ops, list):
                return "Error: must be a list"

            for i, op in enumerate(ops):
                if not isinstance(op, dict):
                    return f"Error: op #{i+1} not a dict"
                op_type = op.get("type", "")
                if op_type in ("edit", "test"):
                    return f"BLOCKED: '{op_type}' not available in read-only phase"
                if op_type not in _READONLY_TYPES:
                    return f"Error: type '{op_type}' not allowed. Valid: {', '.join(sorted(_READONLY_TYPES))}"
                if op_type == "shell":
                    command = op.get("command", "")
                    if not is_readonly_command(command):
                        return f"BLOCKED: write command in op #{i+1}"

            return await self._batch_tool._arun(
                operations=ops,
                timeout=kwargs.get("timeout", 120),
            )

    return StubBatchReadonly(batch_tool)
