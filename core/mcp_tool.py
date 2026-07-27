"""MCPTool — BaseTool wrapper for MCP tool calls.

Extracted from app.py to allow test imports without chainlit dependency.
The core logic: unwrap kwargs variations from LLMs, call the MCP session.
"""
import asyncio
import json
import logging
from typing import Any, Dict

from langchain.tools import BaseTool

logger = logging.getLogger(__name__)


def unwrap_llm_kwargs(kwargs: dict) -> dict:
    """Normalize the various ways LLMs wrap tool arguments.

    LLMs may pass args as:
    - Direct kwargs: {param1: val1, param2: val2}
    - Wrapped in "kwargs" key as dict: {"kwargs": {param1: val1}}
    - Wrapped in "kwargs" key as JSON string: {"kwargs": '{"param1": "val1"}'}
    - Single dict value: {"input": {param1: val1}}
    """
    args = kwargs

    if "kwargs" in kwargs:
        val = kwargs["kwargs"]
        if isinstance(val, dict):
            args = val
        elif isinstance(val, str):
            try:
                parsed = json.loads(val, strict=False)
                if isinstance(parsed, dict):
                    args = parsed
            except (json.JSONDecodeError, ValueError):
                pass
    elif len(kwargs) == 1 and isinstance(next(iter(kwargs.values())), dict):
        args = next(iter(kwargs.values()))

    # Safety net: nested "kwargs" wrapper
    if "kwargs" in args and len(args) == 1:
        val = args["kwargs"]
        if isinstance(val, dict):
            args = val
        elif isinstance(val, str):
            try:
                parsed = json.loads(val, strict=False)
                if isinstance(parsed, dict):
                    args = parsed
            except (json.JSONDecodeError, ValueError):
                pass

    return args


class MCPTool(BaseTool):
    """BaseTool that delegates to an MCP session's call_tool().

    This is the production wrapper used in app.py (extracted here for testability).
    The mcp_session must implement: call_tool(tool_name: str, args: dict)
    """
    name: str
    description: str
    mcp_session: Any = None
    input_schema_raw: Dict[str, Any] = None

    def _run(self, **kwargs):
        raise NotImplementedError("Use async _arun")

    async def _arun(self, **kwargs: Any) -> Any:
        if self.mcp_session is None:
            raise RuntimeError(f"No MCP session for tool {self.name}")

        args = unwrap_llm_kwargs(kwargs)

        # Strip internal framework keys
        if isinstance(args, dict):
            args.pop("_output_mode", None)
            for key in ("config", "run_manager", "callbacks"):
                args.pop(key, None)

        # Timeout config
        _LONG_RUNNING = {"execute_dynamic_task": 300, "submit_slurm_job": 7200, "batch": 600}
        _default_timeout = _LONG_RUNNING.get(self.name, 60)
        _tool_timeout = args.get("timeout", _default_timeout) if isinstance(args, dict) else _default_timeout
        call_timeout = max(60, _tool_timeout + 30)

        try:
            result = await asyncio.wait_for(
                self.mcp_session.call_tool(self.name, args),
                timeout=call_timeout,
            )
            return result.model_dump() if hasattr(result, "model_dump") else result
        except asyncio.TimeoutError:
            return {"error": f"Tool '{self.name}' timed out after {call_timeout}s"}
        except Exception as e:
            return {"error": f"Tool '{self.name}' failed: {type(e).__name__}: {e}"}
