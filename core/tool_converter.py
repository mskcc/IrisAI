"""Convert LangChain tool definitions to Anthropic native tool format.

Anthropic's Messages API expects tools in this format:
    {"name": str, "description": str, "input_schema": {...JSON Schema...}}

LangChain tools (BaseTool, StructuredTool) carry the same info via:
    .name, .description, .args_schema (Pydantic model)

This module bridges the two formats.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def langchain_tool_to_anthropic(tool: Any) -> dict:
    """Convert a single LangChain BaseTool to Anthropic tool format.

    Args:
        tool: A LangChain BaseTool or StructuredTool instance.

    Returns:
        Dict with keys: name, description, input_schema.
    """
    schema: dict = {"type": "object", "properties": {}}

    if hasattr(tool, "args_schema") and tool.args_schema is not None:
        try:
            schema = tool.args_schema.model_json_schema()
        except AttributeError:
            # Pydantic v1 fallback
            schema = tool.args_schema.schema()

        # Remove Pydantic metadata that Anthropic doesn't need
        schema.pop("title", None)
        schema.pop("description", None)

        # Clean up 'definitions' / '$defs' if empty
        if not schema.get("$defs"):
            schema.pop("$defs", None)
        if not schema.get("definitions"):
            schema.pop("definitions", None)

    return {
        "name": tool.name,
        "description": getattr(tool, "description", "") or "",
        "input_schema": schema,
    }


def langchain_tools_to_anthropic(
    tools: list[Any],
    cache_last: bool = True,
) -> list[dict]:
    """Convert a list of LangChain tools to Anthropic format.

    Optionally adds cache_control to the last tool definition so the entire
    tools prefix (system + all tools) is cached by Anthropic's prompt caching.

    Args:
        tools: List of LangChain BaseTool/StructuredTool instances.
        cache_last: If True, add cache_control to the last tool for prompt caching.

    Returns:
        List of tool dicts in Anthropic format.
    """
    if not tools:
        return []

    # Deduplicate by tool name — last definition wins (preserves override semantics).
    # Guards against duplicate tool objects from multiple registration sources
    # (e.g., MCP + built-in modules) which cause API 400 "Tool names must be unique".
    seen: dict[str, dict] = {}
    for t in tools:
        converted_tool = langchain_tool_to_anthropic(t)
        seen[converted_tool["name"]] = converted_tool

    converted = list(seen.values())

    if cache_last and converted:
        converted[-1]["cache_control"] = {"type": "ephemeral"}

    return converted
