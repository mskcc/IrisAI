"""Mid-loop thinking display callback for LangChain AgentExecutor.

Shows the agent's reasoning and tool usage in the Chainlit UI during
execution. Uses on_agent_action to capture the agent's log (which includes
reasoning text from the AIMessage content alongside tool calls).

Attaches to AgentExecutor via config={"callbacks": [thinking_cb]}.
All methods are non-fatal — exceptions are caught and logged.
"""

import re
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import chainlit as cl
from langchain_core.agents import AgentAction
from langchain_core.callbacks import AsyncCallbackHandler


_MAX_DISPLAY = 200

# Pattern to extract reasoning from the log field
# Log format: "\nInvoking: `tool_name` with `{args}`\nthinking: <reasoning>\n\n"
#         or: "\nInvoking: `tool_name` with `{args}`\nresponded: <text>\n\n"
_THINKING_RE = re.compile(r"thinking:\s*(.+?)(?:\n\n|$)", re.DOTALL)
_RESPONDED_RE = re.compile(r"responded:\s*(.+?)(?:\n\n|$)", re.DOTALL)
_INVOKING_RE = re.compile(r"Invoking:\s*`([^`]+)`\s*with\s*`(.+?)`", re.DOTALL)


def _extract_reasoning(log: str) -> tuple[str, bool]:
    """Extract reasoning text from AgentAction.log.

    Returns (text, is_thinking) where is_thinking=True means real extended
    thinking was used, False means it's regular response text.
    """
    match = _THINKING_RE.search(log)
    if match:
        reasoning = match.group(1).strip()
        if reasoning and len(reasoning) > 10:
            return reasoning, True

    match = _RESPONDED_RE.search(log)
    if match:
        reasoning = match.group(1).strip()
        if reasoning and len(reasoning) > 10:
            return reasoning, False

    return "", False


def _extract_tool_info(log: str) -> tuple:
    """Extract tool name and args summary from AgentAction.log."""
    match = _INVOKING_RE.search(log)
    if match:
        return match.group(1), match.group(2)
    return "", ""


def _summarize_tool_call(tool_name: str, tool_input: Any) -> str:
    """Create a human-friendly summary of what a tool call does."""
    name = tool_name.replace("_", " ")

    # Parse common tool patterns for better descriptions
    if isinstance(tool_input, dict):
        if "filename" in tool_input:
            return f"Reading **{tool_input['filename']}**"
        if "path" in tool_input:
            path = str(tool_input["path"])
            # Shorten long paths
            if len(path) > 50:
                path = "..." + path[-40:]
            return f"{name}: `{path}`"
        if "query" in tool_input:
            q = str(tool_input["query"])[:80]
            return f"Searching: *{q}*"
        if "job_name" in tool_input:
            return f"Submitting job: **{tool_input['job_name']}**"
        if "command" in tool_input:
            cmd = str(tool_input["command"])[:60]
            return f"Running: `{cmd}`"
        if "task_description" in tool_input:
            desc = str(tool_input["task_description"])[:80]
            return f"Task: *{desc}*"

    return f"Using **{name}**"


def _truncate(text: str, max_len: int = _MAX_DISPLAY) -> str:
    """Truncate text at word boundary."""
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rfind(" ")
    if cut > max_len // 2:
        return text[:cut] + "..."
    return text[:max_len] + "..."


class ThinkingDisplayCallback(AsyncCallbackHandler):
    """Shows agent reasoning and tool usage as visible messages in the chat.

    Uses on_agent_action which fires for each tool call the agent makes,
    including the reasoning text (in action.log) and tool arguments.

    Usage:
        thinking_cb = ThinkingDisplayCallback(status_msg)
        result = await executor.ainvoke(..., config={"callbacks": [thinking_cb, ...]})
        await thinking_cb.cleanup()
    """

    def __init__(self, status_msg) -> None:
        super().__init__()
        self._status_msg = status_msg
        self._thinking_msg: Optional[cl.Message] = None
        self._iteration = 0
        self._last_reasoning: str = ""

    async def _show_thinking(self, text: str) -> None:
        """Show a thinking message in the chat (replaces previous one)."""
        try:
            if self._thinking_msg:
                try:
                    await self._thinking_msg.remove()
                except Exception:
                    pass
            self._thinking_msg = cl.Message(content=text)
            await self._thinking_msg.send()
        except Exception:
            pass

    async def _update_status(self, content: str) -> None:
        """Update the status bar message."""
        if self._status_msg is None:
            return
        try:
            self._status_msg.content = content
            await self._status_msg.update()
        except Exception:
            pass

    async def cleanup(self) -> None:
        """Remove the last thinking message. Call after execution completes."""
        if self._thinking_msg:
            try:
                await self._thinking_msg.remove()
            except Exception:
                pass
            self._thinking_msg = None

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Show reasoning from each agent action (tool display handled by Chainlit)."""
        self._iteration += 1
        try:
            log = getattr(action, "log", "")
            tool_name = getattr(action, "tool", "")
            tool_input = getattr(action, "tool_input", "")

            reasoning, is_thinking = _extract_reasoning(log)

            if reasoning and reasoning != self._last_reasoning:
                self._last_reasoning = reasoning
                # Collapse newlines and take first meaningful sentence
                clean = " ".join(reasoning.split())
                for prefix in ("I'll ", "I will ", "Let me ", "Now I'll ", "Now I will ", "Now let me "):
                    if clean.startswith(prefix):
                        clean = clean[len(prefix):]
                        clean = clean[0].upper() + clean[1:] if clean else clean
                        break
                # ✨ = real extended thinking, 💬 = regular response text
                icon = "✨" if is_thinking else "💬"
                await self._show_thinking(f"{icon} *{_truncate(clean, 180)}*")

            # Update status bar with tool context (not shown as message)
            tool_summary = _summarize_tool_call(tool_name, tool_input)
            await self._update_status(f"🔧 {tool_summary}")

        except Exception:
            pass
