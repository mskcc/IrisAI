"""Research sub-agent for phased execution.

Spawns an isolated NativeAgentExecutor with ONLY read-only tools + write_findings.
The sub-agent's context is discarded after completion — only the findings are
returned to the main orchestrator agent. This provides:

1. Structural tool enforcement (write tools literally don't exist in the schema)
2. Context isolation (50+ tool calls don't pollute main agent's history)
3. Deterministic phase transition (sub-agent ends when write_findings is called)
"""

import asyncio
import logging
import os
from typing import Any, Type

from core.phase_config import PHASE_SYSTEM_PROMPTS

logger = logging.getLogger(__name__)

# Research sub-agent uses Sonnet for thorough exploration
RESEARCH_AGENT_MODEL = os.environ.get(
    "RESEARCH_AGENT_MODEL", "anthropic.claude-sonnet-4-6"
)
RESEARCH_AGENT_TIMEOUT = int(os.environ.get("RESEARCH_AGENT_TIMEOUT", "600"))
RESEARCH_MAX_ITERATIONS = int(os.environ.get("RESEARCH_MAX_ITERATIONS", "30"))
RESEARCH_THINKING_BUDGET = int(os.environ.get("RESEARCH_THINKING_BUDGET", "2048"))
# Use unified threshold from sub_agent — all executors use the same limit.
# Intent-aware summarization handles anything above this cleanly.
from core.sub_agent import MAX_TOOL_OBSERVATION_CHARS as RESEARCH_MAX_OBSERVATION_CHARS

RESEARCH_SYSTEM_PROMPT = (
    "You are a research agent. Your job is to thoroughly investigate a topic "
    "by reading files, running read-only commands, and gathering facts.\n\n"
    "RULES:\n"
    "- Explore broadly first, then drill into relevant areas\n"
    "- Read actual file contents — don't guess based on names\n"
    "- Run shell commands to check system state (squeue, module list, etc.)\n"
    "- Note exact file paths, line numbers, and configurations\n"
    "- When you have enough context, call write_findings() with ALL discoveries\n"
    "- Your findings should enable someone else to implement without re-reading files\n"
    "- Include: what exists, what's relevant, current state, constraints, dependencies\n"
    "- Do NOT include implementation plans or suggestions — just facts\n"
    + PHASE_SYSTEM_PROMPTS["research"]
)


def create_research_agent_tool(
    all_tools: list,
    findings_dir: str,
    findings_name: str,
    cost_tracker=None,
    step_callback=None,
):
    """Create the run_research_agent tool for the main executor.

    Args:
        all_tools: Full tool pool (will be filtered to read-only subset)
        findings_dir: Directory to write findings to
        findings_name: Filename for findings
        cost_tracker: Optional cost tracker to merge sub-agent costs into
        step_callback: Optional Chainlit step callback for UI rendering

    Returns:
        A BaseTool instance (or stub) for run_research_agent.
    """
    try:
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, Field
    except ImportError:
        logger.warning("langchain_core not available — returning stub research agent")
        return _create_stub(all_tools, findings_dir, findings_name, cost_tracker)

    class ResearchAgentInput(BaseModel):
        task: str = Field(
            description="The research question or investigation topic. Be specific about what to look for."
        )
        context: str = Field(
            default="",
            description="Additional context: project directory, relevant file paths, prior knowledge."
        )

    class ResearchAgentTool(BaseTool):
        """Spawn a research sub-agent to investigate a topic and return findings."""
        name: str = "run_research_agent"
        description: str = (
            "Spawn a focused research sub-agent that investigates a topic using "
            "read-only tools (file reads, shell commands, grep, find, slurm queries). "
            "The sub-agent runs in its own context window and returns structured "
            "findings. Use this when you need to explore the codebase, check system "
            "state, or gather information before planning/executing. "
            "The sub-agent CANNOT modify files or run destructive commands. "
            "Returns: complete research findings including file paths, current state, "
            "configurations, and relevant facts."
        )
        args_schema: Type[BaseModel] = ResearchAgentInput
        _all_tools: list = []
        _findings_dir: str = ""
        _findings_name: str = ""
        _cost_tracker: Any = None
        _step_callback: Any = None
        _step_callback_factory: Any = None
        _pel_ref: Any = None

        class Config:
            arbitrary_types_allowed = True

        def _run(self, **kwargs) -> str:
            raise NotImplementedError("Use async _arun")

        async def _arun(self, **kwargs: Any) -> str:
            from core.sub_agent import _unwrap_kwargs
            args = _unwrap_kwargs(kwargs, tool_name="run_research_agent")
            task = args.get("task", "")
            context = args.get("context", "")

            if not task or not task.strip():
                return "Error: No research task provided."

            _cb = self._step_callback
            if _cb is None and callable(self._step_callback_factory):
                try:
                    _cb = self._step_callback_factory()
                except Exception:
                    pass

            saved_counts = None
            saved_total = 0
            if self._pel_ref is not None:
                saved_counts = self._pel_ref._turn_call_counts.copy()
                saved_total = self._pel_ref._total_turn_calls
                self._pel_ref._turn_call_counts = {}
                self._pel_ref._total_turn_calls = 0
                logger.info("[RESEARCH-AGENT] PEL turn counters reset for sub-agent context")

            try:
                return await _run_research_sub_agent(
                    task=task,
                    context=context,
                    all_tools=self._all_tools,
                    findings_dir=self._findings_dir,
                    findings_name=self._findings_name,
                    cost_tracker=self._cost_tracker,
                    step_callback=_cb,
                )
            finally:
                if self._pel_ref is not None and saved_counts is not None:
                    self._pel_ref._turn_call_counts = saved_counts
                    self._pel_ref._total_turn_calls = saved_total
                    logger.info("[RESEARCH-AGENT] PEL turn counters restored after sub-agent")

    tool = ResearchAgentTool()
    tool._all_tools = all_tools
    tool._findings_dir = findings_dir
    tool._findings_name = findings_name
    tool._cost_tracker = cost_tracker
    tool._step_callback = step_callback
    return tool


async def _run_research_sub_agent(
    task: str,
    context: str,
    all_tools: list,
    findings_dir: str,
    findings_name: str,
    cost_tracker=None,
    step_callback=None,
) -> str:
    """Execute the research sub-agent with isolated context and read-only tools."""
    from core.llm_provider import get_provider
    from core.cost_tracker import CostTracker
    from core.native_executor import NativeAgentExecutor
    from core.stuck_detection_callback import StuckInterrupt
    from core.single_agent import SkillEscalationInterrupt
    from core.readonly_shell import create_readonly_shell_tool
    from core.researcher import create_write_findings_tool
    from core.phase_config import READONLY_TOOL_NAMES, _ALWAYS_AVAILABLE_TOOLS

    # Filter tools to read-only subset + always-available + web search tools
    _WEB_TOOLS = {"web_search", "fetch_url_content"}
    _allowed = READONLY_TOOL_NAMES | _ALWAYS_AVAILABLE_TOOLS | _WEB_TOOLS
    research_tools = []
    shell_tool = None

    for t in all_tools:
        if t.name in _allowed and t.name != "execute_shell_readonly":
            research_tools.append(t)
        elif t.name == "execute_dynamic_task":
            shell_tool = t

    # Add read-only shell wrapper (replaces execute_dynamic_task)
    if shell_tool:
        readonly_shell = create_readonly_shell_tool(shell_tool)
        research_tools.append(readonly_shell)

    # Add write_findings tool (the only "write" tool — signals completion)
    write_findings_tool = create_write_findings_tool(findings_dir, findings_name)
    research_tools.append(write_findings_tool)

    if not research_tools:
        return "Error: No research tools available. Cannot spawn research sub-agent."

    logger.info(
        f"[RESEARCH_AGENT] Starting with {len(research_tools)} tools: "
        f"{[t.name for t in research_tools]}"
    )

    # Create isolated provider + executor
    provider = get_provider(
        "anthropic",
        model_id=RESEARCH_AGENT_MODEL,
        base_url=os.environ.get("LITELLM_URL", "http://localhost:8080"),
        api_key=os.environ.get("LITELLM_VIRTUAL_KEY", "not-set"),
        temperature=0,
        max_tokens=16384,
        thinking_budget=RESEARCH_THINKING_BUDGET,
        timeout=RESEARCH_AGENT_TIMEOUT,
    )

    sub_cost_tracker = CostTracker()

    executor = NativeAgentExecutor(
        provider=provider,
        tools=research_tools,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        max_iterations=RESEARCH_MAX_ITERATIONS,
        cost_tracker=sub_cost_tracker,
        step_callback=step_callback,
        max_observation_chars=RESEARCH_MAX_OBSERVATION_CHARS,
    )

    # Build input
    input_parts = [f"RESEARCH TASK:\n{task}"]
    if context:
        input_parts.append(f"\nCONTEXT:\n{context}")
    input_text = "\n".join(input_parts)

    try:
        result = await asyncio.wait_for(
            executor.ainvoke(
                {"input": input_text, "chat_history": [], "agent_scratchpad": []},
            ),
            timeout=RESEARCH_AGENT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(
            f"[RESEARCH_AGENT] Timed out after {RESEARCH_AGENT_TIMEOUT}s"
        )
        return (
            f"[Research agent timed out after {RESEARCH_AGENT_TIMEOUT}s. "
            "Partial findings may be on disk. Check the findings directory.]"
        )
    except StuckInterrupt as stuck:
        logger.warning(
            f"[RESEARCH_AGENT] Stuck on '{stuck.tool_name}' ({stuck.failure_count}x)"
        )
        return (
            f"[Research agent stuck: repeated errors on '{stuck.tool_name}'. "
            f"Error: {stuck.error_snippet}. "
            f"Suggestion: {stuck.suggested_query}]"
        )
    except SkillEscalationInterrupt:
        return "[Research agent requested skill escalation — not available in research mode.]"

    # Merge costs
    if cost_tracker is not None:
        cost_tracker.merge(sub_cost_tracker)

    output = result.get("output", "")
    steps = result.get("intermediate_steps", [])

    logger.info(
        f"[RESEARCH_AGENT] Complete: {len(steps)} tool calls, "
        f"output: {len(output)} chars, cost: ${sub_cost_tracker.total_cost:.4f}"
    )

    return (
        f"[Research Agent Report]\n"
        f"Tool calls: {len(steps)} | Model: {RESEARCH_AGENT_MODEL}\n\n"
        f"{output}"
    )


def _create_stub(all_tools, findings_dir, findings_name, cost_tracker):
    """Fallback stub for environments without langchain_core."""

    class StubResearchAgent:
        name = "run_research_agent"
        description = "Spawn a research sub-agent (stub mode)."

        async def _arun(self, **kwargs):
            task = kwargs.get("task", "")
            context = kwargs.get("context", "")
            return await _run_research_sub_agent(
                task=task, context=context,
                all_tools=all_tools,
                findings_dir=findings_dir,
                findings_name=findings_name,
                cost_tracker=cost_tracker,
            )

    return StubResearchAgent()
