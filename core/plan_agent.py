"""Plan sub-agent for phased execution.

Spawns an isolated NativeAgentExecutor with ONLY read-only tools + write_plan.
Receives research findings as input context and produces a step-by-step plan.
The sub-agent's context is discarded after completion — only the plan is
returned to the main orchestrator agent.
"""

import asyncio
import logging
import os
from typing import Any, Type

from core.phase_config import PHASE_SYSTEM_PROMPTS

logger = logging.getLogger(__name__)

# Plan sub-agent uses Opus for high-quality planning (or Sonnet as fallback)
PLAN_AGENT_MODEL = os.environ.get(
    "PLAN_AGENT_MODEL", "anthropic.claude-sonnet-4-6"
)
PLAN_AGENT_TIMEOUT = int(os.environ.get("PLAN_AGENT_TIMEOUT", "300"))
PLAN_MAX_ITERATIONS = int(os.environ.get("PLAN_MAX_ITERATIONS", "15"))
PLAN_THINKING_BUDGET = int(os.environ.get("PLAN_THINKING_BUDGET", "4096"))
PLAN_MAX_OBSERVATION_CHARS = 30000

PLAN_SYSTEM_PROMPT = (
    "You are a planning agent. Your job is to create a detailed, step-by-step "
    "implementation plan based on the information provided below.\n\n"
    "RULES:\n"
    "- The RESEARCH FINDINGS in your input are authoritative — trust them and "
    "proceed directly to writing your plan.\n"
    "- Do NOT re-read, re-search, or re-discover information already present "
    "in your input (e.g., do not call find_files for files already referenced).\n"
    "- Only use read tools if your input explicitly flags something as unknown, "
    "or if the findings section is absent/empty.\n"
    "- Your plan must be concrete: exact file paths, exact commands, exact changes\n"
    "- For code changes, describe what to find and what to replace with\n"
    "- Include verification/test steps where appropriate\n"
    "- Number steps sequentially\n"
    "- Mark steps that can run in parallel with [PARALLEL_GROUP]\n"
    "- When your plan is complete, call write_plan() with the full plan content\n"
    "- Do NOT implement anything — just produce the plan\n"
    + PHASE_SYSTEM_PROMPTS["plan"]
)


def create_plan_agent_tool(
    all_tools: list,
    plans_dir: str,
    plan_name: str,
    cost_tracker=None,
    step_callback=None,
):
    """Create the run_plan_agent tool for the main executor.

    Args:
        all_tools: Full tool pool (will be filtered to read-only subset + write_plan)
        plans_dir: Directory to write plan to
        plan_name: Filename for plan
        cost_tracker: Optional cost tracker to merge sub-agent costs into
        step_callback: Optional Chainlit step callback for UI rendering

    Returns:
        A BaseTool instance (or stub) for run_plan_agent.
    """
    try:
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, Field
    except ImportError:
        logger.warning("langchain_core not available — returning stub plan agent")
        return _create_stub(all_tools, plans_dir, plan_name, cost_tracker)

    class PlanAgentInput(BaseModel):
        task: str = Field(
            description="The original user request or task description."
        )
        findings: str = Field(
            description="Research findings to base the plan on. Include file paths, state, constraints."
        )
        constraints: str = Field(
            default="",
            description="Any additional constraints (e.g., 'must not break existing tests', 'use existing patterns')."
        )

    class PlanAgentTool(BaseTool):
        """Spawn a planning sub-agent to create an implementation plan from research findings."""
        name: str = "run_plan_agent"
        description: str = (
            "Spawn a focused planning sub-agent that creates a step-by-step "
            "implementation plan based on research findings. The sub-agent can "
            "read files to verify details but CANNOT modify anything. "
            "Returns: a complete plan with exact commands, file changes, and "
            "verification steps. The plan will be shown to the user for approval "
            "before execution begins."
        )
        args_schema: Type[BaseModel] = PlanAgentInput
        _all_tools: list = []
        _plans_dir: str = ""
        _plan_name: str = ""
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
            args = _unwrap_kwargs(kwargs, tool_name="run_plan_agent")
            task = args.get("task", "")
            findings = args.get("findings", "")
            constraints = args.get("constraints", "")

            if not task or not task.strip():
                return "Error: No task provided to plan agent."
            if not findings or not findings.strip():
                return "Error: No research findings provided. Run research first."

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
                logger.info("[PLAN-AGENT] PEL turn counters reset for sub-agent context")

            try:
                return await _run_plan_sub_agent(
                    task=task,
                    findings=findings,
                    constraints=constraints,
                    all_tools=self._all_tools,
                    plans_dir=self._plans_dir,
                    plan_name=self._plan_name,
                    cost_tracker=self._cost_tracker,
                    step_callback=_cb,
                )
            finally:
                if self._pel_ref is not None and saved_counts is not None:
                    self._pel_ref._turn_call_counts = saved_counts
                    self._pel_ref._total_turn_calls = saved_total
                    logger.info("[PLAN-AGENT] PEL turn counters restored after sub-agent")

    tool = PlanAgentTool()
    tool._all_tools = all_tools
    tool._plans_dir = plans_dir
    tool._plan_name = plan_name
    tool._cost_tracker = cost_tracker
    tool._step_callback = step_callback
    return tool


async def _run_plan_sub_agent(
    task: str,
    findings: str,
    constraints: str,
    all_tools: list,
    plans_dir: str,
    plan_name: str,
    cost_tracker=None,
    step_callback=None,
) -> str:
    """Execute the plan sub-agent with isolated context and read-only tools + write_plan."""
    from core.llm_provider import get_provider
    from core.cost_tracker import CostTracker
    from core.native_executor import NativeAgentExecutor
    from core.stuck_detection_callback import StuckInterrupt
    from core.single_agent import SkillEscalationInterrupt
    from core.readonly_shell import create_readonly_shell_tool
    from core.opus_planner import create_write_plan_tool
    from core.phase_config import READONLY_TOOL_NAMES

    # Filter tools to read-only subset
    plan_tools = []
    shell_tool = None

    for t in all_tools:
        if t.name in READONLY_TOOL_NAMES and t.name != "execute_shell_readonly":
            plan_tools.append(t)
        elif t.name == "execute_dynamic_task":
            shell_tool = t

    # Add read-only shell wrapper
    if shell_tool:
        readonly_shell = create_readonly_shell_tool(shell_tool)
        plan_tools.append(readonly_shell)

    # Add write_plan tool (the only "write" tool — signals completion)
    write_plan_tool = create_write_plan_tool(plans_dir, plan_name)
    plan_tools.append(write_plan_tool)

    if not plan_tools:
        return "Error: No plan tools available. Cannot spawn plan sub-agent."

    logger.info(
        f"[PLAN_AGENT] Starting with {len(plan_tools)} tools: "
        f"{[t.name for t in plan_tools]}"
    )

    # Create isolated provider + executor
    provider = get_provider(
        "anthropic",
        model_id=PLAN_AGENT_MODEL,
        base_url=os.environ.get("LITELLM_URL", "http://localhost:8080"),
        api_key=os.environ.get("LITELLM_VIRTUAL_KEY", "not-set"),
        temperature=0,
        max_tokens=16384,
        thinking_budget=PLAN_THINKING_BUDGET,
        timeout=PLAN_AGENT_TIMEOUT,
    )

    sub_cost_tracker = CostTracker()

    executor = NativeAgentExecutor(
        provider=provider,
        tools=plan_tools,
        system_prompt=PLAN_SYSTEM_PROMPT,
        max_iterations=PLAN_MAX_ITERATIONS,
        cost_tracker=sub_cost_tracker,
        step_callback=step_callback,
        max_observation_chars=PLAN_MAX_OBSERVATION_CHARS,
    )

    # Build input with findings as primary context
    input_parts = [
        f"USER REQUEST:\n───\n{task}\n───\n",
        f"\nRESEARCH FINDINGS:\n═══\n{findings}\n═══\n",
    ]
    if constraints:
        input_parts.append(f"\nCONSTRAINTS:\n{constraints}\n")
    input_parts.append(
        "\nCreate a step-by-step implementation plan. Include exact file paths, "
        "commands, and code changes. Include verification/test steps. "
        "When complete, call write_plan() with the full plan."
    )
    input_text = "\n".join(input_parts)

    try:
        result = await asyncio.wait_for(
            executor.ainvoke(
                {"input": input_text, "chat_history": [], "agent_scratchpad": []},
            ),
            timeout=PLAN_AGENT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(f"[PLAN_AGENT] Timed out after {PLAN_AGENT_TIMEOUT}s")
        return (
            f"[Plan agent timed out after {PLAN_AGENT_TIMEOUT}s. "
            "Partial plan may be on disk. Check the plans directory.]"
        )
    except StuckInterrupt as stuck:
        logger.warning(
            f"[PLAN_AGENT] Stuck on '{stuck.tool_name}' ({stuck.failure_count}x)"
        )
        return (
            f"[Plan agent stuck: repeated errors on '{stuck.tool_name}'. "
            f"Error: {stuck.error_snippet}]"
        )
    except SkillEscalationInterrupt:
        return "[Plan agent requested skill escalation — not available in plan mode.]"

    # Merge costs
    if cost_tracker is not None:
        cost_tracker.merge(sub_cost_tracker)

    output = result.get("output", "")
    steps = result.get("intermediate_steps", [])

    logger.info(
        f"[PLAN_AGENT] Complete: {len(steps)} tool calls, "
        f"output: {len(output)} chars, cost: ${sub_cost_tracker.total_cost:.4f}"
    )

    return (
        f"[Plan Agent Report]\n"
        f"Tool calls: {len(steps)} | Model: {PLAN_AGENT_MODEL}\n\n"
        f"{output}"
    )


def _create_stub(all_tools, plans_dir, plan_name, cost_tracker):
    """Fallback stub for environments without langchain_core."""

    class StubPlanAgent:
        name = "run_plan_agent"
        description = "Spawn a plan sub-agent (stub mode)."

        async def _arun(self, **kwargs):
            task = kwargs.get("task", "")
            findings = kwargs.get("findings", "")
            constraints = kwargs.get("constraints", "")
            return await _run_plan_sub_agent(
                task=task, findings=findings, constraints=constraints,
                all_tools=all_tools,
                plans_dir=plans_dir,
                plan_name=plan_name,
                cost_tracker=cost_tracker,
            )

    return StubPlanAgent()
