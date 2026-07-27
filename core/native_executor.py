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

"""Native Agent Executor — replaces LangChain's AgentExecutor.

Uses AnthropicProvider.create_message() directly for the main execution loop.
Gains prompt caching across iterations (system + tools cached on iter 2+),
extended thinking for Opus, and native tool format.

Same input/output interface as LangChain AgentExecutor:
  - ainvoke({"input": str, "chat_history": list, "agent_scratchpad": []}, config={...})
  - Returns {"output": str, "intermediate_steps": list[tuple(NativeAgentAction, str)]}

Feature flag: IRIS_USE_NATIVE_EXECUTOR (default "1"). Set to "0" to fall back.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.llm_provider import AnthropicProvider, OpenAIProvider, LLMResponse, ToolCall
from core.cost_tracker import CostTracker
from core.tool_converter import langchain_tools_to_anthropic
from core.sub_agent import MAX_TOOL_OBSERVATION_CHARS
from pydantic import ValidationError
from core.stuck_detection_callback import (
    _is_error_observation, _make_fingerprint, _suggest_query, StuckInterrupt,
    SCHEMA_ERROR_PREFIX,
)
from core.single_agent import SkillEscalationInterrupt, ESCALATION_MARKER

logger = logging.getLogger(__name__)

_DEBUG = os.environ.get("IRIS_DEBUG_EXECUTOR", "0") == "1"

DEFAULT_TOOL_TIMEOUT = 300  # 5 min
LONG_RUNNING_TOOL_TIMEOUTS = {
    "submit_slurm_job": 7200,
    "run_worker_agent": 900,
    "upload_weights_to_fixed_location": 9000,
}

# Intra-turn token ceiling — 83% of Bedrock's 180K hard limit.
# Leaves ~30K tokens for: system prompt (~8K), tools (~5K), one more LLM
# response (~4K), and safety margin (~13K).
INTRA_TURN_TOKEN_CEILING = 150_000


@dataclass
class NativeAgentAction:
    """Drop-in compatible with langchain_core.agents.AgentAction.

    Has .tool, .tool_input, .log attributes expected by:
    - detect_escalation_in_result()
    - format_intermediate_steps_for_handoff()
    - format_tool_call_record()
    - validate_agent_result()
    - ThinkingDisplayCallback.on_agent_action()
    """
    tool: str
    tool_input: dict
    log: str = ""


class NativeAgentExecutor:
    """Custom agent executor using native Anthropic provider.

    Replaces LangChain's AgentExecutor with direct Anthropic Messages API
    calls. Gains prompt caching (40-60% token savings), extended thinking,
    and direct token tracking.

    Phase-aware mode (when phase_config is provided):
    - Tools and system prompt change dynamically based on current phase
    - Phase transitions are detected at harness level (completion tool calls)
    - The LLM never sees tools outside its current phase's allowed set
    """

    def __init__(
        self,
        provider: "AnthropicProvider | OpenAIProvider",
        tools: list,
        system_prompt: str,
        max_iterations: int = 15,
        callbacks: list = None,
        cost_tracker: CostTracker = None,
        step_callback=None,
        max_observation_chars: int = None,
        iteration_callback=None,
        tool_timeouts: dict = None,
        phase_config=None,
        websearch_enabled: bool = False,
        active_plan_path: str = "",
        research_provider: "AnthropicProvider | OpenAIProvider | None" = None,
        pel=None,
    ):
        self.provider = provider
        self._research_provider = research_provider
        self.tools = tools
        self.tool_map = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.callbacks = callbacks or []
        self.cost_tracker = cost_tracker or CostTracker()
        self.step_callback = step_callback
        self.iteration_callback = iteration_callback
        self._max_observation_chars = max_observation_chars or MAX_TOOL_OBSERVATION_CHARS
        self._pel = pel

        # Phase-aware execution support
        self._phase_config = phase_config
        self._current_phase = phase_config.initial_phase if phase_config else None
        self._websearch_enabled = websearch_enabled
        self._active_plan_path = active_plan_path
        self._plan_nudge_given = False

        if phase_config:
            # Pre-compute tool sets per phase for fast switching
            self._phase_tools = {}
            self._phase_tool_maps = {}
            for phase in phase_config.phases:
                from core.phase_config import filter_tools_for_phase
                phase_tool_list = filter_tools_for_phase(tools, phase, websearch_enabled=websearch_enabled)
                self._phase_tools[phase] = langchain_tools_to_anthropic(
                    phase_tool_list, cache_last=True
                )
                self._phase_tool_maps[phase] = {t.name: t for t in phase_tool_list}
            self._anthropic_tools = self._phase_tools[self._current_phase]
        else:
            self._anthropic_tools = langchain_tools_to_anthropic(tools, cache_last=True)
            self._phase_tools = None
            self._phase_tool_maps = None

        self._tool_timeouts = {**LONG_RUNNING_TOOL_TIMEOUTS}
        if tool_timeouts:
            self._tool_timeouts.update(tool_timeouts)

        self._stuck_counts: dict[str, int] = defaultdict(int)
        self._stuck_details: dict[str, tuple] = {}
        self._stuck_threshold = 3
        self._schema_error_fingerprints: set[str] = set()
        self._phase_validation_failure: str | None = None

    def _get_current_tools(self):
        """Get the Anthropic-format tool list for the current phase."""
        if self._phase_tools and self._current_phase:
            return self._phase_tools.get(self._current_phase, self._anthropic_tools)
        return self._anthropic_tools

    def _get_current_tool_map(self):
        """Get the tool map (name → tool) for the current phase."""
        if self._phase_tool_maps and self._current_phase:
            return self._phase_tool_maps.get(self._current_phase, self.tool_map)
        return self.tool_map

    def _get_current_system_prompt(self):
        """Get the system prompt with phase-specific addition appended."""
        if self._phase_config and self._current_phase:
            addition = self._phase_config.get_system_prompt_addition(
                self._current_phase, websearch_enabled=self._websearch_enabled
            )
            return self.system_prompt + addition
        return self.system_prompt

    def _estimate_messages_tokens(self, messages: list) -> int:
        """Rough token estimate for accumulated messages in the executor loop.

        Uses chars/3 heuristic (same as history.py CHARS_PER_TOKEN=3).
        """
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total_chars += len(block.get("text", ""))
                        c = block.get("content", "")
                        if isinstance(c, str):
                            total_chars += len(c)
                        if "input" in block and isinstance(block["input"], dict):
                            total_chars += len(json.dumps(block["input"], default=str))
        return total_chars // 3 + 1

    def _check_phase_transition(self, tool_name: str, observation: str) -> bool:
        """Check if a tool call signals phase completion. Returns True if phase advanced."""
        if not self._phase_config or not self._current_phase:
            return False

        completion_tool = self._phase_config.get_completion_tool(self._current_phase)
        if tool_name != completion_tool:
            return False

        # Validate output quality before advancing
        if not self._phase_config.validate_phase_output(self._current_phase, observation):
            logger.info(
                f"[PHASE_GATE] Phase '{self._current_phase}' output validation failed — "
                f"staying in current phase"
            )
            self._phase_validation_failure = self._current_phase
            return False
        self._phase_validation_failure = None

        next_phase = self._phase_config.next_phase(self._current_phase)
        if next_phase:
            old_phase = self._current_phase
            self._current_phase = next_phase
            # Update tool references for new phase
            self._anthropic_tools = self._get_current_tools()
            logger.info(f"[PHASE_TRANSITION] {old_phase} → {next_phase}")
            return True

        return False

    async def ainvoke(self, inputs: dict, config: dict = None) -> dict:
        """Execute the tool-calling loop.

        Args:
            inputs: {"input": str, "chat_history": list, "agent_scratchpad": []}
            config: {"callbacks": list}

        Returns:
            {"output": str, "intermediate_steps": list[tuple(NativeAgentAction, str)]}
        """
        user_input = inputs["input"]
        chat_history = inputs.get("chat_history", [])

        # Reset stuck detection state for fresh invocation
        self._stuck_counts.clear()
        self._stuck_details.clear()
        self._schema_error_fingerprints.clear()

        runtime_callbacks = (config or {}).get("callbacks", [])
        all_callbacks = self.callbacks + runtime_callbacks

        messages = self._build_anthropic_messages(user_input, chat_history)
        intermediate_steps: list[tuple] = []
        response = None

        # Add cache breakpoint on the last message (user's current request).
        # This allows iterations 2+ to cache the entire prefix (system + tools +
        # history) and only pay full price for new tool_use/tool_result pairs.
        if messages:
            self._add_cache_breakpoint(messages[-1])

        for iteration in range(self.max_iterations):
            if self.iteration_callback:
                try:
                    await self.iteration_callback(iteration, len(intermediate_steps))
                except Exception:
                    pass

            # Phase-aware: use current phase's tools and system prompt
            current_tools = self._get_current_tools()
            current_system = self._get_current_system_prompt()

            # Use cheaper model for research phase (tool calls only, no complex reasoning needed)
            active_provider = self.provider
            if self._research_provider and self._current_phase == "research":
                active_provider = self._research_provider

            response = await active_provider.create_message(
                system=current_system,
                messages=messages,
                tools=current_tools if current_tools else None,
                cache_system=True,
            )

            self.cost_tracker.accumulate(response.usage, model=response.model)

            if _DEBUG:
                _tc_names = [tc.name for tc in response.tool_calls] if response.tool_calls else []
                _cache_r = response.usage.cache_read_tokens
                _cache_c = response.usage.cache_creation_tokens
                _cache_str = f" cache_read={_cache_r} cache_create={_cache_c}" if (_cache_r or _cache_c) else ""
                logger.info(
                    f"[EXECUTOR] iter={iteration+1}/{self.max_iterations} "
                    f"stop_reason={response.stop_reason} "
                    f"tokens_out={response.usage.output_tokens} "
                    f"tokens_in={response.usage.input_tokens}"
                    f"{_cache_str} "
                    f"tool_calls={_tc_names} "
                    f"thinking={'yes' if response.thinking else 'no'}"
                )

            if response.tool_calls and len(response.tool_calls) > 1:
                logger.info(
                    f"[PARALLEL] {len(response.tool_calls)} tool calls in single response: "
                    f"{[tc.name for tc in response.tool_calls]}"
                )

            if response.stop_reason == "end_turn" or not response.tool_calls:
                # In research/plan phases, don't exit on text-only — nudge the LLM
                if (self._phase_config and self._current_phase in ("research", "plan")
                        and iteration < self.max_iterations - 1):
                    completion_tool = self._phase_config.get_completion_tool(self._current_phase)
                    nudge = (
                        f"[SYSTEM: You responded with text but did not call any tools. "
                        f"You are in {self._current_phase} phase — you MUST call "
                        f"{completion_tool}() to advance. Use your tools to gather information, "
                        f"then call {completion_tool}() with your findings/plan.]"
                    )
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": response.content or ""}]})
                    messages.append({"role": "user", "content": [{"type": "text", "text": nudge}]})
                    logger.info(f"[PHASE_NUDGE] Text-only response in {self._current_phase} — nudging LLM")
                    continue

                # In execution phase with active plan: nudge model to mark steps
                if (self._active_plan_path and not self._plan_nudge_given
                        and intermediate_steps and iteration < self.max_iterations - 1):
                    from core.plan_verification import plan_has_unchecked_steps
                    try:
                        _plan_content = Path(self._active_plan_path).read_text()
                    except Exception:
                        _plan_content = ""
                    if _plan_content and plan_has_unchecked_steps(_plan_content):
                        self._plan_nudge_given = True
                        nudge = (
                            f"[SYSTEM: You stopped but the plan still has unchecked steps. "
                            f"You MUST call edit_plan() to mark completed steps as [x] before finishing. "
                            f"Review what you accomplished and update the plan NOW.]"
                        )
                        messages.append({"role": "assistant", "content": [{"type": "text", "text": response.content or ""}]})
                        messages.append({"role": "user", "content": [{"type": "text", "text": nudge}]})
                        logger.info(f"[PLAN_NUDGE] Executor stopped with unchecked plan steps — nudging to mark")
                        continue

                return {
                    "output": response.content,
                    "intermediate_steps": intermediate_steps,
                }

            # Detect output truncation — tool calls are likely incomplete
            if response.stop_reason == "max_tokens" and response.tool_calls:
                logger.warning(
                    f"[TRUNCATION] stop_reason=max_tokens with {len(response.tool_calls)} "
                    f"tool call(s) — output was cut off mid-generation. "
                    f"output_tokens={response.usage.output_tokens}, "
                    f"tools=[{', '.join(tc.name for tc in response.tool_calls)}]"
                )
                assistant_content = self._build_assistant_content(response)
                messages.append({"role": "assistant", "content": assistant_content})
                tool_results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": "ERROR: Output was truncated — this tool call is incomplete.",
                        "is_error": True,
                    }
                    for tc in response.tool_calls
                ]
                tool_results.append({
                    "type": "text",
                    "text": (
                        "[SYSTEM: Your response was truncated (hit output token limit). "
                        "The tool call was incomplete — required parameters were missing. "
                        "For large file writes, split the content: write the first ~200 lines "
                        "with write_text_file, then use edit_file to append remaining sections. "
                        "Do NOT retry the same large single-call write.]"
                    ),
                })
                messages.append({"role": "user", "content": tool_results})
                continue

            assistant_content = self._build_assistant_content(response)
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            phase_transitioned = False

            for tool_call in response.tool_calls:
                sanitized_input = self._sanitize_tool_input(tool_call.input)
                action = NativeAgentAction(
                    tool=tool_call.name,
                    tool_input=sanitized_input,
                    log=self._build_action_log(response, tool_call),
                )

                await self._fire_on_agent_action(action, all_callbacks)

                # Fire step_callback for UI rendering (Chainlit steps)
                step_ctx = None
                if self.step_callback:
                    try:
                        step_ctx = await self.step_callback(
                            tool_call.name, sanitized_input, "start"
                        )
                    except Exception:
                        pass

                # Use phase-aware tool map for execution (only when phase_config is active)
                try:
                    if self._phase_config and self._current_phase:
                        current_tool_map = self._get_current_tool_map()
                        if tool_call.name not in current_tool_map:
                            observation = (
                                f"Error: `{tool_call.name}` is not available in the current "
                                f"phase ({self._current_phase}). "
                                f"Available tools: {', '.join(sorted(current_tool_map.keys())[:15])}"
                            )
                        else:
                            observation = await self._execute_tool(tool_call.name, sanitized_input)
                    else:
                        observation = await self._execute_tool(tool_call.name, sanitized_input)
                except SkillEscalationInterrupt as _esc:
                    # Close the UI step before propagating
                    if step_ctx and self.step_callback:
                        try:
                            await self.step_callback(
                                tool_call.name,
                                f"Escalating to skill: {_esc.skill_name}",
                                "end", step_ctx,
                            )
                        except Exception:
                            pass
                    intermediate_steps.append(
                        (action, f"[ESCALATION] Requested skill: {_esc.skill_name}")
                    )
                    _esc.intermediate_steps = intermediate_steps
                    raise

                if step_ctx and self.step_callback:
                    try:
                        await self.step_callback(
                            tool_call.name, observation, "end", step_ctx
                        )
                    except Exception:
                        pass

                if _DEBUG:
                    _input_size = len(json.dumps(tool_call.input, default=str))
                    _obs_size = len(str(observation))
                    _is_err = "error" in str(observation)[:200].lower()
                    logger.info(
                        f"[TOOL_CALL] {tool_call.name} "
                        f"input_keys={list(tool_call.input.keys())} "
                        f"input_size={_input_size} "
                        f"result_size={_obs_size} "
                        f"is_error={_is_err}"
                    )

                intermediate_steps.append((action, observation))

                # Check for phase transition AFTER successful tool execution
                if self._check_phase_transition(tool_call.name, observation):
                    phase_transitioned = True
                    # Add the tool result for this call
                    trimmed_obs = await self._trim_observation(
                        observation, tool_call.name,
                        intent=getattr(response, 'thinking', '') or '',
                        tool_input=sanitized_input,
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": trimmed_obs,
                    })
                    break  # Stop processing remaining tool calls in this batch

                # If phase validation failed, tell the LLM why so it can fix it
                if self._phase_validation_failure:
                    phase = self._phase_validation_failure
                    self._phase_validation_failure = None
                    from core.phase_config import MIN_RESEARCH_CHARS, MIN_PLAN_STEPS
                    if phase == "research":
                        hint = (
                            f"[PHASE_GATE REJECTED] Your {tool_call.name} output did not "
                            f"meet the quality bar. Requirements: (1) must be >{MIN_RESEARCH_CHARS} "
                            f"characters, (2) must include concrete file paths (containing '/'). "
                            f"Your output was {len(observation)} chars. "
                            f"Please call {tool_call.name} again with more substantive findings."
                        )
                    elif phase == "plan":
                        hint = (
                            f"[PHASE_GATE REJECTED] Your {tool_call.name} output did not "
                            f"meet the quality bar. Requirements: must have at least "
                            f"{MIN_PLAN_STEPS} numbered or bulleted steps. "
                            f"Please call {tool_call.name} again with a more detailed plan."
                        )
                    else:
                        hint = ""
                    if hint:
                        observation = f"{observation}\n\n{hint}"

                self._detect_stuck(tool_call.name, observation)

                trimmed_obs = await self._trim_observation(
                    observation, tool_call.name,
                    intent=getattr(response, 'thinking', '') or '',
                    tool_input=sanitized_input,
                )
                content = self._format_tool_content(trimmed_obs)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": content,
                })

            messages.append({"role": "user", "content": tool_results})

            # ── Intra-turn context budget check ─────────────────────────────
            loop_tokens = self._estimate_messages_tokens(messages)
            if loop_tokens > INTRA_TURN_TOKEN_CEILING:
                logger.warning(
                    f"[CONTEXT_BUDGET] {loop_tokens} tokens exceeds ceiling "
                    f"({INTRA_TURN_TOKEN_CEILING}). Forcing completion."
                )
                if self._phase_config and self._current_phase in ("research", "plan"):
                    completion_tool = self._phase_config.get_completion_tool(self._current_phase)
                    budget_msg = {
                        "type": "text",
                        "text": (
                            f"[SYSTEM: Context budget reached ({loop_tokens} tokens). "
                            f"You MUST call {completion_tool}() NOW with everything you "
                            f"have gathered so far. Do not make any more tool calls — "
                            f"summarize and call {completion_tool}() immediately.]"
                        ),
                    }
                else:
                    budget_msg = {
                        "type": "text",
                        "text": (
                            f"[SYSTEM: Context budget reached ({loop_tokens} tokens). "
                            f"Complete your current task NOW. Provide your final answer "
                            f"with whatever information you have gathered so far. "
                            f"Do not make further tool calls.]"
                        ),
                    }
                messages[-1]["content"].append(budget_msg)

            # Phase transition: return to user for approval before advancing
            if phase_transitioned:
                logger.info(
                    f"[PHASE_PAUSE] Pausing after phase transition to "
                    f"'{self._current_phase}' — returning to user for approval"
                )
                return {
                    "output": observation,
                    "intermediate_steps": intermediate_steps,
                    "_phase_paused": True,
                    "_paused_at_phase": self._current_phase,
                }

            # Budget warning: tell the model to wrap up when 3 iterations remain.
            # Phase-aware: in research/plan phases, force the completion tool call.
            if iteration >= self.max_iterations - 3:
                if self._phase_config and self._current_phase in ("research", "plan"):
                    completion_tool = self._phase_config.get_completion_tool(self._current_phase)
                    budget_hint = {
                        "type": "text",
                        "text": (
                            f"[SYSTEM: You have {self.max_iterations - iteration - 1} iterations remaining. "
                            f"You MUST call {completion_tool}() NOW with everything you have gathered so far. "
                            f"This is not optional — the phase cannot advance without it. "
                            f"Summarize your findings/plan from the tool calls above and call "
                            f"{completion_tool}() immediately.]"
                        ),
                    }
                else:
                    budget_hint = {
                        "type": "text",
                        "text": (
                            "[SYSTEM: You are running low on iteration budget. "
                            "Stop making tool calls and provide your final answer NOW "
                            "with whatever information you have gathered so far.]"
                        ),
                    }
                messages[-1]["content"].append(budget_hint)

        # Early stopping: give the model one final chance.
        # Phase-aware: keep tools available so the agent can still call the
        # completion tool (write_findings/write_plan) on this final attempt.
        try:
            if self._phase_config and self._current_phase in ("research", "plan"):
                completion_tool = self._phase_config.get_completion_tool(self._current_phase)
                messages.append({"role": "user", "content": [{
                    "type": "text",
                    "text": (
                        f"[SYSTEM: Maximum iterations reached. You MUST call {completion_tool}() "
                        f"NOW. Summarize ALL information from your tool calls above into a single "
                        f"{completion_tool}() call. This is your LAST chance — if you do not call "
                        f"{completion_tool}(), all your work in this phase will be lost.]"
                    ),
                }]})
                final_response = await self.provider.create_message(
                    system=self._get_current_system_prompt(),
                    messages=messages,
                    tools=self._get_current_tools(),
                    cache_system=True,
                )
                self.cost_tracker.accumulate(final_response.usage, model=final_response.model)

                # Process tool calls from the final response to catch completion tool
                if final_response.tool_calls:
                    for tool_call in final_response.tool_calls:
                        if tool_call.name == completion_tool:
                            fc_input = self._sanitize_tool_input(tool_call.input)
                            observation = await self._execute_tool(tool_call.name, fc_input)
                            action = NativeAgentAction(
                                tool=tool_call.name,
                                tool_input=fc_input,
                                log=f"[forced-completion] {tool_call.name}",
                            )
                            intermediate_steps.append((action, observation))
                            self._check_phase_transition(tool_call.name, observation)
                            break

                return {
                    "output": final_response.content or "",
                    "intermediate_steps": intermediate_steps,
                }
            else:
                messages.append({"role": "user", "content": [{
                    "type": "text",
                    "text": (
                        "[SYSTEM: Maximum iterations reached. You MUST respond with your "
                        "final answer NOW. Summarize all findings from your tool calls above. "
                        "Do NOT call any more tools.]"
                    ),
                }]})
                final_response = await self.provider.create_message(
                    system=self._get_current_system_prompt(),
                    messages=messages,
                    tools=None,
                    cache_system=True,
                )
                self.cost_tracker.accumulate(final_response.usage, model=final_response.model)
                return {
                    "output": final_response.content or "",
                    "intermediate_steps": intermediate_steps,
                }
        except Exception:
            warning = (
                f"\n\n[Agent reached maximum iteration limit ({self.max_iterations}). "
                "Please continue in the next turn.]"
            )
            final_output = (response.content or "") + warning if response else warning
            return {
                "output": final_output,
                "intermediate_steps": intermediate_steps,
            }

    def _build_anthropic_messages(self, user_input: str, chat_history: list) -> list:
        """Convert LangChain history + user input to Anthropic messages format."""
        messages = []

        for msg in chat_history:
            from langchain_core.messages import HumanMessage, AIMessage
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "user"
            content = getattr(msg, "content", str(msg))
            if content and content.strip():
                messages.append({"role": role, "content": content})

        messages = self._enforce_alternation(messages)
        messages.append({"role": "user", "content": user_input})
        return messages

    @staticmethod
    def _add_cache_breakpoint(message: dict):
        """Add cache_control breakpoint to a message for prompt caching.

        Converts string content to block format if needed, since cache_control
        requires content blocks (not plain strings).
        """
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = [{
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }]
        elif isinstance(content, list) and content:
            content[-1]["cache_control"] = {"type": "ephemeral"}

    def _enforce_alternation(self, messages: list) -> list:
        """Ensure strict user/assistant alternation required by Anthropic API."""
        if not messages:
            return messages

        result = [messages[0]]
        for msg in messages[1:]:
            if msg["role"] == result[-1]["role"]:
                prev_content = result[-1]["content"]
                curr_content = msg["content"]
                result[-1]["content"] = f"{prev_content}\n\n{curr_content}"
            else:
                result.append(msg)
        return result

    def _build_assistant_content(self, response: LLMResponse) -> list:
        """Build assistant message content blocks from LLM response.

        Includes text and tool_use blocks. Thinking blocks are NOT included
        (Anthropic API rejects them in follow-up messages).
        """
        content = []

        if response.content:
            content.append({"type": "text", "text": response.content})

        for tc in response.tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": self._sanitize_tool_input(tc.input),
            })

        return content

    def _build_action_log(self, response: LLMResponse, tool_call: ToolCall) -> str:
        """Build action.log compatible with ThinkingDisplayCallback.

        Format matches LangChain's OpenAI functions agent log format:
        "\\nInvoking: `tool_name` with `{args}`\\nresponded: <reasoning>\\n\\n"

        Uses "thinking:" prefix when real extended thinking is present,
        "responded:" for regular text content.
        """
        args_str = json.dumps(tool_call.input, default=str)
        log = f"\nInvoking: `{tool_call.name}` with `{args_str}`\n"

        if response.thinking:
            log += f"thinking: {response.thinking}\n\n"
        elif response.content:
            log += f"responded: {response.content}\n\n"

        return log

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool by name with timeout protection.

        Returns observation string. Propagates SkillEscalationInterrupt.
        TimeoutError and other exceptions are caught and returned as error messages.
        """
        if tool_name not in self.tool_map:
            return (
                f"{ESCALATION_MARKER} skill_escalation_pending\n"
                f"Error: `{tool_name}` is not a valid tool. "
                "Available tools: " + ", ".join(sorted(self.tool_map.keys())[:20]) +
                ". If you need a tool from another skill, call request_additional_skill."
            )

        tool = self.tool_map[tool_name]
        timeout = self._tool_timeouts.get(tool_name, DEFAULT_TOOL_TIMEOUT)

        param_error = self._check_required_params(tool_name, tool_input)
        if param_error:
            prefixed = f"{SCHEMA_ERROR_PREFIX}{param_error}"
            self._schema_error_fingerprints.add(_make_fingerprint(tool_name, prefixed))
            logger.warning(f"[PARAM_CHECK] {tool_name}: {param_error}")
            return prefixed

        if self._pel is not None:
            policy_result = self._pel.check(tool_name, tool_input)
            if not policy_result.allowed:
                error_text = policy_result.to_tool_error()
                logger.warning(f"[PEL] Blocked {tool_name}: {error_text[:200]}")
                return error_text

        _kwargs = {**tool_input, "config": {}}
        if self._pel is not None and hasattr(tool, "mcp_session"):
            _kwargs["_pel_checked"] = True

        try:
            result = await asyncio.wait_for(
                tool._arun(**_kwargs),
                timeout=timeout,
            )
            if isinstance(result, dict):
                error_val = result.get("error", "")
                if error_val and ("validation error" in error_val.lower()
                                  or "unexpected keyword" in error_val.lower()
                                  or "missing required" in error_val.lower()):
                    return self._format_schema_error_from_result(
                        tool_name, tool_input, error_val
                    )
                return json.dumps(result, indent=2, default=str)
            return str(result) if result is not None else ""
        except SkillEscalationInterrupt:
            raise
        except asyncio.TimeoutError:
            error_msg = (
                f"Error: Tool '{tool_name}' timed out after {timeout} seconds. "
                f"You may retry with simpler parameters or try a different approach."
            )
            logger.warning(f"[TIMEOUT] {tool_name} exceeded {timeout}s timeout")
            return error_msg
        except (ValidationError, TypeError) as e:
            tool = self.tool_map.get(tool_name)
            all_params = self._get_all_param_names(tool) if tool else []
            required = self._get_required_params(tool) if tool else []
            provided = [k for k in tool_input.keys() if not k.startswith("_")]
            invalid = [k for k in provided if k not in all_params]
            valid_display = ", ".join(
                f"{p} (required)" if p in required else p for p in all_params
            )
            example = self._build_example_format(tool, required) if tool and required else ""
            parts = [f"{tool_name}: {type(e).__name__}: {str(e)[:200]}"]
            if invalid:
                parts.append(f"INVALID parameters you sent: {', '.join(invalid)}")
            if all_params:
                parts.append(f"VALID parameters for this tool: {valid_display}")
            if example:
                parts.append(example)
            parts.append(
                "ACTION: Retry this same tool with corrected parameter names. "
                "Do NOT switch to a different tool."
            )
            error_msg = "\n".join(parts)
            prefixed = f"{SCHEMA_ERROR_PREFIX}{error_msg}"
            self._schema_error_fingerprints.add(_make_fingerprint(tool_name, prefixed))
            logger.warning(f"[SCHEMA_ERROR] {tool_name}: {type(e).__name__}")
            return prefixed
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {type(e).__name__}: {str(e)}"
            logger.warning(error_msg)
            return error_msg

    def _format_tool_content(self, observation: str) -> str | list:
        """Convert tool observation to content block(s). Detects image markers.

        If the observation is a JSON string with __image__: true, converts it
        to a multimodal content block list (image + text) for the Anthropic API.
        Otherwise returns the observation string as-is.
        """
        if observation and observation.startswith('{"__image__":'):
            try:
                import json as _json
                data = _json.loads(observation)
                if data.get("__image__"):
                    return [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": data["media_type"],
                            "data": data["base64"],
                        }},
                        {"type": "text", "text": data.get("description", "Image fetched from web")},
                    ]
            except (ValueError, KeyError):
                pass
        return observation

    async def _trim_observation(
        self, observation: str, tool_name: str,
        intent: str = "", tool_input: dict | None = None,
    ) -> str:
        """Archive large tool outputs and summarize with intent-aware Haiku extraction.

        Outputs <= threshold stay fully in context. Outputs > threshold get:
        1. Archived to disk (always, for auditability)
        2. Summarized by Haiku using agent's intent (thinking + tool_input)
        3. On Haiku failure: deterministic head(65%) + tail(30%) fallback
        """
        if not observation or len(observation) <= self._max_observation_chars:
            return observation
        if observation.startswith('{"__image__":'):
            return observation

        # Break infinite loop: re-read of already-archived output
        if tool_name == "read_text_file" and tool_input:
            path = tool_input.get("path", "")
            if ".tool_outputs/" in path:
                from core.intent_summarizer import deterministic_truncate
                return deterministic_truncate(observation, self._max_observation_chars)

        # Fast-path: write-tool results in execute phase don't need smart summarization
        _WRITE_TOOLS = {"write_text_file", "edit_file", "batch_file_edit", "create_file"}
        if tool_name in _WRITE_TOOLS and self._current_phase == "execute":
            from core.intent_summarizer import deterministic_truncate
            return deterministic_truncate(observation, self._max_observation_chars)

        # Archive to disk (always)
        output_path = self._archive_to_disk(observation, tool_name)
        if not output_path:
            from core.intent_summarizer import deterministic_truncate
            return deterministic_truncate(observation, self._max_observation_chars)

        # Intent-aware summarization via Haiku
        try:
            from core.intent_summarizer import (
                build_intent_string, summarize_with_intent,
            )
            intent_str = build_intent_string(intent, tool_name, tool_input)
            summary = await summarize_with_intent(
                content=observation,
                intent=intent_str,
                max_output=self._max_observation_chars - 500,
                timeout=30,
            )
            logger.info(
                f"[TOOL_OUTPUT] {tool_name}: {len(observation):,} chars → "
                f"{len(summary):,} chars via intent summarization "
                f"(archived: {output_path})"
            )
            return (
                f"┌─ SUMMARIZED: {tool_name} ({len(observation):,} chars → {len(summary):,}) ─┐\n"
                f"{summary}\n"
                f"└─ Full output: {output_path} ─┘"
            )
        except Exception as e:
            logger.warning(
                f"[TOOL_OUTPUT] Summarization failed for {tool_name}, using fallback: {e}"
            )
            from core.intent_summarizer import deterministic_truncate
            return deterministic_truncate(
                observation, self._max_observation_chars, str(output_path)
            )

    def _archive_to_disk(self, observation: str, tool_name: str):
        """Save tool output to disk for auditability. Returns path or None."""
        from core.persistence import get_work_dir
        _wd = get_work_dir()
        if not _wd:
            _wd = "/tmp"
        output_dir = Path(_wd) / "dynamic_tasks" / ".tool_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{tool_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}.txt"
        output_path = output_dir / filename

        try:
            output_path.write_text(observation, encoding="utf-8")
            return output_path
        except Exception as e:
            logger.warning(f"[TOOL_OUTPUT] Failed to save to disk: {e}")
            return None

    @staticmethod
    def _sanitize_tool_input(tool_input: dict) -> dict:
        """Sanitize tool input from LLM responses.

        Handles two known model quirks:
        1. Bedrock/LiteLLM empty-string keys: {"": {}} → stripped
        2. Stringified arrays (Nemotron, some open models): '["a","b"]' → ["a","b"]
           These models serialize list parameters as JSON strings instead of native arrays.
        """
        if not tool_input:
            return {}
        sanitized = {}
        for k, v in tool_input.items():
            if k == "":
                continue
            if isinstance(v, str) and len(v) > 1 and v[0] == "[" and v[-1] == "]":
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        v = parsed
                except (json.JSONDecodeError, ValueError):
                    pass
            sanitized[k] = v
        return sanitized

    def _get_tool_schema(self, tool) -> dict | None:
        """Get JSON schema from tool (args_schema or input_schema_raw)."""
        if hasattr(tool, "args_schema") and tool.args_schema is not None:
            try:
                return tool.args_schema.model_json_schema()
            except (AttributeError, Exception):
                pass
        if hasattr(tool, "input_schema_raw") and tool.input_schema_raw:
            return tool.input_schema_raw
        return None

    def _get_required_params(self, tool) -> list[str]:
        """Extract required parameter names from a tool's schema."""
        schema = self._get_tool_schema(tool)
        if not schema:
            return []
        required = schema.get("required", [])
        return [r for r in required if not r.startswith("_")]

    def _get_all_param_names(self, tool) -> list[str]:
        """Extract all valid parameter names from a tool's schema."""
        schema = self._get_tool_schema(tool)
        if not schema:
            return []
        props = schema.get("properties", {})
        return [p for p in props if not p.startswith("_")]

    def _check_required_params(self, tool_name: str, tool_input: dict) -> str | None:
        """Pre-validate required params. Returns error string or None."""
        if tool_name not in self.tool_map:
            return None
        tool = self.tool_map[tool_name]
        required = self._get_required_params(tool)
        if not required:
            return None
        missing = [p for p in required if p not in tool_input]
        if not missing:
            return None
        provided = [k for k in tool_input.keys() if not k.startswith("_")]
        all_params = self._get_all_param_names(tool)
        invalid = [k for k in provided if k not in all_params]
        valid_display = ", ".join(
            f"{p} (required)" if p in required else p for p in all_params
        )
        example = self._build_example_format(tool, required)
        parts = [f"{tool_name}: parameter error."]
        if invalid:
            parts.append(f"INVALID parameters you sent: {', '.join(invalid)}")
        parts.append(f"VALID parameters for this tool: {valid_display}")
        if example:
            parts.append(example)
        parts.append(
            "ACTION: Retry this same tool with corrected parameter names. "
            "Do NOT switch to a different tool."
        )
        return "\n".join(parts)

    def _build_example_format(self, tool, required: list[str]) -> str:
        """Build a CORRECT FORMAT example from tool schema for error messages."""
        try:
            schema = self._get_tool_schema(tool)
            if not schema:
                return ""
            props = schema.get("properties", {})
            example_parts = []
            for param in required:
                prop = props.get(param, {})
                ptype = prop.get("type", "string")
                if ptype == "string":
                    example_parts.append(f'"{param}": "your value here"')
                elif ptype == "integer":
                    example_parts.append(f'"{param}": 1')
                elif ptype == "boolean":
                    example_parts.append(f'"{param}": true')
                else:
                    example_parts.append(f'"{param}": ...')
            if example_parts:
                return f"CORRECT FORMAT: {{{', '.join(example_parts)}}}"
        except Exception:
            pass
        return ""

    def _format_schema_error_from_result(
        self, tool_name: str, tool_input: dict, error_val: str
    ) -> str:
        """Format a schema error returned as a dict result into a directive message."""
        tool = self.tool_map.get(tool_name)
        all_params = self._get_all_param_names(tool) if tool else []
        required = self._get_required_params(tool) if tool else []
        provided = [k for k in tool_input.keys() if not k.startswith("_")]
        invalid = [k for k in provided if k not in all_params] if all_params else []
        valid_display = ", ".join(
            f"{p} (required)" if p in required else p for p in all_params
        )
        example = self._build_example_format(tool, required) if tool and required else ""

        parts = [f"{tool_name}: {error_val[:200]}"]
        if invalid:
            parts.append(f"INVALID parameters you sent: {', '.join(invalid)}")
        if all_params:
            parts.append(f"VALID parameters for this tool: {valid_display}")
        if example:
            parts.append(example)
        parts.append(
            "ACTION: Retry this same tool with corrected parameter names. "
            "Do NOT switch to a different tool."
        )
        error_msg = "\n".join(parts)
        prefixed = f"{SCHEMA_ERROR_PREFIX}{error_msg}"
        self._schema_error_fingerprints.add(_make_fingerprint(tool_name, prefixed))
        logger.warning(f"[SCHEMA_ERROR] {tool_name}: schema error in result")
        return prefixed

    def _detect_stuck(self, tool_name: str, observation: str) -> None:
        """Check for repeated error patterns. Raises StuckInterrupt on threshold."""
        obs_str = str(observation)

        if "is not a valid tool" in obs_str:
            return
        if not _is_error_observation(obs_str):
            return

        fingerprint = _make_fingerprint(tool_name, obs_str)
        self._stuck_counts[fingerprint] += 1

        if fingerprint not in self._stuck_details:
            self._stuck_details[fingerprint] = (tool_name, obs_str[:200])

        if self._stuck_counts[fingerprint] >= self._stuck_threshold:
            stored_name, error_snippet = self._stuck_details[fingerprint]
            raise StuckInterrupt(
                tool_name=stored_name,
                error_fingerprint=fingerprint,
                failure_count=self._stuck_counts[fingerprint],
                suggested_query=_suggest_query(stored_name, error_snippet),
                error_snippet=error_snippet,
                is_internal=True,
            )

    async def _fire_on_agent_action(self, action: NativeAgentAction, callbacks: list) -> None:
        """Fire on_agent_action on all callbacks that support it."""
        run_id = uuid.uuid4()
        for cb in callbacks:
            if hasattr(cb, "on_agent_action"):
                try:
                    await cb.on_agent_action(
                        action,
                        run_id=run_id,
                        parent_run_id=None,
                        tags=None,
                    )
                except Exception:
                    pass
