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

import os
import json
import yaml
import logging
import chainlit as cl
from chainlit.server import connect_mcp as cl_connect_mcp
from chainlit.types import ConnectStreamableHttpMCPRequest
from typing import Dict, List, Any, Optional
import datetime
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
import traceback
import asyncio
import re
import litellm
import aiohttp
from pathlib import Path

logger = logging.getLogger(__name__)

EXECUTOR_SESSION_TIMEOUT = 1800  # 30 min safety net for executor.ainvoke()

# ── P1: Import from core/ modules (tested, not inline copies) ──────────────────
from core.history import (
    history_to_text,
    trim_history_by_tokens,
    history_to_text_with_budget,
    estimate_tokens,
    truncate_oversized_messages,
    async_truncate_oversized_messages,
    enforce_total_token_budget,
    sanitize_history,
    build_sliding_window_context,
    async_build_sliding_window_context,
    MAX_SINGLE_MESSAGE_TOKENS,
    DEFAULT_AGENT_TOKEN_BUDGET,
    BEDROCK_HARD_TOKEN_LIMIT,
    SLIDING_WINDOW_SIZE,
    CHARS_PER_TOKEN,
    async_compact_history,
    COMPACTION_TOKEN_THRESHOLD,
    format_tool_call_record,
)
from core.serialization import make_json_serializable
from core.config import load_mcp_server_config, load_user_extension_configs
from core.agent_utils import (
    classify_error,
    get_retry_params,
    validate_agent_result,
    build_tool_enforcement_retry_prompt,
    MAX_HALLUCINATION_RETRIES,
    validate_workflow_completion,
    check_workflow_environment,
)
from core.skill_loader import SkillLoader
from core.policy_enforcement import PolicyEnforcementLayer
from core.playbook import PlaybookMemory
from core.single_agent import (
    create_request_additional_skill_tool,
    build_skill_selection_prompt,
    async_build_skill_selection_prompt,
    format_history_for_skill_selector,
    SKILL_SELECTION_PROMPT_TEMPLATE,
    parse_skill_selection,
    filter_tools_for_skills,
    build_agent_system_prompt,
    get_agent_config_for_skills,
    create_skill_based_agent,
    SkillSelection,
    detect_escalation_in_result,
    format_intermediate_steps_for_handoff,
    async_format_intermediate_steps_for_handoff,
    ESCALATION_MARKER,
    MAX_ESCALATION_ITERATIONS,
)
from core.sub_agent import summarize_escalation_handoff  # kept for potential future use
from core.persistence import (
    get_user_data_dir,
    get_user_settings,
    save_user_settings,
    update_work_dir_from_env,
    bootstrap_work_dir_from_env,
    get_work_dir,
)
from core.memory import build_memory_context_block, get_protocols_dir
from core.checkpointing import (
    checkpoint_tool_call,
    checkpoint_tool_result,
    checkpoint_tool_error,
    get_completed_tool_calls,
    clear_session_checkpoints,
)
from core.fresh_start import handle_fresh_start
from core.session_log import (
    init_session_log,
    append_message as session_log_append,
    append_phase_marker,
    get_session_log_dir,
    load_history_from_session_log,
)
from core.protocol_recorder import (
    ProtocolRecorder,
    DiskFullError,
    should_nudge as should_nudge_recording,
)
from core.protocol_player import (
    ProtocolPlayer,
    PlayMode,
    PlayState,
    list_available_protocols,
    find_protocol_by_name,
    detect_slurm_checkpoints,
)
from core.protocol_refiner import (
    ProtocolRefiner,
    has_golden_protocol,
    load_golden_protocol,
    substitute_variables,
)
from core.disk_monitor import (
    run_startup_check,
    check_disk_space,
    should_pause_recording,
    format_startup_toast,
    DiskStatus,
)
import time as _time

# ── Model ID mapping for user-requested model switches ──────────────────────────
# Detection is handled by the Haiku skill selector (SkillSelection.requested_model).
# This maps the extracted short name to the actual LiteLLM model ID.
_MODEL_ID_MAP = {
    "nemotron": "nvidia.nemotron-super-3-120b",
    "gpt-oss-20b": "openai.gpt-oss-20b-1",
    "gpt-oss-120b": "openai.gpt-oss-120b-1",
    "gpt-oss": "openai.gpt-oss-120b-1",
}


# ── SIGTERM handler: flush session facts to disk on SLURM kill ─────────────────
import signal

def _handle_sigterm(signum, frame):
    """Graceful shutdown on SLURM SIGTERM — session log is already on disk per-turn."""
    print("[SIGTERM] Received — session log persisted per-turn (crash-safe). Exiting.")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, _handle_sigterm)


# ── Cost Tracking Callback ─────────────────────────────────────────────────────
# Accumulates token usage and cost across all LLM calls in a single user turn.
# Uses litellm.completion_cost() for pricing — no static pricing dict needed.
# LiteLLM knows the cost of every model and keeps pricing up to date.
#
# IMPORTANT: All LLMs must be created with disable_streaming=True so that
# on_llm_end receives full token_usage data. When ChatOpenAI uses streaming
# (the default), AgentExecutor's internal LLM calls return empty llm_output.
class CostTrackingCallback(BaseCallbackHandler):
    """Accumulates token usage and cost across all LLM calls in a turn.

    Works for both direct calls (skill selector) and
    intermediate AgentExecutor calls — no DB queries, pure in-memory.
    Uses litellm.completion_cost() for accurate, maintenance-free pricing.
    """

    def __init__(self):
        super().__init__()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.num_calls = 0

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        """Called after every LLM call — the only end hook in LangChain's callback system.

        Extracts token usage from llm_output and uses litellm.completion_cost()
        to calculate the cost. No static pricing dict needed.
        """
        self.num_calls += 1
        try:
            llm_output = getattr(response, "llm_output", None) or {}
            usage = llm_output.get("token_usage") or llm_output.get("usage", {})
            model_name = llm_output.get("model_name", "unknown")

            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            if input_tokens or output_tokens:
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens

                # Let LiteLLM calculate the cost — it knows all model pricing.
                # Use cost_per_token() which accepts prompt_tokens/completion_tokens directly.
                # Strip the "us." cross-region prefix if present so LiteLLM recognises the model.
                try:
                    clean_model = model_name.lstrip("us.") if model_name.startswith("us.") else model_name
                    prompt_cost, completion_cost_val = litellm.cost_per_token(
                        model=clean_model,
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                    )
                    cost = prompt_cost + completion_cost_val
                    if cost == 0.0 and (input_tokens > 0 or output_tokens > 0):
                        print(f"[COST_TRACK] Warning: Zero cost for model '{model_name}' "
                              f"(tried '{clean_model}') — model may not be in LiteLLM price list.")
                except Exception as cost_err:
                    print(f"[COST_TRACK] litellm.cost_per_token() failed for model '{model_name}': {cost_err}")
                    cost = 0.0

                self.total_cost += cost
            else:
                # Token data missing — log for debugging but don't break
                print(f"[COST_TRACK] LLM Call #{self.num_calls}: No token data. "
                      f"model={model_name}, llm_output keys={list(llm_output.keys())}")

        except Exception as e:
            print(f"[COST_TRACK] Error in on_llm_end: {e}")

    def get_summary(self) -> dict:
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "cost": self.total_cost,
            "num_calls": self.num_calls,
        }

    def format_cost_line(self) -> str | None:
        """Format a one-line cost summary for display to the user.
        Returns None if no LLM calls were tracked.
        """
        if self.num_calls == 0:
            return None
        s = self.get_summary()
        return (
            f"💰 ~${s['cost']:.4f} · {s['total_tokens']:,} tokens "
            f"({s['input_tokens']:,} in / {s['output_tokens']:,} out) · "
            f"{s['num_calls']} LLM call{'s' if s['num_calls'] != 1 else ''}"
        )


# ── Starter Message on startup ─────────────────────────────────────────
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Submit AlphaFold job",
            message="I have a FASTA file ready. Can you help me submit an AlphaFold job for protein structure prediction?"
        ),
        cl.Starter(
            label="Check my job status",
            message="Can you check the status of my last AlphaFold job?"
        ),
        cl.Starter(
            label="Show results from a job",
            message="Can you help show me the results and PDB structures from my completed AlphaFold job. Let me know if i need to give job id"
        ),
        cl.Starter(
            label="Find Fasta files I can access",
            message="Can you find all the fasta files in the directories that i have access to in a particular path? Can you check what directories i have access to on the filesystem?"
        ),
        cl.Starter(
            label="Explore a directory",
            message="Can you help list the contents of my home directory and tell me what kind of files are there. If there are FASTA or PDB files, show me one. Let me know if i need to give you a path"
        ),
        cl.Starter(
            label="Visualize a PDB file",
            message="Can you help me visualize PDB files that are generated by alphafold and tell me what the structure looks like. Let me know if i need to give a path or directory"
        ),
        cl.Starter(
            label="Show my conversation history",
            message="Can you show me a summary of my recent conversations and what we discussed?"
        ),
    ]

# ── LiteLLM Proxy Configuration ────────────────────────────────────────
api_key = os.environ.get("LITELLM_VIRTUAL_KEY")
if not api_key:
    raise RuntimeError("LITELLM_VIRTUAL_KEY not set")

LITELLM_URL = os.environ.get("LITELLM_URL", "http://localhost:8080")

# Feature flag: use native Anthropic SDK for supported call sites (skill selection, sub-agents)
IRIS_USE_NATIVE_ANTHROPIC = os.environ.get("IRIS_USE_NATIVE_ANTHROPIC", "1") == "1"

# ── Load MCP servers from config ────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.getenv("MCP_CONFIG_PATH", os.path.join(_APP_DIR, "config", "mcp_servers.yaml"))

# P1: MCP server loading now uses core.config.load_mcp_server_config
def load_mcp_servers():
    servers = load_mcp_server_config(CONFIG_PATH)
    if servers:
        print(f"[INFO] Loaded {len(servers)} MCP servers from config")
    else:
        print(f"[WARNING] No MCP servers loaded from {CONFIG_PATH}")
    return servers

MCP_SERVERS = load_mcp_servers()

# Policy Enforcement Layer — universal tool governance
_pel = PolicyEnforcementLayer(
    policy_path=str(Path(__file__).parent / "config" / "policy.yaml"),
    environment_path=str(Path(__file__).parent / "config" / "environment.yaml"),
)

# Playbook Memory — persistent record of tool outcomes (what worked/failed)
_playbook = PlaybookMemory(
    playbook_path=str(Path(__file__).parent / "logs" / "playbook.jsonl")
)



def _iter_response_chunks(text: str, chunk_size: int = 12):
    """Yield chunks that respect word boundaries for typewriter streaming."""
    i = 0
    while i < len(text):
        end = min(i + chunk_size, len(text))
        if end < len(text) and text[end] not in (' ', '\n', '\t', '.', ',', ':', ';', '!', '?'):
            next_break = text.find(' ', end)
            if next_break != -1 and next_break - i < chunk_size * 2:
                end = next_break + 1
        yield text[i:end]
        i = end


async def _stream_response(content: str, chunk_size: int = 40) -> "cl.Message":
    """Send a response with typewriter streaming effect."""
    msg = cl.Message(content="")
    await msg.send()
    for chunk in _iter_response_chunks(content, chunk_size):
        await msg.stream_token(chunk)
        await asyncio.sleep(0.015)
    await msg.update()
    return msg




async def _update_status(status_msg, label: str) -> None:
    """Update the status message with a new label (no-op if status_msg is None)."""
    if status_msg is not None:
        try:
            status_msg.content = label
            await status_msg.update()
        except Exception:
            pass


def _store_user_feedback(user_message: str, project: str):
    """Store user feedback as a preference in knowledge.md immediately.

    Failure-safe: logs warning on error, never raises.
    Called when is_refinement=True to persist preferences before execution.
    Appends directly to the knowledge file (no LLM call, sub-millisecond).
    """
    try:
        from datetime import datetime
        from pathlib import Path

        quote = user_message[:200].replace("\n", " ").strip()
        entry = (
            f"\n- [USER_PREFERENCE]: User expressed dissatisfaction\n"
            f"  Raw: \"{quote}\"\n"
            f"  Date: {datetime.now().strftime('%Y-%m-%d')}\n"
            f"  Context: Triggered refinement mode — agent should try different approach\n"
        )

        # Resolve memory path (same logic as MCP update_memory)
        import pwd
        username = pwd.getpwuid(os.getuid()).pw_name
        app_name = os.environ.get("IRISAI_APP_NAME", "IrisAIdev")
        memory_root = Path(f"/home/{username}/{app_name}/memory")

        if not project or project == "general":
            fpath = memory_root / "knowledge.md"
        else:
            fpath = memory_root / "projects" / project / "knowledge.md"

        fpath.parent.mkdir(parents=True, exist_ok=True)
        if not fpath.exists():
            fpath.write_text("# Knowledge\n\n", encoding="utf-8")

        # Append (not replace) — preserves existing knowledge
        with open(fpath, "a", encoding="utf-8") as f:
            f.write(entry)

        target_label = f"_global" if (not project or project == "general") else project
        print(f"[REFINEMENT] Preference stored in {target_label}/knowledge.md")
    except Exception as e:
        print(f"[REFINEMENT] WARNING: Failed to store preference: {e}")


def _check_plan_needs_pipeline(plan_text: str) -> bool:
    """Check if the active plan explicitly requires pipeline script execution.

    Only returns True when the plan contains clear indicators that a multi-stage
    Python orchestration script is needed (not just multi-command bash).
    """
    if not plan_text:
        return False
    plan_lower = plan_text.lower()
    pipeline_indicators = [
        "run_pipeline_script",
        "pipeline script",
        "iris.mcp_call",
        "iris.run_shell",
        "iris.submit_slurm",
        "write a python script that orchestrates",
        "multi-stage python",
    ]
    return any(indicator in plan_lower for indicator in pipeline_indicators)


# ── MCP Tool Schema Builder ────────────────────────────────────────────
def _build_args_schema(tool_name: str, input_schema: dict) -> type:
    """Build a Pydantic model from MCP tool inputSchema.

    This ensures the LLM sees actual parameter names (path, content, commands)
    instead of a generic 'kwargs' object — preventing the common validation error
    where the LLM wraps params as {'kwargs': '{"path": "..."}' }.
    """
    from pydantic import create_model

    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    properties = input_schema.get("properties", {})
    if not properties:
        return None

    required = set(input_schema.get("required", []))
    fields = {}
    for name, prop in properties.items():
        py_type = _TYPE_MAP.get(prop.get("type", "string"), Any)
        if name in required:
            fields[name] = (py_type, ...)
        else:
            default = prop.get("default", None)
            fields[name] = (Optional[py_type], default)

    try:
        return create_model(f"{tool_name}_schema", **fields)
    except Exception:
        return None


# ── MCP Tool Wrapper with Checkpointing ─────────────────────────────────
class MCPTool(BaseTool):
    name: str
    description: str
    mcp_session: Any = None # your MCP session
    _server_name: str = ""

    def _run(self, **kwargs):
        raise NotImplementedError("Use async _arun")

    async def _arun(self, **kwargs: Any) -> Dict[str, Any]:
        if self.mcp_session is None:
            raise RuntimeError(f"No MCP session for tool {self.name}")

        # Unwrap kwargs that the LLM may have wrapped in various ways
        args = kwargs

        if "kwargs" in kwargs:
            val = kwargs["kwargs"]
            # Case 2: LLM wrapped everything in "kwargs" (as dict)
            if isinstance(val, dict):
                args = val
            # Case 3: LLM wrapped everything in "kwargs" (as JSON string)
            elif isinstance(val, str):
                try:
                    parsed = json.loads(val, strict=False)
                    if isinstance(parsed, dict):
                        args = parsed
                        logger.info(f"[KWARGS] Unwrapped '{self.name}' using strict=False")
                except json.JSONDecodeError as e:
                    # Attempt prefix extraction: valid JSON may end before trailing garbage
                    if hasattr(e, 'pos') and e.pos > 2:
                        try:
                            parsed = json.loads(val[:e.pos], strict=False)
                            if isinstance(parsed, dict):
                                args = parsed
                                logger.info(f"[KWARGS] Recovered '{self.name}' via prefix extraction at pos {e.pos}")
                        except (json.JSONDecodeError, ValueError):
                            pass
                    if args is kwargs:
                        logger.warning(
                            f"[KWARGS] Failed to parse kwargs for '{self.name}': "
                            f"{type(e).__name__}: {e} | preview={val[:200]!r}"
                        )
                except ValueError as e:
                    logger.warning(
                        f"[KWARGS] Failed to parse kwargs for '{self.name}': "
                        f"{type(e).__name__}: {e} | preview={val[:200]!r}"
                    )
        # Case 4: LLM passed a single dict as first arg (only if no "kwargs" key)
        elif len(kwargs) == 1 and isinstance(next(iter(kwargs.values())), dict):
            args = next(iter(kwargs.values()))

        # Final safety net: if args still has a "kwargs" wrapper, unwrap it
        if "kwargs" in args and len(args) == 1:
            val = args["kwargs"]
            if isinstance(val, dict):
                args = val
            elif isinstance(val, str):
                try:
                    parsed = json.loads(val, strict=False)
                    if isinstance(parsed, dict):
                        args = parsed
                        logger.info(f"[KWARGS] Safety-net unwrap '{self.name}' using strict=False")
                except json.JSONDecodeError as e:
                    if hasattr(e, 'pos') and e.pos > 2:
                        try:
                            parsed = json.loads(val[:e.pos], strict=False)
                            if isinstance(parsed, dict):
                                args = parsed
                                logger.info(f"[KWARGS] Safety-net recovered '{self.name}' via prefix at pos {e.pos}")
                        except (json.JSONDecodeError, ValueError):
                            pass
                    if "kwargs" in args and len(args) == 1:
                        logger.warning(
                            f"[KWARGS] Safety-net parse failed for '{self.name}': "
                            f"{type(e).__name__}: {e} | preview={val[:200]!r}"
                        )
                except ValueError as e:
                    logger.warning(
                        f"[KWARGS] Safety-net parse failed for '{self.name}': "
                        f"{type(e).__name__}: {e} | preview={val[:200]!r}"
                    )

        # ── Strip internal kwargs before MCP call ─────
        _pel_already_checked = False
        if isinstance(args, dict):
            _pel_already_checked = args.pop("_pel_checked", False)
            for _internal_key in ("config", "run_manager", "callbacks", "_output_mode"):
                args.pop(_internal_key, None)

        # ── POLICY ENFORCEMENT (universal, all tools) ──────────────────
        # Skip when native executor already ran PEL (avoids double-counting budgets)
        if _pel_already_checked:
            policy_result = type('_R', (), {'allowed': True, 'warnings': []})()
        else:
            policy_result = _pel.check(self.name, args)
        if not policy_result.allowed:
            # Check if this is an approval-required violation (not hard block)
            violation = policy_result.violations[0]
            if violation.rule_type == "approval_required":
                # Show Yes/No approval button to user
                try:
                    res = await cl.AskActionMessage(
                        content=(
                            f"🔒 **Policy Approval Required**\n\n"
                            f"**Tool:** `{self.name}`\n\n"
                            f"**Reason:** {violation.reason}\n\n"
                            + (
                                f"**Detected command:** `{violation.matched_input}`\n\n"
                                if violation.matched_input else ""
                            ) +
                            f"Do you approve this action?"
                        ),
                        actions=[
                            cl.Action(name="approve", payload={"value": "approve"}, label="✅ Approve"),
                            cl.Action(name="deny", payload={"value": "deny"}, label="❌ Deny"),
                        ],
                        timeout=120,
                    ).send()

                    # Extract action from response
                    action = "deny"  # default if timeout/None
                    if res is not None:
                        if isinstance(res, dict):
                            payload = res.get("payload", res)
                            action = payload.get("value", res.get("name", "deny")) if isinstance(payload, dict) else res.get("name", "deny")
                        elif hasattr(res, "name"):
                            action = res.name
                        elif hasattr(res, "value"):
                            action = res.value

                    if action == "approve":
                        # Find the pattern that triggered this and approve it
                        blocked_rules = _pel.policy.get("blocked_patterns", [])
                        input_strings = _pel._extract_strings(args)
                        combined_input = " ".join(input_strings)
                        for rule in blocked_rules:
                            p = rule.get("pattern", "")
                            if p and rule.get("requires_approval") and p in combined_input:
                                _pel.approve_pattern(p)
                        logger.info(f"[PEL] User approved {self.name} — retrying")
                        # Re-check policy (should now pass since pattern is approved)
                        policy_result = _pel.check(self.name, args)
                        if not policy_result.allowed:
                            # Still blocked by a different rule
                            logger.warning(
                                f"[PEL] Still blocked after approval: {policy_result.violations[0].reason}"
                            )
                            return {
                                "content": [{"type": "text", "text": policy_result.to_tool_error()}],
                                "isError": True
                            }
                        # Approved and re-check passed — fall through to execute
                    else:
                        # User denied
                        logger.info(f"[PEL] User denied {self.name}")
                        _playbook.record_failure(
                            self.name, args,
                            reason="user_denied",
                            category="permanent",
                        )
                        return {
                            "content": [{"type": "text", "text": "❌ Action denied by user."}],
                            "isError": True
                        }
                except Exception as e:
                    logger.error(f"[PEL] Approval dialog error: {e}")
                    return {
                        "content": [{"type": "text", "text": f"❌ Approval dialog failed: {e}"}],
                        "isError": True
                    }
            else:
                # Hard block — no approval possible
                logger.warning(
                    f"[PEL] Blocked {self.name}: {violation.reason}"
                )
                # Record failure in playbook for future avoidance
                _playbook.record_failure(
                    self.name, args,
                    reason=violation.rule_type,
                    category="permanent" if violation.rule_type == "pattern" else "transient",
                )
                return {
                    "content": [{
                        "type": "text",
                        "text": policy_result.to_tool_error()
                    }],
                    "isError": True
                }
        # ── END POLICY ENFORCEMENT ─────────────────────────────────

        # Retry configuration for MCP session disconnections
        MAX_MCP_RETRIES = 3
        RETRY_DELAYS = [1.0, 2.0, 4.0]  # Exponential backoff
        # Long-running tools need much larger timeouts than the default 60s
        _LONG_RUNNING_TOOL_TIMEOUTS = {
            "execute_dynamic_task": 300,
            "submit_slurm_job": 7200,
        }
        _default_timeout = _LONG_RUNNING_TOOL_TIMEOUTS.get(self.name, 60)
        _tool_timeout = args.get("timeout", _default_timeout) if isinstance(args, dict) else _default_timeout
        MCP_CALL_TIMEOUT = max(60, _tool_timeout + 30)

        username = cl.context.session.user.identifier
        last_error = None
        _tool_start = _time.time()

        for attempt in range(MAX_MCP_RETRIES):
            try:
                # Checkpoint before tool call (sync — no await needed)
                if attempt == 0:
                    checkpoint_tool_call(username, self.name, args)

                _tool_start = _time.time()
                result = await asyncio.wait_for(
                    self.mcp_session.call_tool(self.name, args),
                    timeout=MCP_CALL_TIMEOUT
                )
                _tool_duration_ms = (_time.time() - _tool_start) * 1000

                # Checkpoint after successful tool call (sync)
                checkpoint_tool_result(username, self.name, args, result)

                # Record success in playbook for future reference
                _playbook.record_success(self.name, args)

                # ── Protocol recording: capture step if recording is active ──
                _proto_recorder = cl.user_session.get("protocol_recorder")
                if _proto_recorder and _proto_recorder.is_active:
                    try:
                        _result_for_proto = result.model_dump() if hasattr(result, "model_dump") else result
                        if self.name in ("web_search", "fetch_url_content"):
                            _query = args.get("query", args.get("url", ""))
                            _proto_recorder.record_reference(_query, [])
                        else:
                            _proto_recorder.record_step(self.name, args, _result_for_proto, _tool_duration_ms)
                    except DiskFullError:
                        try:
                            await cl.context.emitter.send_toast(
                                message="Protocol recording paused: disk full!",
                                type="error",
                            )
                        except Exception:
                            print("[PROTOCOL] Disk full — recording paused")
                elif should_nudge_recording(self.name, False, cl.user_session.get("protocol_nudged", False)):
                    cl.user_session.set("protocol_nudged", True)
                    try:
                        await cl.context.emitter.send_toast(
                            message="Procedural action detected — click Protocol button to start recording.",
                            type="info",
                        )
                    except Exception:
                        pass

                result_data = result.model_dump() if hasattr(result, "model_dump") else result
                if isinstance(result_data, dict):
                    result_data.pop("structuredContent", None)

                # ── Refresh session cache if work_dir was changed ────────
                if self.name == "set_user_work_directory":
                    _rd = result_data if isinstance(result_data, dict) else {}
                    _content = _rd.get("content", [])
                    if isinstance(_content, list):
                        for _item in _content:
                            if isinstance(_item, dict) and _item.get("type") == "text":
                                try:
                                    _parsed = json.loads(_item.get("text", ""))
                                    if _parsed.get("success") and _parsed.get("work_dir"):
                                        cl.user_session.set("work_dir", _parsed["work_dir"])
                                except (json.JSONDecodeError, TypeError):
                                    pass

                return result_data

            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                is_timeout = isinstance(e, asyncio.TimeoutError)
                error_msg = (
                    f"(timed out after {MCP_CALL_TIMEOUT}s — MCP server likely disconnected)"
                    if is_timeout
                    else str(e) or "(empty — likely MCP session disconnection)"
                )

                # Determine if this is a retryable error (connection/session issues)
                retryable_indicators = [
                    "disconnect", "closed", "broken", "eof",
                    "connection", "transport", "timeout", "reset",
                ]
                is_retryable = (
                    is_timeout  # Timeout = dead connection, always retry
                    or not str(e)  # Empty error string = MCP disconnect
                    or any(ind in str(e).lower() for ind in retryable_indicators)
                )

                if is_retryable and attempt < MAX_MCP_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    print(
                        f"[MCP-RETRY] Tool '{self.name}' failed (attempt {attempt + 1}/{MAX_MCP_RETRIES}): "
                        f"{error_type}: {error_msg}. Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)

                    # ── MCP RECONNECTION: attempt to re-establish session ──
                    # If the MCP server disconnected (e.g. transient network blip,
                    # server crash + restart), retrying on the same dead session
                    # will always fail. Try to reconnect before the next attempt.
                    reconnected = await self._attempt_reconnect()
                    if reconnected:
                        print(f"[MCP-RECONNECT] Successfully reconnected to MCP server for tool '{self.name}'")

                    continue
                else:
                    # Final attempt failed or non-retryable error
                    rich_error = (
                        f"MCP tool '{self.name}' failed after {attempt + 1} attempt(s). "
                        f"Error type: {error_type}. "
                        f"Details: {error_msg}. "
                        f"Session state: {'connected' if self.mcp_session else 'None'}"
                    )
                    print(f"[MCP-ERROR] {rich_error}")
                    checkpoint_tool_error(username, self.name, args, rich_error)
                    # Protocol recording: capture failed step
                    _proto_recorder = cl.user_session.get("protocol_recorder")
                    if _proto_recorder and _proto_recorder.is_active:
                        _err_duration = (_time.time() - _tool_start) * 1000
                        try:
                            _proto_recorder.record_step(self.name, args, None, _err_duration, error=rich_error)
                        except DiskFullError:
                            pass
                    return {"error": rich_error}

    async def _attempt_reconnect(self) -> bool:
        """Attempt to reconnect the MCP session for this tool.

        Delegates to the shared _reconnect_mcp_server() helper which handles
        config lookup, reconnection, and sibling tool propagation.
        """
        server_name = self._server_name or None
        if not server_name:
            mcp_sessions = cl.user_session.get("mcp_sessions", {})
            for name, session in mcp_sessions.items():
                if session is self.mcp_session:
                    server_name = name
                    break
        if not server_name:
            print(f"[MCP-RECONNECT] Could not identify server for tool '{self.name}' — skipping")
            return False

        result = await _reconnect_mcp_server(server_name)
        if result:
            updated_sessions = cl.user_session.get("mcp_sessions", {})
            new_session = updated_sessions.get(server_name)
            if new_session:
                self.mcp_session = new_session
        return result

# ── Phase 1: Dynamic Skill Loader ────────────────────────────────────────
# Replaces the old AGENT_REGISTRY + agents.yaml + load_agent_registry().
# Skills are auto-discovered from skills/*.md files — no registration needed.
# Adding/modifying/removing a skill = adding/editing/deleting a .md file.
SKILLS_DIR = os.getenv("SKILLS_DIR", os.path.join(_APP_DIR, "skills"))

# Discover user extension skill files from /home/{user}/{IRISAI_APP_NAME}/extensions/*/
_username = os.environ.get("USER", os.environ.get("username", ""))
_irisai_app_name = os.environ.get("IRISAI_APP_NAME", "IrisAIdev")
_user_ext_dir = Path(f"/home/{_username}/{_irisai_app_name}/extensions")
_extra_skill_dirs: list = []
if _user_ext_dir.exists() and _user_ext_dir.is_dir():
    for _ext_subdir in sorted(_user_ext_dir.iterdir()):
        if _ext_subdir.is_dir() and (_ext_subdir / "skill.md").exists():
            _extra_skill_dirs.append(_ext_subdir)
    if _extra_skill_dirs:
        print(f"[INFO] Found {len(_extra_skill_dirs)} user extension skill(s)")

skill_loader = SkillLoader(SKILLS_DIR, extra_dirs=_extra_skill_dirs)
print(f"[INFO] Loaded {len(skill_loader.list_skill_names())} skills: {skill_loader.list_skill_names()}")

# Create the request_additional_skill tool (dynamic skill escalation)
try:
    _request_skill_tool = create_request_additional_skill_tool(skill_loader)
    print(f"[INFO] Created request_additional_skill tool for dynamic escalation")
except Exception as e:
    _request_skill_tool = None
    print(f"[WARN] Could not create request_additional_skill tool: {e}")


# ──────────────────────────────────────────────────────────────
# Helper: Core logic — skill selection + agent execution
# ── Mid-loop stuck handler ────────────────────────────────────────────────────
# Called when StuckDetectionCallback raises StuckInterrupt inside ainvoke().
# Mirrors the UX of normal web_search approval:
#   - globe OFF → AskActionMessage "I'm stuck, search for X?" with ✅/❌
#   - globe ON  → auto_web_search() immediately, inject result, re-invoke agent
# ──────────────────────────────────────────────────────────────────────────────
async def _handle_stuck_mid_loop(stuck, agent_input, recent_history, executor, agent_callbacks):
    """Handle a StuckInterrupt raised mid-loop by StuckDetectionCallback.

    Args:
        stuck: The StuckInterrupt exception (has .tool_name, .suggested_query,
               .failure_count, .error_snippet attributes).
        agent_input: The original agent input string (for re-invoke after search).
        recent_history: The chat history list (for re-invoke after search).
        executor: The AgentExecutor (for re-invoke after search).
        agent_callbacks: The callbacks list (for re-invoke after search).
    """
    from core.websearch_tools import _approval_gate, _get_approval_lock, auto_web_search
    from core.stuck_detection_callback import StuckInterrupt

    websearch_on = cl.user_session.get("websearch_enabled", False)
    tool_name = stuck.tool_name
    failure_count = stuck.failure_count
    suggested_query = stuck.suggested_query
    error_snippet = stuck.error_snippet[:200]

    if stuck.is_internal:
        correction = (
            f"SYSTEM CORRECTION: Your last {failure_count} calls to `{tool_name}` "
            f"failed with the same internal error:\n"
            f"> {error_snippet}\n\n"
            f"This is a parameter/schema error. "
            f"Read the error carefully and provide ALL required parameters "
            f"with correct types and real values. "
            f"If you cannot fix it, try a different approach."
        )
        await cl.Message(
            content=(
                f"⚠️ **Parameter error detected** — retrying with correction.\n\n"
                f"Tool `{tool_name}` failed {failure_count}x: "
                f"`{error_snippet[:100]}`"
            )
        ).send()
        enriched_input = f"{correction}\n\n---\nOriginal task:\n{agent_input}"
        try:
            from core.stuck_detection_callback import StuckDetectionCallback
            fresh_stuck_cb = StuckDetectionCallback(threshold=3)
            retry_result = await executor.ainvoke(
                {"input": enriched_input, "chat_history": recent_history, "agent_scratchpad": []},
                config={"callbacks": agent_callbacks + [fresh_stuck_cb], "handle_parsing_errors": True}
            )
            retry_output = retry_result.get("output", "")
            if retry_output:
                await cl.Message(content=retry_output).send()
        except StuckInterrupt as _nested:
            await cl.Message(
                content=(
                    f"⚠️ **Still unable to fix parameter error.** "
                    f"`{_nested.tool_name}` continues to fail:\n"
                    f"> `{_nested.error_snippet[:200]}`\n\n"
                    f"Please rephrase your request or provide the content directly."
                )
            ).send()
        except Exception as _e:
            logger.error(f"Re-invoke after internal error correction failed: {_e}")
        return

    if stuck.is_local:
        await cl.Message(
            content=(
                f"⚠️ **Local system error** — this is not something web search can resolve.\n\n"
                f"Tool `{tool_name}` failed {failure_count}x with:\n"
                f"> `{error_snippet}`\n\n"
                f"This looks like a **permissions or filesystem issue** on the cluster. "
                f"Please check file/directory permissions, paths, and disk space, "
                f"then try again."
            )
        ).send()
        return

    if not websearch_on:
        # ── Case A: globe OFF — ask user to approve enabling web search ──────
        # Use the same AskActionMessage pattern as _approval_gate() so the UX
        # is consistent with normal web search approval.
        lock = _get_approval_lock()
        async with lock:
            res = await cl.AskActionMessage(
                content=(
                    f"🔍 **I'm stuck mid-task and need web search to continue.**\n\n"
                    f"I've tried `{tool_name}` **{failure_count}x** with the same error:\n"
                    f"> `{error_snippet}`\n\n"
                    f"**Suggested search:** `{suggested_query}`\n\n"
                    f"Enable web search and run this query now?"
                ),
                actions=[
                    cl.Action(name="approve", payload={"value": "approve"}, label="✅ Search Now"),
                    cl.Action(name="cancel", payload={"value": "cancel"}, label="❌ Skip"),
                ],
                timeout=120,
            ).send()

        # Extract action value (mirrors _extract_action_value in websearch_tools.py)
        action = "cancel"
        if res:
            payload = res.get("payload", res) if isinstance(res, dict) else res
            val = payload.get("value") if isinstance(payload, dict) else None
            if val is None:
                val = res.get("name") if isinstance(res, dict) else getattr(res, "name", None)
            action = val or "cancel"

        if action != "approve":
            # User declined — store query for next turn and return
            await cl.Message(
                content=(
                    f"⏸️ **Web search skipped.** If you'd like me to search for:\n"
                    f"> `{suggested_query}`\n\n"
                    f"Enable the 🌐 web search button and ask me to continue."
                )
            ).send()
            cl.user_session.set("pending_websearch_query", suggested_query)
            return

        # User approved — enable websearch and run the search
        cl.user_session.set("websearch_enabled", True)
        await cl.Message(content=f"🔍 **Searching:** `{suggested_query}`...").send()
        search_result = await auto_web_search(suggested_query)

    else:
        # ── Case B: globe ON — auto-search without approval gate ─────────────
        await cl.Message(
            content=f"🔍 **Stuck on same error {failure_count}x — auto-searching:** `{suggested_query}`"
        ).send()
        search_result = await auto_web_search(suggested_query)

    # ── Inject search result and re-invoke agent ──────────────────────────────
    if search_result and "error" not in search_result.lower()[:50]:
        await cl.Message(
            content=f"✅ **Search complete.** Re-running with results injected...\n\n"
                    f"*If the issue persists, please share more details.*"
        ).send()
        # Prepend search result to agent input so the agent can use it
        enriched_input = (
            f"Web search result for '{suggested_query}':\n{search_result[:2000]}\n\n"
            f"---\nOriginal task:\n{agent_input}"
        )
        try:
            from core.stuck_detection_callback import StuckDetectionCallback
            # Fresh callback so the re-invoke starts with a clean error counter
            fresh_stuck_cb = StuckDetectionCallback(threshold=3)
            retry_result = await executor.ainvoke(
                {
                    "input": enriched_input,
                    "chat_history": recent_history,
                    "agent_scratchpad": [],
                },
                config={
                    "callbacks": agent_callbacks + [fresh_stuck_cb],
                    "handle_parsing_errors": True,
                }
            )
            # Deliver the retry result
            retry_output = retry_result.get("output", "")
            if retry_output:
                await cl.Message(content=retry_output).send()
        except StuckInterrupt as _nested:
            # Still stuck even after web search — give up gracefully
            await cl.Message(
                content=(
                    f"⚠️ **Still stuck after web search.** The error persists:\n"
                    f"> `{_nested.error_snippet[:200]}`\n\n"
                    f"Please provide more context or try a different approach."
                )
            ).send()
        except Exception as _e:
            logger.error(f"Re-invoke after stuck search failed: {_e}")
            await cl.Message(
                content=f"⚠️ **Re-invoke failed after web search:** {str(_e)[:200]}"
            ).send()
    else:
        await cl.Message(
            content=(
                f"⚠️ **Auto-search failed** "
                f"(`{search_result[:100] if search_result else 'no result'}`). "
                f"Please try searching manually or provide more context."
            )
        ).send()


# ── Chainlit step callback for native executor ──────────────────────────
# Renders tool calls as collapsible steps in the Chainlit UI, matching
# the behavior of LangchainCallbackHandler for the old AgentExecutor path.


def _extract_step_output(data) -> str:
    """Extract display-friendly text from tool observation data.

    Handles nested formats:
    1. MCP-wrapped: {"meta": null, "content": [{"type": "text", "text": "..."}]}
       where text may itself be a JSON task result
    2. Task result: {"success": bool, "stdout": str, "stderr": str, "return_code": int, ...}
    3. Structured result: {"success": bool, "key": "value", ...} (no stdout)
    4. Plain string
    """
    if not data:
        return ""
    raw = str(data)
    try:
        import json as _j
        parsed = _j.loads(raw) if isinstance(raw, str) else raw
        if isinstance(parsed, dict):
            if "content" in parsed:
                texts = []
                for block in parsed["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))
                if texts:
                    combined = "\n".join(texts)
                    return _unwrap_task_result(combined, _j) or combined
            if "stdout" in parsed or "success" in parsed:
                return _format_task_result(parsed)
    except (ValueError, TypeError, KeyError):
        pass
    return raw


def _unwrap_task_result(text: str, _j=None) -> str:
    """If text is a JSON task result, extract stdout/stderr. Returns '' if not."""
    if _j is None:
        import json as _j
    try:
        inner = _j.loads(text)
        if isinstance(inner, dict) and ("stdout" in inner or "success" in inner):
            return _format_task_result(inner)
    except (ValueError, TypeError):
        pass
    return ""


def _summarize_dict(d: dict, max_keys: int = 8) -> str:
    """Produce a readable multi-line summary of a dict's interesting fields."""
    _SKIP_KEYS = {"success", "return_code", "task_id", "task_dir",
                  "isError", "meta", "output_truncated", "files_created"}
    status = "✓" if d.get("success", True) else "✗"
    interesting = [(k, v) for k, v in d.items() if k not in _SKIP_KEYS]
    if not interesting:
        return ""
    lines = [status]
    for key, val in interesting[:max_keys]:
        if isinstance(val, list):
            if len(val) <= 3 and all(isinstance(v, str) for v in val):
                lines.append(f"  {key}: {', '.join(val)}")
            else:
                lines.append(f"  {key}: [{len(val)} items]")
        elif isinstance(val, dict):
            lines.append(f"  {key}: {{...}}")
        elif isinstance(val, str) and len(val) > 120:
            lines.append(f"  {key}: {val[:120]}...")
        else:
            lines.append(f"  {key}: {val}")
    return "\n".join(lines)


def _format_task_result(parsed: dict) -> str:
    """Format a task result dict into display-friendly text."""
    import json as _j
    parts = []
    if parsed.get("stdout"):
        stdout = parsed["stdout"].strip()
        try:
            inner = _j.loads(stdout)
            if isinstance(inner, dict):
                summary = _summarize_dict(inner)
                parts.append(summary if summary else stdout)
            elif isinstance(inner, list):
                if len(inner) <= 5:
                    parts.append(_j.dumps(inner, indent=2))
                else:
                    parts.append(f"[{len(inner)} items]\n" + _j.dumps(inner[:3], indent=2) + "\n...")
            else:
                parts.append(stdout)
        except (ValueError, TypeError):
            parts.append(stdout)
    if parsed.get("stderr"):
        parts.append(f"[stderr] {parsed['stderr'].strip()}")
    if not parts:
        summary = _summarize_dict(parsed)
        if summary:
            parts.append(summary)
        else:
            status = "success" if parsed.get("success") else "failed"
            parts.append(f"[{status}, rc={parsed.get('return_code', '?')}]")
    return "\n".join(parts)


_TOOL_EMOJIS = {
    "web_search": "🔍", "fetch_url_content": "🌐",
    "execute_dynamic_task": "⚡", "submit_slurm_job": "🖥️",
    "read_text_file": "📄", "grep_file": "📄", "analyze_files": "📄",
    "write_text_file": "✏️", "edit_file": "✏️",
    "list_directory": "📁", "run_worker_agent": "🤖",
    "run_pipeline_script": "🔧", "query_slurm_cluster": "📊",
    "slurm_monitor_job": "📊", "read_memory": "🧠", "update_memory": "🧠",
    "list_projects": "🧠", "add_project": "🧠", "remove_project": "🧠",
    "write_plan": "📋", "write_findings": "📝",
    "request_additional_skill": "🔀",
    "search_conversation_transcripts": "💬",
    "slurm_cancel_job": "🛑",
    "batch": "📦", "batch_readonly": "📦",
}
_STEP_OUTPUT_MAX = 1500


async def _chainlit_step_callback(tool_name: str, data, event: str, step=None):
    """Callback for NativeAgentExecutor to render tool steps in Chainlit UI."""
    import json as _json
    import time as _t
    from datetime import datetime, timezone

    def _utc_now():
        return datetime.now(timezone.utc).isoformat()

    if event == "start":
        _parent = cl.user_session.get("_tool_steps_parent_id")
        # Lazy-create parent step on first tool call
        if not _parent:
            _msg_id = cl.user_session.get("_tools_parent_msg_id")
            _label = cl.user_session.get("_tools_parent_label", "Tools")
            if _msg_id:
                try:
                    _parent_step = cl.Step(
                        name=f"⚙️ {_label}",
                        type="run",
                        parent_id=_msg_id,
                    )
                    _parent_step.start = _utc_now()
                    await _parent_step.send()
                    _parent = _parent_step.id
                    cl.user_session.set("_tool_steps_parent_id", _parent)
                    cl.user_session.set("_tools_parent_step", _parent_step)
                    print(f"[STEP_UI] Created parent step: {_label}")
                except Exception as _pe:
                    print(f"[STEP_UI] Failed to create parent step: {_pe}")
            else:
                print(f"[STEP_UI] No _tools_parent_msg_id in session — skipping parent step")
        _emoji = _TOOL_EMOJIS.get(tool_name, "⚙️")
        _step = cl.Step(name=f"{_emoji} {tool_name}", type="tool", parent_id=_parent, show_input="json")
        _step.start = _utc_now()
        _step._tool_start_time = _t.time()
        if isinstance(data, dict) and data:
            _step.input = _json.dumps(data, indent=2, default=str)
        elif data:
            _step.input = str(data)
        else:
            _step.show_input = False
        await _step.send()
        cl.user_session.set("_current_tool_step_id", _step.id)
        cl.user_session.set("_current_tool_name", tool_name)
        cl.user_session.set("_current_tool_step_obj", _step)
        return _step

    elif event == "end" and step:
        output = _extract_step_output(data)
        if len(output) > _STEP_OUTPUT_MAX:
            step.output = output[:_STEP_OUTPUT_MAX] + "\n\n... (truncated, full result in context)"
        else:
            step.output = output
        step.end = _utc_now()
        _elapsed = int(_t.time() - getattr(step, '_tool_start_time', _t.time()))
        if _elapsed > 0:
            step.name = f"{step.name} ({_elapsed}s)"
        await step.update()
        cl.user_session.set("_current_tool_step_id", None)
        cl.user_session.set("_current_tool_name", None)
        cl.user_session.set("_current_tool_step_obj", None)
        return None


def _make_nested_step_callback(parent_step_id: str = None):
    """Factory that creates a step callback nesting tool steps under a parent."""
    import json as _json
    import time as _t
    from datetime import datetime, timezone

    def _utc_now():
        return datetime.now(timezone.utc).isoformat()

    async def _nested_cb(tool_name: str, data, event: str, step=None):
        if event == "start":
            _emoji = _TOOL_EMOJIS.get(tool_name, "⚙️")
            kwargs = {"name": f"{_emoji} {tool_name}", "type": "tool", "show_input": "json"}
            if parent_step_id:
                kwargs["parent_id"] = parent_step_id
            _step = cl.Step(**kwargs)
            _step.start = _utc_now()
            _step._tool_start_time = _t.time()
            if isinstance(data, dict) and data:
                _step.input = _json.dumps(data, indent=2, default=str)
            elif data:
                _step.input = str(data)
            else:
                _step.show_input = False
            await _step.send()
            return _step
        elif event == "end" and step:
            output = _extract_step_output(data)
            if len(output) > _STEP_OUTPUT_MAX:
                step.output = output[:_STEP_OUTPUT_MAX] + "\n\n... (truncated, full result in context)"
            else:
                step.output = output
            step.end = _utc_now()
            _elapsed = int(_t.time() - getattr(step, '_tool_start_time', _t.time()))
            if _elapsed > 0:
                step.name = f"{step.name} ({_elapsed}s)"
            await step.update()
            return None

    return _nested_cb


# Replaces the old execute_supervisor_and_agent() which used
# supervisor LLM call + DECISION parsing + safety check.
# Now uses structured output for skill selection (1 LLM call)
# then creates a skill-based agent with filtered tools.
# ──────────────────────────────────────────────────────────────
async def execute_skill_based_turn(
    llm,
    history: list,
    input_content: str,
    tools: list,
    username: str,
    session_id: str,
    cost_tracker: CostTrackingCallback = None
):
    """
    Phase 1 replacement for execute_supervisor_and_agent().

    Flow:
        1. Sanitize history
        2. Prepare history context (Haiku summary + recent window) — ONCE
        3. Select skill(s) via structured output LLM call (with summary)
        4. Assemble tool pool (MCP + built-in + conditional websearch)
        5. Create skill-based AgentExecutor
        6. Execute with hallucination guard + retry
    """
    # Log user query to stdout (appears in output.log)
    print(f"[USER] {input_content[:500]}")
    cl.user_session.set("_last_user_input", input_content)

    # ── POLICY ENFORCEMENT: Reset per-turn counters ─────────────────────
    _pel.reset_turn()

    # ── Bedrock blank-message guard (defense-in-depth) ─────────────────
    history = sanitize_history(history)

    # Build callback config — used for all LLM calls in this turn
    cb_config = {}
    if cost_tracker:
        cb_config = {"callbacks": [cost_tracker]}

    # Per-turn native cost tracker — accumulates Haiku skill selection + executor costs
    from core.cost_tracker import CostTracker as NativeCostTracker
    _turn_cost_tracker = NativeCostTracker()

    # Immediate UI feedback — show processing started
    _init_status = cl.Message(content="⚡ Processing your request...")
    await _init_status.send()

    # ── Step 0: Prepare history context ONCE (shared by skill selector + agent) ──
    # TOKEN-AWARE HISTORY CAP
    recent_history, hist_tokens = trim_history_by_tokens(
        history, max_tokens=DEFAULT_AGENT_TOKEN_BUDGET
    )

    # FINAL GUARD: Enforce total token budget on agent history
    recent_history = enforce_total_token_budget(
        recent_history, max_total_tokens=BEDROCK_HARD_TOKEN_LIMIT
    )
    recent_history = sanitize_history(recent_history)
    hist_tokens = sum(estimate_tokens(getattr(m, 'content', '')) for m in recent_history)

    print(f"[TOKEN_CAP] History: {len(history)} msgs -> {len(recent_history)} msgs, ~{hist_tokens} est tokens (budget: {DEFAULT_AGENT_TOKEN_BUDGET})")

    # ── SINGLE-STEP COMPACTION (Claude Code style) ─────────────────────────
    # When history exceeds threshold, produce one comprehensive summary using
    # the main model (Sonnet/Opus). No progressive cleanup layers — go straight
    # to high-quality compaction that preserves all critical information.
    main_llm = cl.user_session.get("llm", None)

    _early_status = None
    if hist_tokens > COMPACTION_TOKEN_THRESHOLD:
        _early_status = cl.Message(content="🧠 Compacting conversation context...")
        await _early_status.send()

        # Load existing knowledge for dedup in compaction's knowledge extraction
        _compaction_project = cl.user_session.get("project_name", "")
        _existing_knowledge = ""
        if _compaction_project and _compaction_project != "general":
            try:
                from core.memory_state import get_project_dir
                _kb_path = get_project_dir(_compaction_project) / "knowledge.md"
                if _kb_path.exists():
                    _existing_knowledge = _kb_path.read_text(encoding="utf-8")
            except Exception:
                pass

        recent_history = await async_compact_history(
            recent_history,
            token_threshold=COMPACTION_TOKEN_THRESHOLD,
            recent_window=SLIDING_WINDOW_SIZE,
            llm=main_llm,
            existing_knowledge=_existing_knowledge,
        )
        compacted_tokens = sum(estimate_tokens(getattr(m, 'content', '')) for m in recent_history)
        if compacted_tokens < hist_tokens:
            print(f"[COMPACT] History compacted: ~{hist_tokens} tokens -> ~{compacted_tokens} tokens ({len(recent_history)} msgs)")
        hist_tokens = compacted_tokens
        # Invalidate summary cache after compaction
        cl.user_session.set("_cached_conversation_summary", None)
        cl.user_session.set("_cached_summary_msg_count", None)

        # ── Post-compaction disk write-back (belt + suspenders) ───────────
        # Extract knowledge from compaction summary and persist to knowledge.md.
        # Next turn, build_memory_context_block() re-reads it from disk.
        if _compaction_project and _compaction_project != "general" and recent_history:
            try:
                from core.memory_state import (
                    extract_knowledge_from_compaction_summary,
                    append_compaction_knowledge,
                )
                _compaction_content = getattr(recent_history[0], 'content', '')
                _knowledge_extract = extract_knowledge_from_compaction_summary(
                    _compaction_content
                )
                if _knowledge_extract:
                    append_compaction_knowledge(_compaction_project, _knowledge_extract)
                    print(f"[COMPACTION_KB] Persisted {len(_knowledge_extract)} chars to knowledge.md")
            except Exception as _kb_err:
                print(f"[COMPACTION_KB] Write-back failed (non-fatal): {_kb_err}")

        # Strip knowledge extract from in-context summary (disk-only metadata)
        if recent_history and "===KNOWLEDGE_EXTRACT===" in getattr(recent_history[0], 'content', ''):
            _cleaned = getattr(recent_history[0], 'content', '').split("===KNOWLEDGE_EXTRACT===", 1)[0].rstrip()
            recent_history[0].content = _cleaned

    # ── CONTEXT BUILDING: conversation summary + skill selector ──────────
    import asyncio as _aio

    # Retrieve cached summary for incremental updates
    _cached_summary = cl.user_session.get("_cached_conversation_summary", None)
    _cached_summary_count = cl.user_session.get("_cached_summary_msg_count", None)

    # Sliding window: summarizes OLD messages beyond token budget for context
    (conversation_summary, _recent_context) = await async_build_sliding_window_context(
        recent_history,
        recent_window=SLIDING_WINDOW_SIZE,
        max_tokens=DEFAULT_AGENT_TOKEN_BUDGET // 3,
        llm=main_llm,
        cached_summary=_cached_summary,
        cached_summary_msg_count=_cached_summary_count,
    )

    # Cache the summary for next turn's incremental update
    if conversation_summary:
        from core.history import SLIDING_WINDOW_TOKEN_BUDGET, estimate_message_tokens
        _older_count = 0
        _running = 0
        for i in range(len(recent_history) - 1, -1, -1):
            _running += estimate_message_tokens(recent_history[i])
            if _running > SLIDING_WINDOW_TOKEN_BUDGET:
                _older_count = i
                break
        cl.user_session.set("_cached_conversation_summary", conversation_summary)
        cl.user_session.set("_cached_summary_msg_count", _older_count)

    # Skill selector: just last 3 user messages (session facts provide the rest)
    _selector_formatted_history = format_history_for_skill_selector(recent_history)

    # Remove early status if it was shown
    if _early_status:
        try:
            await _early_status.remove()
        except Exception:
            pass

    print(f"[CONTEXT] conversation_summary: {len(conversation_summary)} chars, "
          f"history: {len(recent_history)} msgs, ~{hist_tokens} tokens")

    # ── Step 1: Skill Selection via structured output ──────────────────
    _init_status.content = "🧠 Analyzing request..."
    await _init_status.update()
    websearch_enabled = cl.user_session.get("websearch_enabled", False)

    # Fetch project_name and work_dir early — needed for skill selection prompt
    # and context switch state saving.
    project_name = cl.user_session.get("project_name", "")
    work_dir = cl.user_session.get("work_dir", "")

    # Load known projects with descriptions from unified memory system
    from core.memory_state import get_known_projects_with_descriptions, load_project_state
    _known_projects_str = get_known_projects_with_descriptions()

    # Load active project status for skill selector — gives Haiku visibility into
    # project complexity, active plans, blockers, and failure history so it can
    # make informed needs_planning/needs_research decisions on resume requests.
    _project_status_for_selector = ""
    if project_name and project_name != "general" and work_dir:
        _project_status_for_selector = load_project_state(work_dir, project_name) or ""

    # Build skill selection prompt using pre-computed formatted history (no second Haiku call)
    _summary_parts = []
    if conversation_summary and conversation_summary.strip():
        _summary_parts.append(
            f"Conversation summary (from earlier in this session):\n{conversation_summary}"
        )
    _recent_history_for_selector = cl.user_session.get("history", [])
    if _recent_history_for_selector and len(_recent_history_for_selector) > 2:
        _recent_lines = []
        for m in _recent_history_for_selector[-4:]:
            _mtype = getattr(m, 'type', '?')
            _mcontent = getattr(m, 'content', '') or ''
            if _mtype == 'ai':
                _first_line = _mcontent.split('\n')[0][:150]
                _recent_lines.append(f"[{_mtype}]: {_first_line}")
            else:
                _recent_lines.append(f"[{_mtype}]: {_mcontent[:150]}")
        _recent_ctx_str = "\n".join(_recent_lines)
        _summary_parts.append(f"Recent turns:\n{_recent_ctx_str}")
    if _project_status_for_selector and _project_status_for_selector.strip():
        _summary_parts.append(
            f"Active project status (from memory):\n{_project_status_for_selector}"
        )
    _summary_section = "\n\n".join(_summary_parts)

    selection_prompt = SKILL_SELECTION_PROMPT_TEMPLATE.format(
        manifest=skill_loader.get_manifest(),
        user_input=input_content,
        recent_history=_selector_formatted_history,
        conversation_summary=_summary_section,
        active_project=project_name or "(none)",
        known_projects=_known_projects_str or "general (default — non-project-specific queries)",
    )
    selection_messages = [HumanMessage(content=selection_prompt)]

    try:
        # Use Haiku for skill selection — it's pure classification (structured output).
        # Haiku is ~5x cheaper than Sonnet and handles JSON schema perfectly.
        _haiku_provider = cl.user_session.get("haiku_provider") if IRIS_USE_NATIVE_ANTHROPIC else None

        if _haiku_provider:
            # Native Anthropic path: forced tool_choice for structured output
            raw_selection = await _haiku_provider.create_structured(
                messages=[{"role": "user", "content": selection_prompt}],
                schema=SkillSelection,
            )
            # Track Haiku cost into per-turn tracker
            if hasattr(raw_selection, '_usage'):
                _turn_cost_tracker.accumulate(raw_selection._usage, "anthropic.claude-haiku-4-5-20251001-v1")
        else:
            # LangChain fallback path
            _selector_base_llm = cl.user_session.get("haiku_llm") or llm
            selector_llm = _selector_base_llm.with_structured_output(SkillSelection)
            raw_selection = await selector_llm.ainvoke(
                selection_messages, config=cb_config if cb_config else None
            )

        parsed = parse_skill_selection(raw_selection, skill_loader)
    except Exception as sel_err:
        print(f"[SKILL_SELECT] Structured output failed: {sel_err}. Falling back to conversational.")
        parsed = {
            "skills": ["conversational"],
            "primary_skill": "conversational",
            "reasoning": f"Fallback due to selection error: {sel_err}",
            "fallback_used": True,
        }

    skill_names = parsed["skills"]
    primary_skill = parsed["primary_skill"]

    # ── Project context switch: auto-save outgoing project state ──────
    _new_project_context = parsed.get("project_context")
    _is_context_switch = parsed.get("is_context_switch", False)

    # ── Unified project confirmation: always ask user before switching ──
    from core.memory_state import (
        get_known_project_names, register_project, resolve_project_name,
    )

    _need_confirmation = False
    _project_just_switched = None
    _turn_count = cl.user_session.get("_turn_count", 0)

    if _new_project_context == "__ask_user__":
        _need_confirmation = True
        _new_project_context = None  # No guess — show picker without pre-selection
    elif _turn_count > 1 and _is_context_switch and _new_project_context and _new_project_context != project_name:
        _resolved_switch = resolve_project_name(_new_project_context)
        _known = get_known_project_names()
        if _resolved_switch in _known:
            project_name = await _perform_project_switch(project_name, _resolved_switch, work_dir, parsed)
            _project_just_switched = project_name
        else:
            _need_confirmation = True

    if _need_confirmation and cl.user_session.get("_project_confirmed"):
        _known = get_known_project_names()
        _selected = await _ask_project_confirmation(
            guess=_new_project_context,
            current=project_name,
            known_projects=_known,
            is_new_session=False,
        )

        if _selected and _selected != project_name:
            project_name = await _perform_project_switch(project_name, _selected, work_dir, parsed)
            _project_just_switched = project_name

    # ── Websearch routing ─────────────────────────────────────────────────
    # Web search tools are always available (tool-level approval gate handles
    # enabling). Just ensure the websearch skill is in the pool when needed
    # so the agent gets search-related system prompt context.
    _needs_websearch = parsed.get("needs_websearch", False)
    if _needs_websearch or websearch_enabled:
        if _needs_websearch:
            print(f"[WEBSEARCH] Skill selector flagged needs_websearch=True")
        if websearch_enabled:
            print(f"[WEBSEARCH] Globe ON — injecting websearch skill into pool")
        if "websearch" not in skill_names:
            skill_names.append("websearch")

    # Ensure we always have at least one skill
    if not skill_names:
        skill_names = ["conversational"]
        primary_skill = "conversational"

    print(f"[SKILL_SELECT] Selected: {skill_names} (primary: {primary_skill}, "
          f"reasoning: {parsed['reasoning'][:100]})")
    print(f"[SKILL_SELECT] needs_research={parsed.get('needs_research', False)}, "
          f"needs_planning={parsed.get('needs_planning', False)}, "
          f"needs_slurm={parsed.get('needs_slurm', False)}, "
          f"parallel_subtasks={parsed.get('parallel_subtasks', False)}, "
          f"needs_websearch={_needs_websearch}")

    # ── Step 2: Assemble tool pool ─────────────────────────────────────
    # Start with MCP tools, add built-in tools from core modules.
    # Deduplicate by name — MCP reconnects or multiple servers can produce dupes.
    _seen_tool_names = set()
    all_tools = []
    for _t in tools:
        _tname = getattr(_t, 'name', None)
        if _tname and _tname not in _seen_tool_names:
            all_tools.append(_t)
            _seen_tool_names.add(_tname)

    # Add spend tools (always available)
    try:
        from core.spend_tools import SPEND_TOOLS
        all_tools.extend(SPEND_TOOLS)
    except ImportError as e:
        print(f"[TOOLS] Warning: Could not import spend tools: {e}")

    # Add chainlit tools (always available)
    try:
        from core.chainlit_tools import CHAINLIT_TOOLS
        all_tools.extend(CHAINLIT_TOOLS)
    except ImportError as e:
        print(f"[TOOLS] Warning: Could not import chainlit tools: {e}")

    # Always inject web search tools — they have their own enable + approval gate
    try:
        from core.websearch_tools import WEBSEARCH_TOOLS
        all_tools.extend(WEBSEARCH_TOOLS)
        print(f"[WEBSEARCH] web_search + fetch_url_content available (gate: tool-level approval)")
    except ImportError as e:
        print(f"[WEBSEARCH] Warning: Could not import web search tools: {e}")

    # Add sub-agent tools (always available — Phase 3 context isolation)
    try:
        from core.sub_agent import SUB_AGENT_TOOLS
        all_tools.extend(SUB_AGENT_TOOLS)
        print(f"[SUB-AGENT] Injected {len(SUB_AGENT_TOOLS)} sub-agent tools into tool pool")
    except ImportError as e:
        print(f"[SUB-AGENT] Warning: Could not import sub-agent tools: {e}")

    # ── Worker agent tool (Task tool pattern — injected after full pool) ──
    # WorkerAgentTool receives only the skill-filtered subset of tools, not the
    # full pool. This improves Haiku's tool selection accuracy and reduces prompt
    # token cost. The worker's internal plan-text filter then further narrows to
    # the 3-8 tools most relevant to the specific task.
    _worker_agent_tool = None
    if os.environ.get("IRIS_WORKER_AGENT", "1") == "1":
        try:
            from core.sub_agent import worker_agent as _wa
            # Use skill-filtered tools instead of full pool (cost + accuracy fix)
            _worker_tools = filter_tools_for_skills(all_tools, skill_names, skill_loader)
            _wa.worker_tools = _worker_tools
            _wa._pel_ref = _pel  # Fix 2 & 3: give worker access to shared PEL
            all_tools.append(_wa)
            _worker_agent_tool = _wa
            print(f"[WORKER-AGENT] WorkerAgentTool injected with {len(_wa.worker_tools)} skill-filtered tools (from {len(all_tools)-1} total)")
        except (ImportError, AttributeError) as e:
            print(f"[WORKER-AGENT] Warning: Could not inject WorkerAgentTool: {e}")

    # Add batch tools (composite operations — reduce tool call count)
    try:
        from core.batch_tools import BATCH_TOOLS
        all_tools.extend(BATCH_TOOLS)
        print(f"[BATCH-TOOLS] Injected {len(BATCH_TOOLS)} batch tools into tool pool")
    except ImportError as e:
        print(f"[BATCH-TOOLS] Warning: Could not import batch tools: {e}")

    # Note: run_pipeline_script is included in BATCH_TOOLS via PIPELINE_TOOLS export

    # Add request_additional_skill tool (dynamic skill escalation)
    if _request_skill_tool is not None:
        all_tools.append(_request_skill_tool)
        print(f"[SKILL_ESCALATION] Added request_additional_skill to tool pool")

    # Add write_findings and write_plan tools (single-executor phase tools)
    # Stored under project memory path (persistent, survives work_dir changes)
    try:
        from core.researcher import create_write_findings_tool, get_findings_dir, generate_findings_name
        from core.opus_planner import create_write_plan_tool, create_edit_plan_tool, get_plans_dir, generate_plan_name
        _proj = cl.user_session.get("project_name", "") or "_no_project"
        _findings_dir = str(get_findings_dir(_proj))
        _findings_name = generate_findings_name(input_content[:200])
        all_tools.append(create_write_findings_tool(_findings_dir, _findings_name))
        _plans_dir = str(get_plans_dir(_proj))
        _plan_name = generate_plan_name(input_content[:200])
        all_tools.append(create_write_plan_tool(_plans_dir, _plan_name))
        _pending_plan_full_path = str(Path(_plans_dir) / _plan_name)
        if not cl.user_session.get("_plan_execution_pending", False):
            all_tools.append(create_edit_plan_tool(_pending_plan_full_path))
        cl.user_session.set("_pending_findings_path", str(Path(_findings_dir) / _findings_name))
        cl.user_session.set("_pending_plan_path", _pending_plan_full_path)
        print(f"[PHASE_TOOLS] Injected write_findings + write_plan + edit_plan (project: {_proj})")
    except Exception as e:
        print(f"[PHASE_TOOLS] Warning: Could not inject phase tools: {e}")

    # ── Step 3: Create skill-based agent ───────────────────────────────
    # Tier 1 UI: Create an updatable status message that shows progress
    # through each phase (planning → executing → compressing → done).
    # This eliminates the "blank screen" problem where users see nothing
    # for 2-8+ minutes.
    skills_display = ", ".join(f"**{s}**" for s in skill_names)

    # Replace init status with the main status message
    try:
        await _init_status.remove()
    except Exception:
        pass
    status_msg = cl.Message(content=f"🛸 Crew assembled: {skills_display} reporting for duty!")
    await status_msg.send()
    # Parent tool step is created lazily on first tool call (avoids empty "Working..." UI)
    _skills_label = ", ".join(skill_names)
    cl.user_session.set("_tool_steps_parent_id", None)
    cl.user_session.set("_tools_parent_step", None)
    cl.user_session.set("_tools_parent_msg_id", status_msg.id)
    cl.user_session.set("_tools_parent_label", _skills_label)

    # Build user context block for system prompt injection.
    # This replaces the old pattern where every skill had to call
    # get_user_settings as its first action — now it's automatic.
    work_dir = cl.user_session.get("work_dir", "")
    project_name = cl.user_session.get("project_name", "")

    # Project detection is now handled by SkillSelection (project_context field).
    # Update last_active in the unified memory index when a project is active.
    if project_name:
        try:
            from core.memory_state import register_project
            register_project(project_name, work_dir=work_dir)
        except Exception:
            pass

    user_context = build_memory_context_block(
        username=username,
        work_dir=work_dir,
        project_name=project_name,
        skill_names=skill_names,
        task_keywords=[],
        environment_index=_pel.get_environment_index(),
        weights_path=cl.user_session.get("weights_path", ""),
    )

    # Inject playbook context for tools likely to be used by selected skills
    _SKILL_TOOL_MAP = {
        "bioinformatics": ["submit_slurm_job"],
        "hpc_cluster": ["submit_slurm_job", "execute_dynamic_task"],
        "code_execution": ["execute_dynamic_task"],
        "dev": ["execute_dynamic_task"],
    }
    playbook_tools = set()
    for skill in skill_names:
        playbook_tools.update(_SKILL_TOOL_MAP.get(skill, []))
    if playbook_tools:
        playbook_sections = []
        for tool in playbook_tools:
            ctx = _playbook.get_context_for_tool(tool, {})
            if ctx:
                playbook_sections.append(ctx)
        if playbook_sections:
            user_context += "\n\n" + "\n\n".join(playbook_sections)

    print(f"[USER_CONTEXT] Injected {len(user_context)} chars into system prompt "
          f"for skills {skill_names}")

    dev_llm = cl.user_session.get("dev_llm", llm)
    # Extract complexity for model selection (defaults to 'standard' if not present)
    task_complexity = parsed.get("complexity", "standard") if isinstance(parsed, dict) else getattr(parsed, 'complexity', 'standard')

    # ── Model selection: persistent until user changes it ────────────────────
    # The Haiku skill selector detects model switch requests via `requested_model`.
    # Once set, the model persists across turns until the user explicitly changes it.

    _requested_model = parsed.get("requested_model")

    if _requested_model:
        _requested_model_lower = _requested_model.lower().strip()

        if _requested_model_lower in ("opus", "best", "strongest", "most capable"):
            cl.user_session.set("opus_sticky", True)
            cl.user_session.set("model_override", None)
            task_complexity = "complex"
            print(f"[MODEL_SELECT] User requested Opus — set permanently until changed")
        elif _requested_model_lower in ("sonnet", "default", "cheaper"):
            cl.user_session.set("opus_sticky", False)
            cl.user_session.set("model_override", None)
            print(f"[MODEL_SELECT] User requested Sonnet — set permanently until changed")
        else:
            _model_id = _MODEL_ID_MAP.get(_requested_model_lower, _requested_model_lower)
            cl.user_session.set("model_override", _model_id)
            cl.user_session.set("opus_sticky", False)
            print(f"[MODEL_SELECT] User requested {_model_id} — set permanently until changed")

    # Apply persistent model state
    _model_override = cl.user_session.get("model_override", None)
    if _model_override is None:
        _model_override = os.environ.get("IRIS_DEFAULT_MODEL") or None

    _use_opus_for_executor = cl.user_session.get("opus_sticky", False)
    if _use_opus_for_executor:
        task_complexity = "complex"

    print(f"[MODEL_SELECT] Task complexity: {task_complexity} | opus={_use_opus_for_executor} | model_override: {_model_override}")

    # ── Refinement detection: user dissatisfied → escalate + store preference ──
    _is_refinement = parsed.get("is_refinement", False) if isinstance(parsed, dict) else getattr(parsed, 'is_refinement', False)

    # Safety net: detect indirect dissatisfaction that the skill selector may miss
    _turn_count = cl.user_session.get("_turn_count", 0)
    if not _is_refinement and _turn_count > 1:
        _msg_lower = input_content.lower()
        _criticism_patterns = [
            "homework", "amateurish", "not professional", "not publication",
            "looks cheap", "too basic", "too simple", "undergraduate",
            "low quality", "needs to be better", "nature paper",
            "publication-ready", "publication quality", "print quality",
            "not good enough", "looks terrible", "looks awful",
        ]
        _improvement_patterns = [
            "proper", "professional", "better", "improve", "300 dpi",
            "publication", "high quality", "nature", "svg", "export",
            "print", "typography", "color palette",
        ]
        _has_criticism = any(p in _msg_lower for p in _criticism_patterns)
        _has_improvement = any(p in _msg_lower for p in _improvement_patterns)
        if _has_criticism and _has_improvement:
            _is_refinement = True
            print(f"[REFINEMENT] Safety net triggered — indirect dissatisfaction detected")

    if _is_refinement:
        print(f"[REFINEMENT] User dissatisfied — escalating + storing preference")

        # Auto-escalate to Opus (regardless of model_override — user wants BETTER)
        if not cl.user_session.get("opus_sticky"):
            cl.user_session.set("opus_sticky", True)
            cl.user_session.set("model_override", None)
            _model_override = None
            _use_opus_for_executor = True
            task_complexity = "complex"
            print(f"[REFINEMENT] Auto-escalated to Opus via complexity='complex'")

        # Clear stale phase state — refinement starts fresh
        cl.user_session.set("_unfulfilled_phase_gates", [])
        cl.user_session.set("_resume_from_phase", "")

        # Enable web search
        cl.user_session.set("websearch_enabled", True)
        needs_websearch = True

        # Store user preference immediately (survives session crash)
        _store_user_feedback(
            user_message=input_content,
            project=cl.user_session.get("project_name", ""),
        )

        # Set refinement mode (system prompt injection)
        cl.user_session.set("_refinement_mode", True)
    else:
        cl.user_session.set("_refinement_mode", False)

    # Inject refinement protocol into system prompt when active
    if cl.user_session.get("_refinement_mode"):
        _refinement_block = (
            "\n\n<refinement-protocol>\n"
            "REFINEMENT MODE — The user was not satisfied with the previous result.\n\n"
            "BEFORE proceeding with this task:\n"
            "1. PARSE FEEDBACK: What specifically did the user not like? Quote their exact words.\n"
            "2. DIAGNOSE ROOT CAUSE:\n"
            "   - Wrong information? → Search for better sources (web_search if available)\n"
            "   - Wrong tool/library? → Check get_environment_info('packages') for alternatives\n"
            "   - Poor execution quality? → Read knowledge files: get_environment_info('package:<name>')\n"
            "   - Missing elements? → Re-read the original request carefully\n"
            "3. STATE YOUR NEW APPROACH: What will you do differently this time?\n"
            "4. ACT on your improved understanding — do NOT repeat the same approach\n\n"
            "IMPORTANT:\n"
            "- Check read_memory() for any USER_PREFERENCE entries — they contain prior feedback\n"
            "- If a different package/approach would produce better results, SWITCH to it\n"
            "- When you find what works, call update_memory to store the successful approach as a USER_PREFERENCE\n"
            "</refinement-protocol>"
        )
        user_context = _refinement_block + "\n" + (user_context or "")
        print(f"[REFINEMENT] Injected refinement protocol into system prompt")

    # Pipeline gate: plan-gated injection
    # Pipeline is only available when the active plan explicitly requires it.
    # This prevents the executor LLM from defaulting to pipeline for everything.
    _active_plan_for_pipeline = cl.user_session.get("_active_plan_text", "")
    _plan_needs_pipeline = _check_plan_needs_pipeline(_active_plan_for_pipeline)
    _pipeline_exclude = None
    _pipeline_include_only = None
    if not _plan_needs_pipeline:
        _pipeline_exclude = {"run_pipeline_script"}
        print(f"[PIPELINE_GATE] Pipeline EXCLUDED (plan does not require it)")
    else:
        print(f"[PIPELINE_GATE] Pipeline AVAILABLE (plan explicitly requires multi-stage script)")
    print(f"[PIPELINE_GATE] complexity={parsed.get('complexity')}")

    # Slurm gate: when Haiku classifies the task as needing HPC resources
    _needs_slurm = parsed.get("needs_slurm", False)
    if _needs_slurm:
        if "hpc_cluster" not in skill_names:
            skill_names.append("hpc_cluster")
        print(f"[SLURM_GATE] needs_slurm=True: auto-added hpc_cluster skill, will inject SLURM REQUIRED notice")
    else:
        print(f"[SLURM_GATE] needs_slurm=False")

    print(f"[MODEL_SELECT] Executor LLM: {'opus' if _use_opus_for_executor else _model_override or 'sonnet'}")

    # Determine if executor should check memory before acting.
    # Only needed when no research/planning produced context for this turn.
    _existing_findings = cl.user_session.get("active_findings_paths") or []
    _findings_available = bool(_existing_findings)
    _plan_available = bool(cl.user_session.get("_active_plan_text"))
    _research_will_run = parsed.get("needs_research", False)
    _include_memory_check = not _findings_available and not _plan_available and not _research_will_run
    if _include_memory_check:
        print("[MEMORY_CHECK] No research/planning context — executor will check memory before acting")

    # ── Phased Execution: Structural Enforcement ─────────────────────────
    # When research or planning is needed, the executor receives ONLY the
    # corresponding sub-agent tool. This is structural — the LLM literally
    # cannot see execution tools until the phase completes (file on disk).
    _phase_exclude = set(_pipeline_exclude) if _pipeline_exclude else set()

    _needs_research_raw = parsed.get("needs_research", False)
    _needs_planning_raw = parsed.get("needs_planning", False)

    # Guidance mode override — user forced phased execution via compass button
    _guidance_mode = cl.user_session.get("guidance_mode", False)
    if _guidance_mode:
        _needs_research_raw = True
        _needs_planning_raw = True
        print(f"[GUIDANCE_MODE] Forcing research + planning phases")

    # Websearch globe override — force research phase so model proactively searches
    if websearch_enabled and not _needs_research_raw:
        _needs_research_raw = True
        print(f"[WEBSEARCH] Globe ON — forcing research phase for proactive web search")

    # Track what Haiku/guidance originally wanted (before overrides)
    _originally_wanted_phases = _needs_research_raw or _needs_planning_raw

    # If plan execution is pending (incomplete plan from prior turn), skip phasing
    # BUT if the selector says this is a NEW complex task (needs both research + planning),
    # the user moved on — abandon the old plan rather than blindly continuing it.
    _plan_execution_pending = cl.user_session.get("_plan_execution_pending", False)
    _phase_revision_pending = False
    _revision_target = ""
    if _plan_execution_pending and cl.user_session.get("active_plan_path") and not _originally_wanted_phases:
        _needs_research_raw = False
        _needs_planning_raw = False
        print(f"[PHASED_EXEC] Plan execution pending — direct execution mode")
        _log_path_str = cl.user_session.get("session_log_path", "")
        if _log_path_str:
            append_phase_marker(
                Path(_log_path_str), "execute", "started",
                metadata={"plan_path": cl.user_session.get("active_plan_path", "")},
            )
    elif _plan_execution_pending and _originally_wanted_phases:
        # User moved on to a new task — abandon the old incomplete plan
        cl.user_session.set("_plan_execution_pending", False)
        cl.user_session.set("active_plan_path", "")
        print(f"[PHASED_EXEC] Abandoned incomplete plan — user started new task (selector wants research+planning)")
    else:
        # Restore unfulfilled phase state from previous turn
        _prior_unfulfilled = cl.user_session.get("_unfulfilled_phase_gates") or []
        if "write_plan" in _prior_unfulfilled and not cl.user_session.get("active_plan_path"):
            _needs_planning_raw = True
        if "write_findings" in _prior_unfulfilled and not _existing_findings:
            _needs_research_raw = True

        # Phase revision: user clicked "Revise Findings/Plan" on previous turn
        _phase_revision_pending = cl.user_session.get("_phase_revision_pending", False)
        _revision_target = cl.user_session.get("_phase_revision_target", "")
        if _phase_revision_pending:
            cl.user_session.set("_phase_revision_pending", False)
            cl.user_session.set("_phase_revision_target", "")
            if _revision_target == "research":
                _needs_research_raw = True
            elif _revision_target == "plan":
                _needs_planning_raw = True
            print(f"[PHASE_GATE] Revision mode: re-entering {_revision_target} with existing artifact + user guidance")

        # Clear stale state from old non-blocking button flow (backwards compat)
        _resume_from_phase = cl.user_session.get("_resume_from_phase", "")
        if _resume_from_phase:
            cl.user_session.set("_resume_from_phase", "")
            print(f"[PHASE_GATE] Stale _resume_from_phase='{_resume_from_phase}' cleared")
        cl.user_session.set("_phase_user_action", "")

    # If prior findings exist, research is already done (unless revising)
    if _existing_findings and not _phase_revision_pending:
        _needs_research_raw = False
    # If prior plan exists, planning is already done (unless revising)
    if cl.user_session.get("active_plan_path") and not _phase_revision_pending:
        _needs_planning_raw = False

    # Notify if phases were expected but overridden (e.g. plan already pending)
    if _originally_wanted_phases and not _needs_research_raw and not _needs_planning_raw:
        await send_toast("Direct execution (plan already active)", "info")

    # ── Phase execution loop ─────────────────────────────────────────────
    # Wraps executor build + invoke. Loops when user clicks "Continue" or
    # "Skip to Execute" after a phase pause (same-turn continuation).
    # Breaks on: normal completion, "Revise", "Done for now", or timeout.
    _base_all_tools = list(all_tools)
    _phase_loop_iteration = 0

    while True:
        _phase_loop_iteration += 1
        all_tools = list(_base_all_tools)

        # ── Determine current phase and structurally restrict tools ──────────
        _current_phase = "execute"
        _phase_include_only = None  # None = all tools (execute phase)
        _phase_config = None  # Set when phased execution is active
    
        if _needs_research_raw or _needs_planning_raw:
            _phases_active = []
            if _needs_research_raw:
                _phases_active.append("Research")
            if _needs_planning_raw:
                _phases_active.append("Plan")
            _phase_source = "Guidance mode" if _guidance_mode else "Auto-detected"
            await send_toast(f"{_phase_source}: {' → '.join(_phases_active)} → Execute", "info")
    
            try:
                _proj = cl.user_session.get("project_name", "") or "_no_project"
                from core.researcher import get_findings_dir, generate_findings_name
                from core.opus_planner import get_plans_dir, generate_plan_name
                # Reuse paths already set at tool-injection time (lines 1843-1844)
                # to avoid UUID mismatch between tool write path and session tracking.
                _pending_f = cl.user_session.get("_pending_findings_path", "")
                _pending_p = cl.user_session.get("_pending_plan_path", "")
                _findings_dir = str(Path(_pending_f).parent) if _pending_f else str(get_findings_dir(_proj))
                _findings_name = Path(_pending_f).name if _pending_f else generate_findings_name(input_content[:200])
                _plans_dir = str(Path(_pending_p).parent) if _pending_p else str(get_plans_dir(_proj))
                _plan_name = Path(_pending_p).name if _pending_p else generate_plan_name(input_content[:200])
    
                # Always inject sub-agent tools into the pool (they may be
                # filtered to by include_only_tools for structural enforcement)
                if _needs_research_raw:
                    from core.research_agent import create_research_agent_tool
                    _research_tool = create_research_agent_tool(
                        all_tools=all_tools,
                        findings_dir=_findings_dir,
                        findings_name=_findings_name,
                        cost_tracker=_turn_cost_tracker,
                        step_callback=None,
                    )
                    _research_tool._step_callback_factory = lambda: _make_nested_step_callback(
                        cl.user_session.get("_current_tool_step_id")
                    )
                    _research_tool._pel_ref = _pel
                    all_tools.append(_research_tool)
                    print(f"[PHASED_EXEC] Injected run_research_agent (read-only sub-agent)")
    
                if _needs_planning_raw:
                    from core.plan_agent import create_plan_agent_tool
                    _plan_tool = create_plan_agent_tool(
                        all_tools=all_tools,
                        plans_dir=_plans_dir,
                        plan_name=_plan_name,
                        cost_tracker=_turn_cost_tracker,
                        step_callback=None,
                    )
                    _plan_tool._step_callback_factory = lambda: _make_nested_step_callback(
                        cl.user_session.get("_current_tool_step_id")
                    )
                    _plan_tool._pel_ref = _pel
                    all_tools.append(_plan_tool)
                    print(f"[PHASED_EXEC] Injected run_plan_agent (read-only sub-agent)")
    
                # Same-context phase execution: main agent stays in control with
                # tool restrictions per phase. PhaseConfig handles dynamic filtering
                # inside NativeAgentExecutor (tools change as phases advance).
                from core.phase_config import PhaseConfig
                _phase_config = PhaseConfig(
                    needs_research=_needs_research_raw,
                    needs_planning=_needs_planning_raw,
                )
                _current_phase = _phase_config.initial_phase
                _phase_include_only = None  # PhaseConfig handles tool filtering
                print(f"[PHASED_EXEC] Phase: {_current_phase.upper()} — same-context tool restriction via PhaseConfig")
    
                # Inject read-only wrappers for research/plan phases
                from core.readonly_shell import create_readonly_shell_tool, create_readonly_batch_tool
                _shell_tool = next((t for t in all_tools if t.name == "execute_dynamic_task"), None)
                if _shell_tool:
                    _readonly_shell = create_readonly_shell_tool(_shell_tool)
                    all_tools.append(_readonly_shell)
                    print(f"[PHASED_EXEC] Injected execute_shell_readonly for phase tool restriction")
                _has_batch_readonly = any(t.name == "batch_readonly" for t in all_tools)
                if not _has_batch_readonly:
                    _batch_tool = next((t for t in all_tools if t.name == "batch"), None)
                    if _batch_tool:
                        _readonly_batch = create_readonly_batch_tool(_batch_tool)
                        all_tools.append(_readonly_batch)
                        print(f"[PHASED_EXEC] Injected batch_readonly wrapper (MCP version not found)")
                else:
                    print(f"[PHASED_EXEC] batch_readonly already provided by MCP server")
    
                print(f"[PHASED_EXEC] Phases: {_phases_active} → Execute "
                      f"(research={_needs_research_raw}, planning={_needs_planning_raw})")
    
                _log_path_str = cl.user_session.get("session_log_path", "")
                if _log_path_str:
                    append_phase_marker(
                        Path(_log_path_str), _current_phase, "started",
                        metadata={"phases_planned": [p.lower() for p in _phases_active] + ["execute"]},
                    )
            except Exception as e:
                print(f"[PHASED_EXEC] Warning: Could not set up phased execution: {e}")
                _current_phase = "execute"
                _phase_include_only = None
                _phase_config = None
        else:
            # Remove read-only tool variants — they only exist for research/plan phases
            # and confuse the model during direct execution (it picks batch_readonly over batch).
            from core.phase_config import EXECUTE_EXCLUDED_TOOLS
            _pre_filter_count = len(all_tools)
            all_tools = [t for t in all_tools if t.name not in EXECUTE_EXCLUDED_TOOLS]
            _removed = _pre_filter_count - len(all_tools)
            if _removed:
                print(f"[PHASED_EXEC] Direct execution — removed {_removed} read-only tools: {sorted(EXECUTE_EXCLUDED_TOOLS)}")
            else:
                print(f"[PHASED_EXEC] Direct execution (no research/planning needed)")
    
        # Store current phase for post-execution checks and auto-advance
        cl.user_session.set("_current_phase", _current_phase)
        cl.user_session.set("_needs_research_raw", _needs_research_raw)
        cl.user_session.set("_needs_planning_raw", _needs_planning_raw)
    
        # Pipeline exclusions (for non-phase use cases like pipeline mode)
        # When _phase_config is active, it handles tool filtering inside the executor.
        _effective_include_only = _phase_include_only if _phase_include_only else _pipeline_include_only
    
        # Inject edit_plan for active plan BEFORE executor creation (executor snapshots tools at construction)
        _active_plan_path_for_edit = cl.user_session.get("active_plan_path", "")
        if _active_plan_path_for_edit and cl.user_session.get("_plan_execution_pending", False):
            from core.opus_planner import create_edit_plan_tool
            all_tools.append(create_edit_plan_tool(_active_plan_path_for_edit))
            print(f"[PLAN_TOOL] Injected edit_plan for: {_active_plan_path_for_edit}")
    
        executor, filtered_tool_count, model_display_name = create_skill_based_agent(
            llm=llm,
            all_tools=all_tools,
            skill_names=skill_names,
            skill_loader=skill_loader,
            dev_llm=dev_llm,
            user_context=user_context,
            complexity=task_complexity,
            websearch_enabled=websearch_enabled,
            exclude_tools=_phase_exclude or None,
            include_only_tools=_effective_include_only,
            phase_config=_phase_config,
            use_opus=_use_opus_for_executor,
            include_memory_check=_include_memory_check,
            model_override=_model_override,
            active_plan_path=cl.user_session.get("active_plan_path", ""),
            pel=_pel,
        )

        # Inject Chainlit step callback for native executor UI rendering
        if hasattr(executor, 'step_callback'):
            executor.step_callback = _chainlit_step_callback
    
        # Live elapsed-time ticker — updates status_msg every second during execution
        import time as _time_mod
        _exec_start = _time_mod.time()
        _ticker_steps = [0]  # mutable counter updated by iteration_callback
    
        _model_label = model_display_name.capitalize()
    
        async def _status_ticker():
            try:
                while True:
                    await asyncio.sleep(1)
                    elapsed = int(_time_mod.time() - _exec_start)
                    n = _ticker_steps[0]
                    if n:
                        _content = f"💭 {_model_label} thinking ({elapsed}s · {n} tool call{'s' if n != 1 else ''})"
                    else:
                        _content = f"💭 {_model_label} thinking ({elapsed}s)"
                    _active_tool = cl.user_session.get("_current_tool_name")
                    if _active_tool:
                        _t_emoji = _TOOL_EMOJIS.get(_active_tool, "⚙️")
                        _content += f"\n{_t_emoji} {_active_tool}"
                    status_msg.content = _content
                    await status_msg.update()
            except asyncio.CancelledError:
                pass
    
        _ticker_task = asyncio.create_task(_status_ticker())
    
        # Iteration callback updates step count (ticker reads it)
        if hasattr(executor, 'iteration_callback'):
            async def _iteration_cb(iteration: int, steps_so_far: int):
                _ticker_steps[0] = steps_so_far
            executor.iteration_callback = _iteration_cb
    
        # Inject per-turn cost tracker + nested step callback factory into worker tool.
        # The factory reads parent_step_id from session at call time (set by
        # _chainlit_step_callback before _execute_tool runs the worker's _arun).
        if _worker_agent_tool is not None:
            _worker_agent_tool._cost_tracker_ref = _turn_cost_tracker
    
            def _get_worker_step_callback():
                parent_id = cl.user_session.get("_current_tool_step_id")
                return _make_nested_step_callback(parent_id)
    
            _worker_agent_tool._chainlit_callback_ref = _get_worker_step_callback
    
        # ── Step 4: Build agent input using pre-computed context ───────────
        # Only include conversation_summary (older messages) in agent_input.
        # Recent messages are already in {chat_history} — including them here
        # too would duplicate them in every LLM call (paying for them twice).
        if conversation_summary:
            agent_input = f"{conversation_summary}\n\nCurrent request: {input_content}"
        else:
            agent_input = input_content
    
        _raw_user_request = input_content
    
        if _project_just_switched:
            agent_input = (
                f"[System: Project was just switched to '{_project_just_switched}' "
                f"via user selection. Acknowledge briefly and handle any follow-up in the message.]\n\n"
                f"{agent_input}"
            )
    
        # ── Resume Intent Detection ──────────────────────────────────────────
        # When user asks to resume/continue, proactively inject last_turn + project
        # state from disk. Don't rely on the agent to call read_memory — inject it directly.
        _is_resume_request = False
        _input_lower = input_content.lower().strip()
        if len(_input_lower) < 150:
            _resume_triggers = (
                "resume", "continue", "where did we leave", "where were we",
                "pick up where", "last session", "what were we doing",
                "what happened last", "where did we stop", "left off",
                "what was i doing", "what did we do",
            )
            _is_resume_request = any(t in _input_lower for t in _resume_triggers)
    
        if _is_resume_request:
            from core.memory_state import get_memory_root
            import json as _json
            _resume_context_parts = []
    
            # Layer 1: last_turn.json (most recent finding from previous session)
            _last_turn_path = get_memory_root() / "session" / "last_turn.json"
            if _last_turn_path.exists():
                try:
                    _lt = _json.loads(_last_turn_path.read_text(encoding="utf-8"))
                    _lt_summary = (
                        f"LAST TURN (from previous session, {_lt.get('timestamp', 'unknown')}):\n"
                        f"  Project: {_lt.get('project', 'unknown')}\n"
                        f"  User asked: {_lt.get('user', '')}\n"
                        f"  Assistant found: {_lt.get('assistant', '')}\n"
                        f"  Tools used: {', '.join(_lt.get('tool_calls_this_turn', []))}\n"
                        f"  Running jobs: {[j.get('job_id', '') for j in _lt.get('running_jobs', [])]}"
                    )
                    _resume_context_parts.append(_lt_summary)
                except Exception:
                    pass
    
            # Layer 2: Project state (status + knowledge — full cross-session context)
            if project_name and work_dir:
                from core.memory_state import get_state_for_prompt
                _project_state = get_state_for_prompt(work_dir, project_name)
                if _project_state:
                    _resume_context_parts.append(
                        f"PROJECT STATE (paths, constraints, current status):\n{_project_state}"
                    )
    
            if _resume_context_parts:
                _resume_block = "\n\n".join(_resume_context_parts)
                agent_input = (
                    f"═══ RESUME CONTEXT (proactively loaded from previous session) ═══\n"
                    f"{_resume_block}\n"
                    f"═══ END RESUME CONTEXT ═══\n\n"
                    f"IMPORTANT: Review the resume context above carefully before taking action. "
                    f"The LAST TURN shows what was just discovered/happening. Do NOT repeat work "
                    f"that was already done. If you need more detail, call read_memory(project=...).\n\n{agent_input}"
                )
                print(f"[RESUME] Injected {len(_resume_block)} chars of resume context into agent input")
    
        # Inject project state when no research phase loaded it.
        # The researcher calls get_state_for_prompt() internally; when it doesn't run,
        # the executor needs it injected directly so it has project context.
        # NOTE: This MUST also run for resume requests — resume context (last_turn.json)
        # only has a truncated snapshot, not the full project paths/constraints.
        if _include_memory_check and project_name and work_dir:
            from core.memory_state import get_state_for_prompt
            _project_state_for_executor = get_state_for_prompt(work_dir, project_name)
            if _project_state_for_executor:
                agent_input = f"{_project_state_for_executor}\n\n{agent_input}"
    
        # Inject SLURM REQUIRED notice when Haiku classified task as needing HPC
        if _needs_slurm:
            _slurm_notice = (
                "SLURM REQUIRED: This task involves heavy/long-running compute. "
                "You MUST use submit_slurm_job (or iris.submit_slurm in pipelines) "
                "for the computationally intensive parts. DO NOT run them via "
                "execute_dynamic_task or iris.run_shell — those have a hard 5-minute kill. "
                "Quick setup commands (mkdir, pip install, checking paths) are fine directly."
            )
            agent_input = f"{_slurm_notice}\n\n{agent_input}"
    
        # If history skill is active and a project was detected, provide paths
        # so the resume strategy can read PROJECT_STATE.md directly
        if "history" in skill_names and project_name and work_dir:
            from core.memory_state import get_project_dir
            _project_state_path = str(get_project_dir(project_name) / "status.md")
            agent_input += (
                f"\n\n[PROJECT CONTEXT: project_name='{project_name}', "
                f"work_dir='{work_dir}', "
                f"project_state_path='{_project_state_path}', "
                f"plans_dir='{work_dir}/plans/']"
            )
    
        # Build agent callbacks list
        agent_callbacks = [cl.LangchainCallbackHandler()]
        # Only add LangChain cost callback for non-native executor (native tracks via _turn_cost_tracker)
        _is_native_executor = hasattr(executor, 'cost_tracker')
        if cost_tracker and not _is_native_executor:
            agent_callbacks.append(cost_tracker)
    
        # Thinking display: show agent reasoning + tool usage in status message
        from core.thinking_callback import ThinkingDisplayCallback
        _thinking_cb = ThinkingDisplayCallback(status_msg)
        agent_callbacks.append(_thinking_cb)
    
        # ── Step 4: Phase context + task reframing ───────────────────────────
        # Phased execution is now enforced structurally via sub-agents (research/plan
        # sub-agents have read-only tool sets). PEL phase gates are no longer needed.
        # This section handles: loading prior findings/plan from disk, and injecting
        # existing plan for execution resumption.
        from core.researcher import get_findings_for_prompt
        from core.opus_planner import get_plan_for_prompt, read_plan_file
    
        _active_plan_path = cl.user_session.get("active_plan_path", "")
        _all_findings_paths = cl.user_session.get("active_findings_paths") or []
    
        # Clear any legacy PEL phase gates (they're no longer the enforcement mechanism)
        _pel.clear_phase_gates()
    
        # ── Inject existing plan from disk (memory-based resumption) ──
        if _active_plan_path:
            try:
                _disk_plan = read_plan_file(_active_plan_path) or ""
                if _disk_plan:
                    agent_input = (
                        f"{agent_input}\n\n"
                        f"═══ ACTIVE PLAN at {_active_plan_path} (follow this, update checkboxes) ═══\n"
                        f"{_disk_plan}\n"
                        f"═══ END PLAN ═══\n"
                        f"Execute steps in order. Use edit_plan to mark completed: - [ ] → - [x]. "
                        f"Use run_worker_agent for PARALLEL_GROUP steps.\n"
                    )
                    cl.user_session.set("_active_plan_text", _disk_plan)
            except Exception:
                pass
    
        # ── Step 5: Task reframing (phase-aware) ────────────────────────────
        # When in a phased execution, give the executor clear guidance on its role.
        # Tools are structurally restricted by PhaseConfig — this reinforces behavior.
        if _current_phase == "research":
            _findings_context = ""
            if _phase_revision_pending and _revision_target == "research" and _all_findings_paths:
                _existing_findings_text = get_findings_for_prompt(_all_findings_paths)
                agent_input = (
                    f"USER REVISION REQUEST:\n───\n{_raw_user_request}\n───\n\n"
                    f"YOUR ROLE: REVISE the research findings below based on the user's feedback.\n\n"
                    f"EXISTING FINDINGS:\n═══\n{_existing_findings_text.strip() if _existing_findings_text else '(none)'}\n═══\n\n"
                    f"Read the existing findings, apply the user's requested changes, then call "
                    f"write_findings() with the complete updated findings.\n"
                )
                print(f"[TASK_REFRAME] Phase: research REVISION — editing existing findings")
            else:
                agent_input = (
                    f"USER REQUEST:\n───\n{_raw_user_request}\n───\n\n"
                    f"YOUR ROLE: You are in the RESEARCH phase. Explore the environment to gather "
                    f"all facts needed to plan and implement the user's request.\n\n"
                    f"You have read-only tools available. Use them to understand:\n"
                    f"- What files/code exist relevant to this task\n"
                    f"- Current state of the system (configs, paths, versions)\n"
                    f"- Constraints and dependencies\n\n"
                    f"When you have enough information, call write_findings() with your complete "
                    f"discoveries. Report FACTS ONLY — no plans, no implementation suggestions.\n\n"
                    f"If you need clarification from the user before researching, ask them."
                )
                print(f"[TASK_REFRAME] Phase: research — direct tool access")
        elif _current_phase == "plan":
            _findings_context = ""
            if _all_findings_paths:
                _findings_block = get_findings_for_prompt(_all_findings_paths)
                if _findings_block:
                    _findings_context = f"\nRESEARCH FINDINGS:\n═══\n{_findings_block.strip()}\n═══\n"

            if _phase_revision_pending and _revision_target == "plan":
                _existing_plan_text = ""
                _active_plan_path = cl.user_session.get("active_plan_path", "")
                if _active_plan_path:
                    _existing_plan_text = read_plan_file(_active_plan_path) or ""
                agent_input = (
                    f"USER REVISION REQUEST:\n───\n{_raw_user_request}\n───\n\n"
                    f"{_findings_context}"
                    f"EXISTING PLAN:\n═══\n{_existing_plan_text.strip() if _existing_plan_text else '(none)'}\n═══\n\n"
                    f"YOUR ROLE: REVISE the plan above based on the user's feedback.\n"
                    f"Apply the requested changes, then call write_plan() with the complete updated plan.\n"
                )
                print(f"[TASK_REFRAME] Phase: plan REVISION — editing existing plan")
            else:
                agent_input = (
                    f"USER REQUEST:\n───\n{_raw_user_request}\n───\n\n"
                    f"{_findings_context}"
                    f"YOUR ROLE: You are in the PLANNING phase. Create a detailed, step-by-step "
                    f"implementation plan based on the findings above and your project knowledge.\n\n"
                    f"Your plan must be concrete: exact file paths, exact commands, exact changes.\n"
                    f"Do NOT implement anything — produce the plan only.\n"
                    f"When complete, call write_plan() with the full plan.\n\n"
                    f"If you need to verify a detail, use your read-only tools. "
                    f"If you need clarification from the user, ask them."
                )
                print(f"[TASK_REFRAME] Phase: plan — direct tool access")
        else:
            # Direct execution or plan resumption
            from core.plan_verification import plan_has_unchecked_steps, count_plan_progress
            _plan_text_for_exec = cl.user_session.get("_active_plan_text", "")
            if _plan_text_for_exec and plan_has_unchecked_steps(_plan_text_for_exec):
                _done, _total = count_plan_progress(_plan_text_for_exec)
                _exec_parts = [
                    f"YOUR TASK: Execute the plan below to completion. "
                    f"You MUST complete ALL remaining steps ({_total - _done} of {_total} remaining).\n"
                ]
                if _all_findings_paths:
                    _findings_block = get_findings_for_prompt(_all_findings_paths)
                    if _findings_block:
                        _exec_parts.append(_findings_block.strip())
                        _exec_parts.append("")
                _exec_parts.append(
                    f"═══ ACTIVE PLAN at {_active_plan_path} (execute all unchecked steps) ═══\n"
                    f"{_plan_text_for_exec}\n"
                    f"═══ END PLAN ═══\n"
                )
                _exec_parts.append(
                    f"USER REQUEST:\n───\n{_raw_user_request}\n───\n\n"
                    f"INSTRUCTIONS:\n"
                    f"- Execute each unchecked step (- [ ]) in order\n"
                    f"- After completing a step, update the plan file: mark - [ ] → - [x]\n"
                    f"- Do NOT stop until ALL steps are marked [x]\n"
                    f"- If a step fails, note the error and continue to the next step\n"
                    f"- Use run_worker_agent for PARALLEL_GROUP steps\n"
                )
                agent_input = "\n".join(_exec_parts)
                print(f"[TASK_REFRAME] Execution phase — {_total - _done}/{_total} plan steps remaining")
            else:
                if _all_findings_paths:
                    _findings_block = get_findings_for_prompt(_all_findings_paths)
                    if _findings_block:
                        agent_input = agent_input + _findings_block
                print(f"[TASK_REFRAME] Direct execution — no phasing needed")
    
        # ── Step 6: Execute agent (main path — Opus/Sonnet with tools) ────
        status_msg.content = f"🚀 Engines ignited! {model_display_name.capitalize()} is on the case with {filtered_tool_count} tools ({skills_display})..."
        await status_msg.update()
    
        from core.stuck_detection_callback import StuckInterrupt
        from core.single_agent import SkillEscalationInterrupt
        _escalation_interrupt_skill = None
    
        try:
            result = await asyncio.wait_for(
                executor.ainvoke(
                    {
                        "input": agent_input,
                        "chat_history": recent_history,
                        "agent_scratchpad": [],
                    },
                    config={
                        "callbacks": agent_callbacks,
                        "handle_parsing_errors": True,
                    }
                ),
                timeout=EXECUTOR_SESSION_TIMEOUT,
            )
        except SkillEscalationInterrupt as _esc:
            _ticker_task.cancel()
            await _thinking_cb.cleanup()
            _escalation_interrupt_skill = _esc.skill_name
            print(f"[SKILL_ESCALATION] Immediate interrupt: '{_esc.skill_name}' requested")
            _pre_escalation_steps = getattr(_esc, 'intermediate_steps', [])
            result = {"output": "", "intermediate_steps": _pre_escalation_steps}
            # Close orphaned UI step (the native executor already closed it via
            # step_callback("end") before re-raising, but clear session state as backup)
            _orphan_step = cl.user_session.get("_current_tool_step_obj")
            if _orphan_step:
                try:
                    if not getattr(_orphan_step, 'end', None):
                        _orphan_step.output = f"Escalating to skill: {_esc.skill_name}"
                        _orphan_step.end = datetime.datetime.now(datetime.timezone.utc).isoformat()
                        await _orphan_step.update()
                except Exception:
                    pass
                cl.user_session.set("_current_tool_step_obj", None)
                cl.user_session.set("_current_tool_step_id", None)
                cl.user_session.set("_current_tool_name", None)
        except asyncio.TimeoutError:
            _ticker_task.cancel()
            await _thinking_cb.cleanup()
            _tools_parent_step = cl.user_session.get("_tools_parent_step")
            if _tools_parent_step:
                try:
                    _tools_parent_step.name = "🔧 Timed out"
                    _tools_parent_step.end = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    await _tools_parent_step.update()
                except Exception:
                    pass
            logger.error(f"[SESSION_TIMEOUT] Executor exceeded {EXECUTOR_SESSION_TIMEOUT}s")
            timeout_msg = (
                f"⏱️ This operation exceeded the {EXECUTOR_SESSION_TIMEOUT // 60}-minute session time limit. "
                f"Please try breaking the task into smaller steps."
            )
            await cl.Message(content=timeout_msg).send()
            return
        except StuckInterrupt as _stuck:
            _ticker_task.cancel()
            await _thinking_cb.cleanup()
            # Close parent tool step so the UI spinner stops
            _tools_parent_step = cl.user_session.get("_tools_parent_step")
            if _tools_parent_step:
                try:
                    _tools_parent_step.name = "🔧 Interrupted"
                    _tools_parent_step.end = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    await _tools_parent_step.update()
                except Exception:
                    pass
            if status_msg:
                try:
                    status_msg.content = "⚠️ Agent stuck — requesting help"
                    await status_msg.update()
                except Exception:
                    pass
            await _handle_stuck_mid_loop(_stuck, agent_input, recent_history, executor, agent_callbacks)
            return
        except asyncio.CancelledError:
            _ticker_task.cancel()
            await _thinking_cb.cleanup()
            _tools_parent_step = cl.user_session.get("_tools_parent_step")
            if _tools_parent_step:
                try:
                    _tools_parent_step.name = "🔧 Stopped"
                    _tools_parent_step.end = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    await _tools_parent_step.update()
                except Exception:
                    pass
            if status_msg:
                try:
                    status_msg.content = "⏹️ Stopped by user"
                    await status_msg.update()
                except Exception:
                    pass
            return
        except Exception as _exec_err:
            _ticker_task.cancel()
            await _thinking_cb.cleanup()
            _tools_parent_step = cl.user_session.get("_tools_parent_step")
            if _tools_parent_step:
                try:
                    _tools_parent_step.name = "🔧 Error"
                    _tools_parent_step.end = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    await _tools_parent_step.update()
                except Exception:
                    pass
            raise _exec_err
    
        # Stop ticker
        _ticker_task.cancel()
    
        # Merge executor cost into turn tracker and store on session for unified display
        if hasattr(executor, 'cost_tracker'):
            _turn_cost_tracker.merge(executor.cost_tracker)
        elif cost_tracker and cost_tracker.num_calls > 0:
            # LangChain fallback path: feed callback data into native tracker
            from core.llm_provider import Usage
            _turn_cost_tracker.accumulate(
                Usage(
                    input_tokens=cost_tracker.total_input_tokens,
                    output_tokens=cost_tracker.total_output_tokens,
                    cache_creation_tokens=0,
                    cache_read_tokens=0,
                ),
                model="langchain-fallback",
            )
            # Override cost with LangChain's own calculation (more accurate for its path)
            _turn_cost_tracker.total_cost += cost_tracker.total_cost - _turn_cost_tracker._call_details[-1]["cost"]
            _turn_cost_tracker._call_details[-1]["cost"] = cost_tracker.total_cost
        cl.user_session.set("_last_native_cost_tracker", _turn_cost_tracker)
    
        # NOTE: WEBSEARCH_NEEDED marker handling removed — web search tools are now
        # always available with tool-level approval gate. The agent can directly
        # call web_search() and the tool will ask the user to enable if needed.
    
    
        # ── Step 5a: HYBRID CONTEXT — compact records in history ─────────────
        # Tool call records (~300-500 chars/tool) go into history via
        # format_tool_call_record(). Full outputs are in agent_scratchpad
        # during the turn and on disk via file-reference. Compaction fires
        # at the START of the next turn when approaching the token budget.
        _pending_compression_steps = None  # Kept for compatibility with downstream code
    
        # ── Step 5b: STUCK DETECTION — now handled MID-LOOP via StuckDetectionCallback ──
        # StuckDetectionCallback (attached to AgentExecutor in create_skill_based_agent)
        # raises StuckInterrupt after the 3rd repeated error, BEFORE ainvoke() returns.
        # The try/except StuckInterrupt block above (wrapping ainvoke) catches it and
        # calls _handle_stuck_mid_loop() — which shows the same approval dialog as
        # normal web_search. This replaces the old post-loop check that fired too late.
        # (No code needed here — handled above.)
    
        # ── Step 5c: DYNAMIC SKILL ESCALATION (same-turn re-creation) ─────
        # If the agent called request_additional_skill, we detect it here,
        # merge the requested skill(s) into the current skill set, re-create
        # a brand new AgentExecutor with the expanded tools + guardrails,
        # and re-invoke it. This solves the core problem: LangChain's
        # AgentExecutor binds tools at construction time, so you can't
        # just append tools — you must create a new executor.
        #
        # Primary path: SkillEscalationInterrupt raised by the tool (immediate).
        # Fallback path: detect_escalation_in_result scans intermediate_steps
        # (handles edge cases where the exception didn't propagate).
    
        # Pre-escalation: register findings/plan from initial executor before
        # escalation can overwrite `result`. Also save phase pause state.
        _initial_called_tools = set()
        for step in result.get("intermediate_steps", []):
            _action = step[0] if step else None
            if _action:
                _initial_called_tools.add(getattr(_action, "tool", ""))
    
        if "write_findings" in _initial_called_tools:
            _pending_findings = cl.user_session.get("_pending_findings_path", "")
            if _pending_findings and Path(_pending_findings).is_file():
                _existing = cl.user_session.get("active_findings_paths") or []
                if _pending_findings not in _existing:
                    _existing.append(_pending_findings)
                cl.user_session.set("active_findings_paths", _existing)
                cl.user_session.set("active_findings_path", _pending_findings)
                print(f"[PRE_ESCALATION] Registered write_findings → {_pending_findings}")
    
        if "write_plan" in _initial_called_tools:
            _pending_plan = cl.user_session.get("_pending_plan_path", "")
            if _pending_plan and Path(_pending_plan).is_file():
                cl.user_session.set("active_plan_path", _pending_plan)
                cl.user_session.set("_plan_execution_pending", True)
                print(f"[PRE_ESCALATION] Registered write_plan → {_pending_plan}")
    
        _original_phase_paused = result.get("_phase_paused", False)
        _original_paused_at_phase = result.get("_paused_at_phase", "execute")
    
        if _escalation_interrupt_skill:
            escalation_info = {
                "escalation_detected": True,
                "requested_skills": [_escalation_interrupt_skill],
            }
        else:
            escalation_info = detect_escalation_in_result(result)
        escalation_iteration = 0
    
        while (escalation_info["escalation_detected"]
               and escalation_iteration < MAX_ESCALATION_ITERATIONS):
            escalation_iteration += 1
            new_skills = escalation_info["requested_skills"]
    
            # Filter to only valid skills that aren't already active
            valid_new_skills = [
                s for s in new_skills
                if s in skill_loader.list_skill_names() and s not in skill_names
            ]
    
            if not valid_new_skills:
                print(f"[SKILL_ESCALATION] Escalation requested but no new valid skills "
                      f"to add (requested: {new_skills}, current: {skill_names})")
                break
    
            # Merge new skills into the active set
            skill_names = skill_names + valid_new_skills
            print(f"[SKILL_ESCALATION] Iteration {escalation_iteration}: "
                  f"Merging skills {valid_new_skills} -> active skills now: {skill_names}")
    
            # Notify the user that we're loading additional capabilities
            await cl.Message(
                f"Loading additional skill(s): {', '.join(valid_new_skills)}"
            ).send()
    
            # Store escalated skills in session for next-turn fallback
            prev_escalated = cl.user_session.get("escalated_skills", [])
            cl.user_session.set(
                "escalated_skills",
                list(set(prev_escalated + valid_new_skills))
            )
    
            # Re-build user context with expanded skill set
            user_context = build_memory_context_block(
                username=username,
                work_dir=work_dir,
                project_name=project_name,
                skill_names=skill_names,
                task_keywords=[],
                environment_index=_pel.get_environment_index(),
                weights_path=cl.user_session.get("weights_path", ""),
            )
    
            # Re-create a BRAND NEW executor with the expanded skill set.
            # This is the key fix: LangChain's create_tool_calling_agent()
            # bakes tools into the prompt at construction time. You cannot
            # append tools to an existing executor — the agent's internal
            # routing won't know about them. A new executor is required.
            executor, filtered_tool_count, model_display_name = create_skill_based_agent(
                llm=llm,
                all_tools=all_tools,
                skill_names=skill_names,
                skill_loader=skill_loader,
                dev_llm=dev_llm,
                user_context=user_context,
                complexity=task_complexity,
                websearch_enabled=websearch_enabled,
                exclude_tools=_pipeline_exclude,
                include_only_tools=_pipeline_include_only,
                use_opus=_use_opus_for_executor,
                include_memory_check=_include_memory_check,
                model_override=_model_override,
                pel=_pel,
            )

            if hasattr(executor, 'step_callback'):
                executor.step_callback = _chainlit_step_callback
            if hasattr(executor, 'iteration_callback'):
                executor.iteration_callback = _iteration_cb
    
            print(f"[SKILL_ESCALATION] Re-created executor with skills {skill_names}")
    
            # Update worker agent tools with expanded skill set
            if _worker_agent_tool is not None:
                _worker_tools = filter_tools_for_skills(all_tools, skill_names, skill_loader)
                _worker_agent_tool.worker_tools = _worker_tools
                print(f"[SKILL_ESCALATION] Worker agent tools updated: {len(_worker_tools)} tools")
    
            # Build rich handoff context from previous executor's work.
            # This preserves ALL tool calls and results so the new agent
            # knows exactly what was done and can continue without repeating.
            previous_skills = [s for s in skill_names if s not in valid_new_skills]
            raw_handoff = await async_format_intermediate_steps_for_handoff(
                result=result,
                user_input=agent_input,
                previous_skills=previous_skills,
                new_skills=valid_new_skills,
            )
    
            # Pass full handoff directly — no summarization. The handoff is ephemeral
            # (not stored in history) and even 50K chars is only ~16K tokens, well
            # within the 200K API limit. Summarization loses critical details.
            print(f"[SKILL_ESCALATION] Raw handoff: {len(raw_handoff)} chars (~{len(raw_handoff)//3} tokens)")
            escalation_context = raw_handoff
    
            # Frame the handoff as a colleague briefing — the narrative style
            # naturally conveys "this is done, continue from here" without rigid
            # rules that might prevent legitimate tool reuse.
            _prev_skills_str = ", ".join(previous_skills)
            _new_skills_str = ", ".join(valid_new_skills)
    
            escalation_directive = (
                f"--- HANDOFF FROM PREVIOUS AGENT ---\n"
                f"You are picking up mid-task. A colleague with [{_prev_skills_str}] "
                f"skills worked on this request and got as far as they could. They "
                f"called for your [{_new_skills_str}] skills to finish the job.\n\n"
                f"Read their briefing below carefully — it contains everything they "
                f"did, what they found, and what they need you to do next. The "
                f"briefing includes exact file paths, values, and conclusions that "
                f"you should use directly.\n\n"
                f"Your job: pick up where they left off using your new capabilities. "
                f"Their conclusions and edits are solid — build on them rather than "
                f"re-investigating from scratch. If something in the briefing already "
                f"answers a question, use that answer.\n\n"
                f"{escalation_context}\n"
                f"--- END HANDOFF ---"
            )
            escalated_input = f"{agent_input}\n\n{escalation_directive}"
    
            try:
                result = await asyncio.wait_for(
                    executor.ainvoke(
                        {
                            "input": escalated_input,
                            "chat_history": recent_history,
                            "agent_scratchpad": [],
                        },
                        config={
                            "callbacks": agent_callbacks,
                            "handle_parsing_errors": True,
                        }
                    ),
                    timeout=EXECUTOR_SESSION_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.error(f"[SESSION_TIMEOUT] Escalated executor exceeded {EXECUTOR_SESSION_TIMEOUT}s")
                result = {"output": "Operation timed out during escalation.",
                          "intermediate_steps": _pre_escalation_steps}
                break
            except SkillEscalationInterrupt as _esc2:
                # Escalated executor requested yet another skill
                print(f"[SKILL_ESCALATION] Chained escalation: '{_esc2.skill_name}'")
                result = {"output": "", "intermediate_steps": _pre_escalation_steps}
                escalation_info = {
                    "escalation_detected": True,
                    "requested_skills": [_esc2.skill_name],
                }
                continue
    
            # Escalated tool results stay in agent_scratchpad for the current turn.
            # Compact records go into history; full outputs on disk via file-reference.
    
            # Check if the new result also requests escalation (loop continues)
            escalation_info = detect_escalation_in_result(result)
            if escalation_info["escalation_detected"]:
                print(f"[SKILL_ESCALATION] Agent requested further escalation: "
                      f"{escalation_info['requested_skills']}")
    
        if escalation_iteration > 0:
            print(f"[SKILL_ESCALATION] Completed {escalation_iteration} escalation iteration(s). "
                  f"Final skill set: {skill_names}")
            # The escalated executor already compressed its steps inline (inside
            # the while loop).  Clear the pending pointer so the background
            # compressor does NOT overwrite the new executor's steps with the
            # old executor's stale steps.
            _pending_compression_steps = None
    
            # Escalation enrichment: if the original executor paused after writing
            # findings/plan, append escalation output to the artifact so it becomes
            # the complete ground truth. Then restore phase pause state.
            if _original_phase_paused and result.get("output"):
                _artifact_path = ""
                if "write_findings" in _initial_called_tools:
                    _artifact_path = cl.user_session.get("_pending_findings_path", "")
                elif "write_plan" in _initial_called_tools:
                    _artifact_path = cl.user_session.get("_pending_plan_path", "")
    
                if _artifact_path and Path(_artifact_path).is_file():
                    existing = Path(_artifact_path).read_text(encoding="utf-8")
                    supplement = result["output"].strip()
                    if supplement and supplement not in existing:
                        updated = existing + f"\n\n## Additional Research (Escalation)\n\n{supplement}\n"
                        Path(_artifact_path).write_text(updated, encoding="utf-8")
                        print(f"[ESCALATION_ENRICH] Updated {Path(_artifact_path).name} "
                              f"with escalation output ({len(supplement)} chars)")
    
                # Restore phase pause so downstream code processes it correctly
                result["_phase_paused"] = True
                result["_paused_at_phase"] = _original_paused_at_phase
                # Present complete artifact to user (not just escalation fragment)
                if _artifact_path and Path(_artifact_path).is_file():
                    result["output"] = Path(_artifact_path).read_text(encoding="utf-8")
    
        out = result["output"]
    
        # ── Phase pause: executor stopped for user approval after phase transition ──
        _phase_paused = result.get("_phase_paused", False)
        if _phase_paused:
            _paused_at = result.get("_paused_at_phase", "execute")
            cl.user_session.set("_current_phase", _paused_at)
            print(f"[PHASE_PAUSE] Executor paused — now in '{_paused_at}' phase, awaiting user approval")
            _log_path_str = cl.user_session.get("session_log_path", "")
            if _log_path_str:
                _completed_phase = "plan" if "write_plan" in _initial_called_tools else "research"
                _artifact = cl.user_session.get("_pending_plan_path", "") if "write_plan" in _initial_called_tools else cl.user_session.get("_pending_findings_path", "")
                append_phase_marker(
                    Path(_log_path_str), _completed_phase, "completed",
                    metadata={"artifact": _artifact} if _artifact else None,
                )
                append_phase_marker(
                    Path(_log_path_str), _paused_at, "awaiting_approval",
                    metadata={"next_phase": _paused_at, "artifact": _artifact} if _artifact else None,
                )
    
        # ── Post-execution: detect write_findings/write_plan via tool call + file existence ──
        _called_tools = set()
        for step in result.get("intermediate_steps", []):
            action = step[0] if step else None
            if action:
                _called_tools.add(getattr(action, "tool", ""))
    
        if "write_plan" in _called_tools:
            _pending_plan = cl.user_session.get("_pending_plan_path", "")
            if _pending_plan and Path(_pending_plan).is_file():
                cl.user_session.set("active_plan_path", _pending_plan)
                cl.user_session.set("_plan_execution_pending", True)
                cl.user_session.set("_unfulfilled_phase_gates", [])
                print(f"[POST_EXEC] Detected write_plan → {_pending_plan}")
                print(f"[POST_EXEC] Plan execution pending — next turn will execute")
                if not _phase_paused:
                    _log_path_str = cl.user_session.get("session_log_path", "")
                    if _log_path_str:
                        append_phase_marker(
                            Path(_log_path_str), "plan", "completed",
                            metadata={"artifact": _pending_plan},
                        )
    
        if "write_findings" in _called_tools:
            _pending_findings = cl.user_session.get("_pending_findings_path", "")
            if _pending_findings and Path(_pending_findings).is_file():
                _existing = cl.user_session.get("active_findings_paths") or []
                if _pending_findings not in _existing:
                    _existing.append(_pending_findings)
                cl.user_session.set("active_findings_paths", _existing)
                cl.user_session.set("active_findings_path", _pending_findings)
                print(f"[POST_EXEC] Detected write_findings → {_pending_findings}")
                if not _phase_paused:
                    _log_path_str = cl.user_session.get("session_log_path", "")
                    if _log_path_str:
                        append_phase_marker(
                            Path(_log_path_str), "research", "completed",
                            metadata={"artifact": _pending_findings},
                        )
    
        # Sub-agents write to disk internally but their tool calls don't appear in
        # the main executor's intermediate_steps. Detect via disk existence.
        if "run_research_agent" in _called_tools:
            _pending_findings = cl.user_session.get("_pending_findings_path", "")
            if _pending_findings and Path(_pending_findings).is_file():
                _existing = cl.user_session.get("active_findings_paths") or []
                if _pending_findings not in _existing:
                    _existing.append(_pending_findings)
                cl.user_session.set("active_findings_paths", _existing)
                cl.user_session.set("active_findings_path", _pending_findings)
                print(f"[POST_EXEC] Sub-agent wrote findings → {_pending_findings}")
    
        if "run_plan_agent" in _called_tools:
            _pending_plan = cl.user_session.get("_pending_plan_path", "")
            if _pending_plan and Path(_pending_plan).is_file():
                cl.user_session.set("active_plan_path", _pending_plan)
                cl.user_session.set("_plan_execution_pending", True)
                cl.user_session.set("_unfulfilled_phase_gates", [])
                print(f"[POST_EXEC] Sub-agent wrote plan → {_pending_plan}")
    
        # Phase gate persistence: track unfulfilled phases for next turn.
        # If a gated phase was active but the sub-agent didn't produce output,
        # carry the requirement forward so next turn re-enters the same phase.
        _unfulfilled = []
        _cur_phase = cl.user_session.get("_current_phase", "execute")
        if _cur_phase == "research":
            _pending_findings = cl.user_session.get("_pending_findings_path", "")
            if not (_pending_findings and Path(_pending_findings).is_file()):
                _unfulfilled.append("write_findings")
                print(f"[PHASE_GATE] Research phase unfulfilled — will retry next turn")
        elif _cur_phase == "plan":
            _pending_plan = cl.user_session.get("_pending_plan_path", "")
            if not (_pending_plan and Path(_pending_plan).is_file()):
                _unfulfilled.append("write_plan")
                print(f"[PHASE_GATE] Plan phase unfulfilled — will retry next turn")
        cl.user_session.set("_unfulfilled_phase_gates", _unfulfilled)
    
        # ── ESCALATION MARKER STRIPPING ──────────────────────────────────────
        # The request_additional_skill tool returns text containing the
        # ESCALATION_MARKER and boilerplate instructions. If the agent echoes
        # this in its final output (return_direct=True), strip it before the
        # user sees it. This is an internal mechanism only.
        if ESCALATION_MARKER in out:
            out = "\n".join(
                l for l in out.split("\n")
                if ESCALATION_MARKER not in l
                and "SKILL ESCALATION ACKNOWLEDGED" not in l
                and "IMPORTANT: The system will automatically handle this escalation" not in l
                and "Do NOT ask the user to 'continue' or" not in l
                and "'send their request again'" not in l
                and "Simply acknowledge that you need the additional tools" not in l
                and "will take care of the rest." not in l
            ).strip()
            print(f"[ESCALATION_STRIP] Stripped escalation boilerplate from output. "
                  f"Remaining output length: {len(out)}")
    
        # ── HALLUCINATION GUARD: Detect zero-tool-call responses ──────────
        TOOL_FREE_SKILLS = {"conversational"}
        _guard_steps = result.get("intermediate_steps", [])
        _guard_step_count = len(_guard_steps)
        if not TOOL_FREE_SKILLS.issuperset(set(skill_names)):
            _step_tools = []
            for _s in _guard_steps:
                try:
                    _a = _s[0]
                    _t = getattr(_a, "tool", None)
                    _step_tools.append(f"{_t}(type={type(_a).__name__})")
                except Exception as _e:
                    _step_tools.append(f"ERR:{_e}")
            print(f"[HALLUCINATION_GUARD_DEBUG] steps={_guard_step_count} "
                  f"tools={_step_tools} skills={skill_names} "
                  f"escalation_iteration={escalation_iteration}")
    
        status_msg.content = f"🔍 Hallucination guard checking the work..."
        await status_msg.update()
    
        validation = validate_agent_result(result)
        # Skills that legitimately respond without tool calls (e.g. follow-up small talk).
        # Guard is skipped ONLY when the ENTIRE active skill set is tool-free.
        # After escalation (e.g. conversational + hpc_cluster), the guard stays active.
        guard_required = not TOOL_FREE_SKILLS.issuperset(set(skill_names))
        if validation.get("is_confirmation") and guard_required:
            print(f"[HALLUCINATION_GUARD] Skills {skill_names} returned zero tool calls but output "
                  f"is a confirmation prompt — allowing through. Output length: {len(out)}")
        elif validation["is_suspicious"] and guard_required:
            # ── SKIP REASON escape hatch: agent explicitly declared why no tools needed ──
            SKIP_REASON_MARKER = "SKIP REASON:"
            if SKIP_REASON_MARKER in out:
                skip_lines = [l.strip() for l in out.split("\n") if SKIP_REASON_MARKER in l]
                skip_reason = skip_lines[0] if skip_lines else "(no reason extracted)"
                print(f"[HALLUCINATION_GUARD] Agent declared skip reason: {skip_reason}. "
                      f"Allowing through without retry. Output length: {len(out)}")
                # Strip SKIP REASON line(s) from output — internal mechanism, not shown to user
                out = "\n".join(l for l in out.split("\n") if SKIP_REASON_MARKER not in l).strip()
            else:
                print(f"[HALLUCINATION_GUARD] Skills {skill_names} returned output with ZERO tool calls. "
                      f"Response may be fabricated. Output length: {len(out)}. Retrying...")
    
                # RETRY LOOP with escalating tool-enforcement prompts
                retry_succeeded = False
                for retry_attempt in range(1, MAX_HALLUCINATION_RETRIES + 1):
                    retry_instruction = build_tool_enforcement_retry_prompt(
                        user_query=input_content,
                        attempt=retry_attempt,
                        max_attempts=MAX_HALLUCINATION_RETRIES,
                    )
                    retry_input = f"{agent_input}\n\n{retry_instruction}"
    
                    print(f"[HALLUCINATION_GUARD] Retry {retry_attempt}/{MAX_HALLUCINATION_RETRIES} "
                          f"for skills {skill_names}...")
    
                    try:
                        retry_result = await executor.ainvoke(
                            {
                                "input": retry_input,
                                "chat_history": recent_history,
                                "agent_scratchpad": [],
                            },
                            config={
                                "callbacks": agent_callbacks,
                                "handle_parsing_errors": True,
                            }
                        )
    
                        retry_validation = validate_agent_result(retry_result)
                        if not retry_validation["is_suspicious"]:
                            # Retry succeeded — agent used tools
                            out = retry_result["output"]
                            result = retry_result  # Update result for task state saving
                            print(f"[HALLUCINATION_GUARD] Retry {retry_attempt} succeeded — "
                                  f"agent used {retry_validation['tool_count']} tool(s): "
                                  f"{retry_validation['tools_called'][:5]}")
                            retry_succeeded = True
                            break
                        else:
                            print(f"[HALLUCINATION_GUARD] Retry {retry_attempt} still produced "
                                  f"zero tool calls for skills {skill_names}")
                    except SkillEscalationInterrupt as _esc_retry:
                        # Agent correctly identified it needs a different skill —
                        # honor the escalation instead of wasting retries
                        print(f"[HALLUCINATION_GUARD] Retry {retry_attempt} triggered skill "
                              f"escalation to '{_esc_retry.skill_name}' — honoring escalation")
                        _escalation_interrupt_skill = _esc_retry.skill_name
                        result = {"output": "", "intermediate_steps": []}
                        retry_succeeded = True
                        break
                    except Exception as retry_err:
                        print(f"[HALLUCINATION_GUARD] Retry {retry_attempt} failed with error: {retry_err}")
    
                if not retry_succeeded:
                    # All retries exhausted — reject the hallucinated content entirely
                    print(f"[HALLUCINATION_GUARD] All {MAX_HALLUCINATION_RETRIES} retries exhausted "
                          f"for skills {skill_names}. Rejecting hallucinated response.")
                    out = (
                        "⚠️ **I was unable to complete this task using my available tools.** "
                        "After multiple attempts, I could not ground my response in real "
                        "tool calls (file reads, command execution, etc.). "
                        "Please try rephrasing your request, or ask me to perform a "
                        "specific action like reading a file or running a command."
                    )
        else:
            print(f"[HALLUCINATION_GUARD] Skills {skill_names} made {validation['tool_count']} tool call(s): "
                  f"{validation['tools_called'][:5]}")
    
        # ── PLAN-AWARE VERIFICATION GATE (replaces old forced-retry WORKFLOW_GATE) ──
        # Instead of blindly enforcing workflow steps by tool-pattern matching and
        # re-invoking the executor (which caused duplicate SLURM submissions),
        # this gate is now advisory-only. Verification is plan-driven:
        # - If a plan exists and is complete → gate passes.
        # - If a plan exists but is incomplete → defer to plan-verification below.
        # - If no plan exists → advisory log only, never retry.
        _active_plan_path = cl.user_session.get("active_plan_path", "")
    
        workflow_requirements = skill_loader.get_merged_workflow_requirements(skill_names)
        if workflow_requirements:
            workflow_check = validate_workflow_completion(result, workflow_requirements)
            if not workflow_check["workflow_complete"]:
                missing = workflow_check["missing_steps"]
                triggered = workflow_check["triggered_skills"]
    
                if _active_plan_path:
                    from core.plan_verification import plan_has_unchecked_steps
                    from core.opus_planner import read_plan_file
    
                    disk_plan = read_plan_file(_active_plan_path)
                    if disk_plan and not plan_has_unchecked_steps(disk_plan):
                        print(f"[WORKFLOW_GATE] Plan complete (all [x]) — "
                              f"overriding workflow check (missing: {missing})")
                    else:
                        print(f"[WORKFLOW_GATE] Plan in progress — "
                              f"deferring to plan-driven verification. "
                              f"(Workflow missing: {missing})")
                else:
                    print(f"[WORKFLOW_GATE] ADVISORY: Skills {triggered} triggered "
                          f"workflow but skipped: {missing}. "
                          f"No plan active — not forcing retry.")
    
        # ── POST-EXECUTION PLAN VERIFICATION (disk check) ───────────
        # Two-layer verification:
        # 1. Disk check: Read PLAN.md — if unchecked '- [ ]' steps remain, plan is
        #    still in progress. Skip Haiku entirely (free, reliable).
        # 2. Haiku check: When disk says all steps are [x], confirm with strict
        #    per-step evidence mapping. Catches cases where executor marked [x]
        #    without actually completing the step.
        _active_plan = cl.user_session.get("_active_plan_text", "")
        if _active_plan and result.get("intermediate_steps"):
            from core.plan_verification import plan_has_unchecked_steps, count_plan_progress
    
            # Layer 1: Disk-based check — read plan for unchecked steps
            _disk_plan_now = None
            try:
                _active_plan_path = cl.user_session.get("active_plan_path", "")
                if _active_plan_path:
                    _disk_plan_now = read_plan_file(_active_plan_path)
            except Exception:
                pass
    
            _plan_to_check = _disk_plan_now or _active_plan
    
            if plan_has_unchecked_steps(_plan_to_check):
                _completed, _total = count_plan_progress(_plan_to_check)
                _step_count = len(result.get("intermediate_steps", []))
                _edit_plan_called = any(
                    getattr(s[0], "tool", "") == "edit_plan"
                    for s in result.get("intermediate_steps", [])
                )
                if _step_count > 0 and not _edit_plan_called:
                    print(f"[PLAN_VERIFY] Plan incomplete ({_completed}/{_total} steps done) — "
                          f"executor made {_step_count} tool calls but NEVER called edit_plan")
                else:
                    print(f"[PLAN_VERIFY] Plan incomplete ({_completed}/{_total} steps done) — "
                          f"execution will continue next turn")
                cl.user_session.set("_plan_execution_pending", True)
                cl.user_session.set("_active_plan_text", _plan_to_check)
            else:
                # All steps marked [x] on disk — plan is complete. Trust the disk state.
                cl.user_session.set("_active_plan_text", "")
                cl.user_session.set("active_plan_path", "")
                cl.user_session.set("_plan_execution_pending", False)
                cl.user_session.set("_unfulfilled_phase_gates", [])
                print(f"[PLAN_VERIFY] All plan steps marked [x] on disk — plan complete ✓")
    
        # ── FINAL SKIP REASON STRIPPING (foolproof catch-all) ──────────────
        # Regardless of which code path was taken above (hallucination guard,
        # workflow gate, retries, etc.), strip any remaining "SKIP REASON:" lines
        # before the output reaches the user. This is an internal signaling
        # mechanism that must NEVER appear in user-visible output.
        SKIP_REASON_MARKER_FINAL = "SKIP REASON:"
        if SKIP_REASON_MARKER_FINAL in out:
            out = "\n".join(
                l for l in out.split("\n") if SKIP_REASON_MARKER_FINAL not in l
            ).strip()
    
        if out and out.strip():
            # ── HYBRID HISTORY: AI response + compact tool-call records ───────
            # Store AI response + compact tool records (~300-500 chars/tool).
            # Full outputs are in agent_scratchpad during the turn, and on disk
            # via the file-reference system. History gets just enough for the
            # LLM to know what happened (tool + outcome + first line).
            _tool_record = format_tool_call_record(
                result.get("intermediate_steps", [])
            )
            history_content = out + _tool_record if _tool_record else out
            history.append(AIMessage(content=history_content))
        else:
            # Synthesize a short marker if agent did work but produced no text.
            # This prevents consecutive HumanMessages in saved history.
            intermediate_steps = result.get("intermediate_steps", [])
            if intermediate_steps:
                tool_names = [step[0].tool for step in intermediate_steps[:5]]
                suffix = f" (+{len(intermediate_steps) - 5} more)" if len(intermediate_steps) > 5 else ""
                out = f"[Executed: {', '.join(tool_names)}{suffix}]"
                _tool_record = format_tool_call_record(intermediate_steps)
                history_content = out + _tool_record if _tool_record else out
                print(f"[WARN] Agent returned empty text — synthesized from {len(intermediate_steps)} tool call(s): {out[:80]}")
            else:
                print("[WARN] Agent returned empty response with no tool calls")
                out = "I wasn't able to generate a response. Please try again."
                history_content = out
            history.append(AIMessage(content=history_content))
    
        # ── Incremental session log: persist tool calls + assistant message ──
        _log_path = cl.user_session.get("session_log_path")
        if _log_path:
            _log_p = Path(_log_path)
            # Log each tool call and its output (the raw evidence)
            for _step in result.get("intermediate_steps", []):
                try:
                    _action = _step[0]
                    _observation = str(_step[1]) if len(_step) > 1 else ""
                    _tool_name = getattr(_action, "tool", "unknown")
                    _tool_input = getattr(_action, "tool_input", "")
                    if isinstance(_tool_input, dict):
                        _tool_input = json.dumps(_tool_input, ensure_ascii=False)
                    session_log_append(
                        _log_p, "tool",
                        _observation,
                        metadata={"tool": _tool_name, "tool_input": str(_tool_input)[:500]},
                    )
                except Exception:
                    pass
            # Log the final assistant response
            session_log_append(_log_p, "assistant", out)

        # ── Per-turn transcript: write full details to .md for disk backup ──
        from core.history import write_turn_transcript
        from core.persistence import get_work_dir
        _work_dir = get_work_dir()
        if _work_dir:
            _turn_num = cl.user_session.get("_turn_number", 0) + 1
            cl.user_session.set("_turn_number", _turn_num)
            _user_msg = cl.user_session.get("_last_user_input", "")
            write_turn_transcript(
                work_dir=_work_dir,
                turn_number=_turn_num,
                user_input=_user_msg,
                intermediate_steps=result.get("intermediate_steps", []),
                ai_response=out,
            )

        # Remove thinking message and update status before streaming
        await _thinking_cb.cleanup()
        _step_count = len(result.get("intermediate_steps", []))
        _total_elapsed = int(_time_mod.time() - _exec_start)
    
        # Close the parent tool step with final summary (only exists if tools were used)
        _tools_parent_step = cl.user_session.get("_tools_parent_step")
        if _tools_parent_step:
            try:
                _skills_label = cl.user_session.get("_tools_parent_label", "Tools")
                _tools_parent_step.name = f"✓ {_skills_label} · {_step_count} call{'s' if _step_count != 1 else ''} · {_total_elapsed}s"
                _tools_parent_step.end = datetime.datetime.now(datetime.timezone.utc).isoformat()
                await _tools_parent_step.update()
            except Exception:
                pass
    
        if status_msg and _step_count:
            try:
                status_msg.content = f"✓ {_model_label} · {_step_count} tool call{'s' if _step_count != 1 else ''} · {_total_elapsed}s"
                await status_msg.update()
            except Exception:
                pass
    
        # Send the agent's response with typewriter streaming effect
        await _stream_response(out)
    
        # ── Phase approval gate: blocking AskActionMessage ─────────────────
        if not _phase_paused:
            break  # Normal completion — exit phase loop

        _completed_phase = "plan" if "write_plan" in _called_tools or "run_plan_agent" in _called_tools else "research"
        if _completed_phase == "research":
            _phase_actions = [
                cl.Action(name="continue", payload={"value": "continue"}, label="▶️ Continue to Plan"),
                cl.Action(name="skip_execute", payload={"value": "skip_execute"}, label="⚡ Skip to Execute"),
                cl.Action(name="revise", payload={"value": "revise"}, label="✏️ Revise Findings"),
                cl.Action(name="defer", payload={"value": "defer"}, label="⏸️ Done for now"),
            ]
        else:
            _phase_actions = [
                cl.Action(name="continue", payload={"value": "continue"}, label="▶️ Execute Plan"),
                cl.Action(name="revise", payload={"value": "revise"}, label="✏️ Revise Plan"),
                cl.Action(name="defer", payload={"value": "defer"}, label="⏸️ Done for now"),
            ]

        _phase_res = await cl.AskActionMessage(
            content="What would you like to do next?",
            actions=_phase_actions,
            timeout=600,
        ).send()

        # Extract action (same pattern as PEL approval gate at line ~693)
        _phase_action = "defer"
        if _phase_res is not None:
            if isinstance(_phase_res, dict):
                _payload = _phase_res.get("payload", _phase_res)
                _phase_action = (
                    _payload.get("value", _phase_res.get("name", "defer"))
                    if isinstance(_payload, dict)
                    else _phase_res.get("name", "defer")
                )
            elif hasattr(_phase_res, "name"):
                _phase_action = _phase_res.name
            elif hasattr(_phase_res, "value"):
                _phase_action = _phase_res.value

        print(f"[PHASE_GATE] User chose: {_phase_action} (after {_completed_phase} phase)")

        if _phase_action in ("continue", "skip_execute"):
            if _phase_action == "continue":
                if _completed_phase == "research":
                    _needs_planning_raw = True
                    _needs_research_raw = False
                else:
                    _needs_planning_raw = False
                    _needs_research_raw = False
                    cl.user_session.set("_plan_execution_pending", True)
            elif _phase_action == "skip_execute":
                _needs_planning_raw = False
                _needs_research_raw = False
            continue  # Loop back — rebuild executor for next phase

        elif _phase_action == "revise":
            cl.user_session.set("_phase_revision_pending", True)
            cl.user_session.set("_phase_revision_target", _completed_phase)
            await cl.Message(content="What would you like me to change?").send()
            break  # Return — user's next message provides revision instructions

        else:  # "defer" or timeout
            cl.user_session.set("_resume_from_phase", "")
            cl.user_session.set("_plan_execution_pending", False)
            cl.user_session.set("_unfulfilled_phase_gates", [])
            cl.user_session.set("_phase_revision_pending", False)
            await cl.Message(content="Got it. We can pick this up whenever you're ready.").send()
            break

    # ── Activity indicator: show user that post-processing is happening ──
    if status_msg and _step_count:
        try:
            status_msg.content = f"✓ {_model_label} · {_step_count} tool call{'s' if _step_count != 1 else ''} · {_total_elapsed}s\n⏳ Saving session..."
            await status_msg.update()
        except Exception:
            pass

    _turn_count = cl.user_session.get("_turn_count", 1)

    # ── PER-TURN GUARANTEED PERSISTENCE (no LLM, <2ms) ──
    # These survive session kills — even if on_chat_end never fires.
    _steps = result.get("intermediate_steps", [])
    if out and len(out) > 50:
        try:
            from core.memory_state import (
                save_last_turn, load_last_turn,
            )

            # Extract plan progress (done/pending steps from markdown checkboxes)
            _plan_text = cl.user_session.get("_active_plan_text", "")
            _plan_done = []
            _plan_pending = []
            if _plan_text:
                for line in _plan_text.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith("- [x]") or stripped.startswith("- [X]"):
                        _plan_done.append(stripped[6:].strip()[:100])
                    elif stripped.startswith("- [ ]"):
                        _plan_pending.append(stripped[6:].strip()[:100])

            # Extract tool names used this turn
            _tools_this_turn = []
            if _steps:
                for step in _steps:
                    try:
                        _tools_this_turn.append(getattr(step[0], "tool", "unknown"))
                    except (IndexError, AttributeError):
                        pass

            # Running jobs: no longer regex-extracted (was brittle and captured stale IDs).
            # The executor's assistant summary in last_turn.json carries this info naturally.
            # Future: executor writes structured job state to status.md directly.
            _running_jobs = []

            # Current state for milestone detection
            _current_project = cl.user_session.get("project_name", "")
            _current_plan_path = cl.user_session.get("active_plan_path", "")
            _current_findings_path = cl.user_session.get("active_findings_path", "")
            _current_session_id = cl.context.session.id

            # Load previous turn state BEFORE overwriting (for milestone comparison)
            _prev_turn = load_last_turn()

            save_last_turn(
                project_name=_current_project,
                user_msg=input_content[:300] if input_content else "",
                assistant_summary=out[:500] if out else "",
                session_id=_current_session_id,
                plan_steps_done=_plan_done,
                plan_steps_pending=_plan_pending,
                running_jobs=_running_jobs,
                tool_calls_this_turn=_tools_this_turn,
                active_plan_path=_current_plan_path,
                active_findings_path=_current_findings_path,
            )

            # ── MILESTONE DETECTION (cheap comparisons, no LLM) ──
            _prev_session = _prev_turn.get("session_id", "") if _prev_turn else ""
            _prev_project = _prev_turn.get("project", "") if _prev_turn else ""
            _prev_plan_path = _prev_turn.get("active_plan_path", "") if _prev_turn else ""
            _prev_findings_path = _prev_turn.get("active_findings_path", "") if _prev_turn else ""
            _prev_steps_done = _prev_turn.get("plan_steps_done", []) if _prev_turn else []


        except Exception as _persist_err:
            print(f"[PERSIST] Per-turn save failed (non-fatal): {_persist_err}")

    # ── ATTEMPTS LOG: Track what was tried (append-only, never consolidated) ──
    _attempt_project = cl.user_session.get("project_name", "")
    if _attempt_project and _steps:
        try:
            from core.memory_state import append_attempt
            # Summarize what this turn did as a single attempt entry
            _tool_names = [getattr(s[0], "tool", "?") for s in _steps[:5]]
            _action_summary = f"Tools: {', '.join(_tool_names)}"
            if input_content:
                _action_summary = f"{input_content[:100]} | {_action_summary}"
            # Check if there was an error
            _attempt_error = ""
            for step in reversed(_steps):
                obs = str(step[1]) if len(step) > 1 else ""
                if "error" in obs.lower() or "failed" in obs.lower():
                    _attempt_error = obs[:200]
                    break
            _attempt_result = out[:200] if out else ""
            append_attempt(
                project_name=_attempt_project,
                action=_action_summary,
                result=_attempt_result,
                error=_attempt_error,
            )
        except Exception:
            pass

    # Finalize status message: restore clean state or remove if no tools
    if status_msg is not None:
        try:
            _step_count = len(result.get("intermediate_steps", []))
            if not _step_count:
                await status_msg.remove()
            else:
                status_msg.content = f"✓ {_model_label} · {_step_count} tool call{'s' if _step_count != 1 else ''} · {_total_elapsed}s"
                await status_msg.update()
        except Exception:
            pass

    return history


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    env_password = os.environ.get("CHAINLIT_AUTH_SECRET")
    env_username = os.environ.get("USER")
    if (username,password) == (env_username,env_password):
        print(f"[DEBUG] Successfully authenticated User: {username}")
        return cl.User(
            identifier=env_username, metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

# ── Toast notification helper ───────────────────────────────────────────
_TOAST_STAGGER = 2.5  # seconds between consecutive toasts

async def send_toast(message: str, toast_type: str = "info", delay: float = 0, stagger: bool = True):
    """Send a popup toast notification using Chainlit emitter API.
    toast_type: 'info', 'success', 'warning', 'error'
    delay: explicit seconds to wait BEFORE showing (overrides stagger if > 0)
    stagger: if True, auto-space from last toast to avoid overlap
    """
    try:
        if delay > 0:
            await asyncio.sleep(delay)
        elif stagger:
            _last = getattr(send_toast, "_last_ts", 0)
            _now = asyncio.get_event_loop().time()
            _elapsed = _now - _last if _last else _TOAST_STAGGER
            if _elapsed < _TOAST_STAGGER:
                await asyncio.sleep(_TOAST_STAGGER - _elapsed)
        send_toast._last_ts = asyncio.get_event_loop().time()
        await cl.context.emitter.send_toast(
            message=message,
            type=toast_type,
        )
    except Exception as e:
        # Fallback: just log it, don't clutter the chat
        print(f"[TOAST-{toast_type.upper()}] {message}")
        print(f"[TOAST-ERROR-DETAIL] {type(e).__name__}: {e}")


# ── Project session-log helper ─────────────────────────────────────────

def _log_project_to_session(project_name: str, work_dir: str = "") -> None:
    """Write a __PROJECT_SWITCH__ marker to the session log."""
    if not project_name:
        return
    _log_path = cl.user_session.get("session_log_path")
    if not _log_path:
        return
    session_log_append(
        Path(_log_path), "system",
        f"__PROJECT_SWITCH__ {project_name}",
        metadata={"type": "project_switch", "project": project_name, "work_dir": work_dir},
    )


# ── Curation progress toasts ───────────────────────────────────────────

async def make_curation_progress_callback():
    """Create a toast-based progress callback for curate_pending_session_logs."""

    async def _progress(event: str, **kwargs):
        if event == "found":
            items = kwargs["items"]
            if items:
                n = len(items)
                await send_toast(f"📝 Curating {n} previous session{'s' if n != 1 else ''}...", "info")
        elif event == "project_done":
            project = kwargs.get("project", "unknown")
            entry_count = kwargs.get("entry_count", 0)
            await send_toast(f"✅ Curated: {project} ({entry_count} messages)", "success")
        elif event == "project_failed":
            project = kwargs.get("project", "unknown")
            await send_toast(f"❌ Failed to curate: {project}", "error")
        elif event == "skip":
            await send_toast("⏭️ Skipped legacy session (no project marker)", "info")

    return _progress


# ── Rename internal LangChain step names to friendly UI labels ──────────
@cl.author_rename
def rename_authors(orig_author: str) -> str:
    """Map internal LangChain class names to friendly display names."""
    rename_map = {
        "AgentExecutor": "Iris AI",
        "ChatOpenAI": "Claude 💭",
        "ChatLiteLLM": "Claude 💭",
        "ChatBedrock": "Claude 💭",
        "Chatbot": "IRIS",
    }
    return rename_map.get(orig_author, orig_author)


@cl.on_chat_start
async def start():
    cl.user_session.set("mcp_sessions", {})
    cl.user_session.set("lc_tools", [])
    username = cl.context.session.user.identifier
    session_id = cl.context.session.id

    # ── RECONNECT DETECTION ─────────────────────────────────────────
    # If a session log already exists for this session_id, we're reconnecting
    # within the same Slurm job. Auto-load history from the session log.
    # Fallback chain: session log → conversation file → lazy loading.
    history_msg = None
    reconnected = False
    message_factory = {"human": HumanMessage, "ai": AIMessage}

    session_log_path = get_session_log_dir(username) / f"{session_id}.jsonl"
    if session_log_path.exists():
        try:
            reconnect_history = load_history_from_session_log(
                username, session_id, message_factory=message_factory
            )
            if reconnect_history:
                cl.user_session.set("history", reconnect_history)
                reconnected = True
                print(f"[RECONNECT] Session {session_id[:8]} — auto-loaded "
                      f"{len(reconnect_history)} messages from session log")
        except Exception as e:
            print(f"[RECONNECT] Session log parse failed: {e}")

    if reconnected:
        history_msg = (
            f"🔄 Session reconnected — {len(cl.user_session.get('history', []))} messages restored"
        )
    else:
        cl.user_session.set("history", [])

        # ── Clear stale checkpoints (only on fresh start, not reconnect) ──
        try:
            cleared = clear_session_checkpoints(username)
            if cleared > 0:
                print(f"[CHECKPOINT] Cleared {cleared} stale checkpoint files for {username}")
        except Exception as e:
            print(f"[CHECKPOINT] Warning: could not clear checkpoints: {e}")

    # ── Deferred curation + Crash recovery ──────────────────────────────────
    # Blocks startup so memory state is current before user interaction.
    # Toast notifications show per-project curation progress.
    if not reconnected:
        try:
            from core.memory_state import curate_pending_session_logs
            progress_cb = await make_curation_progress_callback()
            _recovery_msgs = await curate_pending_session_logs(
                username, session_id, progress_cb=progress_cb
            )
            if _recovery_msgs:
                if history_msg:
                    history_msg += f"\n{_recovery_msgs}"
                else:
                    history_msg = _recovery_msgs
        except Exception as _cr_err:
            print(f"[DEFERRED_CURATE] Recovery failed (non-fatal): {_cr_err}")
            await send_toast("Session recovery failed (non-fatal)", "warning")

    # ── LiteLLM proxy LLMs ──────────────────────────────────────────
    # Initialize LLMs BEFORE MCP connections — ensures user can always chat
    # even if MCP servers fail to connect.
    # disable_streaming=True: on_llm_end receives full token_usage from LiteLLM.

    # Sonnet: main agent execution (all skills except dev)
    llm = ChatOpenAI(
        model="anthropic.claude-sonnet-4-6",
        temperature=0,
        openai_api_key=api_key,
        openai_api_base=LITELLM_URL,
        request_timeout=300,
        disable_streaming=True,
    )
    cl.user_session.set("llm", llm)

    # Opus: only used when user explicitly requests it or advisor recommends it
    dev_llm = ChatOpenAI(
        model="anthropic.claude-opus-4-6-v1",
        temperature=0,
        openai_api_key=api_key,
        openai_api_base=LITELLM_URL,
        request_timeout=300,
        disable_streaming=True,
    )
    cl.user_session.set("dev_llm", dev_llm)

    # Haiku: skill selection (classification) + summarization (cheap, fast)
    haiku_llm = ChatOpenAI(
        model="anthropic.claude-haiku-4-5-20251001-v1",
        temperature=0,
        openai_api_key=api_key,
        openai_api_base=LITELLM_URL,
        request_timeout=300,
        disable_streaming=True,
    )
    cl.user_session.set("haiku_llm", haiku_llm)

    # ── Connect to MCP servers (with timeout — non-fatal) ─────────────
    # Entire section wrapped: if MCP fails, user can still chat (no tools).
    failed_servers = []
    try:
        token = os.environ.get("MCP_SHARED_BEARER_TOKEN")
        for i, server in enumerate(MCP_SERVERS):
            try:
                if i > 0:
                    await asyncio.sleep(0.3)
                headers = {"Authorization": f"Bearer {token}"}
                conn = ConnectStreamableHttpMCPRequest(
                    sessionId=cl.context.session.id,
                    clientType="streamable-http",
                    name=server["name"],
                    url=server["url"],
                    headers=headers
                )
                await asyncio.wait_for(
                    cl_connect_mcp(conn, cl.context.session.user),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                print(f"[MCP-CONNECT] Timeout connecting to {server['name']} (15s)")
                failed_servers.append(server["name"])
            except Exception as e:
                print(f"[MCP-CONNECT] Failed to connect {server['name']}: {type(e).__name__}: {e}")
                failed_servers.append(server["name"])

        # ── Connect to User Extension MCP servers ────────────────────────────
        if _user_ext_dir.exists():
            user_servers = load_user_extension_configs(str(_user_ext_dir))
            for server in user_servers:
                try:
                    await asyncio.sleep(0.3)
                    headers = {"Authorization": f"Bearer {token}"}
                    conn = ConnectStreamableHttpMCPRequest(
                        sessionId=cl.context.session.id,
                        clientType="streamable-http",
                        name=server["name"],
                        url=server["url"],
                        headers=headers
                    )
                    await asyncio.wait_for(
                        cl_connect_mcp(conn, cl.context.session.user),
                        timeout=15.0
                    )
                except (asyncio.TimeoutError, Exception) as e:
                    print(f"[MCP-CONNECT] Failed user extension {server['name']}: "
                          f"{type(e).__name__}: {e}")
    except Exception as _mcp_err:
        print(f"[MCP-CONNECT] MCP connection phase failed (non-fatal): {_mcp_err}")

    if failed_servers:
        await send_toast(f"Some tools unavailable: {', '.join(failed_servers)}", "warning")

    # ── Native Anthropic providers (Phase 1: skill selection + sub-agents) ──
    if IRIS_USE_NATIVE_ANTHROPIC:
        from core.llm_provider import get_provider
        _sonnet_provider = get_provider(
            "anthropic", model_id="anthropic.claude-sonnet-4-6",
            temperature=0, max_tokens=8192, timeout=300,
        )
        _opus_provider = get_provider(
            "anthropic", model_id="anthropic.claude-opus-4-6-v1",
            temperature=0, max_tokens=16384, thinking_budget=10000, timeout=300,
        )
        _haiku_provider = get_provider(
            "anthropic", model_id="anthropic.claude-haiku-4-5-20251001-v1",
            temperature=0, max_tokens=4096, timeout=300,
        )
        cl.user_session.set("sonnet_provider", _sonnet_provider)
        cl.user_session.set("opus_provider", _opus_provider)
        cl.user_session.set("haiku_provider", _haiku_provider)

    # ── Work directory setup (settings file is single source of truth) ──
    bootstrap_work_dir_from_env(username)
    work_dir = get_work_dir(username)
    user_data = get_user_settings(username)

    if work_dir:
        cl.user_session.set("work_dir", work_dir)

    # ── Weights path: persist in session so system prompt always has it ──
    _weights_path = user_data.get("weights_path", "")
    if not _weights_path and work_dir:
        _default_weights = os.path.join(work_dir, "alphafold3", "weights")
        if os.path.isdir(_default_weights) and os.listdir(_default_weights):
            _weights_path = _default_weights
    if _weights_path:
        cl.user_session.set("weights_path", _weights_path)

    # Don't auto-load stale project_name from settings — start fresh.
    # Project is set on turn 1 via deterministic project picker (AskActionMessage),
    # then on subsequent turns via the Haiku skill selector's project_context field.
    cl.user_session.set("project_name", "")

    # ── Incremental session log (crash-safe JSONL) ─────────────────────
    try:
        log_path = init_session_log(username, session_id, {"work_dir": work_dir})
        cl.user_session.set("session_log_path", str(log_path))
        print(f"[SESSION_LOG] Initialized: {log_path}")
    except Exception as e:
        cl.user_session.set("session_log_path", "")
        print(f"[SESSION_LOG] Warning: could not initialize session log: {e}")

    # ── Send staggered toast notifications ──────────────────────────────
    # Auto-stagger ensures each toast waits 2.5s after the previous one.
    await send_toast(f"👋 Hello {username}! Welcome to IrisAI", "success")

    if work_dir:
        if os.access(work_dir, os.W_OK):
            await send_toast(f"📁 Work directory: {work_dir}", "info")
        else:
            await send_toast(f"❌ Work directory is not writable: {work_dir}", "error")
    else:
        await send_toast("⚠️ No work directory set", "warning")

    if history_msg:
        toast_type = "warning" if "Could not" in history_msg else "info"
        await send_toast(history_msg, toast_type)

    # Toast 4: MCP server failures — already shown at connection time (line ~3202)

    # Toast 5: Budget status (non-blocking — fetched from LiteLLM /user/info)
    try:
        budget_headers = {"Authorization": f"Bearer {api_key}"}
        async with aiohttp.ClientSession() as budget_session:
            async with budget_session.get(
                f"{LITELLM_URL}/user/info",
                headers=budget_headers,
                params={"user_id": username},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as budget_resp:
                if budget_resp.status == 200:
                    budget_data = await budget_resp.json()
                    user_info = budget_data.get("user_info", budget_data)
                    spend = user_info.get("spend", 0) or 0
                    max_budget = user_info.get("max_budget")
                    if max_budget and max_budget > 0:
                        pct = (spend / max_budget) * 100
                        remaining = max_budget - spend
                        if pct > 80:
                            await send_toast(
                                f"⚠️ Budget: ${spend:.2f} / ${max_budget:.2f} ({pct:.0f}% used, ${remaining:.2f} remaining)",
                                "warning"
                            )
                        else:
                            await send_toast(
                                f"💰 Budget: ${spend:.2f} / ${max_budget:.2f} ({pct:.0f}% used, ${remaining:.2f} remaining)",
                                "info"
                            )
                    else:
                        await send_toast(f"💰 Spend: ${spend:.2f} (no budget limit)", "info")
    except Exception as budget_err:
        print(f"[BUDGET_TOAST] Failed to fetch budget info: {budget_err}")

    # ── Disk space check ────────────────────────────────────────────────
    try:
        _home_dir = os.path.expanduser("~")
        _work_dir = cl.user_session.get("work_dir", "")
        _disk_results = run_startup_check(_home_dir, _work_dir)
        _disk_toast = format_startup_toast(_disk_results)
        if _disk_toast:
            await send_toast(_disk_toast["message"], _disk_toast["type"])
            print(f"[DISK] Startup check: {_disk_toast['message']}")
        cl.user_session.set("_disk_check_results", _disk_results)
        # Start background disk monitor
        asyncio.get_event_loop().create_task(
            _background_disk_monitor(username, _home_dir, _work_dir)
        )
    except Exception as _disk_err:
        print(f"[DISK] Startup check failed: {_disk_err}")


    # ── Walltime monitor ────────────────────────────────────────────────
    asyncio.get_event_loop().create_task(_background_walltime_monitor())

    # ── MCP keepalive monitor ────────────────────────────────────────────
    asyncio.get_event_loop().create_task(_background_mcp_keepalive())

    # ── Register command buttons (web search + fresh start) ─────────────
    # The globe button appears near the text input. Clicking it toggles
    # websearch_enabled in the session — when ON, the skill selector can
    # route to the websearch skill. When OFF, websearch tools are not in pool.
    try:
        await cl.context.emitter.set_commands([
            {"id": "Search", "icon": "globe", "description": "Search the web safely via SearXNG", "button": True, "persistent": True},
            {"id": "Guidance", "icon": "compass", "description": "Force research→plan→approve before execution", "button": True, "persistent": True},
            {"id": "Protocol", "icon": "circle-dot", "description": "Start/stop protocol recording for reproducibility", "button": True, "persistent": True},
            {"id": "Play", "icon": "play-circle", "description": "Replay a recorded protocol (Reproduce/Transfer)", "button": True, "persistent": False},
            {"id": "FreshStart", "icon": "rotate-ccw", "description": "Save conversation and start fresh", "button": True, "persistent": False}
        ])
        print("[DEBUG] Commands registered (Search, Guidance, Protocol, Play, FreshStart)")
    except Exception as cmd_err:
        print(f"[WARN] Failed to register commands: {cmd_err}")

    # Initialize web search as disabled — user must click globe to enable
    cl.user_session.set("websearch_enabled", False)
    # Initialize guidance mode as disabled — user must click compass to enable
    cl.user_session.set("guidance_mode", False)

    # Initialize protocol recorder — user must click button to enable
    cl.user_session.set("protocol_recorder", None)
    cl.user_session.set("protocol_nudged", False)
    cl.user_session.set("protocol_player", None)
    try:
        _proto_dir = get_protocols_dir(username)
        _proto_dir.mkdir(parents=True, exist_ok=True)
        _current_session_id = cl.context.session.id

        # Auto-cleanup stale recordings from dead sessions
        _cleanup_result = ProtocolRecorder.cleanup_stale(_proto_dir, _current_session_id)
        if _cleanup_result["drafted"]:
            _draft_names = ", ".join(_cleanup_result["drafted"][:3])
            await send_toast(
                f"Saved {len(_cleanup_result['drafted'])} incomplete protocol(s) from previous session(s) to _drafts/: {_draft_names}",
                "info", delay=2
            )
        if _cleanup_result["deleted"]:
            print(f"[PROTOCOL] Deleted {len(_cleanup_result['deleted'])} empty stale recording(s)")

        # Check for incomplete recordings from THIS session (reconnect recovery)
        _incomplete = ProtocolRecorder.find_incomplete(_proto_dir)
        _own_incomplete = [
            p for p in _incomplete
            if ProtocolRecorder.read_header(p).get("session_id") == _current_session_id
        ]
        if _own_incomplete:
            await send_toast(
                f"Found {len(_own_incomplete)} interrupted protocol(s) from this session. "
                "Say 'resume protocol' to continue.",
                "warning", delay=2
            )

        # Check for play-mode SLURM checkpoints
        _play_checkpoints = detect_slurm_checkpoints(get_user_data_dir(username))
        if _play_checkpoints:
            await send_toast(
                f"Found {len(_play_checkpoints)} paused play session(s) awaiting SLURM. "
                "Say 'resume play' to continue.",
                "info", delay=3
            )
    except Exception as _proto_err:
        print(f"[PROTOCOL] Error checking incomplete protocols: {_proto_err}")

    # Toast 6: Skills loaded
    skill_count = len(skill_loader.list_skill_names())
    await send_toast(f"🧠 {skill_count} skills loaded", "info")


@cl.on_chat_resume
async def on_chat_resume(thread):
    """Restore session state: curate memory from session_log, load recent messages.

    When the user's browser goes idle and reconnects, Chainlit calls this
    instead of on_chat_start. We:
    1. Run mid-session Sonnet curation (status.md + knowledge.md, skip history.md)
    2. Load last 20 user+assistant messages (tool observations skipped)
    3. Re-create LLMs and session variables
    """
    try:
        username = cl.context.session.user.identifier
        session_id = cl.context.session.id
        print(f"[CHAT_RESUME] Resuming session for {username}, session_id={session_id}")

        await send_toast("🔄 Resuming session — updating memory...", "info")

        # 1. Run mid-session curation (skip history.md — session not over yet)
        _log_path = get_session_log_dir(username) / f"{session_id}.jsonl"
        if _log_path.exists():
            try:
                from core.memory_state import _parse_session_log_state, session_end_curate
                state = _parse_session_log_state(str(_log_path))
                _last_project = ""
                for proj_name, proj_data in state.get("projects", {}).items():
                    if not proj_name or proj_name == "general":
                        continue
                    if len(proj_data["entries"]) < 3:
                        continue
                    conversation_text = "\n".join(
                        f"[{e.get('role', 'unknown')}]: {e.get('content', '')}"
                        for e in proj_data["entries"]
                    )
                    await session_end_curate(
                        project_name=proj_name,
                        work_dir=proj_data["work_dir"],
                        conversation_text=conversation_text,
                        skip_history=True,
                        entries=proj_data["entries"],
                    )
                    _last_project = proj_name
                    print(f"[CHAT_RESUME] Curated project '{proj_name}' ({len(proj_data['entries'])} msgs)")
            except Exception as _curate_err:
                print(f"[CHAT_RESUME] Curation failed (non-fatal): {_curate_err}")
                _last_project = ""

        # 2. Load last 20 user+assistant messages (tools skipped by load_history_from_session_log)
        message_factory = {"human": HumanMessage, "ai": AIMessage}
        previous_history = load_history_from_session_log(
            username, session_id, message_factory=message_factory
        )
        if previous_history:
            capped = previous_history[-20:] if len(previous_history) > 20 else previous_history
            cl.user_session.set("history", capped)
            print(f"[CHAT_RESUME] Loaded {len(capped)} messages (of {len(previous_history)} total)")
        else:
            cl.user_session.set("history", [])
            print("[CHAT_RESUME] No previous history found on disk")

        # 3. Re-initialize session variables
        cl.user_session.set("mcp_sessions", {})
        cl.user_session.set("lc_tools", [])
        cl.user_session.set("websearch_enabled", False)

        # Re-initialize work directory and user settings
        bootstrap_work_dir_from_env(username)
        work_dir = get_work_dir(username)
        if work_dir:
            cl.user_session.set("work_dir", work_dir)
        cl.user_session.set("project_name", _last_project if _last_project else "")
        cl.user_session.set("session_log_path", str(_log_path) if _log_path.exists() else "")

        # 4. Restore phase state from session log markers
        if _log_path.exists() and _last_project:
            try:
                _proj_data = state.get("projects", {}).get(_last_project, {})
                _proj_entries = _proj_data.get("entries", [])
                _markers = [e for e in _proj_entries if e.get("type") == "phase_marker"]
                if _markers:
                    _last_marker = _markers[-1]
                    _last_phase = _last_marker.get("phase", "execute")
                    _last_event = _last_marker.get("event", "")
                    _meta = _last_marker.get("metadata", {})

                    if _last_event == "awaiting_approval":
                        cl.user_session.set("_resume_from_phase", _last_phase)
                        cl.user_session.set("_current_phase", _last_phase)
                        print(f"[CHAT_RESUME] Restored phase: awaiting approval for '{_last_phase}'")
                    elif _last_event == "started" and _last_phase != "execute":
                        _gate_tool = "write_findings" if _last_phase == "research" else "write_plan"
                        cl.user_session.set("_unfulfilled_phase_gates", [_gate_tool])
                        cl.user_session.set("_current_phase", _last_phase)
                        print(f"[CHAT_RESUME] Restored phase: '{_last_phase}' in progress")
                    elif _last_event == "completed" and _last_phase == "plan":
                        _plan_path = _meta.get("artifact", "")
                        if _plan_path and Path(_plan_path).is_file():
                            cl.user_session.set("active_plan_path", _plan_path)
                            cl.user_session.set("_plan_execution_pending", True)
                            print(f"[CHAT_RESUME] Restored: plan execution pending")
                    elif _last_event == "completed" and _last_phase == "research":
                        _artifact = _meta.get("artifact", "")
                        if _artifact and Path(_artifact).is_file():
                            cl.user_session.set("active_findings_paths", [_artifact])
                            cl.user_session.set("active_findings_path", _artifact)
                            _planned_phases = []
                            for m in _markers:
                                if m.get("event") == "started":
                                    _pp = m.get("metadata", {}).get("phases_planned", [])
                                    if _pp:
                                        _planned_phases = _pp
                            if "plan" in _planned_phases:
                                cl.user_session.set("_resume_from_phase", "plan")
                                print(f"[CHAT_RESUME] Restored: research done, plan pending")
                            else:
                                print(f"[CHAT_RESUME] Restored: research findings active")
            except Exception as _phase_err:
                print(f"[CHAT_RESUME] Phase state restore failed (non-fatal): {_phase_err}")
        if _last_project:
            _log_project_to_session(_last_project, work_dir or "")
            await send_toast(f"📂 Resumed project: {_last_project}", "info")

        # Re-create LLMs (these are not serializable — must be recreated)
        api_key = os.environ.get("LITELLM_VIRTUAL_KEY", "")
        llm = ChatOpenAI(
            model="anthropic.claude-sonnet-4-6",
            temperature=0,
            openai_api_key=api_key,
            openai_api_base=LITELLM_URL,
            request_timeout=300,
            disable_streaming=True,
        )
        cl.user_session.set("llm", llm)

        dev_llm = ChatOpenAI(
            model="anthropic.claude-opus-4-6-v1",
            temperature=0,
            openai_api_key=api_key,
            openai_api_base=LITELLM_URL,
            request_timeout=300,
            disable_streaming=True,
        )
        cl.user_session.set("dev_llm", dev_llm)

        haiku_llm = ChatOpenAI(
            model="anthropic.claude-haiku-4-5-20251001-v1",
            temperature=0,
            openai_api_key=api_key,
            openai_api_base=LITELLM_URL,
            request_timeout=300,
            disable_streaming=True,
        )
        cl.user_session.set("haiku_llm", haiku_llm)

        msg_count = len(capped) if previous_history else 0
        await send_toast(
            f"🔄 Session resumed — {msg_count} messages restored, memory updated",
            "success"
        )
        print(f"[CHAT_RESUME] Session fully restored for {username}")

    except Exception as e:
        print(f"[CHAT_RESUME] Error resuming session: {e}")
        cl.user_session.set("history", [])
        cl.user_session.set("mcp_sessions", {})
        cl.user_session.set("lc_tools", [])
        await send_toast("⚠️ Could not restore previous session — starting fresh", "warning")



@cl.on_mcp_connect
async def on_mcp_connect(connection, session):
    print(f"[DEBUG] Connected to MCP server: {connection.name}")
    mcp_sessions = cl.user_session.get("mcp_sessions", {})
    mcp_sessions[connection.name] = session
    cl.user_session.set("mcp_sessions", mcp_sessions)
    try:
        tools_result = await session.list_tools()
    except Exception as e:
        print(f"[MCP-CONNECT] Failed to list tools from {connection.name}: {type(e).__name__}: {e}")
        return
    lc_tools = cl.user_session.get("lc_tools", [])
    existing_names = {t.name for t in lc_tools if hasattr(t, 'name')}
    added = 0
    for t in tools_result.tools:
        if t.name and t.description:
            if t.name in existing_names:
                lc_tools = [lt for lt in lc_tools if getattr(lt, 'name', None) != t.name]
                existing_names.discard(t.name)
            schema_cls = _build_args_schema(t.name, t.inputSchema) if t.inputSchema else None
            tool = MCPTool(
                name=t.name,
                description=t.description,
                args_schema=schema_cls,
            )
            tool.mcp_session = session
            tool._server_name = connection.name
            lc_tools.append(tool)
            existing_names.add(t.name)
            added += 1
    cl.user_session.set("lc_tools", lc_tools)
    print(f"[DEBUG] Loaded {added} tools from {connection.name} (total: {len(lc_tools)})")




# ── Background Disk Space Monitor ────────────────────────────────────────────


async def _background_disk_monitor(username: str, home_dir: str, work_dir: str):
    """Periodically check disk space and warn/pause recording if critical."""
    _INTERVAL = 300  # 5 minutes
    _last_status = DiskStatus.OK
    while True:
        await asyncio.sleep(_INTERVAL)
        try:
            from core.disk_monitor import get_worst_status
            results = run_startup_check(home_dir, work_dir)
            current = get_worst_status(results)

            if current != _last_status and current != DiskStatus.OK:
                toast = format_startup_toast(results)
                if toast:
                    await send_toast(toast["message"], toast["type"])

                # Auto-pause protocol recording at CRITICAL
                if current == DiskStatus.CRITICAL:
                    _recorder = cl.user_session.get("protocol_recorder")
                    if _recorder and _recorder.is_active:
                        _recorder.pause(reason="disk_space_critical")
                        await send_toast(
                            "Protocol recording PAUSED — disk space critical!",
                            "error",
                        )
                        print(f"[DISK] Auto-paused recording for {username}")

            _last_status = current
        except Exception as e:
            print(f"[DISK] Monitor error: {e}")


async def _background_walltime_monitor():
    """Monitor SLURM job remaining walltime and warn at thresholds."""
    from core.walltime_monitor import walltime_monitor_loop
    await walltime_monitor_loop(send_toast_fn=send_toast)


# ── MCP Session Keepalive ────────────────────────────────────────────────────


async def _reconnect_mcp_server(server_name: str) -> bool:
    """Reconnect to an MCP server by name and update all tool references."""
    server_config = None
    for server in MCP_SERVERS:
        if server["name"] == server_name:
            server_config = server
            break
    if not server_config:
        from core.config import load_user_extension_configs
        _user_ext_dir = Path(os.path.expanduser("~")) / ".irisai" / "extensions"
        if _user_ext_dir.exists():
            for s in load_user_extension_configs(str(_user_ext_dir)):
                if s["name"] == server_name:
                    server_config = s
                    break
    if not server_config:
        print(f"[MCP-RECONNECT] No config for server '{server_name}'")
        return False

    try:
        token = os.environ.get("MCP_SHARED_BEARER_TOKEN")
        headers = {"Authorization": f"Bearer {token}"}
        conn = ConnectStreamableHttpMCPRequest(
            sessionId=cl.context.session.id,
            clientType="streamable-http",
            name=server_config["name"],
            url=server_config["url"],
            headers=headers,
        )
        await asyncio.wait_for(
            cl_connect_mcp(conn, cl.context.session.user),
            timeout=15.0,
        )
        updated_sessions = cl.user_session.get("mcp_sessions", {})
        new_session = updated_sessions.get(server_name)
        if new_session:
            lc_tools = cl.user_session.get("lc_tools", [])
            for tool in lc_tools:
                if isinstance(tool, MCPTool) and tool._server_name == server_name:
                    tool.mcp_session = new_session
            print(f"[MCP-RECONNECT] Proactively reconnected '{server_name}'")
            return True
        return False
    except Exception as e:
        print(f"[MCP-RECONNECT] Failed to reconnect '{server_name}': {type(e).__name__}: {e}")
        return False


async def _background_mcp_keepalive():
    """Ping MCP sessions every 2 min to prevent idle-timeout disconnections.

    StreamableHttp connections idle-timeout after ~5 minutes. Sub-agent
    operations (research, plan, worker) routinely take 5-10 min, during
    which no main-agent MCP calls keep the connection alive.
    """
    _INTERVAL = 120
    while True:
        await asyncio.sleep(_INTERVAL)
        try:
            mcp_sessions = cl.user_session.get("mcp_sessions", {})
            if not mcp_sessions:
                continue
            for server_name, session in list(mcp_sessions.items()):
                if session is None:
                    continue
                try:
                    await asyncio.wait_for(session.list_tools(), timeout=10.0)
                except asyncio.TimeoutError:
                    print(f"[MCP-KEEPALIVE] Ping timeout for '{server_name}' — attempting reconnect")
                    await _reconnect_mcp_server(server_name)
                except Exception as e:
                    err_str = str(e).lower()
                    if any(k in err_str for k in ("closed", "disconnect", "broken", "eof")):
                        print(f"[MCP-KEEPALIVE] Session dead for '{server_name}': {type(e).__name__} — reconnecting")
                        await _reconnect_mcp_server(server_name)
                    else:
                        print(f"[MCP-KEEPALIVE] Ping error for '{server_name}': {type(e).__name__}: {e}")
        except Exception as e:
            print(f"[MCP-KEEPALIVE] Monitor error: {e}")


# ── Play Mode Helpers ─────────────────────────────────────────────────────────


def _detect_protocol_path(text: str) -> Optional[Path]:
    """Detect if the message text contains a path to a protocol directory or file.

    Returns Path if found and valid (directory with recording.jsonl, or a .jsonl file),
    None otherwise.
    """
    if not text:
        return None

    # Match absolute paths (common on HPC: /home/..., /data1/..., /scratch/...)
    path_candidates = re.findall(r'(/[^\s,;\"\']+)', text)

    for candidate in path_candidates:
        p = Path(candidate.rstrip("/"))
        # Directory containing recording.jsonl
        if p.is_dir() and (p / "recording.jsonl").exists():
            return p
        # Direct path to a recording.jsonl file
        if p.is_file() and p.name == "recording.jsonl":
            return p.parent
        # Path to a protocol.yaml — parent should have recording.jsonl
        if p.is_file() and p.name == "protocol.yaml" and (p.parent / "recording.jsonl").exists():
            return p.parent

    return None


async def _import_protocol_to_dir(source: Path, protocols_dir: Path) -> Optional[Path]:
    """Copy a protocol directory into the user's protocols directory.

    Returns the destination Path on success, None on failure.
    """
    import shutil

    if not source.exists():
        return None

    # Validate: source must be a directory with recording.jsonl
    if source.is_file() and source.name == "recording.jsonl":
        source = source.parent
    if not source.is_dir() or not (source / "recording.jsonl").exists():
        return None

    # Determine destination name
    dest_name = source.name
    # If source doesn't follow naming convention, create one from directory name
    if not re.match(r"^.+_v\d+\.\d+\.\d+$", dest_name):
        dest_name = f"{dest_name}_v1.0.0"

    dest = protocols_dir / dest_name
    if dest.exists():
        # Already imported — use it directly
        return dest

    # Copy the protocol directory
    shutil.copytree(str(source), str(dest))
    return dest


async def _import_uploaded_protocol(uploaded_files: list, protocols_dir: Path) -> Optional[Path]:
    """Process uploaded protocol files and import into protocols directory.

    Handles: raw recording.jsonl, .tar.gz/.zip archives containing protocol dir.
    """
    import shutil
    import tempfile
    import tarfile
    import zipfile
    from datetime import datetime

    for file in uploaded_files:
        source_path = Path(file.path)
        if not source_path.exists():
            continue

        fname = file.name.lower()

        # Raw .jsonl file — create a protocol directory for it
        if fname.endswith(".jsonl"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_dir = protocols_dir / f"imported_{timestamp}_v1.0.0"
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source_path), str(dest_dir / "recording.jsonl"))
            return dest_dir

        # Archive — extract and look for protocol dir inside
        if fname.endswith((".tar.gz", ".tgz", ".gz", ".zip")):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                try:
                    if fname.endswith(".zip"):
                        with zipfile.ZipFile(str(source_path), "r") as zf:
                            zf.extractall(tmp)
                    else:
                        with tarfile.open(str(source_path), "r:*") as tf:
                            tf.extractall(tmp)
                except (tarfile.TarError, zipfile.BadZipFile):
                    continue

                # Find recording.jsonl in extracted contents
                for jsonl in tmp.rglob("recording.jsonl"):
                    proto_src = jsonl.parent
                    return await _import_protocol_to_dir(proto_src, protocols_dir)

    return None


# ── Refine Mode ──────────────────────────────────────────────────────────────


async def _run_refine_mode(protocol_dir: Path, username: str):
    """Drive the refine workflow: raw recording → golden protocol.

    Flow:
    1. Load raw recording and generate analysis prompt
    2. Send to LLM (Sonnet) for golden path identification
    3. Present plan to user for approval
    4. Execute the golden path
    5. Detect variables
    6. Produce golden_protocol.json
    """
    from core.sub_agent import _call_sub_agent_llm, WORKER_AGENT_MODEL

    refiner = ProtocolRefiner(protocol_dir)

    # Step 1: Load raw recording
    try:
        summary = refiner.load_raw_recording()
    except Exception as e:
        await cl.Message(content=f"Failed to load recording: {e}").send()
        return

    await cl.Message(
        content=f"**Refine Mode** — Analyzing recording: {summary['total_steps']} steps "
        f"({summary['success_steps']} succeeded, {summary['error_steps']} had errors).\n\n"
        f"Asking LLM to identify the golden path..."
    ).send()

    # Step 2: LLM analysis
    analysis_prompt = refiner.generate_analysis_prompt()
    try:
        llm_response = await _call_sub_agent_llm(analysis_prompt, model=WORKER_AGENT_MODEL)
    except Exception as e:
        await cl.Message(content=f"LLM analysis failed: {e}").send()
        return

    # Step 3: Parse and present plan
    try:
        golden_steps = refiner.parse_golden_plan(llm_response)
    except ValueError as e:
        await cl.Message(
            content=f"Could not parse LLM's golden path selection: {e}\n\n"
            f"Raw response:\n```\n{llm_response[:2000]}\n```"
        ).send()
        return

    plan_summary = refiner.get_plan_summary()
    await cl.Message(content=plan_summary).send()

    # Ask user to approve
    _approve = await cl.AskActionMessage(
        content=f"**Golden path: {len(golden_steps)} steps.** Execute this plan?",
        actions=[
            cl.Action(name="refine_approve", payload={"approve": "yes"}, label="Execute golden path"),
            cl.Action(name="refine_approve", payload={"approve": "no"}, label="Cancel"),
        ],
        timeout=120,
    ).send()

    if not _approve or _approve.get("approve") != "yes":
        await send_toast("Refine mode cancelled.", "info")
        return

    # Step 4: Execute
    await cl.Message(content="Executing golden path...").send()
    result = await refiner.execute_golden_plan(_play_mode_call_tool)

    if not result.success:
        await cl.Message(
            content=f"**Golden path execution failed** at step {result.failed_step}:\n\n"
            f"{result.failure_detail}\n\n"
            f"You can re-record and try refining again, or manually adjust the recording."
        ).send()
        return

    await cl.Message(
        content=f"**Golden path validated** — all {result.executed_steps} steps succeeded.\n\n"
        f"Now detecting variables for other users..."
    ).send()

    # Step 5: Variable detection
    var_prompt = refiner.generate_variable_detection_prompt()
    try:
        var_response = await _call_sub_agent_llm(var_prompt, model=WORKER_AGENT_MODEL)
        detected_vars = refiner.parse_variables(var_response)
    except Exception as e:
        detected_vars = []
        await send_toast(f"Variable detection had issues: {e}. Continuing with empty variables.", "warning")

    if detected_vars:
        _var_lines = ["**Detected variables for parameterization:**\n"]
        for v in detected_vars:
            _var_lines.append(f"- **{v['name']}** ({v['type']}) — {v['description']}")
            _var_lines.append(f"  Example: `{v.get('example', '')}`")
        await cl.Message(content="\n".join(_var_lines)).send()

        _var_approve = await cl.AskActionMessage(
            content="Accept these variables? (You can edit them later in golden_protocol.json)",
            actions=[
                cl.Action(name="var_approve", payload={"approve": "yes"}, label="Accept variables"),
                cl.Action(name="var_approve", payload={"approve": "no"}, label="Skip variables (none)"),
            ],
            timeout=60,
        ).send()

        if not _var_approve or _var_approve.get("approve") != "yes":
            detected_vars = []

    # Step 6: Produce golden protocol
    golden_path = refiner.produce_golden(
        variables=detected_vars,
        username=username,
    )

    await cl.Message(
        content=f"**Golden protocol created:** `{golden_path}`\n\n"
        f"- {len(refiner._golden_steps)} validated steps\n"
        f"- {len(detected_vars)} variables defined\n"
        f"- Execution mode: strict\n\n"
        f"Other users can now load this protocol in **Golden Execute** mode — "
        f"they fill in variables and the steps run exactly as validated."
    ).send()


# ── Play Mode Execution ──────────────────────────────────────────────────────


async def _play_mode_call_tool(tool_name: str, args: dict) -> Any:
    """Call an MCP tool directly for play mode (bypasses PEL and approval dialogs)."""
    lc_tools = cl.user_session.get("lc_tools", [])
    target = next((t for t in lc_tools if t.name == tool_name), None)
    if not target or not hasattr(target, "mcp_session") or not target.mcp_session:
        raise RuntimeError(f"Tool '{tool_name}' not available in current MCP sessions")

    username = cl.context.session.user.identifier
    checkpoint_tool_call(username, tool_name, args)

    result = await asyncio.wait_for(
        target.mcp_session.call_tool(tool_name, args),
        timeout=3600,
    )
    result_data = result.model_dump() if hasattr(result, "model_dump") else result
    checkpoint_tool_result(username, tool_name, args, result_data)
    return result_data


async def _run_play_mode(player: ProtocolPlayer, username: str):
    """Drive the play-mode execution loop."""
    step_count = 0
    while player.can_execute:
        step_result = await player.execute_next_step(_play_mode_call_tool)
        step_count += 1
        status_emoji = "+" if step_result.status == "success" else "x"
        await send_toast(
            f"[{status_emoji}] Step {step_result.step_number}: {step_result.tool_name} "
            f"({step_result.duration_ms:.0f}ms)",
            "success" if step_result.status == "success" else "error",
        )

        if player.is_waiting_slurm:
            # Save checkpoint and start polling
            _user_data = get_user_data_dir(username)
            checkpoint_path = player.save_checkpoint(_user_data)
            await cl.Message(
                content=f"**Play mode paused** — waiting for SLURM job "
                f"`{player.session.slurm_job_id}` to complete.\n\n"
                f"Checkpoint saved. Say **'resume play'** when the job finishes, "
                f"or I'll check automatically."
            ).send()
            # Start background polling
            asyncio.get_event_loop().create_task(
                _play_mode_slurm_poll(player, username)
            )
            return

        if player.is_paused:
            if player.session.mode == PlayMode.GOLDEN:
                await cl.Message(
                    content=f"**Golden execution paused** at step {step_result.step_number}: "
                    f"{step_result.deviation_detail}\n\n"
                    f"This is a strict golden protocol — the steps cannot change.\n"
                    f"To retry, provide corrected variables as `NAME=value` (one per line), "
                    f"then say **'retry step'**.\n"
                    f"Or say **'abort play'** to stop."
                ).send()
            else:
                await cl.Message(
                    content=f"**Play mode paused** at step {step_result.step_number}: "
                    f"{step_result.deviation_detail}\n\n"
                    f"Say **'resume play'** to continue past this error, or **'abort play'** to stop."
                ).send()
            return

        if player.is_failed:
            break

    # Execution complete or failed
    if player.is_complete or player.is_failed:
        report_path = player.save_report()
        report_content = player.generate_report()
        # Show a summary
        verdict = "COMPLETE" if player.is_complete else "FAILED"
        await cl.Message(
            content=f"**Play mode {verdict}** — {step_count} steps executed.\n\n"
            f"Report saved to: `{report_path}`\n\n"
            f"<details><summary>View Report</summary>\n\n{report_content}\n</details>"
        ).send()
        cl.user_session.set("protocol_player", None)


async def _play_mode_slurm_poll(player: ProtocolPlayer, username: str):
    """Background task: poll SLURM job until completion, then resume."""
    poll_interval = 60
    max_polls = 360  # 6 hours max
    for _ in range(max_polls):
        await asyncio.sleep(poll_interval)
        if player.session.state != PlayState.WAITING_SLURM:
            return  # Player was resumed manually or aborted

        try:
            job_status = await _play_mode_call_tool(
                "slurm_monitor_job", {"job_id": int(player.session.slurm_job_id)}
            )
            # Parse result
            finished = False
            status_str = "UNKNOWN"
            exit_code = -1
            if isinstance(job_status, dict):
                content = job_status.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            try:
                                data = json.loads(item.get("text", ""))
                                finished = data.get("finished", False)
                                status_str = data.get("status", "UNKNOWN")
                                exit_code = data.get("exit_code", -1)
                            except (json.JSONDecodeError, TypeError):
                                pass

            if finished:
                step_result = player.resume_from_slurm({
                    "finished": True, "status": status_str, "exit_code": exit_code
                })
                await send_toast(
                    f"SLURM job {player.session.slurm_job_id} completed: {status_str}",
                    "success" if step_result.status == "success" else "error",
                )
                if player.can_execute:
                    await _run_play_mode(player, username)
                elif player.is_failed or player.is_complete:
                    report_path = player.save_report()
                    await cl.Message(
                        content=f"**Play mode finished after SLURM job.**\n"
                        f"Report: `{report_path}`"
                    ).send()
                    cl.user_session.set("protocol_player", None)
                return
        except Exception as poll_err:
            print(f"[PLAY_MODE] Poll error: {poll_err}")
            continue


@cl.on_message
async def main(message: cl.Message):
    # ── Fresh Start: save history and clear for a new conversation ───
    # One-shot button (persistent=False). Saves current history to file,
    # clears session history, notifies user via toast.
    # If the user also typed a non-blank message, it is processed as a
    # fresh query (no prior history context).
    if getattr(message, "command", None) == "FreshStart":
        history = cl.user_session.get("history", [])
        _proto_recorder = cl.user_session.get("protocol_recorder")
        _proto_player = cl.user_session.get("protocol_player")
        result = await handle_fresh_start(
            history=history,
            username=cl.context.session.user.identifier,
            session_id=cl.context.session.id,
            work_dir=cl.user_session.get("work_dir"),
            project_name=cl.user_session.get("project_name"),
            session_facts=cl.user_session.get("session_facts", ""),
            plan_summary=cl.user_session.get("_active_plan_text", ""),
            protocol_recorder=_proto_recorder,
            protocol_player=_proto_player,
        )
        if result["should_clear_history"]:
            cl.user_session.set("history", [])
            # Reset to session-start state so first-turn project picker fires again
            cl.user_session.set("_turn_count", 0)
            cl.user_session.set("_project_confirmed", False)
            cl.user_session.set("project_name", "")
            cl.user_session.set("_first_turn_project", "")
            cl.user_session.set("session_facts", "")
            cl.user_session.set("_active_plan_text", "")
            cl.user_session.set("active_plan_path", "")
            cl.user_session.set("_pending_plan_path", "")
            cl.user_session.set("_plan_execution_pending", False)
            cl.user_session.set("_cached_conversation_summary", None)
            cl.user_session.set("_cached_summary_msg_count", None)
            cl.user_session.set("opus_sticky", False)
            cl.user_session.set("model_override", None)
            # Clear phased execution state so stale findings/plans don't carry over
            cl.user_session.set("active_findings_paths", [])
            cl.user_session.set("active_findings_path", "")
            cl.user_session.set("_pending_findings_path", "")
            cl.user_session.set("_unfulfilled_phase_gates", [])
            cl.user_session.set("_resume_from_phase", "")
            cl.user_session.set("guidance_mode", False)
        if _proto_recorder and _proto_recorder.active:
            cl.user_session.set("protocol_recorder", None)
        if _proto_player:
            cl.user_session.set("protocol_player", None)
        _cleared = result.get("cleared_project", "")
        _toast = f"✅ {result['message_count']} messages archived (project: {_cleared}) — starting fresh!" if _cleared else result["toast_message"]
        print(f"[FRESH_START] action={result['action']}, messages={result['message_count']}, cleared_project={_cleared}")
        await send_toast(_toast, result["toast_type"])
        # If user typed a non-blank message alongside the Fresh Start button,
        # process it as a new query (history is already cleared above).
        if message.content and message.content.strip():
            print(f"[FRESH_START] Processing accompanying message: {message.content!r}")
            await process_user_input(message.content)
        return

    # ── Protocol Recording: toggle start/stop for reproducibility ─────
    if getattr(message, "command", None) == "Protocol":
        _recorder = cl.user_session.get("protocol_recorder")
        username = cl.context.session.user.identifier
        session_id = cl.context.session.id
        _user_text = (message.content or "").strip()
        if _recorder and _recorder.is_active:
            # Toggle OFF — stop recording and compile
            try:
                output_dir = _recorder.stop()
                cl.user_session.set("protocol_recorder", None)
                await send_toast(f"Protocol saved: {output_dir.name}", "success")
                print(f"[PROTOCOL] Stopped recording, output: {output_dir}")
            except Exception as proto_err:
                await send_toast(f"Error stopping protocol: {proto_err}", "error")
                print(f"[PROTOCOL] Error on stop: {proto_err}")
            # If user typed a message alongside toggling off, process it
            if _user_text:
                try:
                    print(f"[PROTOCOL] Processing accompanying message after stop: {_user_text[:80]!r}")
                    await process_user_input(_user_text)
                except Exception as _query_err:
                    print(f"[PROTOCOL] Error processing query after stop: {_query_err}")
                    traceback.print_exc()
                    await cl.Message(content=f"Protocol saved, but an error occurred processing your message: {_query_err}").send()
        else:
            # Toggle ON — start recording
            # Always auto-generate name from project + timestamp
            _project_label = cl.user_session.get("project_name", "") or "session"
            _name = f"{_project_label}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

            _project = cl.user_session.get("project_name", "") or None
            _proto_dir = get_protocols_dir(username, _project)
            _proto_dir.mkdir(parents=True, exist_ok=True)
            _recorder = ProtocolRecorder(_proto_dir)
            try:
                _recorder.start(_name, username, session_id)
                cl.user_session.set("protocol_recorder", _recorder)
                await send_toast(f"Recording protocol: {_name}", "info")
                print(f"[PROTOCOL] Started recording: {_name}")
            except Exception as proto_err:
                await send_toast(f"Error starting protocol: {proto_err}", "error")
                print(f"[PROTOCOL] Error on start: {proto_err}")
                return
            # If user typed a query alongside enabling protocol, process it
            if _user_text:
                try:
                    print(f"[PROTOCOL] Processing accompanying query: {_user_text[:80]!r}")
                    await process_user_input(_user_text)
                except Exception as _query_err:
                    print(f"[PROTOCOL] Error processing query after start: {_query_err}")
                    traceback.print_exc()
                    await cl.Message(content=f"Protocol recording is active, but an error occurred: {_query_err}").send()
            else:
                # No query — just confirm protocol is ready
                await cl.Message(
                    content="Protocol recording started. All tool calls will be captured. Go ahead with your task."
                ).send()
        return

    # ── Play Mode: replay a recorded protocol ────────────────────────────
    if getattr(message, "command", None) == "Play":
        username = cl.context.session.user.identifier
        _project = cl.user_session.get("project_name", "") or None
        _proto_dir = get_protocols_dir(username, _project)
        _proto_dir.mkdir(parents=True, exist_ok=True)
        _global_proto_dir = get_protocols_dir(username)  # Always available as fallback
        _global_proto_dir.mkdir(parents=True, exist_ok=True)
        _selected_path = None

        # Step 1: Check if user provided a protocol path in the message text
        _msg_text = (message.content or "").strip()
        _detected_path = _detect_protocol_path(_msg_text)

        if _detected_path:
            # User gave a path — import it into project-scoped dir
            _imported = await _import_protocol_to_dir(_detected_path, _proto_dir)
            if _imported:
                _selected_path = _imported
                await cl.Message(content=f"Protocol imported from `{_detected_path}` → `{_imported.name}`").send()
            else:
                await send_toast(f"Path `{_detected_path}` is not a valid protocol (no recording.jsonl found).", "error")
                return
        else:
            # Step 2: Check for existing protocols (project-scoped + global)
            _available = list_available_protocols(_proto_dir)
            _global_available = []
            if _project and _global_proto_dir != _proto_dir:
                _global_available = list_available_protocols(_global_proto_dir)
            _all_available = _available + _global_available

            if not _all_available:
                # No protocols anywhere — ask user how they want to provide one
                _source_choice = await cl.AskActionMessage(
                    content=(
                        "**No recorded protocols found.** How would you like to provide one?\n\n"
                        "You can upload a protocol directory (as a zip/tar), "
                        "provide a file path, or record one first."
                    ),
                    actions=[
                        cl.Action(name="proto_source", payload={"source": "upload"}, label="Upload protocol file"),
                        cl.Action(name="proto_source", payload={"source": "path"}, label="Enter protocol path"),
                        cl.Action(name="proto_source", payload={"source": "cancel"}, label="Cancel"),
                    ],
                    timeout=120,
                ).send()

                if not _source_choice or _source_choice.get("source") == "cancel":
                    await send_toast("Play mode cancelled.", "info")
                    return

                if _source_choice.get("source") == "upload":
                    # Upload flow — similar to weights upload
                    _uploaded = await cl.AskFileMessage(
                        content=(
                            "**Upload a protocol file.**\n\n"
                            "Upload the `recording.jsonl` file (or a `.tar.gz`/`.zip` containing the protocol directory)."
                        ),
                        accept=[".jsonl", ".tar.gz", ".tgz", ".zip", ".gz"],
                        max_files=5,
                        max_size_mb=50,
                        timeout=300,
                    ).send()

                    if not _uploaded:
                        await send_toast("Upload cancelled.", "info")
                        return

                    _imported = await _import_uploaded_protocol(_uploaded, _proto_dir)
                    if _imported:
                        _selected_path = _imported
                        await cl.Message(content=f"Protocol imported → `{_imported.name}`").send()
                    else:
                        await send_toast("Could not import uploaded file as a protocol.", "error")
                        return

                elif _source_choice.get("source") == "path":
                    # Ask for path
                    _path_response = await cl.AskUserMessage(
                        content="Enter the path to the protocol directory (must contain `recording.jsonl`):",
                        timeout=120,
                    ).send()

                    if not _path_response:
                        await send_toast("Play mode cancelled.", "info")
                        return

                    _user_path = (_path_response.get("output", "") or "").strip()
                    _imported = await _import_protocol_to_dir(Path(_user_path), _proto_dir)
                    if _imported:
                        _selected_path = _imported
                        await cl.Message(content=f"Protocol imported from `{_user_path}` → `{_imported.name}`").send()
                    else:
                        await send_toast(f"Path `{_user_path}` is not a valid protocol directory.", "error")
                        return
            else:
                # Protocols available — show picker with project/global labels
                _actions = []
                # Project protocols first (most relevant)
                for p in _available[-8:]:
                    _label = f"{p['name']} v{p['version']} ({p['modified']})"
                    if _project:
                        _label = f"[{_project}] {_label}"
                    _actions.append(cl.Action(
                        name="select_protocol",
                        payload={"path": p["path"], "name": p["name"], "version": p["version"]},
                        label=_label,
                    ))
                # Global protocols (if in project context, show separately)
                for p in _global_available[-5:]:
                    _actions.append(cl.Action(
                        name="select_protocol",
                        payload={"path": p["path"], "name": p["name"], "version": p["version"]},
                        label=f"[global] {p['name']} v{p['version']} ({p['modified']})",
                    ))
                _actions.append(cl.Action(
                    name="select_protocol",
                    payload={"path": "__import__"},
                    label="Import from path or upload...",
                ))

                _picker_header = "**Select a protocol to replay:**"
                if _project:
                    _picker_header = f"**Select a protocol to replay** (project: {_project}):"

                _proto_select = await cl.AskActionMessage(
                    content=_picker_header,
                    actions=_actions,
                    timeout=120,
                ).send()

                if not _proto_select:
                    await send_toast("Play mode cancelled.", "info")
                    return

                if _proto_select.get("path") == "__import__":
                    # Import sub-flow
                    _import_choice = await cl.AskActionMessage(
                        content="How would you like to import a protocol?",
                        actions=[
                            cl.Action(name="import_method", payload={"method": "upload"}, label="Upload protocol file"),
                            cl.Action(name="import_method", payload={"method": "path"}, label="Enter protocol path"),
                        ],
                        timeout=60,
                    ).send()

                    if not _import_choice:
                        await send_toast("Play mode cancelled.", "info")
                        return

                    if _import_choice.get("method") == "upload":
                        _uploaded = await cl.AskFileMessage(
                            content="Upload the `recording.jsonl` or a `.tar.gz`/`.zip` of the protocol directory.",
                            accept=[".jsonl", ".tar.gz", ".tgz", ".zip", ".gz"],
                            max_files=5,
                            max_size_mb=50,
                            timeout=300,
                        ).send()
                        if not _uploaded:
                            return
                        _imported = await _import_uploaded_protocol(_uploaded, _proto_dir)
                        if _imported:
                            _selected_path = _imported
                        else:
                            await send_toast("Could not import uploaded file.", "error")
                            return
                    else:
                        _path_response = await cl.AskUserMessage(
                            content="Enter the path to the protocol directory:",
                            timeout=120,
                        ).send()
                        if not _path_response:
                            return
                        _user_path = (_path_response.get("output", "") or "").strip()
                        _imported = await _import_protocol_to_dir(Path(_user_path), _proto_dir)
                        if _imported:
                            _selected_path = _imported
                        else:
                            await send_toast(f"Invalid protocol path: `{_user_path}`", "error")
                            return
                else:
                    _selected_path = Path(_proto_select.get("path", ""))

        # Check if golden protocol exists — auto-offer golden mode
        _has_golden = has_golden_protocol(_selected_path)

        # Ask for mode
        _mode_actions = [
            cl.Action(name="play_mode", payload={"mode": "reproduce"}, label="Reproduce — verify same results (strict checks)"),
            cl.Action(name="play_mode", payload={"mode": "transfer"}, label="Transfer — apply method to new data (flexible)"),
            cl.Action(name="play_mode", payload={"mode": "refine"}, label="Refine — create golden protocol (author mode)"),
        ]
        if _has_golden:
            _mode_actions.insert(0, cl.Action(
                name="play_mode", payload={"mode": "golden"},
                label="Execute Golden Protocol — strict, fill variables only (recommended)"
            ))

        _mode_select = await cl.AskActionMessage(
            content="**Select replay mode:**" + (" *(Golden protocol available)*" if _has_golden else ""),
            actions=_mode_actions,
            timeout=60,
        ).send()

        if not _mode_select:
            await send_toast("Play mode cancelled.", "info")
            return

        _selected_mode = _mode_select.get("mode", "reproduce")

        # ── Refine mode: LLM-driven golden path extraction ──
        if _selected_mode == "refine":
            await _run_refine_mode(_selected_path, cl.context.session.user.identifier)
            return

        # ── Golden mode: strict execution with variable substitution ──
        if _selected_mode == "golden":
            _mode = PlayMode.GOLDEN
            _player = ProtocolPlayer(_selected_path, _mode)
            try:
                _player.load_golden()
            except Exception as load_err:
                await send_toast(f"Failed to load golden protocol: {load_err}", "error")
                return

            # Prompt for variables
            _golden_vars = _player.golden_variables
            _variables = {}
            if _golden_vars:
                _var_lines = ["**This golden protocol requires the following variables:**\n"]
                for v in _golden_vars:
                    req = " *(required)*" if v.get("required", True) else ""
                    _var_lines.append(f"- **{v['name']}** — {v['description']}{req}")
                    _var_lines.append(f"  Example: `{v.get('example', '')}`")
                await cl.Message(content="\n".join(_var_lines)).send()

                _var_response = await cl.AskUserMessage(
                    content="Enter variable values as `NAME=value`, one per line:",
                    timeout=300,
                ).send()
                if _var_response:
                    for line in (_var_response.get("output", "") or "").split("\n"):
                        if "=" in line:
                            k, v = line.split("=", 1)
                            _variables[k.strip()] = v.strip()

                # Validate required variables are provided
                _missing = [
                    v["name"] for v in _golden_vars
                    if v.get("required", True) and v["name"] not in _variables
                ]
                if _missing:
                    await send_toast(
                        f"Missing required variables: {', '.join(_missing)}. Aborting.", "error"
                    )
                    return

            # Set variables and start execution
            if _variables:
                _player.session.variables = _variables

            await cl.Message(
                content=f"**Executing golden protocol** — {_player.total_steps} steps, "
                f"strict mode. Variables: {len(_variables)} set.\n\n"
                f"*If a step fails, I will ask you to adjust a variable — the procedure will not change.*"
            ).send()

            cl.user_session.set("protocol_player", _player)
            # Skip preflight for golden (variables are the only requirement)
            _player.session.state = PlayState.EXECUTING
            _player.session.started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            await _run_play_mode(_player, cl.context.session.user.identifier)
            return

        # ── Standard modes (Reproduce / Transfer) ──
        _mode = PlayMode(_selected_mode)
        _player = ProtocolPlayer(_selected_path, _mode)

        try:
            _player.load()
        except Exception as load_err:
            await send_toast(f"Failed to load protocol: {load_err}", "error")
            return

        # Transfer mode: detect and prompt for variables
        _variables = {}
        if _mode == PlayMode.TRANSFER:
            _detected_vars = _player.detect_variables()
            if _detected_vars:
                _var_msg = "**Protocol uses variables. Provide substitutions:**\n"
                _var_msg += "\n".join(f"- `{{{{{v}}}}}`" for v in _detected_vars)
                await cl.Message(content=_var_msg).send()
                _var_response = await cl.AskUserMessage(
                    content="Enter variable values as `name=value`, one per line:",
                    timeout=180,
                ).send()
                if _var_response:
                    for line in _var_response.get("output", "").split("\n"):
                        if "=" in line:
                            k, v = line.split("=", 1)
                            _variables[k.strip()] = v.strip()

        # Run preflight
        _preflight = _player.run_preflight(variables=_variables if _variables else None)

        # Display preflight results
        _preflight_lines = [f"## Pre-flight Check — {_mode.value.title()} Mode\n"]
        if _preflight["blockers"]:
            _preflight_lines.append("### Blockers")
            for b in _preflight["blockers"]:
                _preflight_lines.append(f"- {b}")
        if _preflight["warnings"]:
            _preflight_lines.append("\n### Warnings")
            for w in _preflight["warnings"]:
                _preflight_lines.append(f"- {w}")
        if _preflight["passed"]:
            _preflight_lines.append(f"\n**All checks passed.** Starting execution of {_player.total_steps} steps...")
        else:
            _preflight_lines.append(f"\n**Pre-flight FAILED.** {len(_preflight['blockers'])} blocker(s) must be resolved.")

        await cl.Message(content="\n".join(_preflight_lines)).send()

        if not _preflight["passed"]:
            # Offer override in Reproduce mode
            if _mode == PlayMode.REPRODUCE:
                _override = await cl.AskActionMessage(
                    content="Override blockers and proceed anyway? (deviation will be logged in report)",
                    actions=[
                        cl.Action(name="override", payload={"override": "yes"}, label="Override and proceed"),
                        cl.Action(name="override", payload={"override": "no"}, label="Cancel"),
                    ],
                    timeout=60,
                ).send()
                if _override and _override.get("override") == "yes":
                    for b in _preflight["blockers"]:
                        _player.override_blocker(b)
                else:
                    return
            else:
                return

        cl.user_session.set("protocol_player", _player)
        await _run_play_mode(_player, username)
        return

    # ── Play mode text commands: resume/abort ──────────────────────────
    _msg_lower = (message.content or "").strip().lower()
    if _msg_lower in ("resume play", "resume playback"):
        _player = cl.user_session.get("protocol_player")
        if _player and (_player.is_paused or _player.is_waiting_slurm):
            if _player.is_paused:
                _player.resume_from_pause()
                await send_toast("Resuming play mode...", "info")
                await _run_play_mode(_player, cl.context.session.user.identifier)
            elif _player.is_waiting_slurm:
                # Check SLURM job now
                try:
                    _job_status = await _play_mode_call_tool(
                        "slurm_monitor_job", {"job_id": int(_player.session.slurm_job_id)}
                    )
                    await send_toast("Checking SLURM job status...", "info")
                except Exception as _e:
                    await send_toast(f"Could not check job: {_e}", "error")
        else:
            # Check for checkpoint files
            _checkpoints = detect_slurm_checkpoints(get_user_data_dir(cl.context.session.user.identifier))
            if _checkpoints:
                _player = ProtocolPlayer.resume_from_checkpoint(_checkpoints[0])
                cl.user_session.set("protocol_player", _player)
                await send_toast("Resumed from checkpoint. Checking SLURM job...", "info")
                await _run_play_mode(_player, cl.context.session.user.identifier)
            else:
                await send_toast("No active play session to resume.", "warning")
        return

    if _msg_lower in ("abort play", "cancel play", "stop play"):
        _player = cl.user_session.get("protocol_player")
        if _player:
            _player.abort()
            report_path = _player.save_report()
            cl.user_session.set("protocol_player", None)
            await cl.Message(content=f"Play mode aborted. Partial report saved: `{report_path}`").send()
        else:
            await send_toast("No active play session.", "info")
        return

    # ── Golden mode: retry step with updated variables ──────────────────
    if _msg_lower.startswith("retry step"):
        _player = cl.user_session.get("protocol_player")
        if not _player or not _player.is_paused or _player.session.mode != PlayMode.GOLDEN:
            await send_toast("No golden play session paused for retry.", "warning")
            return
        # Parse variable updates from the full message (not just lowered)
        _full_text = (message.content or "").strip()
        _var_lines = [l for l in _full_text.split("\n") if "=" in l and not l.lower().startswith("retry")]
        _updated_vars = {}
        for line in _var_lines:
            if "=" in line:
                k, v = line.split("=", 1)
                _updated_vars[k.strip()] = v.strip()
        if not _updated_vars:
            await cl.Message(
                content="Provide updated variables with the retry command, e.g.:\n```\nretry step\nDATA_PATH=/new/path\nPARTITION=gpushort\n```"
            ).send()
            return
        try:
            _player.retry_with_variables(_updated_vars)
            await send_toast(f"Retrying step with {len(_updated_vars)} updated variable(s)...", "info")
            await _run_play_mode(_player, cl.context.session.user.identifier)
        except Exception as _retry_err:
            await cl.Message(content=f"Retry failed: {_retry_err}").send()
        return

    # ── Web search: sync backend state with frontend toggle ──
    # Persistent button: command="Search" when highlighted, None when not.
    # Clear websearch_enabled when globe is not active so frontend/backend stay in sync.
    if getattr(message, "command", None) != "Search":
        if cl.user_session.get("websearch_enabled", False):
            cl.user_session.set("websearch_enabled", False)
            print("[WEBSEARCH] Globe OFF — disabled web search")

    # When persistent globe is ON and user sends a message,
    # message.command == "Search". We enable websearch and process normally.
    if getattr(message, "command", None) == "Search":
        cl.user_session.set("websearch_enabled", True)
        # Check if there's a pending websearch query from a previous WEBSEARCH_NEEDED response.
        # If the user enabled the globe in response to our request, inject the pending query
        # so the agent knows exactly what to search for.
        _pending_query = cl.user_session.get("pending_websearch_query", "")
        if message.content and message.content.strip():
            # Globe ON + real message → use the message content
            _input = message.content
            if _pending_query:
                # Prepend the pending query context so agent knows what was requested
                _input = (
                    f"{message.content}\n\n"
                    f"[Web search is now enabled. Previously I requested to search for: "
                    f"`{_pending_query}`. Please proceed with that search.]"
                )
                cl.user_session.set("pending_websearch_query", "")  # Clear after use
            print(f"[WEBSEARCH] Web search enabled, processing: {_input[:100]}")
            await process_user_input(_input)
        elif _pending_query:
            # Globe ON + empty message → user enabled globe in response to our WEBSEARCH_NEEDED
            # Re-run the original question with the pending query injected
            _input = (
                f"[Web search is now enabled. Please search for: `{_pending_query}` "
                f"and answer my original question.]"
            )
            cl.user_session.set("pending_websearch_query", "")  # Clear after use
            print(f"[WEBSEARCH] Globe enabled for pending query: {_pending_query}")
            await process_user_input(_input)
        else:
            # Globe ON but empty message and no pending query — ignore silently
            print("[WEBSEARCH] Globe active but empty message — ignoring")
        return

    # ── Guidance mode: compass button forces research→plan→approve ──
    if getattr(message, "command", None) == "Guidance":
        _current_guidance = cl.user_session.get("guidance_mode", False)
        cl.user_session.set("guidance_mode", not _current_guidance)
        _state = "ON" if not _current_guidance else "OFF"
        print(f"[GUIDANCE_MODE] Toggled to {_state}")
        await send_toast(f"Guidance mode {_state}", "info")
        if message.content and message.content.strip():
            await process_user_input(message.content)
        return

    # Globe is OFF — process normally (websearch_enabled persists until user toggles it off)
    await process_user_input(message.content)


async def _perform_project_switch(from_project: str, to_project: str, work_dir: str, parsed: dict) -> str:
    """Switch active project. Auto-saves outgoing state. Returns new project name."""
    from core.memory_state import save_project_state, resolve_project_name, register_project
    try:
        _history = cl.user_session.get("history", [])
        _recent_ctx = "\n".join(
            f"[{getattr(m, 'type', 'unknown')}]: {getattr(m, 'content', '')[:300]}"
            for m in _history[-6:]
        ) if _history else ""
        await save_project_state(
            work_dir=work_dir, project_name=from_project,
            session_facts=_recent_ctx,
            plan_summary=cl.user_session.get("_active_plan_text", ""),
        )
        print(f"[CONTEXT_SWITCH] Auto-saved state for '{from_project}'")
    except Exception as e:
        print(f"[CONTEXT_SWITCH] Failed to save state: {e}")

    _resolved = resolve_project_name(to_project)
    register_project(_resolved, description=parsed.get("reasoning", ""), work_dir=work_dir)
    cl.user_session.set("project_name", _resolved)
    print(f"[CONTEXT_SWITCH] Active project now: '{_resolved}'")
    _log_project_to_session(_resolved, work_dir)
    cl.user_session.set("active_findings_paths", [])
    cl.user_session.set("active_findings_path", "")
    cl.user_session.set("_pending_findings_path", "")
    cl.user_session.set("_pending_plan_path", "")
    cl.user_session.set("_unfulfilled_phase_gates", [])
    cl.user_session.set("_plan_execution_pending", False)
    await send_toast(f"\U0001f4c2 Switched to: {_resolved}", "info")
    return _resolved


async def _ask_project_confirmation(
    guess: str,
    current: str,
    known_projects: list,
    is_new_session: bool,
) -> str:
    """Show project confirmation dialog. Returns selected project name.

    Always asks user to confirm via clickable AskActionMessage.
    Never silently switches or creates projects.
    """
    from core.memory_state import _load_projects_index
    import re as _re

    actions = []

    # Load full project index for descriptions
    _index_data = _load_projects_index()
    _projects_meta = {p["name"]: p for p in _index_data.get("projects", [])}

    # Sort projects by recency — used for BOTH buttons and text display
    _sorted_projects = sorted(
        _index_data.get("projects", []),
        key=lambda p: p.get("date_last_active", ""),
        reverse=True,
    )
    _sorted_names = [p["name"] for p in _sorted_projects if p.get("name") != "general"]

    # Haiku's guess goes first (highlighted)
    if guess and guess in known_projects:
        _label = f"✓ {guess}"
        _desc = _projects_meta.get(guess, {}).get("description", "")
        if _desc:
            _label += f" ({_desc[:25]})"
        actions.append(cl.Action(
            name="project_select", payload={"project": guess},
            label=_label,
        ))
    elif guess and guess not in known_projects:
        actions.append(cl.Action(
            name="project_select", payload={"project": guess},
            label=f"+ Create: {guess}",
        ))

    # Remaining projects sorted by recency (up to 6, excluding guess and general)
    for p in _sorted_names[:6]:
        if p != guess:
            _desc = _projects_meta.get(p, {}).get("description", "")
            _label = p
            if _desc:
                _label += f" ({_desc[:25]})"
            actions.append(cl.Action(
                name="project_select", payload={"project": p},
                label=_label,
            ))

    # "general" as a low-priority option (always available, doesn't steal a project slot)
    if guess != "general":
        actions.append(cl.Action(
            name="project_select", payload={"project": "general"},
            label="general",
        ))

    # Always offer "New project" at the end (unless guess is already a new project)
    if not (guess and guess not in known_projects):
        actions.append(cl.Action(
            name="project_select", payload={"project": "__new__"},
            label="+ New project",
        ))

    # Build message
    if guess and not is_new_session:
        msg = f"Switching project context to **{guess}**. Confirm?"
    elif guess:
        msg = f"I think this is the **{guess}** project. Is that right?"
    else:
        msg = "Which project should I use for this?"

    # Show project details for context (reuses _sorted_projects computed above)
    if _sorted_projects:
        _detail_lines = [msg + "\n"]
        for _p in _sorted_projects[:6]:
            _pname = _p["name"]
            _pdesc = _p.get("description", "")
            _pdate = _p.get("date_last_active", _p.get("date_created", ""))
            _line = f"- **{_pname}**"
            if _pdesc:
                _line += f" — {_pdesc}"
            if _pdate:
                _line += f" _(last active: {_pdate})_"
            _detail_lines.append(_line)
        msg = "\n".join(_detail_lines) + "\n"

    # Loop until user explicitly selects — no silent timeout default
    res = None
    while res is None:
        res = await cl.AskActionMessage(
            content=msg,
            actions=actions,
            timeout=300,
        ).send()
        if res is None:
            msg = "👆 Please select a project to continue."

    # Parse response
    _selected = guess or "general"
    if isinstance(res, dict):
        _payload = res.get("payload", res)
        _selected = _payload.get("project", guess or "general") if isinstance(_payload, dict) else (guess or "general")
    elif hasattr(res, "payload"):
        _selected = getattr(res, "payload", {}).get("project", guess or "general")

    # Handle "new project" selection
    if _selected == "__new__":
        _name_res = await cl.AskUserMessage(
            content="What should I call the new project? (short name, e.g. 'p53_folding')",
            timeout=60,
        ).send()
        if _name_res:
            _raw = _name_res.get("output", "") if isinstance(_name_res, dict) else str(
                _name_res.content if hasattr(_name_res, "content") else _name_res
            )
            _new_name = _re.sub(r"[^a-z0-9_-]", "_", _raw.strip().lower())
            _selected = _new_name if _new_name else "general"
        else:
            _selected = "general"

    return _selected


async def process_user_input(input_content: str):

    # ── Turn counter for tiered post-processing ──
    _turn_count = cl.user_session.get("_turn_count", 0) + 1
    cl.user_session.set("_turn_count", _turn_count)

    # ── First-turn project confirmation (always asks user) ──────
    if _turn_count == 1 and not cl.user_session.get("_project_confirmed"):
        from core.memory_state import (
            get_known_project_names, register_project, resolve_project_name,
        )
        _known = get_known_project_names()
        _selected = await _ask_project_confirmation(
            guess=None,
            current="",
            known_projects=_known,
            is_new_session=True,
        )
        _resolved = resolve_project_name(_selected)
        if _resolved and _resolved != "general":
            _work_dir = cl.user_session.get("work_dir", "")
            register_project(_resolved, work_dir=_work_dir)
        cl.user_session.set("project_name", _resolved or "general")
        cl.user_session.set("_first_turn_project", _resolved or "general")
        cl.user_session.set("_project_confirmed", True)
        _log_project_to_session(_resolved or "general", cl.user_session.get("work_dir", ""))
        await send_toast(f"📂 Project: {_resolved or 'general'}", "info")
        print(f"[PROJECT_SELECT] First-turn selection: '{_resolved or 'general'}'")

    history = cl.user_session.get("history", [])
    history.append(HumanMessage(content=input_content))
    username = cl.context.session.user.identifier
    session_id = cl.context.session.id

    # ── Incremental session log: persist user message immediately ───
    _log_path = cl.user_session.get("session_log_path")
    if _log_path:
        session_log_append(Path(_log_path), "user", input_content)

    llm = cl.user_session.get("llm")
    tools = cl.user_session.get("lc_tools", [])
    if not llm:
        await cl.Message("LLM not ready...").send()
        return

    # P1+P3 FIX: Define metadata once (was commented out in P3 but still referenced)
    metadata = {
        "work_dir": cl.user_session.get("work_dir"),
        "project_name": cl.user_session.get("project_name"),
    }

    # ── Bedrock blank-message guard ───────────────────────────────────────
    # Sanitize history before EVERY call to execute_skill_based_turn().
    history = sanitize_history(history)

    # ── Create cost tracker for this turn ─────────────────────────────
    cost_tracker = CostTrackingCallback()

    # First (normal) attempt
    try:
        updated_history = await execute_skill_based_turn(
            llm=llm,
            history=history,
            input_content=input_content,
            tools=tools,
            username=username,
            session_id=session_id,
            cost_tracker=cost_tracker
        )

        # Success path — or early-return path (WEBSEARCH_NEEDED / StuckInterrupt handled internally)
        # execute_skill_based_turn returns None when it handled the response itself
        # (e.g. WEBSEARCH_NEEDED approval gate, StuckInterrupt mid-loop handler).
        # In that case, the message was already sent — just return cleanly.
        if updated_history is None:
            return

        metadata["session_type"] = "completed"
        metadata["final_message_count"] = len(updated_history)

        # ── Display cost summary to user ──────────────────────────────
        try:
            # Native executor stores its cost tracker on session after execution
            _native_ct = cl.user_session.get("_last_native_cost_tracker")
            if _native_ct and _native_ct.num_calls > 0:
                cost_line = _native_ct.format_cost_line()
            else:
                cost_line = cost_tracker.format_cost_line()
            if cost_line:
                await cl.Message(content=f"---\n{cost_line}").send()
            # Clear for next turn
            cl.user_session.set("_last_native_cost_tracker", None)
        except Exception as e:
            print(f"[COST_TRACK] Failed to display cost: {e}")

        cl.user_session.set("history", updated_history)

        return

    except Exception as general_error:
        error_str = f"{type(general_error).__name__}: {general_error}".lower()
        traceback.print_exc()

        # Error classification uses core.agent_utils.classify_error
        error_type = classify_error(error_str)

        # Save checkpoint with error classification (sync — no await)
        checkpoint_tool_error(
            username=username,
            tool_name="skill_agent",
            args={"input": input_content, "history_length": len(history)},
            error=str(general_error)
        )

        # Retry params from core.agent_utils.get_retry_params
        retry_params = get_retry_params(error_type)
        if retry_params["should_retry"]:
            max_retries = retry_params["max_retries"]
            base_delay = retry_params["base_delay"]
            attempt = 0
            current_history = history.copy()

            while attempt < max_retries:
                attempt += 1
                wait_sec = base_delay * (2 ** (attempt - 1))

                if error_type == "blank_content":
                    # ── BLANK CONTENT FIX: Sanitize and retry immediately ──
                    old_len = len(current_history)
                    current_history = sanitize_history(current_history)
                    new_len = len(current_history)
                    wait_sec = 1  # Override: no need to wait for data issues
                    if old_len != new_len:
                        msg = (f"🔧 Detected blank message in history. "
                               f"Removed {old_len - new_len} poisoned message(s). "
                               f"Retrying immediately (attempt {attempt}/{max_retries})...")
                        print(f"[BLANK_CONTENT_FIX] Sanitized history: {old_len} -> {new_len} messages")
                    else:
                        current_history = current_history[-5:]
                        msg = (f"🔧 Blank content error persists after sanitization. "
                               f"Reducing history to {len(current_history)} messages. "
                               f"Retrying (attempt {attempt}/{max_retries})...")
                        print(f"[BLANK_CONTENT_FIX] Sanitize found nothing — reducing history to {len(current_history)} msgs")

                elif error_type == "context_limit":  # FIX: reduce on EVERY attempt
                    old_len = len(current_history)

                    if attempt == 1:
                        await async_truncate_oversized_messages(current_history, max_tokens_per_message=MAX_SINGLE_MESSAGE_TOKENS)
                        current_history = current_history[-max(5, len(current_history) // 10):]
                    elif attempt == 2:
                        await async_truncate_oversized_messages(current_history, max_tokens_per_message=MAX_SINGLE_MESSAGE_TOKENS // 2)
                        current_history = current_history[-5:]
                    elif attempt == 3:
                        await async_truncate_oversized_messages(current_history, max_tokens_per_message=MAX_SINGLE_MESSAGE_TOKENS // 4)
                        current_history = current_history[-2:]
                    else:
                        await async_truncate_oversized_messages(current_history, max_tokens_per_message=5000)
                        current_history = current_history[-2:]

                    # FINAL GUARD: Enforce total token budget
                    current_history = enforce_total_token_budget(
                        current_history, max_total_tokens=BEDROCK_HARD_TOKEN_LIMIT
                    )

                    total_est = sum(estimate_tokens(getattr(m, 'content', '')) for m in current_history)
                    msg = (f"⚠️ Context length exceeded. Reduced from {old_len} → {len(current_history)} messages "
                           f"(~{total_est} tokens, budget: {BEDROCK_HARD_TOKEN_LIMIT}). "
                           f"Retrying in {wait_sec}s (attempt {attempt}/{max_retries})...")
                else:
                    msg = f"⚠️ {error_type.replace('_', ' ').title()} — Retrying in {wait_sec}s (attempt {attempt}/{max_retries})..."

                await cl.Message(content=msg).send()

                # Save retry attempt checkpoint (sync)
                checkpoint_tool_call(
                    username=username,
                    tool_name=f"retry_attempt_{attempt}",
                    args={
                        "original_error": str(general_error),
                        "attempt": attempt,
                        "wait_seconds": wait_sec,
                        "history_length": len(current_history)
                    }
                )

                await asyncio.sleep(wait_sec)

                try:
                    # FIX: Use original query, NOT bloated retry prompt
                    retry_input = input_content

                    # Bedrock blank-message guard on retry
                    current_history = sanitize_history(current_history)

                    # Create fresh cost tracker for retry
                    retry_cost_tracker = CostTrackingCallback()

                    updated_history = await execute_skill_based_turn(
                        llm=llm,
                        history=current_history,
                        input_content=retry_input,
                        tools=tools,
                        username=username,
                        session_id=session_id,
                        cost_tracker=retry_cost_tracker
                    )

                    # Success after retry - save success checkpoint (sync)
                    # Guard: if execute_skill_based_turn returned None, it handled
                    # the response itself (WEBSEARCH_NEEDED / StuckInterrupt).
                    if updated_history is None:
                        return

                    checkpoint_tool_result(
                        username=username,
                        tool_name=f"retry_success_{attempt}",
                        args={
                            "original_error": str(general_error),
                            "successful_attempt": attempt,
                            "total_wait_time": sum(base_delay * (2 ** i) for i in range(attempt))
                        },
                        result={"status": "completed", "retry_count": attempt}
                    )

                    # Success after retry
                    metadata["session_type"] = "completed_after_retry"
                    metadata["final_message_count"] = len(updated_history)
                    cl.user_session.set("history", updated_history)

                    # ── Display cost summary after retry success ──────────
                    try:
                        _native_ct = cl.user_session.get("_last_native_cost_tracker")
                        if _native_ct and _native_ct.num_calls > 0:
                            cost_line = _native_ct.format_cost_line()
                        else:
                            cost_line = retry_cost_tracker.format_cost_line()
                        if cost_line:
                            await cl.Message(content=f"---\n{cost_line}").send()
                        cl.user_session.set("_last_native_cost_tracker", None)
                    except Exception as e:
                        print(f"[COST_TRACK] Failed to display cost: {e}")

                    return

                except Exception as retry_exc:
                    # Save failed retry checkpoint (sync)
                    checkpoint_tool_error(
                        username=username,
                        tool_name=f"retry_failed_{attempt}",
                        args={
                            "original_error": str(general_error),
                            "retry_error": str(retry_exc),
                            "attempt": attempt
                        },
                        error=str(retry_exc)
                    )

                    # Save even on failed retry attempts
                    print(f"[SAFETY] Failed retry attempt {attempt} — session_log has the record")

                    if attempt == max_retries:
                        general_error = retry_exc
                        break

        # Final failure path — show user-friendly message for budget errors
        if error_type == "budget_exceeded":
            error_msg = (
                "💰 **Budget Exceeded** — Your LiteLLM virtual key budget has been exhausted.\n\n"
                "To continue, please:\n"
                "1. Restart your session (this will generate a new virtual key)\n"
                "2. Or ask your admin to increase the budget limit"
            )
        else:
            error_msg = f"⚠️ Processing failed after retries: {str(general_error)}"
        history.append(AIMessage(content=error_msg))
        await cl.Message(content=error_msg).send()

        metadata["errored"] = True
        metadata["error"] = str(general_error)
        metadata["session_type"] = "failed"

        await cl.Message(
            "You can resume later by saying:\n"
            "• continue where we left off\n"
            "• resume last query"
        ).send()

@cl.on_chat_end
async def on_chat_end():
    """Mark session log as ended (instant, <10ms). Curation deferred to next session start."""
    try:
        username = cl.context.session.user.identifier

        _log_path = cl.user_session.get("session_log_path")
        if _log_path:
            _project_name = cl.user_session.get("project_name", "")
            session_log_append(
                Path(_log_path), "system",
                "__SESSION_ENDED__",
                metadata={
                    "type": "session_ended",
                    "project": _project_name,
                    "work_dir": cl.user_session.get("work_dir", ""),
                },
            )
            print(f"[SESSION_END] Marked session log as ended (curation deferred to next start)")

        from core.session_log import cleanup_old_session_logs
        cleanup_old_session_logs(username, keep_days=30)

    except Exception as e:
        print(f"[ERROR] on_chat_end failed: {e}")
