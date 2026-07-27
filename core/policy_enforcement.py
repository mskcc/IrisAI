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

"""
core/policy_enforcement.py

Policy Enforcement Layer (PEL) for IrisAI.

Universal middleware that intercepts ALL tool calls before execution.
Rules are defined in config/policy.yaml — NOT in individual tool implementations.

Architecture:
    LLM decides to call tool
        → PEL.check(tool_name, tool_input)
            → RequiredParamsChecker
            → BudgetChecker
            → PatternChecker
            → PreconditionChecker
        → if all pass: execute tool
        → if any fail: return PolicyViolation to LLM (tool never runs)

Design Principles:
    1. Universal — applies to ALL tools via single hook point in MCPTool._arun()
    2. Declarative — rules in YAML, not imperative code per tool
    3. Deterministic — 100% enforcement, not probabilistic prompt compliance
    4. Auditable — every blocked call is logged with reason
    5. Extensible — new rules = edit YAML, no code changes

Usage:
    # In app.py, initialize once:
    pel = PolicyEnforcementLayer()

    # In MCPTool._arun(), before mcp_session.call_tool():
    result = pel.check(tool_name, tool_input)
    if not result.allowed:
        return result.to_tool_error()  # structured error back to LLM

    # At start of each user turn:
    pel.reset_turn()
"""

import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, Any

try:
    import yaml
except ImportError:
    yaml = None  # Graceful degradation if yaml not available

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PolicyViolation:
    """A single policy violation — returned when a check fails."""
    tool_name: str
    rule_type: str          # "required_param", "budget", "pattern", "precondition"
    reason: str
    suggestion: str = ""
    is_hard_block: bool = True  # False = warning only, tool still executes
    matched_pattern: str = ""   # The exact pattern/command that triggered this violation
    matched_input: str = ""     # The actual input snippet that matched (e.g. full pip install command)

    def format_error(self) -> str:
        """Format as structured error message for the LLM."""
        lines = [
            f"⛔ POLICY VIOLATION [{self.rule_type.upper()}]",
            f"Tool: {self.tool_name}",
            f"Reason: {self.reason}",
        ]
        if self.suggestion:
            lines.append(f"Suggestion: {self.suggestion}")
        lines.append("")
        lines.append("ACTION REQUIRED: Stop current approach. Do NOT retry this call.")
        lines.append("Report the issue to the user and ask for guidance.")
        return "\n".join(lines)


@dataclass
class PolicyCheckResult:
    """Result of running all policy checks for a tool call."""
    allowed: bool
    violations: list = field(default_factory=list)   # hard blocks
    warnings: list = field(default_factory=list)     # soft warnings

    def to_tool_error(self) -> str:
        """Format the first violation as a tool error string."""
        if self.violations:
            return self.violations[0].format_error()
        return "Policy check failed (unknown reason)"

    def get_warning_text(self) -> str:
        """Format warnings as advisory text (injected into context)."""
        if not self.warnings:
            return ""
        lines = ["⚠️ POLICY WARNINGS:"]
        for w in self.warnings:
            lines.append(f"  - [{w.rule_type}] {w.reason}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE GATES (single-executor enforcement)
# ═══════════════════════════════════════════════════════════════════════════════

PHASE_GATED_TOOLS = frozenset({
    "edit_file", "write_text_file", "remove_file", "batch_file_edit",
    "submit_slurm_job",
    "run_pipeline_script", "submit_alphafold3_job",
})

RESEARCH_GATED_TOOLS = PHASE_GATED_TOOLS

PLAN_GATED_TOOLS = PHASE_GATED_TOOLS

_WRITE_INDICATORS = [
    re.compile(r'[^<\d]>(?![&>])'),
    re.compile(r'>>'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?rm\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?rmdir\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?mv\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?cp\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?chmod\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?chown\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?truncate\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?tee\b'),
    re.compile(r'(?:^|[\s;&|])(?:sudo\s+)?dd\b'),
    re.compile(r'\bsed\b[^|;]*-i'),
    re.compile(r'\bgit\s+(?:reset|push|rebase|merge|commit|stash\s+drop|branch\s+-[dD]|checkout\s+--)'),
    re.compile(r'(?:os\.remove|os\.unlink|shutil\.rmtree|shutil\.move)\b'),
    re.compile(r'open\s*\([^)]*["\']w'),
]

# Patterns that look like writes but are actually safe (stripped before checking)
_SAFE_REDIRECT_PATTERN = re.compile(r'\d*>{1,2}\s*/dev/null')
_SAFE_FD_DUP_PATTERN = re.compile(r'\d*>&\d+')


def _is_readonly_command(tool_input: dict) -> bool:
    """Check whether an execute_dynamic_task call contains only read-only commands.

    Extracts the 'command' parameter and scans for write indicators (redirects,
    destructive commands, git mutations, etc.). Returns True if no write patterns
    are detected.

    Safe patterns like 2>/dev/null and 2>&1 are stripped before checking.
    """
    commands = tool_input.get("command", "") or tool_input.get("commands", "") or ""
    if not commands:
        return True

    # Strip safe redirects that are not actual write operations
    cleaned = _SAFE_REDIRECT_PATTERN.sub('', commands)
    cleaned = _SAFE_FD_DUP_PATTERN.sub('', cleaned)

    for pattern in _WRITE_INDICATORS:
        if pattern.search(cleaned):
            return False
    return True


@dataclass
class PhaseGate:
    """A gate that blocks destructive tools until a required tool is called.

    Configured per-turn based on Haiku's needs_research/needs_planning flags.
    When active, the model must call the required_tool before any tool in
    blocked_tools can execute. Once satisfied, the gate is permanently open
    for the remainder of the session (across turns).

    readonly_pass_tools: tools that bypass the gate IF their input passes
    the read-only check (used to allow execute_dynamic_task for research).
    """
    required_tool: str
    blocked_tools: frozenset = field(default_factory=lambda: PHASE_GATED_TOOLS)
    readonly_pass_tools: frozenset = field(default_factory=frozenset)
    satisfied: bool = False
    message: str = ""

    def check(self, tool_name: str, tool_input: dict = None) -> Optional[PolicyViolation]:
        """Check if this gate blocks the given tool call."""
        if self.satisfied:
            return None
        if tool_name in self.readonly_pass_tools:
            if tool_input is not None and _is_readonly_command(tool_input):
                return None
            return PolicyViolation(
                tool_name=tool_name,
                rule_type="phase_gate",
                reason=(
                    f"{self.message} Write operations via {tool_name} are blocked "
                    f"until {self.required_tool} is called. Read-only commands "
                    f"(ls, cat, grep, git status, find) are allowed."
                ),
                suggestion=(
                    f"Use only read-only commands (ls, cat, grep, git log, find, head, etc.) "
                    f"to gather information, then call {self.required_tool}."
                ),
                is_hard_block=True,
            )
        if tool_name not in self.blocked_tools:
            return None
        return PolicyViolation(
            tool_name=tool_name,
            rule_type="phase_gate",
            reason=self.message,
            suggestion=f"Call {self.required_tool} first to satisfy this requirement.",
            is_hard_block=True,
        )

    def satisfy(self):
        """Mark this gate as satisfied (required tool was called)."""
        self.satisfied = True


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_matched_snippet(combined_input: str, pattern: str, max_len: int = 120) -> str:
    """Extract the actual command snippet around a matched pattern for display in approval dialogs.

    For 'pip install' patterns, extracts the full install command so the user
    can see exactly which packages are being installed.

    Examples:
        combined_input = "pip install plotly kaleido==0.2.1"
        pattern = "pip install"
        → returns "pip install plotly kaleido==0.2.1"
    """
    idx = combined_input.find(pattern)
    if idx == -1:
        return ""
    # Take from the match start to end of line (or max_len chars)
    snippet = combined_input[idx:idx + max_len]
    # Trim at newline if present
    newline_pos = snippet.find("\n")
    if newline_pos != -1:
        snippet = snippet[:newline_pos]
    return snippet.strip()


class PolicyEnforcementLayer:
    """
    Universal middleware for all tool calls in IrisAI.

    Intercepts every tool call BEFORE execution. Checks against:
    1. Required parameters (must be present)
    2. Tool call budgets (per-turn limits)
    3. Blocked patterns (forbidden commands)
    4. Preconditions (files must exist, etc.)

    If any check fails with is_hard_block=True, the tool does NOT execute.
    The LLM receives a structured error explaining what went wrong and what to do.
    """

    def __init__(
        self,
        policy_path: str = "config/policy.yaml",
        environment_path: str = "config/environment.yaml",
    ):
        self.policy_path = policy_path
        self.environment_path = environment_path
        self.policy = self._load_yaml(policy_path, "policy")
        self.environment = self._load_yaml(environment_path, "environment")
        self._turn_call_counts: dict[str, int] = {}
        self._session_call_counts: dict[str, int] = {}  # lifetime counts
        self._total_turn_calls: int = 0  # global counter across ALL tools
        self._last_tool_name: str = ""  # for consecutive-call detection
        self._consecutive_count: int = 0  # how many times in a row same tool called
        self._budget_warnings_delivered: set[str] = set()  # tools that got warn_at message
        self._turn_violations: list = []  # violations this turn (for retry detection)
        self._audit_log_path = (
            self.policy.get("audit", {}).get("log_file", "logs/policy_audit.jsonl")
        )
        # Protocol execution lock (set externally when protocol mode is active)
        self._protocol_lock = None  # type: Optional[Any]
        # Approval tracking: patterns the user has approved this session
        self._approved_patterns: set[str] = set()
        # Phase gates: block destructive tools until research/planning is done
        self._phase_gates: list[PhaseGate] = []
        # Awaiting-approval lock: blocks execution tools after write_plan/write_findings
        # within the same turn, forcing the LLM to present results and stop.
        # Cleared by reset_turn() at the start of the next user message.
        self._awaiting_approval: bool = False
        self._awaiting_approval_tool: str = ""

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def approve_pattern(self, pattern: str) -> None:
        """Mark a blocked pattern as user-approved for this session.

        Once approved, the pattern will no longer trigger a violation.
        Approval persists for the entire session (until process restart).
        """
        self._approved_patterns.add(pattern)
        logger.info(f"[PEL] Pattern approved by user: '{pattern}'")

    def is_pattern_approved(self, pattern: str) -> bool:
        """Check if a pattern has been approved by the user."""
        return pattern in self._approved_patterns

    # ── Phase Gate API ─────────────────────────────────────────────────────────

    def set_phase_gates(self, gates: list) -> None:
        """Configure phase gates for this turn based on Haiku classification.

        Gates block destructive tools until required tools (write_findings,
        write_plan) are called. Once satisfied, gates remain open across turns.
        """
        self._phase_gates = gates
        active = [g.required_tool for g in gates if not g.satisfied]
        if active:
            logger.info(f"[PEL] Phase gates active: {active}")

    def satisfy_gate(self, tool_name: str) -> None:
        """Mark a phase gate as satisfied (required tool was called).

        Called when write_findings or write_plan completes successfully.
        """
        for gate in self._phase_gates:
            if gate.required_tool == tool_name and not gate.satisfied:
                gate.satisfy()
                logger.info(f"[PEL] Phase gate satisfied: {tool_name}")

    def clear_phase_gates(self) -> None:
        """Clear all phase gates (e.g., when resuming with plan already on disk)."""
        for gate in self._phase_gates:
            gate.satisfied = True
        if self._phase_gates:
            logger.info("[PEL] All phase gates cleared (resumption)")

    def get_active_phase_gates(self) -> list:
        """Return list of unsatisfied phase gates (for diagnostics)."""
        return [g for g in self._phase_gates if not g.satisfied]

    def _normalize_tool_input(
        self, tool_name: str, tool_input: dict
    ) -> tuple:
        """Recover actual params from an unparsed kwargs wrapper.

        When upstream json.loads fails, tool_input arrives as
        {"kwargs": "<json string>"}. This attempts a last-resort parse
        so downstream checks can access named parameters.

        Returns:
            (normalized_dict, kwargs_opaque) — kwargs_opaque is True when
            the kwargs string could not be parsed at all.
        """
        if not (len(tool_input) == 1 and "kwargs" in tool_input
                and isinstance(tool_input["kwargs"], str)):
            return tool_input, False

        raw = tool_input["kwargs"]
        for strict in (True, False):
            try:
                parsed = json.loads(raw, strict=strict)
                if isinstance(parsed, dict):
                    logger.info(
                        f"[PEL] Recovered kwargs for '{tool_name}' "
                        f"(strict={strict})"
                    )
                    return parsed, False
            except json.JSONDecodeError as e:
                # Prefix extraction: if valid JSON ends before trailing garbage
                if hasattr(e, 'pos') and e.pos > 2:
                    try:
                        parsed = json.loads(raw[:e.pos], strict=strict)
                        if isinstance(parsed, dict):
                            logger.info(
                                f"[PEL] Recovered kwargs for '{tool_name}' "
                                f"via prefix extraction at pos {e.pos}"
                            )
                            return parsed, False
                    except (json.JSONDecodeError, ValueError):
                        pass
                continue
            except ValueError:
                continue

        logger.warning(
            f"[PEL] Cannot parse kwargs for '{tool_name}' — "
            f"param-based checks will be skipped. "
            f"preview={raw[:200]!r}"
        )
        return tool_input, True

    def check(self, tool_name: str, tool_input: dict) -> PolicyCheckResult:
        """
        Run ALL policy checks for a tool call.

        Call this BEFORE executing any tool. If result.allowed is False,
        do NOT execute the tool — return result.to_tool_error() to the LLM.

        Args:
            tool_name: Name of the tool being called
            tool_input: The arguments/parameters for the tool call

        Returns:
            PolicyCheckResult with .allowed bool and .violations/.warnings lists
        """
        violations = []
        warnings = []

        # Normalize tool_input (recover from upstream kwargs parse failure)
        tool_input, kwargs_opaque = self._normalize_tool_input(tool_name, tool_input)

        # Track consecutive calls FIRST — must update before any early return
        # so that a blocked intermediate tool still resets the consecutive counter.
        if tool_name == self._last_tool_name:
            self._consecutive_count += 1
        else:
            if self._last_tool_name and self._consecutive_count > 1:
                logger.debug(
                    f"[PEL] Consecutive counter reset: {self._last_tool_name} "
                    f"({self._consecutive_count}x) broken by {tool_name}"
                )
            self._last_tool_name = tool_name
            self._consecutive_count = 1

        # -1. Protocol compliance (highest priority — independent of policy.yaml)
        protocol_violation = self._check_protocol_compliance(tool_name, tool_input)
        if protocol_violation:
            violations.append(protocol_violation)
            # Hard stop immediately — no further checks needed
            return PolicyCheckResult(allowed=False, violations=violations, warnings=warnings)

        # -0.5. Phase gates (block destructive tools until research/planning done)
        # Gates are sequential: only the first unsatisfied gate can block.
        # This allows read-only commands during research even when a plan gate exists.
        for gate in self._phase_gates:
            if gate.satisfied:
                continue
            gate_violation = gate.check(tool_name, tool_input)
            if gate_violation:
                violations.append(gate_violation)
                return PolicyCheckResult(allowed=False, violations=violations, warnings=warnings)
            break

        # If this tool call satisfies a phase gate, mark it AND lock execution
        # until the next turn (forces LLM to present plan/findings to user first)
        for gate in self._phase_gates:
            if tool_name == gate.required_tool and not gate.satisfied:
                gate.satisfy()
                self._awaiting_approval = True
                self._awaiting_approval_tool = tool_name
                logger.info(f"[PEL] Phase gate '{gate.required_tool}' satisfied — "
                            f"awaiting user approval before execution")

        # -0.4. Awaiting-approval lock: after write_plan/write_findings satisfies
        # a gate, block execution tools for the rest of this turn. The LLM must
        # present results to the user and stop. Cleared by reset_turn().
        if self._awaiting_approval and tool_name in PHASE_GATED_TOOLS:
            _phase_label = "plan" if "plan" in self._awaiting_approval_tool else "findings"
            violations.append(PolicyViolation(
                tool_name=tool_name,
                rule_type="awaiting_approval",
                reason=(
                    f"You just wrote the {_phase_label}. Present it to the user and "
                    f"ask for their approval before proceeding with execution."
                ),
                suggestion=(
                    f"Show the {_phase_label} content to the user and ask: "
                    f"'Does this look good? Say go ahead to execute, or let me know "
                    f"what you'd like to change.'"
                ),
                is_hard_block=True,
            ))
            return PolicyCheckResult(allowed=False, violations=violations, warnings=warnings)

        if not self.policy:
            # No policy loaded — allow everything (graceful degradation)
            return PolicyCheckResult(allowed=True)

        # 0. Global turn cap (across ALL tools)
        if not violations:
            global_violation = self._check_global_budget()
            if global_violation:
                if global_violation.is_hard_block:
                    violations.append(global_violation)
                else:
                    warnings.append(global_violation)

        # 0b. Consecutive call detection (same tool called N times in a row)
        if not violations:  # skip if already blocked by global cap
            consec_violation = self._check_consecutive(tool_name)
            if consec_violation:
                if consec_violation.is_hard_block:
                    violations.append(consec_violation)
                else:
                    warnings.append(consec_violation)

        # 1. Required parameters
        if not violations:  # skip expensive checks if already blocked
            violations.extend(self._check_required_params(tool_name, tool_input, kwargs_opaque))

        # 2. Tool call budget
        if not violations:
            budget_violation = self._check_budget(tool_name)
            if budget_violation:
                if budget_violation.is_hard_block:
                    violations.append(budget_violation)
                else:
                    warnings.append(budget_violation)

        # 3. Blocked patterns (still runs on opaque kwargs — extracts raw strings)
        if not violations:
            violations.extend(self._check_blocked_patterns(tool_name, tool_input))

        # 4. Preconditions
        if not violations:
            for v in self._check_preconditions(tool_name, tool_input, kwargs_opaque):
                if v.is_hard_block:
                    violations.append(v)
                else:
                    warnings.append(v)

        # Increment budget counters AFTER checks (budget sees count before this call)
        self._turn_call_counts[tool_name] = self._turn_call_counts.get(tool_name, 0) + 1
        self._session_call_counts[tool_name] = self._session_call_counts.get(tool_name, 0) + 1
        self._total_turn_calls += 1

        # Enhance violation messages if this is a repeated pattern this turn
        if violations:
            for v in violations:
                retry_count = self._count_similar_violations(v)
                if retry_count > 0:
                    v.reason += (
                        f"\n\n🔁 NOTE: This is attempt #{retry_count + 1} with the same "
                        f"blocked pattern this turn. A similar operation was already "
                        f"blocked. Change your approach ENTIRELY — do NOT retry with a "
                        f"different path or argument."
                    )
                    v.suggestion = (
                        "STOP retrying this pattern. The issue is the OPERATION, "
                        "not the specific path. Ask the user for an alternative approach."
                    )
            self._turn_violations.extend(violations)

        allowed = len(violations) == 0
        result = PolicyCheckResult(allowed=allowed, violations=violations, warnings=warnings)

        # Audit log
        self._audit(tool_name, tool_input, result)

        return result

    def reset_turn(self):
        """Reset per-turn counters. Call at the start of each new user message."""
        self._turn_call_counts = {}
        self._total_turn_calls = 0
        self._last_tool_name = ""
        self._consecutive_count = 0
        self._budget_warnings_delivered = set()
        self._turn_violations = []
        self._awaiting_approval = False
        self._awaiting_approval_tool = ""

    def _count_similar_violations(self, violation: "PolicyViolation") -> int:
        """Count how many similar violations already occurred this turn.

        'Similar' = same rule_type and same matched_pattern (for pattern rules)
        or same rule_type and same tool_name (for budget/other rule types).
        """
        count = 0
        for prev in self._turn_violations:
            if prev.rule_type != violation.rule_type:
                continue
            if violation.rule_type in ("pattern", "approval_required"):
                if prev.matched_pattern and violation.matched_pattern:
                    if prev.matched_pattern == violation.matched_pattern:
                        count += 1
                elif prev.reason == violation.reason:
                    count += 1
            else:
                if prev.tool_name == violation.tool_name:
                    count += 1
        return count

    def get_remaining_budget(self) -> int:
        """Return how many total tool calls remain in this turn's global budget.

        Used by WorkerAgentTool to decide whether there is enough headroom
        to spawn a worker agent (which needs ≥10 calls to be useful).

        Returns:
            Remaining calls = max_total_per_turn - _total_turn_calls.
            Returns 999 if no global limit is configured.
        """
        max_total = self.policy.get("global_limits", {}).get("max_total_per_turn")
        if max_total is None:
            return 999  # no limit configured
        return max(0, int(max_total) - self._total_turn_calls)

    # ══════════════════════════════════════════════════════════════════════════
    # PROTOCOL MODE — Reproducible execution enforcement
    # ══════════════════════════════════════════════════════════════════════════

    def set_protocol_lock(self, lock) -> None:
        """
        Attach a ProtocolExecutionLock to enforce step-by-step compliance.

        Args:
            lock: A ProtocolExecutionLock instance (or None to disable)
        """
        self._protocol_lock = lock

    def get_protocol_lock(self):
        """Get the current protocol lock (or None if not active)."""
        return self._protocol_lock

    def advance_protocol(self, tool_output: str = "") -> None:
        """Advance the protocol to the next step after successful tool execution."""
        if self._protocol_lock and self._protocol_lock.active:
            self._protocol_lock.advance(tool_output)

    def _check_protocol_compliance(
        self, tool_name: str, tool_input: dict
    ) -> "PolicyViolation | None":
        """
        Check if tool call matches the current protocol step.

        Returns PolicyViolation (hard block) if protocol is active and
        tool deviates. Returns None if no protocol active or compliant.
        """
        if not self._protocol_lock or not self._protocol_lock.active:
            return None

        deviation = self._protocol_lock.check_compliance(tool_name, tool_input)
        if deviation is None:
            return None

        # Protocol deviation — HARD STOP
        return PolicyViolation(
            tool_name=tool_name,
            rule_type="protocol_deviation",
            reason=deviation.format_error(),
            suggestion=(
                f"Follow the protocol: next step requires "
                f"'{deviation.expected_tool}'. Do NOT improvise."
            ),
            is_hard_block=True,
        )

    def get_environment_context(self) -> str:
        """
        Return environment registry as text for injection into system prompt.
        This gives the agent structured knowledge of its environment.
        """
        if not self.environment:
            return ""

        lines = ["\n## ENVIRONMENT REGISTRY (from config/environment.yaml)"]
        lines.append("Consult this BEFORE submitting jobs or using software.\n")

        # Containers
        containers = self.environment.get("containers", {})
        if containers:
            lines.append("### Containers")
            for name, info in containers.items():
                lines.append(f"- **{name}**: `{info.get('path', 'unknown')}`")
                lines.append(f"  Purpose: {info.get('purpose', 'unknown')}")
                if info.get('does_not_have'):
                    lines.append(f"  ⚠️ Does NOT have: {', '.join(info['does_not_have'])}")
            lines.append("")

        # Software
        software = self.environment.get("software", {})
        if software:
            lines.append("### Software")
            for name, info in software.items():
                lines.append(f"- **{name}** v{info.get('version', '?')}: `{info.get('path', 'unknown')}`")
                if info.get('commands_NOT_available'):
                    lines.append(f"  ⚠️ Does NOT support: {', '.join(info['commands_NOT_available'])}")
            lines.append("")

        # Partitions
        partitions = self.environment.get("partitions", {})
        if partitions:
            lines.append("### Partitions")
            for name, info in partitions.items():
                avail = info.get('typical_availability', 'unknown')
                lines.append(f"- **{name}**: GPUs={info.get('gpu_types', [])}, availability={avail}")
            lines.append("")

        return "\n".join(lines)

    def get_environment_index(self) -> str:
        """Return a compact environment index with critical paths for system prompt injection.

        Shows actual paths for software (prevents which/find fallback) and
        container/partition names. Full details via get_environment_info(topic).
        """
        if not self.environment:
            return ""

        containers = list(self.environment.get("containers", {}).keys())
        default_ctr = self.environment.get("default_container", {})
        partitions = list(self.environment.get("partitions", {}).keys())
        sw = self.environment.get("software", {})

        lines = [
            "=== SYSTEM ENVIRONMENT (read-only — call get_environment_info(topic) for full details) ===",
        ]
        if default_ctr:
            lines.append(f"Default container: {default_ctr.get('path', 'unknown')}")
        lines.append(f"Partitions: {', '.join(partitions)}")

        for name, info in sw.items():
            path = info.get("path", "")
            if path:
                lines.append(f"  {name}: {path}")
            else:
                lines.append(f"  {name}: (call get_environment_info('{name}') for path)")

        lines.append(f"Containers available: {', '.join(containers)}")
        if self.environment.get("container_building"):
            lines.append("Container building: fakeroot env vars + best practices available (call get_environment_info('container_building'))")

        # Package registry summary
        pkg_reg = self.environment.get("package_registry", {})
        if pkg_reg:
            reg_path = pkg_reg.get("path", "")
            if reg_path:
                lines.append("")
                lines.append("=== PACKAGE REGISTRY (scientific/visualization software) ===")
                lines.append(f"  Python: {pkg_reg.get('python_bin', 'unknown')}")
                categories = pkg_reg.get("categories_available", [])
                if categories:
                    lines.append(f"  Categories: {', '.join(categories)}")
                # Load package names from registry for compact display
                pkg_names = self._get_package_names_from_registry(reg_path)
                if pkg_names:
                    lines.append(f"  Installed: {', '.join(pkg_names)}")
                lines.append("  Call get_environment_info('packages') for full registry with purposes.")
                lines.append("  Call get_environment_info('package:<name>') for detailed usage + knowledge.")

        lines.append("Call get_environment_info(topic) for container details, GPU specs, constraints.")
        return "\n".join(lines)

    def _get_package_names_from_registry(self, registry_path: str) -> list:
        """Load package names from the package registry YAML file."""
        try:
            reg_file = Path(registry_path)
            if not reg_file.exists():
                return []
            with open(reg_file, "r", encoding="utf-8") as f:
                reg = yaml.safe_load(f) or {}
            packages = reg.get("packages", [])
            return [
                p["name"] for p in packages
                if isinstance(p, dict) and p.get("status", "installed") == "installed"
            ]
        except Exception:
            return []

    # ══════════════════════════════════════════════════════════════════════════
    # CHECKERS
    # ══════════════════════════════════════════════════════════════════════════

    def _check_required_params(
        self, tool_name: str, tool_input: dict, kwargs_opaque: bool = False
    ) -> list[PolicyViolation]:
        """Check that required parameters are present and non-empty."""
        violations = []
        rules = self.policy.get("required_params", {}).get(tool_name, [])

        if kwargs_opaque and rules:
            logger.warning(
                f"[PEL] Skipping required_params check for '{tool_name}' — "
                f"kwargs could not be parsed (params may be present in opaque string)"
            )
            return []

        for rule in rules:
            param = rule["param"]
            value = tool_input.get(param)
            if not value:  # None, empty string, or missing
                violations.append(PolicyViolation(
                    tool_name=tool_name,
                    rule_type="required_param",
                    reason=rule.get("reason", f"Required parameter '{param}' is missing"),
                    suggestion=f"Provide '{param}' before calling {tool_name}. "
                               f"Check config/environment.yaml for valid values.",
                    is_hard_block=True,
                ))
        return violations

    def _check_budget(self, tool_name: str) -> Optional[PolicyViolation]:
        """Check if tool call budget for this turn has been exceeded."""
        budgets = self.policy.get("tool_budgets", {})
        tool_budget = budgets.get(tool_name) or budgets.get("_default")

        if not tool_budget:
            return None

        max_calls = tool_budget.get("max_per_turn", 10)
        current_count = self._turn_call_counts.get(tool_name, 0)

        # Hard block: budget exceeded
        if current_count >= max_calls:
            on_exceed = tool_budget.get("on_exceed", "warn")
            message_template = tool_budget.get(
                "message",
                f"{tool_name} called {{count}}x this turn (max {{max}})"
            )
            message = message_template.replace("{count}", str(current_count + 1))
            message = message.replace("{max}", str(max_calls))
            message = message.replace("{tool_name}", tool_name)

            return PolicyViolation(
                tool_name=tool_name,
                rule_type="budget",
                reason=message.strip(),
                suggestion="Report current status to the user and wait for instructions.",
                is_hard_block=(on_exceed == "error"),
            )

        # Warning threshold: fire once to force batching strategy
        warn_at = tool_budget.get("warn_at")
        if (warn_at is not None
                and current_count >= warn_at
                and tool_name not in self._budget_warnings_delivered):
            self._budget_warnings_delivered.add(tool_name)
            remaining = max_calls - current_count
            warn_template = tool_budget.get(
                "warn_message",
                f"BUDGET WARNING: {tool_name} used {{count}}/{{max}} calls this turn. "
                f"{{remaining}} call(s) remaining. Batch remaining operations."
            )
            warn_msg = warn_template.replace("{count}", str(current_count))
            warn_msg = warn_msg.replace("{max}", str(max_calls))
            warn_msg = warn_msg.replace("{remaining}", str(remaining))
            warn_msg = warn_msg.replace("{tool_name}", tool_name)

            return PolicyViolation(
                tool_name=tool_name,
                rule_type="budget_warning",
                reason=warn_msg.strip(),
                suggestion=(
                    f"STOP. You have used {current_count}/{max_calls} budget. "
                    f"Plan your remaining {remaining} call(s) carefully. "
                    f"Combine all remaining operations into as few calls as possible."
                ),
                is_hard_block=True,
            )

        return None

    def _check_global_budget(self) -> Optional[PolicyViolation]:
        """Check if total tool calls this turn exceeds the global cap."""
        global_config = self.policy.get("global_limits", {})
        max_total = global_config.get("max_total_per_turn", 25)

        if self._total_turn_calls >= max_total:
            return PolicyViolation(
                tool_name="*",
                rule_type="global_budget",
                reason=(
                    f"Total tool calls this turn: {self._total_turn_calls + 1} "
                    f"(hard cap: {max_total}). "
                    f"You have used all available tool calls for this turn."
                ),
                suggestion=(
                    "STOP. Output your current progress to the user immediately. "
                    "Do NOT attempt any more tool calls. Summarize what was accomplished "
                    "and what remains."
                ),
                is_hard_block=True,
            )
        return None

    def _check_consecutive(self, tool_name: str) -> Optional[PolicyViolation]:
        """Check if the same tool is being called too many times consecutively.

        This catches the pattern where the LLM loops on the same tool
        (e.g. grep_file 5 times in a row) instead of using a batch alternative.

        Note: _consecutive_count is updated BEFORE this check runs (at top of
        check()), so it includes the current call. A blocked intermediate tool
        still resets the counter — only truly consecutive calls are counted.
        """
        if tool_name != self._last_tool_name:
            return None  # different tool — no consecutive issue

        consec_limits = self.policy.get("consecutive_limits", {})
        tool_limit = consec_limits.get(tool_name)

        if not tool_limit:
            return None  # no consecutive limit defined for this tool

        max_consecutive = tool_limit.get("max_consecutive", 3)

        # _consecutive_count includes the current call (updated at top of check())
        if self._consecutive_count > max_consecutive:
            redirect_to = tool_limit.get("redirect_to", "")
            reason = (
                f"{tool_name} called {self._consecutive_count}x consecutively "
                f"(max {max_consecutive} in a row). "
                f"This is an inefficient pattern."
            )
            if redirect_to:
                suggestion = (
                    f"Use '{redirect_to}' instead — it handles multiple operations "
                    f"in a single call. Do NOT retry {tool_name}."
                )
            else:
                suggestion = (
                    f"Stop calling {tool_name} repeatedly. Combine your operations "
                    f"or report progress to the user."
                )

            return PolicyViolation(
                tool_name=tool_name,
                rule_type="consecutive",
                reason=reason,
                suggestion=suggestion,
                is_hard_block=True,
            )
        return None

    # Keys that represent executable commands (used by scope: "command" rules)
    _COMMAND_KEYS = frozenset({
        "commands", "command", "cmd", "script", "shell_cmd",
        "shell_command", "bash_command", "code", "exec",
    })

    def _extract_command_strings(self, tool_input: dict) -> list[str]:
        """Extract only string values from command-related parameters.

        Used when a rule has scope: "command" — prevents matching patterns
        in file content, descriptions, or other non-command parameters.
        """
        strings = []
        for key, value in tool_input.items():
            if key.lower() in self._COMMAND_KEYS:
                strings.extend(self._extract_strings(value))
        return strings

    def _check_blocked_patterns(
        self, tool_name: str, tool_input: dict
    ) -> list[PolicyViolation]:
        """Check tool input against blocked command patterns.

        Patterns with requires_approval=true in policy.yaml return a soft
        violation (rule_type='approval_required', is_hard_block=False) that
        the UI layer can intercept to show a Yes/No approval button.
        Once approved via approve_pattern(), the pattern is allowed for
        the rest of the session.

        Scope filtering:
          - scope: "all" (default) — check all string values in tool_input
          - scope: "command" — only check values from command-related keys
            (commands, command, cmd, script, shell_cmd, etc.)
            This prevents false positives when file content contains
            patterns like "pip install" but the tool is write_text_file.
        """
        violations = []
        blocked_rules = self.policy.get("blocked_patterns", [])

        if not blocked_rules:
            return violations

        # Pre-extract both full and command-only strings
        all_strings = self._extract_strings(tool_input)
        combined_all = " ".join(all_strings)
        # Lazy: only compute command strings if a rule needs them
        _combined_command = None

        for rule in blocked_rules:
            pattern = rule.get("pattern", "")
            if not pattern:
                continue

            # Determine which input to check based on scope
            scope = rule.get("scope", "all")
            if scope == "command":
                if _combined_command is None:
                    cmd_strings = self._extract_command_strings(tool_input)
                    _combined_command = " ".join(cmd_strings)
                combined_input = _combined_command
            else:
                combined_input = combined_all

            # Determine match based on exact_boundary flag
            exact_boundary = rule.get("exact_boundary", False)

            if exact_boundary:
                # Pattern must appear followed by a boundary char (not a path continuation)
                idx = combined_input.find(pattern)
                matched = False
                while idx != -1:
                    end_pos = idx + len(pattern)
                    if end_pos >= len(combined_input):
                        matched = True
                        break
                    next_char = combined_input[end_pos]
                    if next_char in (' ', ';', '|', '&', '\n', '\t', '"', "'"):
                        matched = True
                        break
                    idx = combined_input.find(pattern, idx + 1)
            else:
                matched = pattern in combined_input

            if matched:
                # unless_contains: skip violation if a required modifier is present
                unless_contains = rule.get("unless_contains")
                if unless_contains and unless_contains in combined_input:
                    continue

                # min_path_depth: for find patterns, allow if path is deep enough
                min_depth = rule.get("min_path_depth")
                if min_depth and pattern.startswith("find "):
                    base_path = pattern[5:]  # strip "find "
                    idx = combined_input.find(pattern)
                    if idx != -1:
                        # Extract the full path after "find "
                        path_start = idx + 5
                        path_end = combined_input.find(" ", path_start)
                        if path_end == -1:
                            path_end = len(combined_input)
                        full_path = combined_input[path_start:path_end]
                        # Count components beyond the base
                        extra = full_path[len(base_path):].strip("/")
                        depth = len([p for p in extra.split("/") if p]) if extra else 0
                        if depth >= min_depth:
                            continue  # path is deep enough, not a broad search

                # Check exceptions
                exceptions = rule.get("exceptions", [])
                if any(exc in combined_input for exc in exceptions):
                    continue  # exception matched, allow

                # Skip approval prompt if a hard-block already covers this input
                requires_approval = rule.get("requires_approval", False)

                if requires_approval:
                    if any(v.is_hard_block and v.rule_type == "pattern" for v in violations):
                        continue
                    # If already approved this session, skip violation
                    if self.is_pattern_approved(pattern):
                        continue
                    # Return soft violation — UI will show approval button
                    _matched_input = _extract_matched_snippet(combined_input, pattern)
                    violations.append(PolicyViolation(
                        tool_name=tool_name,
                        rule_type="approval_required",
                        reason=rule.get("reason", f"Blocked pattern detected: '{pattern}'"),
                        suggestion=rule.get("suggestion", "Ask the user for guidance."),
                        is_hard_block=False,
                        matched_pattern=pattern,
                        matched_input=_matched_input,
                    ))
                else:
                    violations.append(PolicyViolation(
                        tool_name=tool_name,
                        rule_type="pattern",
                        reason=rule.get("reason", f"Blocked pattern detected: '{pattern}'"),
                        suggestion=rule.get("suggestion", "Ask the user for guidance."),
                        is_hard_block=True,
                        matched_pattern=pattern,
                    ))
        return violations

    def _check_preconditions(
        self, tool_name: str, tool_input: dict, kwargs_opaque: bool = False
    ) -> list[PolicyViolation]:
        """Check preconditions (file exists, in registry, etc.)."""
        violations = []
        rules = self.policy.get("preconditions", {}).get(tool_name, [])

        if kwargs_opaque and rules:
            logger.warning(
                f"[PEL] Skipping precondition checks for '{tool_name}' — "
                f"kwargs could not be parsed"
            )
            return []

        for rule in rules:
            check_type = rule.get("check", "")
            param = rule.get("param", "")
            on_fail = rule.get("on_fail", "error")
            value = tool_input.get(param, "")

            if not value:
                continue  # param not provided — handled by required_params

            violation = self._run_precondition(check_type, value, tool_name, rule)
            if violation:
                violation.is_hard_block = (on_fail == "error")
                violations.append(violation)

        return violations

    def _run_precondition(
        self, check_type: str, value: str, tool_name: str, rule: dict
    ) -> Optional[PolicyViolation]:
        """Execute a single precondition check."""

        if check_type == "file_exists":
            if not os.path.isfile(value):
                suggestion = (
                    "Check config/environment.yaml for the correct path, "
                    "or ask the user."
                )
                param_name = rule.get("param", "")
                if param_name == "container_image" and self.environment:
                    known = self._get_known_container_paths()
                    if known:
                        path_list = "\n".join(
                            f"  - {name}: {path}" for name, path in known
                        )
                        suggestion = (
                            f"File does not exist at '{value}'. "
                            f"Known containers from environment.yaml:\n"
                            f"{path_list}\n\n"
                            f"Use one of the above paths. Do NOT guess other paths."
                        )
                return PolicyViolation(
                    tool_name=tool_name,
                    rule_type="precondition",
                    reason=f"File not found: {value}",
                    suggestion=suggestion,
                )

        elif check_type == "in_environment_registry":
            field_name = rule.get("field", "containers")
            registry_section = self.environment.get(field_name, {})
            known_paths = [
                item.get("path", "") for item in registry_section.values()
                if isinstance(item, dict)
            ]
            if value not in known_paths:
                known_entries = []
                for name, item in registry_section.items():
                    if isinstance(item, dict) and item.get("path"):
                        known_entries.append(f"  - {name}: {item['path']}")
                        purpose = item.get("purpose", item.get("description", ""))
                        if purpose:
                            known_entries.append(f"    Purpose: {purpose}")

                suggestion = (
                    f"'{value}' is NOT in the registry. "
                    f"Known {field_name}:\n"
                    + "\n".join(known_entries)
                    + f"\n\nUse ONLY paths listed above. "
                    f"Do NOT guess — if you need a different {field_name[:-1]}, "
                    f"ask the user."
                )
                if field_name == "containers":
                    default_ctr = self.environment.get("default_container", {})
                    if default_ctr and default_ctr.get("path"):
                        suggestion += (
                            f"\n\nDEFAULT CONTAINER: {default_ctr['path']}"
                        )

                return PolicyViolation(
                    tool_name=tool_name,
                    rule_type="precondition",
                    reason=(
                        f"'{value}' is not registered in config/environment.yaml "
                        f"(section: {field_name})"
                    ),
                    suggestion=suggestion,
                )

        elif check_type == "directory_exists":
            if not os.path.isdir(value):
                return PolicyViolation(
                    tool_name=tool_name,
                    rule_type="precondition",
                    reason=f"Directory not found: {value}",
                    suggestion="Check the path and try again.",
                )

        elif check_type == "script_paths_in_workdir":
            work_dir = os.environ.get("WORK_DIR", "")
            if work_dir and value:
                violation = self._check_script_workdir(value, work_dir, tool_name, rule)
                if violation:
                    return violation

        return None  # check passed

    # Patterns that create/write to filesystem paths (NOT reads)
    _WRITE_PATH_PATTERNS = re.compile(
        r'(?:mkdir\s+(?:-p\s+)?|>>?\s*)(/[^\s;|&><]+)'
        r'|(?:--output[_-]dir|--model[_-]dir|--save[_-]dir|--checkpoint[_-]dir|-o)\s+(/[^\s;|&><]+)'
    )

    # Paths that are always allowed regardless of workdir
    _SAFE_PATH_PREFIXES = ("/dev/null", "/dev/stderr", "/dev/stdout", "/tmp/")

    def _check_script_workdir(
        self, script_content: str, work_dir: str, tool_name: str, rule: dict
    ) -> Optional["PolicyViolation"]:
        """Check that write-target paths in script_content are under work_dir."""
        work_dir_resolved = os.path.realpath(work_dir)
        violations = []

        for match in self._WRITE_PATH_PATTERNS.finditer(script_content):
            path = match.group(1) or match.group(2)
            if not path:
                continue
            if any(path.startswith(safe) for safe in self._SAFE_PATH_PREFIXES):
                continue
            try:
                resolved = os.path.realpath(path)
            except (OSError, ValueError):
                resolved = path
            if not resolved.startswith(work_dir_resolved):
                violations.append(path)

        if violations:
            return PolicyViolation(
                tool_name=tool_name,
                rule_type="precondition",
                reason=(
                    f"Script writes to path(s) outside work directory: "
                    f"{', '.join(violations[:3])}"
                ),
                suggestion=(
                    f"All output paths must be under WORK_DIR ({work_dir}). "
                    f"Change your script to write output inside the work directory."
                ),
                matched_input=violations[0],
            )
        return None

    def _get_known_container_paths(self) -> list[tuple[str, str]]:
        """Return list of (name, path) for all known containers."""
        result = []
        containers = self.environment.get("containers", {})
        for name, info in containers.items():
            if isinstance(info, dict) and info.get("path"):
                result.append((name, info["path"]))
        default_ctr = self.environment.get("default_container", {})
        if default_ctr and default_ctr.get("path"):
            result.append(("default_container", default_ctr["path"]))
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_strings(self, obj: Any, depth: int = 0) -> list[str]:
        """Recursively extract all string values from nested dict/list."""
        if depth > 5:
            return []
        strings = []
        if isinstance(obj, str):
            strings.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                strings.extend(self._extract_strings(v, depth + 1))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                strings.extend(self._extract_strings(item, depth + 1))
        return strings

    def _load_yaml(self, path: str, name: str) -> dict:
        """Load a YAML config file. Returns empty dict on failure (graceful degradation)."""
        if yaml is None:
            logger.warning(f"PyYAML not installed — {name} enforcement disabled")
            return {}
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            logger.warning(f"{name} file not found: {path} — enforcement disabled")
            return {}
        except Exception as e:
            logger.error(f"Failed to load {name} from {path}: {e}")
            return {}

    def _audit(self, tool_name: str, tool_input: dict, result: PolicyCheckResult):
        """Write audit log entry for policy decisions."""
        audit_config = self.policy.get("audit", {})
        if not audit_config.get("enabled", False):
            return

        should_log = (
            (not result.allowed and audit_config.get("log_blocked", True))
            or (result.warnings and audit_config.get("log_warnings", True))
            or (result.allowed and audit_config.get("log_allowed", False))
        )
        if not should_log:
            return

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_name": tool_name,
            "allowed": result.allowed,
            "turn_call_number": self._turn_call_counts.get(tool_name, 0) + 1,
        }
        if result.violations:
            entry["violations"] = [v.reason for v in result.violations]
        if result.warnings:
            entry["warnings"] = [w.reason for w in result.warnings]
        if audit_config.get("include_tool_input", False):
            # Truncate large inputs to prevent log bloat
            entry["tool_input"] = self._truncate_input(tool_input)

        try:
            log_path = Path(self._audit_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'a') as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"Audit log write failed: {e}")

    def _truncate_input(self, tool_input: dict, max_len: int = 500) -> dict:
        """Truncate large string values in tool_input for audit logging."""
        truncated = {}
        for k, v in tool_input.items():
            if isinstance(v, str) and len(v) > max_len:
                truncated[k] = v[:max_len] + f"... ({len(v)} chars total)"
            else:
                truncated[k] = v
        return truncated
