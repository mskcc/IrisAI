"""
Reproducible Protocol Execution for IrisAI.

Two-phase architecture:
  Phase 1 ("Talk Freely"): LLM generates a structured protocol (YAML)
  Phase 2 ("Execute Strictly"): PEL enforces each tool call matches the next step

Protocols are saved to work_dir/protocols/ and can be replayed deterministically.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ProtocolStep:
    """A single step in a reproducible protocol."""

    step_number: int
    tool_name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""
    validation_criteria: str = ""
    timeout_seconds: int = 300


@dataclass
class Protocol:
    """A complete reproducible protocol."""

    protocol_id: str
    title: str
    description: str
    steps: List[ProtocolStep] = field(default_factory=list)
    created_at: str = ""
    created_by: str = ""
    model_version: str = ""
    software_versions: Dict[str, str] = field(default_factory=dict)
    random_seed: Optional[int] = None
    input_hashes: Dict[str, str] = field(default_factory=dict)
    status: str = "pending"  # pending | approved | executing | completed | failed

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml_string(self) -> str:
        """Serialize to YAML-like readable string (no PyYAML dependency)."""
        lines = [
            f"protocol_id: {self.protocol_id}",
            f"title: {self.title}",
            f"description: {self.description}",
            f"created_at: {self.created_at}",
            f"created_by: {self.created_by}",
            f"model_version: {self.model_version}",
            f"status: {self.status}",
        ]
        if self.random_seed is not None:
            lines.append(f"random_seed: {self.random_seed}")
        if self.software_versions:
            lines.append("software_versions:")
            for k, v in self.software_versions.items():
                lines.append(f"  {k}: {v}")
        if self.input_hashes:
            lines.append("input_hashes:")
            for k, v in self.input_hashes.items():
                lines.append(f"  {k}: {v}")
        lines.append("steps:")
        for step in self.steps:
            lines.append(f"  - step_number: {step.step_number}")
            lines.append(f"    tool_name: {step.tool_name}")
            lines.append(f"    description: {step.description}")
            if step.parameters:
                lines.append(f"    parameters: {json.dumps(step.parameters)}")
            if step.expected_outcome:
                lines.append(f"    expected_outcome: {step.expected_outcome}")
            if step.validation_criteria:
                lines.append(f"    validation_criteria: {step.validation_criteria}")
            lines.append(f"    timeout_seconds: {step.timeout_seconds}")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict) -> "Protocol":
        """Reconstruct Protocol from a dictionary."""
        steps_data = data.pop("steps", [])
        steps = [ProtocolStep(**s) for s in steps_data]
        return cls(steps=steps, **data)


# ---------------------------------------------------------------------------
# Protocol Execution Lock — enforces strict step-by-step compliance
# ---------------------------------------------------------------------------


@dataclass
class ProtocolDeviation:
    """Returned when a tool call deviates from the active protocol."""

    step_number: int
    expected_tool: str
    actual_tool: str
    reason: str

    def format_error(self) -> str:
        return (
            f"PROTOCOL DEVIATION at step {self.step_number}: "
            f"Expected tool '{self.expected_tool}', got '{self.actual_tool}'. "
            f"{self.reason}"
        )


class ProtocolExecutionLock:
    """
    Enforces that tool calls follow the protocol step-by-step.

    Usage:
        lock = ProtocolExecutionLock(protocol)
        lock.activate()
        # On each tool call:
        deviation = lock.check_compliance(tool_name, tool_input)
        if deviation:
            # HARD STOP
        else:
            lock.advance()  # Move to next step after successful execution
    """

    def __init__(self, protocol: Protocol):
        self.protocol = protocol
        self.current_step_index: int = 0
        self.active: bool = False
        self.execution_trace: List[Dict[str, Any]] = []
        self._start_time: Optional[float] = None

    def activate(self):
        """Start protocol execution mode."""
        self.active = True
        self.current_step_index = 0
        self.execution_trace = []
        self._start_time = time.time()
        self.protocol.status = "executing"
        logger.info(
            f"Protocol '{self.protocol.protocol_id}' activated — "
            f"{len(self.protocol.steps)} steps to execute"
        )

    def deactivate(self):
        """End protocol execution mode."""
        self.active = False
        logger.info(
            f"Protocol '{self.protocol.protocol_id}' deactivated at step "
            f"{self.current_step_index}/{len(self.protocol.steps)}"
        )

    @property
    def is_complete(self) -> bool:
        """True if all steps have been executed."""
        return self.current_step_index >= len(self.protocol.steps)

    @property
    def current_step(self) -> Optional[ProtocolStep]:
        """Get the current step (next to execute)."""
        if self.is_complete:
            return None
        return self.protocol.steps[self.current_step_index]

    def check_compliance(
        self, tool_name: str, tool_input: dict
    ) -> Optional[ProtocolDeviation]:
        """
        Check if a tool call matches the current protocol step.

        Returns None if compliant, ProtocolDeviation if not.
        """
        if not self.active:
            return None

        if self.is_complete:
            return ProtocolDeviation(
                step_number=self.current_step_index + 1,
                expected_tool="(none — protocol complete)",
                actual_tool=tool_name,
                reason="Protocol execution is complete. No more tool calls expected.",
            )

        expected = self.current_step
        if tool_name != expected.tool_name:
            return ProtocolDeviation(
                step_number=expected.step_number,
                expected_tool=expected.tool_name,
                actual_tool=tool_name,
                reason=(
                    f"Step {expected.step_number} requires '{expected.tool_name}' "
                    f"({expected.description}). HARD STOP — protocol deviation."
                ),
            )

        # Tool name matches — compliance check passed
        return None

    def advance(self, tool_output: str = ""):
        """Advance to the next step after successful execution."""
        if not self.active or self.is_complete:
            return

        step = self.current_step
        trace_entry = {
            "step_number": step.step_number,
            "tool_name": step.tool_name,
            "description": step.description,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "output_preview": tool_output[:500] if tool_output else "",
        }
        self.execution_trace.append(trace_entry)
        self.current_step_index += 1

        if self.is_complete:
            self.protocol.status = "completed"
            elapsed = time.time() - (self._start_time or time.time())
            logger.info(
                f"Protocol '{self.protocol.protocol_id}' completed — "
                f"{len(self.protocol.steps)} steps in {elapsed:.1f}s"
            )

    def get_progress_summary(self) -> str:
        """Human-readable progress string."""
        total = len(self.protocol.steps)
        done = self.current_step_index
        if self.is_complete:
            return f"✅ Protocol complete ({total}/{total} steps)"
        current = self.current_step
        return (
            f"📋 Protocol step {done + 1}/{total}: "
            f"{current.tool_name} — {current.description}"
        )


# ---------------------------------------------------------------------------
# Protocol Persistence — save/load protocols and traces
# ---------------------------------------------------------------------------


class ProtocolStore:
    """Manages protocol storage on disk."""

    def __init__(self, protocols_dir: str):
        self.protocols_dir = protocols_dir
        os.makedirs(protocols_dir, exist_ok=True)

    def save_protocol(self, protocol: Protocol) -> str:
        """Save protocol to disk. Returns file path."""
        filename = f"{protocol.protocol_id}.json"
        filepath = os.path.join(self.protocols_dir, filename)
        with open(filepath, "w") as f:
            json.dump(protocol.to_dict(), f, indent=2)
        logger.info(f"Protocol saved: {filepath}")
        return filepath

    def load_protocol(self, protocol_id: str) -> Optional[Protocol]:
        """Load protocol from disk by ID."""
        filename = f"{protocol_id}.json"
        filepath = os.path.join(self.protocols_dir, filename)
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r") as f:
            data = json.load(f)
        return Protocol.from_dict(data)

    def save_execution_trace(
        self, protocol_id: str, trace: List[Dict[str, Any]]
    ) -> str:
        """Save execution trace alongside protocol."""
        filename = f"{protocol_id}_trace.json"
        filepath = os.path.join(self.protocols_dir, filename)
        with open(filepath, "w") as f:
            json.dump(
                {
                    "protocol_id": protocol_id,
                    "trace": trace,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )
        logger.info(f"Execution trace saved: {filepath}")
        return filepath

    def list_protocols(self) -> List[str]:
        """List all protocol IDs in the store."""
        if not os.path.exists(self.protocols_dir):
            return []
        return [
            f.replace(".json", "")
            for f in os.listdir(self.protocols_dir)
            if f.endswith(".json") and not f.endswith("_trace.json")
        ]


# ---------------------------------------------------------------------------
# Protocol Generation — builds protocol from LLM plan
# ---------------------------------------------------------------------------


def build_protocol_generation_prompt(
    user_request: str, available_tools: List[str]
) -> str:
    """Build prompt for sub-agent to generate a structured protocol."""
    return f"""You are a protocol generator for reproducible research workflows.

Given the user's request, generate a structured protocol as a JSON object.

USER REQUEST:
{user_request}

AVAILABLE TOOLS:
{json.dumps(available_tools, indent=2)}

Generate a JSON object with this EXACT structure:
{{
  "title": "Short descriptive title",
  "description": "What this protocol achieves",
  "steps": [
    {{
      "step_number": 1,
      "tool_name": "exact_tool_name_from_available_tools",
      "description": "What this step does",
      "parameters": {{"param1": "value1"}},
      "expected_outcome": "What success looks like",
      "validation_criteria": "How to verify success",
      "timeout_seconds": 300
    }}
  ],
  "software_versions": {{"tool_name": "version_if_relevant"}},
  "random_seed": null
}}

RULES:
- Use ONLY tools from the AVAILABLE TOOLS list
- Each step must be independently verifiable
- Parameters must be concrete values (no placeholders)
- Order steps logically — each step may depend on previous outputs
- Include validation_criteria for every step

Return ONLY the JSON object, no markdown fences or explanation."""


def parse_protocol_from_llm_output(
    llm_output: str, protocol_id: str, username: str, model_version: str
) -> Optional[Protocol]:
    """Parse LLM-generated protocol JSON into a Protocol object."""
    # Strip markdown fences if present
    text = llm_output.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines[1:] if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse protocol JSON: {e}")
        return None

    steps = []
    for s in data.get("steps", []):
        steps.append(
            ProtocolStep(
                step_number=s.get("step_number", 0),
                tool_name=s.get("tool_name", ""),
                description=s.get("description", ""),
                parameters=s.get("parameters", {}),
                expected_outcome=s.get("expected_outcome", ""),
                validation_criteria=s.get("validation_criteria", ""),
                timeout_seconds=s.get("timeout_seconds", 300),
            )
        )

    return Protocol(
        protocol_id=protocol_id,
        title=data.get("title", "Untitled Protocol"),
        description=data.get("description", ""),
        steps=steps,
        created_by=username,
        model_version=model_version,
        software_versions=data.get("software_versions", {}),
        random_seed=data.get("random_seed"),
        status="pending",
    )


# ---------------------------------------------------------------------------
# Protocol Replay — re-run tools only (no LLM)
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    """Result of replaying a protocol."""

    protocol_id: str
    success: bool
    steps_completed: int
    total_steps: int
    error: str = ""
    trace: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PROTOCOL_CONFIG = {
    "enabled": True,
    "auto_approve": False,
    "protocols_dir": "protocols/",
    "hard_stop_on_deviation": True,
    "max_steps": 50,
}


def load_protocol_config(policy_data: dict) -> dict:
    """Extract protocol config from policy.yaml data."""
    config = dict(DEFAULT_PROTOCOL_CONFIG)
    protocol_section = policy_data.get("protocol", {})
    config.update(protocol_section)
    return config


# ---------------------------------------------------------------------------
# Classification — should this request use protocol mode?
# ---------------------------------------------------------------------------

PROTOCOL_TRIGGER_KEYWORDS = [
    "reproducible",
    "reproducibility",
    "protocol",
    "pipeline",
    "workflow",
    "step by step",
    "step-by-step",
    "same results",
    "deterministic",
    "replicate",
    "replicable",
]

PROTOCOL_SKIP_KEYWORDS = [
    "debug",
    "explore",
    "what is",
    "how do",
    "explain",
    "help me understand",
    "show me",
    "list",
    "check",
]


def should_use_protocol_mode(user_input: str) -> bool:
    """
    Classify whether a request should trigger protocol mode.

    Returns True if the request involves multi-step reproducible work.
    """
    lower = user_input.lower()

    # Explicit triggers — user asked for reproducibility
    for keyword in PROTOCOL_TRIGGER_KEYWORDS:
        if keyword in lower:
            return True

    # Explicit skips — exploratory/simple requests
    for keyword in PROTOCOL_SKIP_KEYWORDS:
        if keyword in lower:
            return False

    # Heuristic: multi-step indicators
    multi_step_indicators = [
        " then ",
        " after that ",
        " next ",
        " finally ",
        " first ",
        "1.",
        "2.",
        "step 1",
        "step 2",
    ]
    indicator_count = sum(1 for ind in multi_step_indicators if ind in lower)
    if indicator_count >= 2:
        return True

    return False
