"""Protocol Refiner — transforms raw recordings into validated golden protocols.

Two-phase lifecycle:
  1. Author loads raw recording → LLM identifies golden path → executes → validates
  2. Consumer loads golden protocol → strict execution with variable substitution

This module handles phase 1 (refine). The player handles phase 2 (golden execution).

No Chainlit dependencies. Uses async LLM calls via LiteLLM proxy.
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

LITELLM_URL = os.environ.get("LITELLM_URL", "http://localhost:8080")
REFINER_MODEL = os.environ.get("REFINER_MODEL", "anthropic.claude-sonnet-4-6")

VARIABLE_HEURISTIC_PATTERNS = [
    (r"/home/([^/]+)/", "username in home path"),
    (r"/data\d*/hpcadmin/([^/]+)/", "username in data path"),
    (r"--partition[= ](\S+)", "SLURM partition"),
    (r"\.sif\b", "Singularity container path"),
    (r"/projects/([^/]+)/", "project-specific path"),
]


class RefineState:
    LOADING = "loading"
    ANALYZING = "analyzing"
    PLAN_READY = "plan_ready"
    EXECUTING = "executing"
    VALIDATING = "validating"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class RefineResult:
    success: bool
    golden_path: Optional[Path] = None
    executed_steps: int = 0
    failed_step: Optional[int] = None
    failure_detail: str = ""
    variables: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GoldenStep:
    step: int
    tool: str
    description: str
    args: Dict[str, Any]
    expected_result_pattern: str = "success"


class ProtocolRefiner:
    """Drives the refine cycle: raw recording -> golden protocol.

    Workflow:
    1. Load raw recording.jsonl (all steps including failures)
    2. Generate analysis prompt for the LLM
    3. LLM identifies golden path, outputs planned steps
    4. Present plan to user for approval
    5. Execute the plan via call_tool_fn
    6. If all succeed -> detect variables -> produce golden_protocol.json
    7. If any fail -> report failure, allow variable adjustment, retry
    """

    def __init__(self, protocol_dir: Path):
        self.protocol_dir = protocol_dir
        self.state = RefineState.LOADING
        self._raw_steps: List[Dict[str, Any]] = []
        self._raw_header: Dict[str, Any] = {}
        self._golden_steps: List[GoldenStep] = []
        self._execution_results: List[Dict[str, Any]] = []

    def load_raw_recording(self) -> Dict[str, Any]:
        """Load the full raw recording with error annotations."""
        jsonl_path = self.protocol_dir / "recording.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"No recording.jsonl in {self.protocol_dir}")

        steps = []
        header = {}

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") == "header":
                    header = entry
                elif entry.get("type") == "step":
                    steps.append(entry)

        self._raw_steps = steps
        self._raw_header = header
        self.state = RefineState.ANALYZING
        return {
            "header": header,
            "steps": steps,
            "total_steps": len(steps),
            "error_steps": sum(1 for s in steps if s.get("error")),
            "success_steps": sum(1 for s in steps if not s.get("error")),
        }

    def generate_analysis_prompt(self) -> str:
        """Build the prompt that asks the LLM to identify the golden path.

        The LLM reads the full recording and outputs ONLY the steps that
        represent the correct, successful approach — skipping failures,
        retries, and dead-end explorations.
        """
        steps_text = []
        for s in self._raw_steps:
            status = "ERROR" if s.get("error") else "OK"
            result_preview = s.get("result_preview", "")[:300]
            error_text = f"\n    ERROR: {s['error']}" if s.get("error") else ""
            steps_text.append(
                f"  Step {s['step_number']} [{status}]: {s['tool_name']}\n"
                f"    Args: {json.dumps(s.get('args', {}), default=str)[:500]}\n"
                f"    Result: {result_preview}{error_text}"
            )

        recording_text = "\n".join(steps_text)

        return f"""You are analyzing a recorded protocol session to extract the GOLDEN PATH — the minimal set of steps that represent the correct, successful approach.

## Raw Recording ({len(self._raw_steps)} steps total, {sum(1 for s in self._raw_steps if s.get('error'))} had errors)

{recording_text}

## Your Task

Analyze this recording and identify which steps form the GOLDEN PATH — the correct approach that should be followed when replaying this protocol.

Rules:
1. SKIP steps that failed (marked [ERROR])
2. SKIP steps that were part of a failed approach (even if they individually succeeded)
   - Example: writing a download script that leads to a failed download = skip the script write too
3. KEEP only steps that are part of the FINAL SUCCESSFUL approach
4. If a step was retried with different parameters and succeeded, keep only the successful version
5. Re-number steps sequentially starting from 1

## Output Format

Respond with a JSON array of the golden path steps. Each step must have:
- "step": sequential number (1, 2, 3...)
- "tool": the tool_name from the recording
- "description": brief human-readable description of what this step does
- "args": the exact args dict from the recording (for the successful version)
- "original_step": which step number in the raw recording this corresponds to

Output ONLY the JSON array, no other text:
```json
[
  {{"step": 1, "tool": "...", "description": "...", "args": {{...}}, "original_step": N}},
  ...
]
```"""

    def parse_golden_plan(self, llm_response: str) -> List[GoldenStep]:
        """Parse the LLM's golden path selection into GoldenStep objects."""
        json_match = re.search(r'\[[\s\S]*\]', llm_response)
        if not json_match:
            raise ValueError("LLM response does not contain a JSON array")

        try:
            steps_data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in LLM response: {e}")

        golden_steps = []
        for i, s in enumerate(steps_data, 1):
            golden_steps.append(GoldenStep(
                step=i,
                tool=s["tool"],
                description=s.get("description", ""),
                args=s.get("args", {}),
                expected_result_pattern=s.get("expected_result_pattern", "success"),
            ))

        self._golden_steps = golden_steps
        self.state = RefineState.PLAN_READY
        return golden_steps

    def get_plan_summary(self) -> str:
        """Return a human-readable summary of the golden path plan for user review."""
        if not self._golden_steps:
            return "No golden path plan generated yet."

        lines = [
            f"## Golden Path Plan ({len(self._golden_steps)} steps)",
            "",
        ]
        for gs in self._golden_steps:
            args_preview = json.dumps(gs.args, default=str)[:200]
            lines.append(f"**Step {gs.step}:** `{gs.tool}` — {gs.description}")
            lines.append(f"  Args: `{args_preview}`")
            lines.append("")

        return "\n".join(lines)

    async def execute_golden_plan(
        self, call_tool_fn: Callable
    ) -> RefineResult:
        """Execute the golden path plan step by step.

        Args:
            call_tool_fn: async (tool_name, args) -> result

        Returns RefineResult with success/failure status.
        """
        if not self._golden_steps:
            return RefineResult(success=False, failure_detail="No golden plan to execute")

        self.state = RefineState.EXECUTING
        self._execution_results = []

        for gs in self._golden_steps:
            try:
                result = await call_tool_fn(gs.tool, gs.args)
                self._execution_results.append({
                    "step": gs.step,
                    "tool": gs.tool,
                    "args": gs.args,
                    "result": result,
                    "success": True,
                    "error": None,
                })
            except Exception as e:
                self._execution_results.append({
                    "step": gs.step,
                    "tool": gs.tool,
                    "args": gs.args,
                    "result": None,
                    "success": False,
                    "error": str(e),
                })
                self.state = RefineState.FAILED
                return RefineResult(
                    success=False,
                    executed_steps=gs.step,
                    failed_step=gs.step,
                    failure_detail=f"Step {gs.step} ({gs.tool}) failed: {e}",
                )

        self.state = RefineState.VALIDATING
        return RefineResult(
            success=True,
            executed_steps=len(self._golden_steps),
        )

    def generate_variable_detection_prompt(self) -> str:
        """Build prompt for LLM to identify which values should be variables.

        Uses heuristic pre-filtering to guide the LLM.
        """
        heuristic_flags = []
        for gs in self._golden_steps:
            args_str = json.dumps(gs.args, default=str)
            for pattern, description in VARIABLE_HEURISTIC_PATTERNS:
                matches = re.findall(pattern, args_str)
                if matches:
                    heuristic_flags.append(
                        f"  Step {gs.step} ({gs.tool}): found {description} — value(s): {matches[:3]}"
                    )

        heuristic_text = "\n".join(heuristic_flags) if heuristic_flags else "  (none detected)"

        steps_json = json.dumps(
            [{"step": gs.step, "tool": gs.tool, "args": gs.args} for gs in self._golden_steps],
            indent=2, default=str
        )

        return f"""Analyze these golden path steps and identify which parameter values should become VARIABLES for other users.

## Golden Path Steps
{steps_json}

## Heuristic Flags (likely variables detected by pattern matching)
{heuristic_text}

## Rules for Variable Detection
1. Paths containing a specific username (e.g., /home/username/, /data1/username/) → VARIABLE
2. SLURM partition names (gpu, cpu, gpushort) → VARIABLE
3. Container .sif file paths → VARIABLE (path may differ per cluster)
4. Dataset-specific paths (input data locations) → VARIABLE
5. Job parameters that scale with data (--mem, --time) → VARIABLE if they depend on dataset size
6. Keep FIXED: tool names, script logic/content structure, flag names, standard paths that are universal

## Output Format
Respond with a JSON array of variables. Each must have:
- "name": UPPER_CASE variable name (e.g., "WORK_DIR", "DATA_PATH", "PARTITION")
- "description": what this variable represents (1 sentence)
- "example": the concrete value from this recording
- "type": one of "path", "slurm_partition", "container_path", "integer", "string"
- "required": true/false
- "found_in_steps": list of step numbers where this value appears

Output ONLY the JSON array:
```json
[
  {{"name": "...", "description": "...", "example": "...", "type": "...", "required": true, "found_in_steps": [1, 3]}}
]
```"""

    def parse_variables(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse LLM's variable detection output."""
        json_match = re.search(r'\[[\s\S]*\]', llm_response)
        if not json_match:
            return []

        try:
            variables = json.loads(json_match.group())
        except json.JSONDecodeError:
            return []

        for var in variables:
            if not all(k in var for k in ("name", "description", "example", "type")):
                continue
            var.setdefault("required", True)
            var.setdefault("found_in_steps", [])

        return variables

    def produce_golden(
        self,
        variables: List[Dict[str, Any]],
        description: str = "",
        username: str = "",
    ) -> Path:
        """Write golden_protocol.json after successful validation.

        Substitutes concrete values with {{VARIABLE_NAME}} placeholders
        in step args based on the detected variables.
        """
        substituted_steps = []
        for gs in self._golden_steps:
            args_str = json.dumps(gs.args, default=str)
            for var in variables:
                example = var.get("example", "")
                if example and example in args_str:
                    args_str = args_str.replace(example, "{{" + var["name"] + "}}")
            try:
                substituted_args = json.loads(args_str)
            except json.JSONDecodeError:
                substituted_args = gs.args

            substituted_steps.append({
                "step": gs.step,
                "tool": gs.tool,
                "description": gs.description,
                "args": substituted_args,
                "expected_result_pattern": gs.expected_result_pattern,
            })

        golden_doc = {
            "type": "golden",
            "version": "1.0.0",
            "name": self._raw_header.get("name", self.protocol_dir.name),
            "validated": True,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "validated_by": username or self._raw_header.get("username", ""),
            "description": description or f"Golden protocol from {self._raw_header.get('name', 'unknown')}",
            "variables": [
                {
                    "name": v["name"],
                    "description": v["description"],
                    "example": v["example"],
                    "type": v["type"],
                    "required": v.get("required", True),
                }
                for v in variables
            ],
            "execution_mode": "strict",
            "on_failure": "adjust_variables_only",
            "steps": substituted_steps,
            "source_recording": "recording.jsonl",
            "total_raw_steps": len(self._raw_steps),
            "golden_steps": len(substituted_steps),
        }

        golden_path = self.protocol_dir / "golden_protocol.json"
        golden_path.write_text(
            json.dumps(golden_doc, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        self.state = RefineState.COMPLETE
        return golden_path


# ── Utility for golden protocol loading ──────────────────────────────────


def load_golden_protocol(protocol_dir: Path) -> Optional[Dict[str, Any]]:
    """Load golden_protocol.json if it exists. Returns None if not found."""
    golden_path = protocol_dir / "golden_protocol.json"
    if not golden_path.exists():
        return None
    try:
        return json.loads(golden_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def has_golden_protocol(protocol_dir: Path) -> bool:
    """Check if a protocol directory contains a validated golden protocol."""
    return (protocol_dir / "golden_protocol.json").exists()


def substitute_variables(args: Dict[str, Any], variables: Dict[str, str]) -> Dict[str, Any]:
    """Replace {{VAR_NAME}} placeholders in args with provided values."""
    args_str = json.dumps(args, default=str)
    for name, value in variables.items():
        args_str = args_str.replace("{{" + name + "}}", value)
    try:
        return json.loads(args_str)
    except json.JSONDecodeError:
        return args
