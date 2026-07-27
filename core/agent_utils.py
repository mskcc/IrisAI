"""Agent utilities — error classification, hallucination detection, retry logic.

Extracted from core/supervisor.py during Phase 1 architecture redesign.
This module keeps the pure utility functions that are still needed:
- Error classification and retry parameters
- Upload intent detection
- Agent result validation (hallucination detection)
- Tool enforcement retry prompts (anti-hallucination)

Removed (now handled by SkillLoader + structured output):
- build_supervisor_prompt() → replaced by skill manifest
- parse_supervisor_decision() → replaced by structured output
- build_safety_check_prompt() → eliminated (no safety check)
- parse_safety_check_response() → eliminated (no safety check)

None of these functions depend on Chainlit, LLM, or MCP.
"""
import os
import subprocess
from typing import Tuple, Optional, List, Dict



# ── Error classification ────────────────────────────────────────────────────────────

def classify_error(error_str: str) -> str:
    """Classify an error string into a category for retry logic.
    
    Args:
        error_str: The error message string (will be lowercased internally)
    
    Returns:
        One of: 'blank_content', 'context_limit', 'token_throttling',
                'request_throttling', 'budget_exceeded', 'code_bug', 'general'
    """
    error_lower = error_str.lower()
    
    # Bedrock blank ContentBlock error — a poisoned message with empty content
    # was sent to Bedrock. The fix is to sanitize history and retry immediately.
    # This MUST be checked BEFORE context_limit because both are 400 errors.
    # Example: "The text field in the ContentBlock object at messages.21.content.0 is blank"
    is_blank_content = any(phrase in error_lower for phrase in [
        "contentblock object",
        "content.0 is blank",
        "text field in the contentblock",
        "is blank. add text to the text field",
    ])
    
    is_context_error = any(phrase in error_lower for phrase in [
        "context length", "context window", "maximum context",
        "exceeds maximum", "input too long", "max input tokens",
        "contextwindowexceeded", "contextwindow exceeded",
    ])
    
    is_token_throttling = (
        "too many tokens" in error_lower and "please wait" in error_lower
    )
    
    is_request_throttling = (
        not is_token_throttling and any(phrase in error_lower for phrase in [
            "throttl", "throttlingexception", "too many requests",
            "request rate is too high", "rate limit", "429",
            "please wait before trying again",
        ])
    )
    
    is_model_overloaded = (
        "model" in error_lower 
        and "overloaded" in error_lower 
        and "try again later" in error_lower
    )
    
    # Budget exceeded — LiteLLM virtual key budget exhausted
    is_budget_exceeded = any(phrase in error_lower for phrase in [
        "budget has been exceeded",
        "budget_exceeded",
        "max budget",
        "max_budget",
    ])

    # Code bugs — deterministic failures that retrying cannot fix
    is_code_bug = any(phrase in error_lower for phrase in [
        "unboundlocalerror",
        "nameerror",
        "typeerror",
        "attributeerror",
        "importerror",
        "syntaxerror",
        "keyerror",
        "indexerror",
        "zerodivisionerror",
        "recursionerror",
    ])

    if is_blank_content:
        return "blank_content"
    elif is_context_error:
        return "context_limit"
    elif is_token_throttling or is_model_overloaded:
        return "token_throttling"
    elif is_request_throttling:
        return "request_throttling"
    elif is_budget_exceeded:
        return "budget_exceeded"
    elif is_code_bug:
        return "code_bug"
    else:
        return "general"


def get_retry_params(error_type: str) -> dict:
    """Get retry parameters based on error type.
    
    Args:
        error_type: One of the classify_error return values
    
    Returns:
        Dict with max_retries, base_delay, should_retry
    """
    if error_type == "blank_content":
        # Blank content errors are caused by poisoned messages in history.
        # The fix is to sanitize history and retry immediately — no delay needed.
        # This is NOT a rate limit or capacity issue, it's a data issue.
        return {
            "should_retry": True,
            "max_retries": 2,
            "base_delay": 1,
        }
    elif error_type in ("request_throttling", "token_throttling", "context_limit"):
        base_delay = 120 if error_type in ("token_throttling", "context_limit") else 60
        return {
            "should_retry": True,
            "max_retries": 4,
            "base_delay": base_delay,
        }
    elif error_type == "budget_exceeded":
        # Budget errors are not transient — retrying is pointless.
        # The user needs to regenerate their virtual key or increase the budget.
        return {
            "should_retry": False,
            "max_retries": 0,
            "base_delay": 0,
        }
    elif error_type == "code_bug":
        # Deterministic code bugs (TypeError, NameError, etc.) will never succeed on retry.
        return {
            "should_retry": False,
            "max_retries": 0,
            "base_delay": 0,
        }
    else:
        # General errors (network blips, transient 500s, etc.) — retry a couple times
        return {
            "should_retry": True,
            "max_retries": 2,
            "base_delay": 10,
        }


# ── Agent result validation (hallucination detection) ─────────────────

# Warning message prepended to agent output when zero tool calls are detected.
# This is a deterministic, code-level check — not a prompt-based heuristic.
ZERO_TOOL_CALLS_WARNING = (
    "⚠️ **Warning: This response may be unreliable.** "
    "The agent produced output without calling any tools. "
    "The information below was generated from the LLM's memory, "
    "not from reading actual files or executing real commands. "
    "Please verify independently or ask the agent to try again.\n\n"
    "---\n\n"
)

# Access claim keywords — if the agent output contains these phrases but
# did NOT call scontrol or check_user_slurm_access, the access claim is
# unverified and should be flagged.
ACCESS_CLAIM_PHRASES = [
    "has access to",
    "can access",
    "has gpu access",
    "has access to gpu",
    "can submit to",
    "has permission",
    "is allowed",
    "partition access",
    # NOTE: "✅" was intentionally removed — it is too broad and causes false positives
    # in any response that uses checkmarks for completed tasks, status lists, etc.
    # The remaining phrases are specific enough to catch real Slurm access claims.
]

ACCESS_VERIFICATION_TOOLS = {
    "check_user_slurm_access",
    "execute_dynamic_task",  # may run scontrol/sacctmgr directly
}

ACCESS_UNVERIFIED_WARNING = (
    "⚠️ **Warning: Partition access claims may be unverified.** "
    "This response contains access statements (e.g. 'has access to gpu') "
    "but the agent did not call `check_user_slurm_access` or run "
    "`scontrol show partition` to verify DenyAccounts. "
    "Access claims based on sacctmgr associations alone are INCOMPLETE — "
    "partitions can deny accounts via DenyAccounts even if the user has a valid account. "
    "Please ask the agent to re-verify using `check_user_slurm_access`.\n\n"
    "---\n\n"
)

# Phrases that indicate the agent is asking for user confirmation before
# performing a destructive action. These are legitimate zero-tool-call
# responses — the agent is correctly waiting for user input before executing.
# The system prompt instructs: "For destructive actions... ALWAYS ask for
# explicit user confirmation FIRST."
#
# IMPORTANT: These phrases must be specific enough to avoid false positives.
# Generic words like "confirm" or "verify" match normal technical prose
# (e.g. "verify the deployment", "confirm the checksum") and cause the guard
# to miss real hallucinations. Only use question-like patterns that the agent
# would use when genuinely asking the user for permission.
# Checked case-insensitively against the agent's output text.
CONFIRMATION_PROMPT_PHRASES = [
    "would you like me to proceed",
    "would you like to proceed",
    "do you want me to",
    "do you want to",
    "shall i proceed",
    "shall i go ahead",
    "should i proceed",
    "should i go ahead",
    "before i proceed",
    "before proceeding",
    "are you sure",
    "yes or no",
    "yes/no",
    "please confirm",
    "just to confirm",
    "can you confirm",
    "confirm with",
    "proceed with all",
    "proceed with the",
]

# Maximum length (in characters) for a response to be considered a genuine
# confirmation prompt. Confirmations listing multiple items (e.g. 5 files
# to delete) can reach 800+ chars while still being legitimate.
# Hallucinated responses are typically 1500+ chars with code blocks and tables.
CONFIRMATION_MAX_LENGTH = 1000


def is_confirmation_prompt(output: str) -> bool:
    """Detect if the agent output is a confirmation prompt.
    
    When the agent asks the user to confirm a destructive action (e.g.
    cancelling a Slurm job, deleting a file, overwriting data), it
    intentionally does NOT call any tools — it's waiting for user input.
    This is correct behavior mandated by the system prompt's safety rules.
    
    The hallucination guard should NOT flag these as suspicious, because
    the agent is doing exactly what it was told: ask before acting.
    
    Uses a two-tier heuristic:
      1. Length check: If the output is longer than CONFIRMATION_MAX_LENGTH,
         it's almost certainly NOT a confirmation prompt — real confirmations
         are short questions. Long outputs with incidental matches (e.g.
         "### One Thing to Verify After Deploy") are hallucinations.
      2. Phrase matching: Checks for specific question-like patterns that
         the agent uses when genuinely asking for permission.
    
    Args:
        output: The agent's output text.
    
    Returns:
        True if the output appears to be a confirmation prompt.
    """
    if not output or not output.strip():
        return False
    
    # Structural guard: long responses are NOT confirmation prompts.
    # A real confirmation is a short question (< 500 chars). Hallucinated
    # outputs that incidentally contain "verify" or "confirm" are typically
    # 1000+ chars with tables, code blocks, and detailed summaries.
    if len(output.strip()) > CONFIRMATION_MAX_LENGTH:
        return False
    
    output_lower = output.lower()
    return any(phrase in output_lower for phrase in CONFIRMATION_PROMPT_PHRASES)


def validate_agent_result(result: dict) -> dict:
    """Validate an AgentExecutor result for signs of hallucination.
    
    Checks whether the agent actually called any tools during execution.
    If intermediate_steps is empty (zero tool calls), the agent's output
    was generated purely from the LLM's memory — not from real file reads,
    command execution, or any grounded action. This is the exact failure
    mode where the agent confidently describes changes it never made.
    
    Exception: Confirmation prompts (where the agent asks the user to
    confirm a destructive action) are legitimate zero-tool-call responses.
    These are detected by is_confirmation_prompt() and exempted from the
    hallucination flag.
    
    This is a deterministic, code-level check. It does NOT rely on the LLM
    to self-report — it inspects the actual execution trace.
    
    Args:
        result: The dict returned by AgentExecutor.ainvoke(), expected to
                contain 'output' (str) and optionally 'intermediate_steps' (list).
    
    Returns:
        Dict with:
            - is_suspicious (bool): True if zero tool calls were made AND
              the output is NOT a confirmation prompt
            - is_confirmation (bool): True if the output is a confirmation prompt
            - tool_count (int): Number of tool calls in intermediate_steps
            - tools_called (list[str]): Names of tools that were called
            - warning (str): Warning message if suspicious, empty string otherwise
            - original_output (str): The raw agent output before any modification
            - output (str): The agent output, prepended with warning if suspicious
    """
    output = result.get("output", "")
    steps = result.get("intermediate_steps", [])

    # Extract tool names from intermediate_steps.
    # Each step is a tuple of (AgentAction, observation).
    # AgentAction has a .tool attribute with the tool name.
    tools_called = []
    for step in steps:
        try:
            action = step[0]
            tool_name = getattr(action, "tool", None)
            if tool_name:
                tools_called.append(tool_name)
        except (IndexError, TypeError):
            # Malformed step — skip it
            continue

    tool_count = len(tools_called)
    if len(steps) > 0 and tool_count == 0:
        print(f"[VALIDATE_DEBUG] {len(steps)} steps but 0 tools extracted! "
              f"step_types={[type(s).__name__ for s in steps[:3]]} "
              f"step0_type={type(steps[0]).__name__ if steps else 'N/A'} "
              f"step0_len={len(steps[0]) if steps and hasattr(steps[0], '__len__') else 'N/A'} "
              f"action_type={type(steps[0][0]).__name__ if steps else 'N/A'} "
              f"action_attrs={dir(steps[0][0])[:10] if steps else 'N/A'}")
    confirmation = is_confirmation_prompt(output)
    
    # Zero tool calls is suspicious UNLESS the agent is asking for confirmation.
    # Confirmation prompts are legitimate — the agent is waiting for user input
    # before performing a destructive action (as instructed by the system prompt).
    is_suspicious = tool_count == 0 and not confirmation
    
    if is_suspicious:
        warning = ZERO_TOOL_CALLS_WARNING
        modified_output = warning + output
    else:
        warning = ""
        modified_output = output

    # Check for unverified access claims: if the output contains access claim
    # phrases (e.g. "has access to", "can submit to") but none of the
    # designated access-verification tools were actually called, prepend a
    # warning so the user knows the claim was not backed by a real tool query.
    output_lower = output.lower()
    has_access_claim = any(phrase in output_lower for phrase in ACCESS_CLAIM_PHRASES)
    used_access_tool = bool(ACCESS_VERIFICATION_TOOLS.intersection(set(tools_called)))
    if has_access_claim and not used_access_tool and not confirmation:
        access_warning = ACCESS_UNVERIFIED_WARNING
        modified_output = access_warning + modified_output
        warning = (warning + "\n" + access_warning).strip() if warning else access_warning
        is_suspicious = True

    return {
        "is_suspicious": is_suspicious,
        "is_confirmation": confirmation,
        "tool_count": tool_count,
        "tools_called": tools_called,
        "warning": warning,
        "original_output": output,
        "output": modified_output,
    }


# ── Tool enforcement retry prompts (anti-hallucination) ───────────────

# Maximum number of retry attempts when an agent returns zero tool calls.
MAX_HALLUCINATION_RETRIES = 3


def build_tool_enforcement_retry_prompt(
    user_query: str,
    attempt: int,
    max_attempts: int = MAX_HALLUCINATION_RETRIES,
) -> str:
    """Build an escalating retry prompt that forces the agent to call tools.
    
    When an agent returns a response without calling any tools, this function
    generates a structured retry prompt that:
    1. Explicitly tells the agent what went wrong (zero tool calls)
    2. Re-states the user's original query so the agent doesn't lose context
    3. Gives clear step-by-step instructions to use tools first
    4. Escalates urgency with each attempt
    
    The prompt is designed to break the LLM out of its "answer from memory"
    pattern by making the tool-call requirement impossible to ignore.
    
    This is a pure function with no side effects — it only builds a string.
    The caller (app.py) is responsible for passing this to the agent executor.
    
    Args:
        user_query: The user's original query text. Must be non-empty.
        attempt: Current retry attempt number (1-indexed). Must be >= 1.
        max_attempts: Total number of retry attempts allowed (default: 3).
    
    Returns:
        A structured retry prompt string that can be appended to the agent input.
    
    Raises:
        ValueError: If user_query is empty or attempt < 1.
    """
    if not user_query or not user_query.strip():
        raise ValueError("user_query must be non-empty")
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    
    # Clamp attempt to max_attempts for escalation lookup
    clamped = min(attempt, max_attempts)
    
    if clamped == 1:
        # First retry: firm but instructive
        return (
            f"Previous attempt: You returned an answer WITHOUT calling any tools.\n"
            f"User query: \"{user_query}\"\n\n"
            f"You MUST call an appropriate tool first. Do NOT answer directly.\n"
            f"Start with a tool call. Think step-by-step:\n"
            f"1. What does the user need?\n"
            f"2. Which tool can provide that information?\n"
            f"3. Call that tool NOW before writing any response."
        )
    elif clamped == 2:
        # Second retry: more forceful, explicit rejection
        return (
            f"MANDATORY TOOL USE — ATTEMPT {attempt}/{max_attempts}:\n"
            f"You have FAILED to use any tools. Your previous response was REJECTED.\n\n"
            f"User query: \"{user_query}\"\n\n"
            f"You MUST call a tool NOW. A response without tool calls will be REJECTED.\n"
            f"Do NOT answer from memory or training data.\n"
            f"Do NOT describe what you would do — actually DO it.\n"
            f"Your VERY FIRST action must be a tool call."
        )
    else:
        # Final retry: last chance, graceful failure option
        return (
            f"FINAL ATTEMPT — {attempt}/{max_attempts}:\n"
            f"You have failed {attempt - 1} times to use your tools.\n\n"
            f"User query: \"{user_query}\"\n\n"
            f"If you can complete this task, you MUST call a tool first.\n"
            f"If you truly cannot complete this task using your available tools, "
            f"say exactly: 'I was unable to complete this task using my "
            f"available tools.' Do NOT fabricate an answer."
        )


# ── Workflow enforcement (skill-agnostic) ─────────────────────────────────────


def validate_workflow_completion(
    result: dict,
    workflow_requirements: List[Dict],
) -> Dict:
    """Check if workflow requirements were completed after trigger tools fired.

    This is a GENERIC validator that works with ANY skill's workflow_required
    frontmatter configuration. It does NOT hardcode any skill name.

    Inspects intermediate_steps to determine whether:
    1. Any trigger tools were called (from workflow_requirements)
    2. All required_after_trigger steps were satisfied

    Args:
        result: The dict returned by AgentExecutor.ainvoke().
        workflow_requirements: List of workflow requirement dicts from
            skill_loader.get_merged_workflow_requirements(). Each dict has:
            - skill_name: which skill this requirement belongs to
            - trigger_tools: list of tool names that activate the workflow
            - required_after_trigger: list of step dicts with:
                - step_name: identifier for the step
                - tool: which tool must be called
                - must_contain_any: list of strings, at least one must appear
                  in the tool_input
                - check_output (optional): if True, check tool output instead
            - skip_allowed: bool
            - skip_requires: string that must appear in output if skipping

    Returns:
        Dict with:
            workflow_complete: bool — True if no triggers fired OR all steps done
            missing_steps: list of step_name strings that were not satisfied
            triggered_skills: list of skill names whose triggers fired
            tools_called: list of all tools that were called
    """
    if not workflow_requirements:
        return {"workflow_complete": True, "missing_steps": [], "triggered_skills": []}

    steps = result.get("intermediate_steps", [])
    tools_called = []
    tool_inputs = {}  # tool_name → list of input strings
    tool_outputs = {}  # tool_name → list of output strings

    for step in steps:
        if hasattr(step, '__len__') and len(step) >= 1:
            action = step[0]
            tool_name = getattr(action, "tool", "")
            tools_called.append(tool_name)
            # Collect inputs
            tool_input = str(getattr(action, "tool_input", ""))
            if tool_name not in tool_inputs:
                tool_inputs[tool_name] = []
            tool_inputs[tool_name].append(tool_input)
            # Collect outputs (second element of the step tuple)
            if len(step) >= 2:
                tool_output = str(step[1]) if step[1] else ""
                if tool_name not in tool_outputs:
                    tool_outputs[tool_name] = []
                tool_outputs[tool_name].append(tool_output)

    tools_called_set = set(tools_called)
    missing_steps = []
    triggered_skills = []

    for wf in workflow_requirements:
        skill_name = wf.get("skill_name", "unknown")
        trigger_tools = wf.get("trigger_tools", [])
        required_steps = wf.get("required_after_trigger", [])

        # Check if any trigger tool was called
        triggered = bool(set(trigger_tools) & tools_called_set)
        if not triggered:
            continue  # This skill's workflow was not activated

        triggered_skills.append(skill_name)

        # Check each required step
        for req_step in required_steps:
            step_name = req_step.get("step_name", "unknown_step")
            is_optional = req_step.get("optional", False)
            req_tool = req_step.get("tool", "")
            must_contain = req_step.get("must_contain_any", [])
            check_output = req_step.get("check_output", False)

            # Skip optional steps — they don't block workflow completion
            if is_optional:
                continue

            # req_tool can be a string OR a list of strings (accept any)
            req_tools = req_tool if isinstance(req_tool, list) else [req_tool]

            # Determine what to search in — union across all accepted tools
            if check_output:
                search_strings = []
                for rt in req_tools:
                    search_strings.extend(tool_outputs.get(rt, []))
            else:
                search_strings = []
                for rt in req_tools:
                    search_strings.extend(tool_inputs.get(rt, []))

            # Check if ANY of the accepted tools was called
            any_tool_called = bool(set(req_tools) & tools_called_set)

            # If must_contain_any is empty, just check if any accepted tool was called
            if not must_contain:
                if not any_tool_called:
                    missing_steps.append(step_name)
            else:
                # Check if any search string contains any of the required patterns
                step_satisfied = any(
                    any(pattern in s for pattern in must_contain)
                    for s in search_strings
                )
                # Also accept if the tool was called with matching test_path arg
                # (run_tests passes test_path directly, not as a substring in commands)
                if not step_satisfied and any_tool_called:
                    # For tools like run_tests where the tool name itself implies testing
                    step_satisfied = any(
                        rt in ("run_tests",) and rt in tools_called_set
                        for rt in req_tools
                    )
                if not step_satisfied:
                    missing_steps.append(step_name)

    return {
        "workflow_complete": len(missing_steps) == 0,
        "missing_steps": missing_steps,
        "triggered_skills": triggered_skills,
        "tools_called": tools_called,
    }


def check_workflow_environment(
    project_dir: str,
    env_checks: List[Dict],
) -> Dict:
    """Check if workflow prerequisites are met in the current environment.

    This is a GENERIC environment checker that interprets env_checks from
    the skill's workflow_required frontmatter. It does NOT hardcode any
    specific checks.

    Supported check types:
        - has_dotgit: checks if .git/ directory exists in project_dir
        - file_exists: checks if a specific file exists (relative to project_dir)
        - command_succeeds: runs a command and checks return code == 0

    Args:
        project_dir: Base directory for relative path checks.
        env_checks: List of check dicts from workflow_required.env_checks.
            Each dict has:
            - name: identifier for the check result
            - check: one of 'has_dotgit', 'file_exists', 'command_succeeds'
            - path: (for file_exists) relative path to check
            - command: (for command_succeeds) shell command to run

    Returns:
        Dict mapping check names to boolean results.
    """
    results = {}  # type: Dict[str, bool]

    for check_def in env_checks:
        name = check_def.get("name", "unknown")
        check_type = check_def.get("check", "")

        if check_type == "has_dotgit":
            results[name] = os.path.isdir(os.path.join(project_dir, ".git"))

        elif check_type == "file_exists":
            rel_path = check_def.get("path", "")
            if rel_path:
                results[name] = os.path.isfile(os.path.join(project_dir, rel_path))
            else:
                results[name] = False

        elif check_type == "command_succeeds":
            command = check_def.get("command", "")
            if command:
                try:
                    proc = subprocess.run(
                        command.split(),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=5,
                    )
                    results[name] = (proc.returncode == 0)
                except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                    results[name] = False
            else:
                results[name] = False

        else:
            # Unknown check type — fail closed
            results[name] = False

    return results


def build_workflow_context_for_planner(
    workflow_requirements: List[Dict],
    project_dir: str,
) -> str:
    """Build a context block for the planner describing available verification.

    Runs env_checks and formats a concise summary that the planner uses to
    decide whether to include verification steps in its plan.

    Args:
        workflow_requirements: Merged workflow requirements from active skills.
        project_dir: Base directory for environment checks.

    Returns:
        Formatted context string, or empty string if no requirements exist.
    """
    if not workflow_requirements:
        return ""

    lines = ["## Verification Context (from active skills)"]

    for wf in workflow_requirements:
        skill_name = wf.get("skill_name", "unknown")
        trigger_tools = wf.get("trigger_tools", [])
        required_steps = wf.get("required_after_trigger", [])
        env_checks_defs = wf.get("env_checks", [])

        env_results = check_workflow_environment(project_dir, env_checks_defs)

        step_names = [s["step_name"] for s in required_steps if not s.get("optional")]
        env_summary = ", ".join(
            f"{k}: {'yes' if v else 'no'}" for k, v in env_results.items()
        )

        lines.append(
            f"- Skill '{skill_name}': expects {step_names} after using "
            f"{trigger_tools}. Environment: [{env_summary}]"
        )

    lines.append("")
    lines.append(
        "Include verification steps in your plan ONLY when appropriate for the task:\n"
        "- Editing tested source code in a git repo with tests → include 'Run tests' step.\n"
        "- Submitting SLURM jobs → include 'Verify job queued' step.\n"
        "- Writing new scripts/configs, no existing tests → do NOT require unit tests.\n"
        "- Use your judgment. The system will NOT force verification beyond your plan."
    )

    return "\n".join(lines)


