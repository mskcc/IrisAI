"""Phase configuration for harness-enforced phased execution.

Defines the three execution phases (research, plan, execute), their tool sets,
system prompt additions, completion triggers, and validation rules.

The main executor and sub-agents use this to structurally enforce phase
boundaries — tools not in the phase's set literally do not exist in the
API call.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# Tools that are ALWAYS available in read-only phases (research + plan).
# These are the tool NAMES as registered in the MCP/LangChain tool pool.
READONLY_TOOL_NAMES = frozenset({
    "read_text_file",
    "grep_file",
    "find_files",
    "list_directory",
    "query_slurm_cluster",
    "check_user_slurm_access",
    "analyze_files",
    "review_codebase_section",
    "summarize_command_output",
    # Read-only wrappers for phase enforcement
    "execute_shell_readonly",
    "batch_readonly",
})

# Tools that signal phase completion
PHASE_COMPLETION_TOOLS = {
    "research": "write_findings",
    "plan": "write_plan",
}

# Tools available per phase (names). None means "all tools available".
# Sub-agent helpers (run_research_agent, run_plan_agent) are optional —
# the main agent can call them for parallel or isolated investigation,
# but is not forced to. The primary path is direct tool use.
# Meta-tools (request_additional_skill, run_worker_agent) are always available
# so the agent can escalate or delegate regardless of phase.
_ALWAYS_AVAILABLE_TOOLS = {
    "request_additional_skill",
    "run_worker_agent",
    "get_environment_info",
    "read_memory",
    "query_software",
    "list_projects",
}
WEBSEARCH_TOOL_NAMES = frozenset({"web_search", "fetch_url_content", "fetch_web_image"})

PHASE_TOOL_SETS = {
    "research": READONLY_TOOL_NAMES | {"write_findings", "run_research_agent"} | _ALWAYS_AVAILABLE_TOOLS,
    "plan": READONLY_TOOL_NAMES | {"write_plan", "edit_plan", "run_plan_agent"} | _ALWAYS_AVAILABLE_TOOLS,
    "execute": None,  # All tools (minus EXECUTE_EXCLUDED below)
}

# Tools to EXCLUDE from execute phase — read-only variants that confuse the model
# when the full-capability versions (batch, execute_dynamic_task) are available.
EXECUTE_EXCLUDED_TOOLS = frozenset({
    "batch_readonly",
    "execute_shell_readonly",
})

# System prompt additions per phase (appended to base system prompt).
# These are strong, unambiguous constraints — not suggestions. The tool
# restriction is structural (tools don't exist in the API schema), and
# these prompts reinforce behavioral expectations.
PHASE_SYSTEM_PROMPTS = {
    "research": (
        "\n\n<phase-constraint>\n"
        "You are in RESEARCH MODE. You are gathering facts and information ONLY.\n\n"
        "WHAT YOU CAN DO:\n"
        "- Read files (read_text_file, grep_file, find_files, list_directory)\n"
        "- Run read-only shell commands (execute_shell_readonly)\n"
        "TOOL RULE: When you need 2+ operations (reads, greps, finds, shell commands), ALWAYS use batch_readonly in ONE call. Do NOT call read_text_file, grep_file, find_files, or execute_shell_readonly multiple times in sequence.\n"
        "- Query SLURM state (query_slurm_cluster, check_user_slurm_access)\n"
        "- Analyze file content (analyze_files, review_codebase_section)\n"
        "- Recall project memory (read_memory, list_projects)\n"
        "- Query software registry (query_software, get_environment_info)\n"
        "- Optionally delegate a sub-task to run_research_agent helper\n\n"
        "EXPLORATION STRATEGY:\n"
        "- For web searches or multi-step web exploration: use run_research_agent with a focused\n"
        "  question — it will search, fetch pages, and return summarized findings\n"
        "- For cluster/system queries that may produce large output (partitions, job lists,\n"
        "  module lists): prefer run_research_agent — it handles large outputs effectively\n"
        "- For small/focused lookups (grep, find_files, list_directory, squeue for your jobs):\n"
        "  call directly — results are compact and stay in your context\n\n"
        "WHAT YOU CANNOT DO:\n"
        "- Write, edit, or create files\n"
        "- Execute destructive or modifying commands\n"
        "- Plan implementation steps (that comes in the next phase)\n"
        "- Implement solutions or write code\n"
        "- Even if the user's request says 'write X' or 'implement Y', you are ONLY\n"
        "  gathering the information needed to do that later\n\n"
        "WHEN DONE:\n"
        "Call write_findings() with ALL your discoveries. Include:\n"
        "- Exact file paths and line numbers\n"
        "- Current state and configurations\n"
        "- Constraints and dependencies\n"
        "- Facts only — no suggestions, no plans, no action items\n\n"
        "Your turn can ONLY end productively by calling write_findings().\n"
        "There is no other way to advance. Every other action is preparation for that call.\n"
        "This constraint is structural — write/edit tools do not exist in this phase.\n"
        "</phase-constraint>"
    ),
    "plan": (
        "\n\n<phase-constraint>\n"
        "You are in PLANNING MODE. You are writing an implementation plan ONLY.\n\n"
        "WHAT YOU CAN DO:\n"
        "- Read files to verify details for your plan\n"
        "- Run read-only shell commands to check current state\n"
        "- Optionally delegate investigation to run_plan_agent helper\n\n"
        "WHAT YOU CANNOT DO:\n"
        "- Implement anything — no file writes, no code execution\n"
        "- Submit jobs, install packages, or modify system state\n"
        "- Even if the user says 'go ahead' or 'do it', you are ONLY writing the plan\n"
        "- Do NOT write the actual script/code — describe what it should contain\n\n"
        "WHEN DONE:\n"
        "Call write_plan() with a step-by-step plan. Include:\n"
        "- Exact file paths and commands for each step\n"
        "- For code changes: describe what to find and what to replace with\n"
        "- Verification/test steps after each major change\n"
        "- Number steps sequentially\n"
        "- Mark parallelizable steps with [PARALLEL_GROUP]\n\n"
        "Your turn can ONLY end productively by calling write_plan().\n"
        "There is no other way to advance. Every other action is preparation for that call.\n"
        "This constraint is structural — write/edit tools do not exist in this phase.\n"
        "</phase-constraint>"
    ),
    "execute": (
        "\n\n<phase-constraint>\n"
        "You are in EXECUTION MODE. All tools are available.\n"
        "Follow the plan step by step, including any verification/test steps.\n"
        "After completing each step, call edit_plan to mark it done (- [ ] → - [x]).\n"
        "Do not stop until all steps are marked [x].\n\n"
        "TOOL RULE: When you need 2+ operations (reads, greps, finds, shell commands, edits, tests), ALWAYS use 'batch' in ONE call. "
        "Do NOT call individual tools multiple times in sequence — batch is faster and keeps context compact.\n\n"
        "BEFORE WRITING CODE:\n"
        "- Check what packages are available: get_environment_info('packages')\n"
        "- Read usage knowledge for your chosen tool: get_environment_info('package:<name>')\n"
        "- The knowledge file contains 'When to Use' and 'When NOT to Use' guidance\n"
        "</phase-constraint>"
    ),
}

# Minimum quality bar for phase output validation
MIN_RESEARCH_CHARS = 200
MIN_PLAN_STEPS = 3


@dataclass
class PhaseConfig:
    """Configuration for phased execution.

    Passed to NativeAgentExecutor to enable phase-aware behavior.
    When None is passed, the executor runs in simple mode (no phase gating).
    """

    needs_research: bool = False
    needs_planning: bool = False
    initial_phase: str = "execute"
    phases: list = field(default_factory=lambda: ["research", "plan", "execute"])

    def __post_init__(self):
        if self.needs_research:
            self.initial_phase = "research"
        elif self.needs_planning:
            self.initial_phase = "plan"
        else:
            self.initial_phase = "execute"

        # Build phase sequence based on what's needed
        active_phases = []
        if self.needs_research:
            active_phases.append("research")
        if self.needs_planning:
            active_phases.append("plan")
        active_phases.append("execute")
        self.phases = active_phases

    def get_tool_names(self, phase: str) -> Optional[frozenset]:
        """Get allowed tool names for a phase. None means all tools allowed."""
        return PHASE_TOOL_SETS.get(phase)

    def get_system_prompt_addition(self, phase: str, websearch_enabled: bool = False) -> str:
        """Get the system prompt text to append for this phase."""
        base = PHASE_SYSTEM_PROMPTS.get(phase, "")
        if phase == "research" and websearch_enabled:
            base = base.replace(
                "- Even if the user's request says 'write X' or 'implement Y', you are ONLY\n"
                "  gathering the information needed to do that later\n\n"
                "WHEN DONE:",
                "- Even if the user's request says 'write X' or 'implement Y', you are ONLY\n"
                "  gathering the information needed to do that later\n\n"
                "WEB SEARCH AVAILABLE:\n"
                "- web_search(query) — search for best practices, reference implementations, style guides\n"
                "- fetch_url_content(url) — read documentation or example code from search results\n"
                "- Proactively search when the task involves: scientific visualization, complex libraries,\n"
                "  publication-quality output, or any domain where web references improve quality\n"
                "- Include key web findings in your write_findings() output\n\n"
                "AVAILABLE SOFTWARE:\n"
                "- Call get_environment_info('packages') to see all installed scientific packages\n"
                "- Call get_environment_info('package:<name>') to read detailed usage knowledge for a specific tool\n"
                "- Call get_environment_info('category:<category>') to find tools by category (e.g., 'visualization', 'svg')\n"
                "- Include available packages relevant to the task in your write_findings() output\n\n"
                "WHEN DONE:",
            )
        return base

    def get_completion_tool(self, phase: str) -> Optional[str]:
        """Get the tool that signals this phase is complete."""
        return PHASE_COMPLETION_TOOLS.get(phase)

    def next_phase(self, current: str) -> Optional[str]:
        """Get the next phase after current. None if at the end."""
        try:
            idx = self.phases.index(current)
            if idx < len(self.phases) - 1:
                return self.phases[idx + 1]
        except ValueError:
            pass
        return None

    def validate_phase_output(self, phase: str, output: str) -> bool:
        """Validate that phase output meets minimum quality bar.

        Returns True if output is sufficient to advance to next phase.
        """
        if not output or not output.strip():
            return False

        if phase == "research":
            # Must have substantive content with concrete references
            has_paths = "/" in output
            has_detail = len(output) > MIN_RESEARCH_CHARS
            return has_paths and has_detail

        if phase == "plan":
            # Must have at least N steps (bullet or numbered)
            bullet_count = output.count("- ") + output.count("- [")
            number_count = sum(1 for c in "123456789" if f"{c}." in output)
            return (bullet_count + number_count) >= MIN_PLAN_STEPS

        return True


def filter_tools_for_phase(all_tools: list, phase: str, websearch_enabled: bool = False) -> list:
    """Filter a tool list to only include tools allowed in the given phase.

    Args:
        all_tools: Full list of tool objects (must have .name attribute)
        phase: Current phase ("research", "plan", or "execute")
        websearch_enabled: When True and phase is "research", include web search tools

    Returns:
        Filtered list of tools. For "execute" phase, returns all tools.
    """
    allowed_names = PHASE_TOOL_SETS.get(phase)
    if allowed_names is None:
        filtered = [t for t in all_tools if t.name not in EXECUTE_EXCLUDED_TOOLS]
        logger.info(
            f"[PHASE_FILTER] Phase '{phase}': {len(filtered)}/{len(all_tools)} tools "
            f"(excluded: {sorted(EXECUTE_EXCLUDED_TOOLS)})"
        )
        return filtered

    if websearch_enabled and phase == "research":
        allowed_names = allowed_names | WEBSEARCH_TOOL_NAMES

    filtered = [t for t in all_tools if t.name in allowed_names]
    logger.info(
        f"[PHASE_FILTER] Phase '{phase}': {len(filtered)}/{len(all_tools)} tools "
        f"(allowed: {sorted(allowed_names)})"
    )
    return filtered
