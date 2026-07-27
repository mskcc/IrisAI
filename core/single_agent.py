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

"""Single agent with dynamic skill selection — Phase 1 core engine.

Replaces the multi-agent + supervisor architecture with a single
AgentExecutor that uses structured output to select one or more skills
per turn. The skill manifest is auto-generated from skills/*.md files
by SkillLoader.

Architecture:
    1. User message arrives
    2. Skill selector (structured output) picks 1+ skills from manifest
    3. System prompt = base instructions + user context + merged skill content
    4. Tools filtered to union of selected skills' allowed_tools
    5. AgentExecutor runs with filtered tools and merged prompt
    6. max_iterations set from primary skill's frontmatter

Pure functions (testable without langchain):
    - build_skill_selection_prompt()
    - filter_tools_for_skills()
    - build_agent_system_prompt()
    - parse_skill_selection()
    - escape_prompt_braces()
    - detect_escalation_in_result()
    - format_intermediate_steps_for_handoff()

LangChain-dependent (runtime only):
    - create_skill_based_agent()
    - request_additional_skill (tool)

Phase 1 of the IrisAI architecture redesign.
"""
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from core.skill_loader import SkillLoader

logger = logging.getLogger("core.single_agent")


# ── Escalation marker for programmatic detection ─────────────────────────
# When the agent calls request_additional_skill, the tool output includes
# this marker followed by the skill name. app.py inspects intermediate_steps
# for this marker to detect escalation and re-create the executor with
# the expanded skill set — all within the SAME turn.
ESCALATION_MARKER = "ESCALATION_REQUESTED:"

# ── Stuck-detection marker for web search suggestion ─────────────────────
# When the agent is stuck (repeated identical errors), app.py detects this
# and prompts the user to enable web search so the agent can look up docs.
WEBSEARCH_NEEDED_MARKER = "WEBSEARCH_NEEDED:"


def detect_stuck_needs_websearch(result: dict, threshold: int = 3) -> Dict[str, Any]:
    """Detect if the agent is stuck in a loop with repeated identical errors.

    Scans intermediate_steps for the pattern: same tool called N+ times
    with the same error substring. If detected, suggests a web search query
    derived from the error context.

    Args:
        result: The dict returned by AgentExecutor.ainvoke(), expected to
                contain 'intermediate_steps'.
        threshold: Number of repeated identical errors to trigger detection.

    Returns:
        Dict with:
            - stuck (bool): True if repeated error pattern detected
            - tool_name (str): The tool that kept failing
            - error_pattern (str): The repeated error substring (truncated)
            - suggested_query (str): A suggested web search query
            - failure_count (int): How many times the same error occurred
    """
    steps = result.get("intermediate_steps", [])
    if len(steps) < threshold:
        return {"stuck": False, "tool_name": "", "error_pattern": "",
                "suggested_query": "", "failure_count": 0}

    # Count (tool_name, error_snippet) occurrences
    error_counts: Dict[str, int] = {}
    error_details: Dict[str, tuple] = {}  # key -> (tool_name, full_error)
    import re as _re  # noqa: PLC0415 — imported here to avoid top-level cost

    for step in steps:
        try:
            action = step[0]
            observation = str(step[1] if len(step) > 1 else "")
            tool_name = getattr(action, "tool", None) or ""

            # First: check for explicit success/failure signals in MCP responses.
            # MCP tool responses contain "success":true or "isError":false when
            # the tool succeeded — even if the *content* mentions error-related
            # words (e.g. reading a Python file that has try/except blocks).
            # Treat these as non-errors regardless of keyword matches.
            obs_lower = observation.lower()
            if '"success":true' in obs_lower or '"success": true' in obs_lower:
                continue  # Tool reported success — not a stuck error
            if '"iserror":false' in obs_lower or '"iserror": false' in obs_lower:
                continue  # Tool reported no error — not a stuck error

            # Heuristic: observation contains error indicators
            is_error = any(kw in obs_lower for kw in [
                "error", "traceback", "exception", "failed", "failure",
                "not found", "no such", "fatal", "permission denied",
                "not readable", "not writable",
                "modulenotfounderror", "importerror",
                "attributeerror", "nameerror", "typeerror",
            ])
            if not is_error:
                continue

            # Normalize fingerprint: strip timestamps, line numbers, hex
            # addresses, and variable path prefixes so the same logical
            # error always maps to the same key even if details vary.
            normalized = observation
            # Strip ISO timestamps (2024-01-01T12:34:56, 2024-01-01 12:34:56)
            normalized = _re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?', '', normalized)
            # Strip Unix timestamps (10-digit numbers)
            normalized = _re.sub(r'\b\d{10,13}\b', '', normalized)
            # Strip hex addresses (0x7f3a...)
            normalized = _re.sub(r'0x[0-9a-fA-F]{4,}', '0xADDR', normalized)
            # Strip line numbers in tracebacks ("line 123,")
            normalized = _re.sub(r'line \d+', 'line N', normalized)
            # Strip UUIDs / job IDs
            normalized = _re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', 'UUID', normalized)
            # Strip Slurm job IDs (pure numbers > 5 digits)
            normalized = _re.sub(r'\bjob\s+\d{5,}\b', 'job N', normalized, flags=_re.IGNORECASE)
            # Collapse whitespace
            normalized = _re.sub(r'\s+', ' ', normalized).strip()

            # Use first 200 chars of normalized error as fingerprint
            error_snippet = normalized[:200]
            key = f"{tool_name}::{error_snippet}"
            error_counts[key] = error_counts.get(key, 0) + 1
            error_details[key] = (tool_name, observation)
        except (IndexError, TypeError, AttributeError):
            continue

    # Find the most repeated error
    if not error_counts:
        return {"stuck": False, "tool_name": "", "error_pattern": "",
                "suggested_query": "", "failure_count": 0}

    max_key = max(error_counts, key=error_counts.get)
    max_count = error_counts[max_key]

    if max_count < threshold:
        return {"stuck": False, "tool_name": "", "error_pattern": "",
                "suggested_query": "", "failure_count": 0}

    tool_name, full_error = error_details[max_key]
    # Truncate error for display
    error_display = full_error[:200].strip()
    if len(full_error) > 200:
        error_display += "..."

    # Skip web search suggestion for local/OS errors (permissions, missing files, disk)
    from core.stuck_detection_callback import _is_local_error
    if _is_local_error(full_error):
        return {"stuck": True, "tool_name": tool_name, "error_pattern": error_display,
                "suggested_query": "", "failure_count": max_count, "is_local": True}

    # Generate suggested search query from error context
    # Extract key terms: module names, function names, error types
    suggested_query = _build_search_query_from_error(tool_name, full_error)

    return {
        "stuck": True,
        "tool_name": tool_name,
        "error_pattern": error_display,
        "suggested_query": suggested_query,
        "failure_count": max_count,
    }


def _build_search_query_from_error(tool_name: str, error_text: str) -> str:
    """Extract a useful web search query from an error message.

    Looks for common patterns like ModuleNotFoundError, AttributeError,
    API method names, package names, etc.
    """
    import re

    query_parts = []

    # Extract Python error type (e.g. "ModuleNotFoundError: No module named 'foo'")
    error_type_match = re.search(
        r"(ModuleNotFoundError|ImportError|AttributeError|TypeError|"
        r"NameError|ValueError|KeyError|FileNotFoundError)"
        r"[:\s]+([^\n]{0,80})", error_text
    )
    if error_type_match:
        query_parts.append(error_type_match.group(0).strip()[:100])

    # Extract module/package names (e.g. "import alphagenome" or "from foo.bar")
    module_match = re.search(
        r"(?:import|from)\s+([\w.]+)", error_text
    )
    if module_match:
        query_parts.append(module_match.group(1))

    # Extract "no attribute 'X'" patterns
    attr_match = re.search(
        r"has no attribute ['\"](\w+)['\"]", error_text
    )
    if attr_match:
        query_parts.append(f"API method {attr_match.group(1)}")

    # If we got useful parts, combine them
    if query_parts:
        return " ".join(query_parts) + " python documentation"

    # Fallback: use first meaningful line of error
    first_line = error_text.split("\n")[-1].strip()[:80]
    if first_line:
        return f"{first_line} python"

    return f"{tool_name} error documentation"

# Maximum number of escalation re-creation loops per turn.
# Prevents infinite loops if the agent keeps requesting skills.
MAX_ESCALATION_ITERATIONS = 2



# ── Base agent instructions (always prepended to system prompt) ──────────

BASE_AGENT_INSTRUCTIONS = """\
You are IRIS, an AI assistant for the IRIS HPC research computing environment.
You help researchers with computational tasks using your available tools.

RULES:
1. NEVER claim success unless the tool returned success. Report exact errors. No fake output.
2. For destructive actions (remove_file, overwrite), ask user confirmation FIRST.
3. CONFIDENCE TAGGING: [VERIFIED], [PARTIALLY VERIFIED], or [UNVERIFIED ASSUMPTION].
4. If answerable from context alone: respond directly with SKIP REASON: <explanation>.

TOOL SELECTION:
- execute_dynamic_task: primary tool — runs bash scripts, pipes, loops, conditionals.
- submit_slurm_job: for tasks >5 minutes or needing GPUs/high memory.
- execute_dynamic_task has a HARD 5-MINUTE (300s) timeout. Heavy compute → Slurm.
- EFFICIENCY: Use the 'batch' tool for multiple operations (reads, greps, shell commands).
  One batch call with 5 operations = 1 round-trip. Five separate tool calls may cost 5 round-trips.
  Reserve execute_dynamic_task ONLY for multi-step scripts where steps depend on each other's output.
- Do NOT re-read files or re-run commands if output already exists in your prior messages.
- PRECISION RULE: When you need to count, sum, rank, or aggregate from large tool output,
  do NOT attempt it yourself — use execute_dynamic_task with shell commands (awk, sort,
  uniq -c, wc -l, grep -c). Shell pipelines are exact; LLM counting from raw text is not.

CORE INFRASTRUCTURE TOOLS (always available regardless of skill):
get_environment_info, read_memory, update_memory, list_projects, add_project,
remove_project, render_image_inline.
- get_environment_info: returns CLUSTER-LEVEL info (shared containers, partitions, system software).
  Use for: "what GPUs exist?", "which container has X?", "what partitions are available?"
- query_software: returns USER-LEVEL installed software (conda envs, personal installs, built containers).
  Use for: "is X already installed?", "where is my samtools?", "do I have a scipy container?"

SOFTWARE REGISTRY PROTOCOL (MANDATORY — register_software / query_software):
These rules apply regardless of which skill is active. They are NON-NEGOTIABLE.

FIRST TOOL CALL for any software/install/container request:
  query_software(search="{name}")
This is your VERY FIRST action — before get_environment_info, before execute_dynamic_task,
before anything else. query_software checks whether the user already has it installed.
- If found: use the existing path. Do NOT reinstall or rebuild. Tell user it's available.
- If not found: proceed with installation or building. Do NOT search the filesystem manually
  (no find, no ls, no grep for binaries). query_software IS the authoritative source.
  If it says "not found", the software is not installed — trust it and move on.

POST-INSTALL REGISTRATION (register_software):
- AFTER successfully installing software, creating a conda env, building a container (.sif),
  or discovering software at a user-provided path: ALWAYS call register_software.
- For containers: source="container", categories=["container", "{domain}"].
- For conda envs: source="conda" with the environment prefix path.
- MANDATORY — no exceptions. Skipping means future sessions cannot find the software.

RULES:
- register_software is the ONLY way to persist software locations.
- update_memory CANNOT substitute — software paths in update_memory are INVISIBLE to future sessions.
- If you are recording a software path, binary, conda env, container .sif, or version → register_software is REQUIRED.

ESCALATION: If you need a tool from "TOOLS IN OTHER SKILLS", call
request_additional_skill as your ONLY tool call.

RESEARCH PHASE: Call write_findings to document discoveries (facts only, no plans).
PLANNING PHASE: Call write_plan with checkbox steps (- [ ] Step N: description).
EXECUTION: Follow the active plan step by step; mark - [x] as you complete each.
"""

# Legacy orchestrator sections — no longer used.
# Phase enforcement is now structural: the executor only receives the sub-agent
# tool during gated phases, so no prompt replacement is needed.

MEMORY_CHECK_PREAMBLE = """\
BEFORE ACTING — CHECK PROJECT MEMORY:
1. Call read_memory(project=PROJECT_NAME) to read status + knowledge + history
2. Call query_software() to see what software is registered (paths, versions, tools)
3. CONFORM your approach to constraints in knowledge.md — they are RULES, not suggestions
4. If your plan would violate a constraint, CHANGE YOUR PLAN — do not proceed

MEMORY OWNERSHIP — PERSIST IMPORTANT DISCOVERIES IMMEDIATELY:
- After a status change (job submitted/completed/failed): call update_memory("status.md", NEW_FULL_CONTENT, project=PROJECT)
- After learning a permanent fact: call update_memory("knowledge.md", APPENDED_CONTENT, project=PROJECT)
- After installing or discovering software: call register_software(...) — NOT update_memory
Do NOT wait for the system to capture these — write them immediately.

When writing to knowledge.md, structure each entry as:
- [TYPE]: <exact fact — real paths, versions, IDs, commands>
  Why: <root cause or reasoning>
  Applies when: <trigger condition for future sessions>
TYPE = CONSTRAINT | VALIDATED_APPROACH | FAILED_ATTEMPT | REFERENCE_PATH | CONFIGURATION
"""

EVIDENCE_GATED_TRUST_RULES = """\
=== MEMORY TRUST PROTOCOL ===

Your context contains project memory (status.md + knowledge.md). Follow these rules:

TRUST LEVELS:
1. Constraints in knowledge.md are BINDING RULES — you MUST follow them.
   Override ONLY if you have concrete tool output THIS SESSION proving they are stale.
   If you override: call update_memory("knowledge.md", corrected_content, project=..., reason="...").

2. Paths, configs, software versions — trust by default.
   If a tool call shows different state → update_memory("knowledge.md", ...) to fix.

3. RECENT ATTEMPTS (shown in read_memory output) — what was tried before.
   If it failed: try a DIFFERENT approach unless evidence shows conditions changed.

CONFLICT RESOLUTION:
- State conflicts explicitly: "Memory says X, but I observed Y"
- Act on the stronger/more current evidence
- WRITE-THROUGH: Call update_memory() to fix stale entries immediately
  - For knowledge.md: APPEND an exception note, do NOT delete original entries
  - For status.md: REPLACE content entirely with current state
- NEVER silently ignore a conflict between memory and observation
=== END MEMORY TRUST PROTOCOL ===
"""


# ── Skill selection prompt template ──────────────────────────────────────────

SKILL_SELECTION_PROMPT_TEMPLATE = """\
You are a skill router for the IRIS AI assistant. Based on the user's \
request, select the most appropriate skill(s) to handle it.

{manifest}

CRITICAL — EACH SKILL HAS LIMITED TOOLS:
Each skill only has access to its own domain-specific tools. The agent \
CANNOT use tools from skills that are not selected. When you select \
multiple skills, their tools are MERGED — giving the agent a combined \
toolset. If a task needs capabilities from two domains, you MUST select \
both skills.

Rules:
- Select ONE skill for focused, single-domain requests.
- Select MULTIPLE skills when the task needs tools from different domains. \
Think about what the agent will need to DO, not just what the topic is.
- The PRIMARY skill drives model selection and iteration limits.
- IMPORTANT: Consider the conversation context below when interpreting \
short or ambiguous messages like "yes", "go ahead", "proceed", "do it", \
"use defaults". These are continuations of the previous topic, NOT new requests.

CRITICAL — capability-based routing:
- Each skill has DIFFERENT tools. Read the skill descriptions carefully — they tell \
you what actions each skill can perform.
- For any user request, ask yourself: "What tools does the agent need to complete \
this task?" Then select the skill(s) that provide those tools.
- "conversational" is ONLY for requests that need NO specialist tools — greetings, \
general knowledge questions, basic info retrieval. If the task requires the agent to \
DO something (any action beyond answering from memory), a specialist skill is required.
- When selecting multiple skills, their tools are merged. Select all skills whose \
tools are needed to complete the full task.

Common multi-skill combinations:
- history + file-operations: when user needs to find/read files related to past sessions
- hpc-submit-job + file-operations: when job needs input files uploaded or created
- alphafold + file-operations: when user needs to upload sequences for structure prediction
- dev + file-operations: when coding tasks need file discovery or creation

RESEARCH DECISION (needs_research field):
- WHAT RESEARCH MEANS: Setting needs_research=true launches an exploration agent \
that reads files, lists directories, and writes a structured findings report BEFORE \
the user's task is attempted. This ensures the execution agent has full context \
and avoids shallow or incorrect answers. Research costs extra time but PREVENTS \
costly mistakes from acting without understanding.
- Set TRUE when ANY of these apply:
  (1) The user asks to INVESTIGATE, UNDERSTAND, EXPLORE, AUDIT, COMPARE, or REVIEW \
something where the answer requires reading multiple files/configs/logs
  (2) The task involves understanding an unfamiliar codebase or system architecture
  (3) The user asks "why" something is happening (root cause analysis)
  (4) The task requires comparing multiple directories, environments, or configurations
  (5) A comprehensive code review or security audit is requested
  Examples: "how does the auth module work?", "why is the build failing?", \
"investigate the memory leak", "what's in the config directory?", "explore this project", \
"review the codebase", "compare these two versions", "audit the configs"
- Set FALSE when:
  * The user explicitly says to create/write/make/build/run/execute/install something
  * Tasks involving "worker agent", "write files", "use X tool to do Y"
  * Multi-step tasks with explicit modification instructions where the target is known
  * Simple questions, greetings, history/save operations
  * The user has already provided all the information needed to act
- PRINCIPLE: It is better to explore and understand BEFORE acting than to act \
blindly and produce shallow results. A diligent agent reads before it writes.
- IMPORTANT: Research phase only allows READ-ONLY tools (read_text_file, grep_file, \
find_files, list_directory, query_slurm_cluster, check_user_slurm_access, analyze_files, \
review_codebase_section, summarize_command_output, execute_shell_readonly). Only set \
needs_research=true if the selected skill(s) provide enough of these tools for the investigation.

PLANNING DECISION (needs_planning field):
- Set needs_planning=true when the task requires multiple coordinated steps that \
produce artifacts or change system state. Examples:
  * Multi-step installs, setting up new environments/pipelines, building from source
  * Complex multi-file debugging, orchestrating multi-system workflows
  * Tasks with multiple independent deliverables (e.g. "write report A AND report B")
  * Tasks that explicitly describe 2+ subtasks that each require exploration/execution
  * Any request with numbered sub-tasks or "(1)...(2)..." structure requiring file writes
- Set needs_planning=false for: questions, status checks, single-file edits, \
quick lookups, continuing existing work, history/save operations, showing/listing \
information, simple multi-step queries that just gather and display info. \
Planning adds latency and cost — only use when genuinely needed.
- IMPORTANT: If the active project status (below) shows an active multi-step plan, \
pending steps, blockers, or a history of failures — set needs_planning=true for \
continuation/resume requests. The user may phrase it simply ("go ahead", "resume") \
but the underlying task is complex.

SLURM DECISION (needs_slurm field):
- Set true when the task involves heavy or long-running compute that should NOT \
run directly. The system enforces a HARD 5-MINUTE KILL on direct execution.
- The key question: could this task take more than 5 minutes, or does it need \
dedicated resources (GPU, high memory, many cores)?
- Set true for: video/audio processing (ffmpeg, encoding), ML training/inference, \
large compilations (make -j, cargo build), genome alignment (bwa, bowtie2, STAR), \
molecular dynamics (gromacs, namd, amber), large data transforms (>1GB input), \
scientific simulations, rendering, batch array jobs, any workload that processes \
large datasets or runs iterative computation.
- Set false for: quick shell commands, pip install, file operations, environment \
setup, status checks, small scripts that finish in seconds, reading/writing files, \
checking cluster status, listing directories.
- When true: the agent will be given hpc-submit-job + Slurm submission tools and \
instructed to use Slurm for the heavy work. Direct execution will be blocked.

PARALLEL SUBTASKS DECISION (parallel_subtasks field):
- Set true when the task requires gathering information from 2+ INDEPENDENT \
sources where one result does NOT change what you'd look at in the others.
- The key question: can work be split into separate threads where NONE \
depends on the output of another?
- Set true for: debugging/investigation (read the failing code + read the \
implementation + run diagnostics — all independent), "compare A and B" \
(2 independent reads), "check status of X, Y, and Z" (3 independent \
queries), code review (read module + read tests + check callers), \
root cause analysis (read logs + read code + check config).
- Set false for: sequential workflows (step 2 needs step 1), single-file \
tasks with one file to read, anything where discovering X determines \
what to look at next.
- When true: research/planning/execution phases will use run_worker_agent to \
dispatch independent sub-tasks in parallel for efficiency.
- COST SIGNAL: If needs_research=true AND the task involves understanding \
multiple files or components, parallel_subtasks should likely be true.
- ADVISORY INPUT: If research findings or a plan exist in the conversation context, \
factor them into your decision alongside your own judgment about the task structure.

WEBSEARCH DECISION (needs_websearch field):
- Set true when web information would SIGNIFICANTLY improve the answer — even \
if a partial answer is possible from local files or LLM knowledge alone.
- Bias toward true: it is better to suggest web search and let the user decline \
than to miss an opportunity for better, more current information.
- Set true for: questions about external tools/software/libraries, documentation \
lookups, troubleshooting errors or unexpected behavior, version compatibility, \
best practices for tools/frameworks, anything where the LLM's knowledge may be \
stale or incomplete, scientific/research topics, configuration guidance for \
third-party software (SLURM, conda, Docker, etc.), "how do I..." questions \
about external systems.
- Set false ONLY for: tasks purely answerable from local files with no external \
context needed, code exploration within the project, file operations, running \
local commands, comparing local directories.
- Examples TRUE: "research AlphaFold3 recent updates", "how to configure SLURM \
for multi-GPU", "what's new in PyTorch 2.5", "find documentation for this library", \
"why is my conda environment failing", "best way to submit array jobs", \
"how does this tool handle X"
- Examples FALSE: "compare these two directories", "read this config file", \
"run the tests", "what's in app.py"

PROJECT CONTEXT SWITCH DETECTION (project_context + is_context_switch fields):
- project_context: Identify which project the user is talking about.
- RULE: You MUST pick from the KNOWN PROJECTS list below whenever possible. \
Only create a brand-new name if the user is discussing genuinely new work that \
does NOT match ANY existing project.
- Use "general" for: greetings, system questions, meta-discussions about IrisAI \
itself, quick one-off questions not tied to any specific project.
- Use "__ask_user__" when: you cannot confidently choose between 2+ projects, \
or the topic is ambiguous. This asks the user to clarify.
- Set to None ONLY for completely content-free messages (empty, pure emoji).
- is_context_switch: Set true ONLY if project_context is different from the currently \
active project shown below. This triggers auto-save of outgoing project state.
- Currently active project: {active_project}
- KNOWN PROJECTS (pick from these — includes descriptions where available): \
{known_projects}

{conversation_summary}

Recent conversation context:
{recent_history}

Current user request: {user_input}

Respond with the skill name(s) and brief reasoning."""


# ── Pydantic models for structured output ────────────────────────────────────

class SkillSelection(BaseModel):
    """Structured output for skill selection.

    The LLM returns this to indicate which skill(s) to activate.
    The 'skills' field accepts any string — validation against loaded
    skill names is done in parse_skill_selection().
    """
    skills: List[str] = Field(
        description="List of skill names to activate. Usually 1, sometimes 2-3 for multi-domain requests."
    )
    primary_skill: str = Field(
        description="The main skill driving model selection and iteration limits."
    )
    reasoning: str = Field(
        description="Brief explanation of why these skills were selected."
    )
    complexity: str = Field(
        default="standard",
        description=(
            "Task complexity: 'simple' (trivial single-line edits), "
            "'standard' (most tasks — DEFAULT), or 'complex' (novel architecture design, "
            "10+ file debugging with non-obvious root causes, user expressing frustration "
            "about repeated failures). "
            "COST: 'complex' triggers a more expensive model — reserve for tasks where "
            "standard would genuinely fail. When in doubt, choose 'standard'."
        )
    )
    needs_research: bool = Field(
        default=False,
        description=(
            "True when the task requires understanding before acting: "
            "investigate, explore, audit, compare, review, root cause analysis. "
            "False for explicit create/write/run tasks or simple questions."
        )
    )
    needs_planning: bool = Field(
        default=False,
        description=(
            "True for multi-step tasks producing artifacts or changing state: "
            "environment installs, pipeline creation, multi-file debugging, "
            "tasks with 2+ independent deliverables. "
            "False for questions, single edits, status checks, continuing existing plans."
        )
    )
    needs_slurm: bool = Field(
        default=False,
        description=(
            "True when compute may exceed 5 minutes or needs HPC resources "
            "(GPU, high memory, parallel cores): ML training, large compilations, "
            "genome alignment, simulations, batch processing. "
            "False for quick commands, file reads, pip installs, status checks."
        )
    )
    parallel_subtasks: bool = Field(
        default=False,
        description=(
            "True when 2+ independent information sources can be explored "
            "simultaneously (e.g. compare A vs B, read logs + code + config). "
            "False for sequential workflows or single-file tasks."
        )
    )
    needs_websearch: bool = Field(
        default=False,
        description=(
            "True when web information would improve the answer: external docs, "
            "troubleshooting errors, version compatibility, best practices. "
            "Bias toward true. False only for purely local file/code tasks."
        )
    )
    project_context: Optional[str] = Field(
        default=None,
        description=(
            "Which project this message is about. MUST pick from the KNOWN PROJECTS "
            "list shown in the prompt whenever possible. Rules:\n"
            "1. Pick an existing project if the user's message clearly relates to it.\n"
            "2. Use 'general' for greetings, system questions, meta-discussions, or "
            "one-off queries not tied to any specific project.\n"
            "3. Use the PREVIOUSLY ACTIVE PROJECT if the user is continuing ('yes', "
            "'do it', 'continue') and doesn't name a new project.\n"
            "4. Set '__ask_user__' if you cannot confidently choose — e.g., the user "
            "mentions a topic that could match 2+ existing projects, or uses ambiguous "
            "terms. This will prompt the user to clarify.\n"
            "5. ONLY create a brand-new name (not in the list) when the user is "
            "starting genuinely new work that does NOT match any existing project. "
            "New names should be short, lowercase, underscore-separated identifiers."
        )
    )
    is_context_switch: bool = Field(
        default=False,
        description=(
            "True if project_context differs from the previously active project. "
            "This triggers auto-save of the outgoing project's state before loading "
            "the new project's context. False if same project, or if project_context is None."
        )
    )
    requested_model: Optional[str] = Field(
        default=None,
        description=(
            "If the user is asking to switch/change the LLM model, extract which model "
            "they want. Valid values: 'opus', 'sonnet', 'nemotron', 'gpt-oss-120b', "
            "'gpt-oss-20b'. Set null if the user is NOT requesting a model switch. "
            "Only set this when the user explicitly asks to change models (e.g. 'use opus', "
            "'switch to nemotron', 'I want the best model', 'change to gpt-oss')."
        )
    )
    is_refinement: bool = Field(
        default=False,
        description=(
            "True if the user is expressing dissatisfaction with the previous result "
            "and wants improvement. This includes BOTH direct and indirect criticism. "
            "DIRECT: 'not good enough', 'try again', 'make it better', 'that's wrong', "
            "'improve this', 'I'm not satisfied', 'do better'. "
            "INDIRECT: comparing output to low quality ('looks like homework', 'too basic', "
            "'not professional', 'amateurish'), requesting a HIGHER standard ('I need this "
            "for a Nature paper', 'publication-ready', 'make it professional'), expressing "
            "disappointment ('colors are ugly', 'too cluttered', 'missing key elements'), "
            "or contrasting output with expectations ('I expected better', 'this isn't what "
            "I meant'). KEY RULE: If the user BOTH criticizes the previous output AND "
            "requests improvement (even implicitly by stating a higher standard), this IS "
            "refinement. False for new requests, follow-up questions, or neutral feedback."
        )
    )


# ── Pure functions (testable without langchain) ──────────────────────────────

def escape_prompt_braces(text: str) -> str:
    """Escape curly braces in text destined for ChatPromptTemplate.

    LangChain's ChatPromptTemplate interprets {variable} as template
    variables. When we inject dynamic content (user context, knowledge
    base, skill content) into the system prompt, any literal curly
    braces must be doubled to avoid being parsed as variables.

    This is the definitive fix for the production error:
        'Input to ChatPromptTemplate is missing variables'
    which occurs when KB entries or skill content contain unescaped
    braces like {name}, {variable}, or {}.

    Args:
        text: Raw text that may contain literal curly braces.

    Returns:
        Text with { → {{ and } → }} so ChatPromptTemplate treats
        them as literal characters.
    """
    # Replace { with {{ and } with }}
    # We must do this in one pass to avoid double-escaping
    return text.replace("{", "{{").replace("}", "}}")


def _format_history_for_skill_selector(
    chat_history: Optional[list],
    max_messages: int = 6,
    max_chars_per_message: int = 1500,
) -> str:
    """Format recent chat history as compact text for the skill selector.

    Includes the last few messages with moderately truncated content —
    enough for the skill selector to understand the conversation context
    and correctly route continuation messages.

    Combined with the conversation_summary (from Haiku), this gives the
    skill selector both long-term context and recent detail.

    Args:
        chat_history: List of message objects (HumanMessage/AIMessage).
        max_messages: Maximum number of recent messages to include.
            Default: 6 (aligned with SLIDING_WINDOW_SIZE).
        max_chars_per_message: Max characters per message content.
            Default: 1500 (enough for domain identification and
            accurate routing decisions).

    Returns:
        Formatted string like:
            Human: I want to submit an AlphaFold job
            AI: Sure! Please upload your FASTA file...
            Human: Here it is
        Or "(No previous conversation)" if empty.
    """
    if not chat_history:
        return "(No previous conversation)"

    recent = chat_history[-max_messages:]
    lines = []
    for msg in recent:
        # Support both LangChain message objects and plain dicts
        if hasattr(msg, "type"):
            role = "Human" if msg.type == "human" else "AI"
        elif hasattr(msg, "content"):
            role = msg.__class__.__name__.replace("Message", "")
        else:
            continue

        content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            # Handle multimodal content (list of dicts)
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = " ".join(text_parts)

        content = content.strip()
        if not content:
            continue

        if len(content) > max_chars_per_message:
            content = content[:max_chars_per_message] + "... [truncated]"
        lines.append(f"{role}: {content}")

    if not lines:
        return "(No previous conversation)"
    return "\n".join(lines)


def build_skill_selection_prompt(
    skill_loader: SkillLoader,
    user_input: str,
    chat_history: Optional[list] = None,
    conversation_summary: Optional[str] = None,
    active_project: Optional[str] = None,
    known_projects_str: Optional[str] = None,
    project_status: Optional[str] = None,
) -> str:
    """Build the skill selection prompt with the current manifest.

    Args:
        skill_loader: Loaded SkillLoader instance.
        user_input: The user's message text.
        chat_history: Optional list of recent chat messages for context.
            Used to disambiguate short messages like "yes", "go ahead".
        conversation_summary: Optional pre-computed conversation summary
            from Haiku LLM summarization (or regex fallback). When provided,
            gives the skill selector long-term context about the conversation
            topic, decisions made, and pending tasks — critical for correctly
            routing continuation messages in long conversations.
        active_project: Currently active project name (for context switch detection).
        known_projects_str: Formatted string of known projects with descriptions.
        project_status: Optional project status.md content for complexity assessment.

    Returns:
        Formatted prompt string for the skill selector.
    """
    manifest = skill_loader.get_manifest()
    recent_history = _format_history_for_skill_selector(chat_history)

    # Format the conversation summary section
    summary_parts = []
    if conversation_summary and conversation_summary.strip():
        summary_parts.append(
            f"Conversation summary (from earlier in this session):\n"
            f"{conversation_summary}"
        )
    if project_status and project_status.strip():
        summary_parts.append(
            f"Active project status (from memory):\n{project_status}"
        )
    summary_section = "\n\n".join(summary_parts)

    return SKILL_SELECTION_PROMPT_TEMPLATE.format(
        manifest=manifest,
        user_input=user_input,
        recent_history=recent_history,
        conversation_summary=summary_section,
        active_project=active_project or "(none)",
        known_projects=known_projects_str or "general (default — non-project-specific queries)",
    )


def format_history_for_skill_selector(
    chat_history: Optional[list],
    max_user_messages: int = 3,
) -> str:
    """Format recent user messages for the skill selector (zero LLM calls).

    Session facts (passed separately in conversation_summary) provide full
    structured context. This only extracts the last few user messages so the
    selector can follow the current thread of conversation.
    """
    if not chat_history:
        return "(No previous conversation)"

    lines = []
    user_count = 0
    for msg in reversed(chat_history):
        if getattr(msg, "type", "") == "human":
            content = getattr(msg, "content", "") or ""
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            content = content.strip()
            if content:
                lines.append(f"User: {content[:500]}")
                user_count += 1
                if user_count >= max_user_messages:
                    break

    lines.reverse()
    return "\n".join(lines) if lines else "(No previous conversation)"


async def async_build_skill_selection_prompt(
    skill_loader: SkillLoader,
    user_input: str,
    chat_history: Optional[list] = None,
    conversation_summary: Optional[str] = None,
    session_facts: Optional[str] = None,
    active_project: Optional[str] = None,
    known_projects_str: Optional[str] = None,
    project_status: Optional[str] = None,
) -> str:
    """Async version of build_skill_selection_prompt with smart compaction.

    Uses Haiku-based compaction for oversized AI messages instead of head-only
    truncation, and injects session_facts for better routing decisions.
    """
    manifest = skill_loader.get_manifest()
    recent_history = format_history_for_skill_selector(chat_history)

    # Format the conversation summary section (include session_facts + project status)
    summary_parts = []
    if conversation_summary and conversation_summary.strip():
        summary_parts.append(
            f"Conversation summary (from earlier in this session):\n"
            f"{conversation_summary}"
        )
    if session_facts and session_facts.strip():
        summary_parts.append(
            f"Known facts from this session:\n{session_facts}"
        )
    if project_status and project_status.strip():
        summary_parts.append(
            f"Active project status (from memory):\n{project_status}"
        )
    summary_section = "\n\n".join(summary_parts)

    return SKILL_SELECTION_PROMPT_TEMPLATE.format(
        manifest=manifest,
        user_input=user_input,
        recent_history=recent_history,
        conversation_summary=summary_section,
        active_project=active_project or "(none)",
        known_projects=known_projects_str or "general (default — non-project-specific queries)",
    )


def parse_skill_selection(
    selection: SkillSelection,
    skill_loader: SkillLoader,
) -> Dict[str, Any]:
    """Validate and normalize a SkillSelection against loaded skills.

    Filters out any skill names that don't exist in the loader.
    Falls back to 'conversational' if no valid skills remain.

    Args:
        selection: The SkillSelection from structured output.
        skill_loader: Loaded SkillLoader instance.

    Returns:
        Dict with:
            - skills: list of validated skill names
            - primary_skill: validated primary skill name
            - reasoning: the original reasoning string
            - fallback_used: True if we had to fall back to conversational
    """
    available = set(skill_loader.list_skill_names())

    # Filter to only valid skill names
    valid_skills = [s for s in selection.skills if s in available]
    valid_primary = selection.primary_skill if selection.primary_skill in available else None

    fallback_used = False

    # If no valid skills, fall back to conversational
    if not valid_skills:
        fallback_skill = "conversational" if "conversational" in available else None
        if fallback_skill:
            valid_skills = [fallback_skill]
            fallback_used = True
        else:
            # Last resort: use first available skill
            if available:
                valid_skills = [sorted(available)[0]]
                fallback_used = True

    # If primary is invalid, use first valid skill
    if valid_primary is None and valid_skills:
        valid_primary = valid_skills[0]
        fallback_used = True

    # Ensure primary is in the skills list
    if valid_primary and valid_primary not in valid_skills:
        valid_skills.insert(0, valid_primary)

    return {
        "skills": valid_skills,
        "primary_skill": valid_primary or (valid_skills[0] if valid_skills else ""),
        "reasoning": selection.reasoning,
        "complexity": getattr(selection, 'complexity', 'standard'),
        "needs_research": getattr(selection, 'needs_research', False),
        "needs_planning": getattr(selection, 'needs_planning', False),
        "needs_slurm": getattr(selection, 'needs_slurm', False),
        "parallel_subtasks": getattr(selection, 'parallel_subtasks', False),
        "needs_websearch": getattr(selection, 'needs_websearch', False),
        "project_context": getattr(selection, 'project_context', None),
        "is_context_switch": getattr(selection, 'is_context_switch', False),
        "requested_model": getattr(selection, 'requested_model', None),
        "fallback_used": fallback_used,
    }




def filter_tools_for_skills(
    all_tools: list,
    skill_names: List[str],
    skill_loader: SkillLoader,
    exclude_tools: set = None,
    include_only_tools: set = None,
) -> list:
    """Filter the master tool list to only tools allowed by selected skills.

    Takes the union of allowed_tools from all selected skills.
    Tools are matched by their .name attribute.

    If a skill has an empty allowed_tools list, ALL tools are included
    (the skill is unrestricted).

    Always includes the 'request_additional_skill' tool if it exists in
    all_tools — this is the dynamic escalation mechanism.

    Args:
        all_tools: Complete list of tool objects (must have .name attribute).
        skill_names: List of selected skill names.
        skill_loader: Loaded SkillLoader instance.
        exclude_tools: Optional set of tool names to forcibly remove from results.
        include_only_tools: Optional set of tool names — when set, ONLY these
            tools are returned (overrides skill-based filtering). Used for
            pipeline-only mode.

    Returns:
        Filtered list of tool objects.
    """
    # Pipeline-only mode: bypass skill-based filtering entirely
    if include_only_tools:
        filtered = [t for t in all_tools if getattr(t, "name", None) in include_only_tools]
        logger.info(
            f"Pipeline-only mode: filtered {len(all_tools)} tools to {len(filtered)} "
            f"(include_only={include_only_tools})"
        )
        return filtered

    # Get union of allowed tool names
    allowed_names = skill_loader.get_merged_tools(skill_names)

    # If any skill has empty allowed_tools, it's unrestricted — return all
    for name in skill_names:
        skill_tools = skill_loader.get_allowed_tools(name)
        if not skill_tools:
            logger.info(
                f"Skill '{name}' has no tool restrictions — returning all tools"
            )
            return list(all_tools)

    if not allowed_names:
        return list(all_tools)

    allowed_set = set(allowed_names)
    # Always include these tools regardless of skill restrictions —
    # but ONLY if they actually exist in all_tools (prevents phantom tool references)
    all_tool_names = {getattr(t, "name", None) for t in all_tools}

    # Meta-tools: escalation and phase management
    _meta_tools = {"request_additional_skill", "run_worker_agent", "write_findings", "write_plan", "edit_plan"}

    # Core infrastructure tools: always available because the system's accuracy
    # and reliability depends on them regardless of which domain skill is active.
    # See SKILLS_RESTRUCTURE_REPORT.md Section 6 for rationale.
    _core_infrastructure_tools = {
        "get_environment_info",   # Software/container discovery — prevents hallucination
        "read_memory",            # Project context recall
        "update_memory",          # Save discoveries for future sessions
        "list_projects",          # Project listing
        "add_project",            # Project creation
        "remove_project",         # Project removal
        "render_image_inline",    # Any skill might produce images
        "query_software",         # Software registry lookup — prevents redundant installs
        "register_software",      # Software registration — must be available for any path discovery
        "batch",                  # Multi-op efficiency — 1 call vs N round-trips
        "batch_readonly",         # Read-only batch for research/plan phases
    }

    _always_include = _meta_tools | _core_infrastructure_tools
    for _meta_tool in _always_include:
        if _meta_tool in all_tool_names:
            allowed_set.add(_meta_tool)



    filtered = [t for t in all_tools if getattr(t, "name", None) in allowed_set]

    if exclude_tools:
        filtered = [t for t in filtered if getattr(t, "name", None) not in exclude_tools]

    logger.info(
        f"Filtered {len(all_tools)} tools to {len(filtered)} for skills {skill_names}"
    )
    return filtered


def build_agent_system_prompt(
    skill_names: List[str],
    skill_loader: SkillLoader,
    base_instructions: str = BASE_AGENT_INSTRUCTIONS,
    user_context: Optional[str] = None,
    include_memory_check: bool = False,
) -> str:
    """Assemble the full system prompt from base instructions + user context + skill content.

    The user_context block (username, work_dir, project_name, knowledge base,
    skill-specific KB) is injected between the base instructions and the
    skill content. This means every agent automatically knows the user's
    environment without needing to call get_user_settings as a first step.

    Also includes the tool registry — a map of tools in OTHER skills that
    the agent can request dynamically if needed.

    Args:
        skill_names: List of selected skill names.
        skill_loader: Loaded SkillLoader instance.
        base_instructions: Base instructions prepended to all prompts.
        user_context: Optional pre-built user context block from
            build_user_context_block(). Contains username, work_dir,
            project_name, global KB summary, and per-skill KB entries.
        include_memory_check: When True, inject MEMORY_CHECK_PREAMBLE so the
            executor checks memory before acting. Only used when no
            research/planning phase ran for this turn.

    Returns:
        Complete system prompt string.
    """
    merged_content = skill_loader.get_merged_content(skill_names)

    parts = [base_instructions]

    if include_memory_check:
        parts.append(MEMORY_CHECK_PREAMBLE)

    if user_context and user_context.strip():
        parts.append(user_context)
        parts.append(EVIDENCE_GATED_TRUST_RULES)

    # Add tool registry showing tools in OTHER skills (for dynamic escalation)
    tool_registry_text = skill_loader.get_tool_registry_text(exclude_skills=skill_names)
    if tool_registry_text:
        parts.append(tool_registry_text)

    parts.append(merged_content)

    return "\n\n".join(parts)


def get_agent_config_for_skills(
    skill_names: List[str],
    skill_loader: SkillLoader,
) -> Dict[str, Any]:
    """Get agent configuration derived from selected skills.

    Returns model override and max_iterations from the primary skill.

    Args:
        skill_names: List of selected skill names (primary first).
        skill_loader: Loaded SkillLoader instance.

    Returns:
        Dict with:
            - model: model string or None
            - max_iterations: integer iteration limit
    """
    return {
        "model": skill_loader.get_primary_model(skill_names),
        "max_iterations": skill_loader.get_primary_max_iterations(skill_names),
    }


# ── Dynamic skill escalation detection (pure function — testable) ────────

def _extract_skill_from_tool_input(action) -> str:
    """Extract skill name from an AgentAction's tool_input (structured).

    This is the PRIMARY extraction method — it reads the argument that
    the LLM passed to the request_additional_skill tool call, which is
    always structured (either a dict with 'skill_name' key, or a plain
    string). This is 100% reliable because LangChain serializes tool
    call arguments as structured data, not free text.

    Args:
        action: An AgentAction-like object with a .tool_input attribute.

    Returns:
        The skill name string, or "" if not extractable.
    """
    tool_input = getattr(action, "tool_input", {})
    if isinstance(tool_input, dict):
        return tool_input.get("skill_name", "")
    elif isinstance(tool_input, str):
        return tool_input
    return ""


def _extract_skill_from_observation(observation) -> str:
    """Extract skill name from tool observation text using ESCALATION_MARKER.

    This is the FALLBACK extraction method — it parses the text output
    of the request_additional_skill tool for the ESCALATION_MARKER prefix.
    Used only when tool_input extraction yields nothing (e.g. if the
    action object lacks tool_input or it's empty).

    Args:
        observation: The tool output (string or stringifiable).

    Returns:
        The skill name string, or "" if not extractable.
    """
    obs_str = str(observation)
    if ESCALATION_MARKER in obs_str:
        marker_idx = obs_str.index(ESCALATION_MARKER)
        after_marker = obs_str[marker_idx + len(ESCALATION_MARKER):]
        parts = after_marker.strip().split()
        if parts:
            return parts[0]
    return ""


def detect_escalation_in_result(result: dict) -> Dict[str, Any]:
    """Detect if the agent called request_additional_skill during execution.

    Inspects the intermediate_steps from an AgentExecutor result to find
    calls to the request_additional_skill tool. Extracts the requested
    skill name using a two-tier strategy:

        1. PRIMARY — action.tool_input (structured): The argument the LLM
           passed to the tool call. Always a dict or string, never ambiguous.
           This is set by LangChain's tool-calling mechanism and is 100%
           reliable.

        2. FALLBACK — ESCALATION_MARKER in observation text: Parses the
           tool's text output for the marker prefix. Used only when
           tool_input is missing or empty (e.g. malformed action objects).

    This is a pure function — no LangChain imports, no side effects.
    It only reads the result dict and returns structured information.

    Args:
        result: The dict returned by AgentExecutor.ainvoke(), expected to
                contain 'intermediate_steps' (list of (AgentAction, observation)
                tuples).

    Returns:
        Dict with:
            - escalation_detected (bool): True if request_additional_skill was called
            - requested_skills (list[str]): Skill names that were requested
    """
    steps = result.get("intermediate_steps", [])
    requested_skills = []

    for step in steps:
        try:
            action = step[0]
            observation = step[1] if len(step) > 1 else ""
            tool_name = getattr(action, "tool", None)

            if tool_name == "request_additional_skill":
                # PRIMARY: Extract from tool_input (structured, reliable)
                skill_name = _extract_skill_from_tool_input(action)

                # FALLBACK: Extract from observation text (marker-based)
                if not skill_name:
                    skill_name = _extract_skill_from_observation(observation)

                if skill_name:
                    requested_skills.append(skill_name)
        except (IndexError, TypeError, AttributeError):
            # Malformed step — skip it
            continue

    # FALLBACK: If intermediate_steps had no escalation, check result["output"].
    # This handles the case where request_additional_skill has return_direct=True —
    # LangChain puts the tool output directly into result["output"] and does NOT
    # add it to intermediate_steps. Without this check, detect_escalation_in_result()
    # would miss the escalation on the first attempt, causing the hallucination guard
    # to fire unnecessarily and the user to see the escalation boilerplate text.
    if not requested_skills:
        output_text = result.get("output", "") if isinstance(result, dict) else ""
        if ESCALATION_MARKER in output_text:
            skill_name = _extract_skill_from_observation(output_text)
            if skill_name:
                requested_skills.append(skill_name)

    # Deduplicate while preserving order
    seen = set()
    unique_skills = []
    for s in requested_skills:
        if s not in seen:
            seen.add(s)
            unique_skills.append(s)

    return {
        "escalation_detected": len(unique_skills) > 0,
        "requested_skills": unique_skills,
    }


# ── Escalation handoff: format intermediate steps for new executor ──────

def _truncate_observation(observation: str, max_chars: int = 4000) -> str:
    """Sync fallback: compact observation using head+tail."""
    if len(observation) <= max_chars:
        return observation
    from core.context_compactor import _fallback_truncate
    return _fallback_truncate(observation, max_chars)


async def _async_truncate_observation(observation: str, max_chars: int = 4000) -> str:
    """Compact a tool observation using Haiku smart compaction."""
    if len(observation) <= max_chars:
        return observation
    from core.context_compactor import async_smart_compact
    return await async_smart_compact(
        observation, max_chars=max_chars, context_type="agent_handoff"
    )


def format_intermediate_steps_for_handoff(
    result: dict,
    user_input: str,
    previous_skills: Optional[List[str]] = None,
    new_skills: Optional[List[str]] = None,
    max_observation_chars: int = 8000,
) -> str:
    """Format intermediate_steps from a completed AgentExecutor into a
    structured handoff text for the next executor after skill escalation.

    This is a pure function — no LLM calls, no side effects. It extracts
    every tool call and its result from the previous executor's run and
    formats them into a clear, structured text block that the new executor
    can use to understand what was already done.

    The output preserves ALL tool calls and their results (with per-tool
    truncation for very large outputs). No information about what was done
    is lost — only very large individual tool outputs are truncated.

    Tools that fetch external content (fetch_url_content, web_search) get
    a higher compaction limit (2x default) to preserve the fetched data
    and avoid the new agent re-fetching the same URLs.

    Args:
        result: The dict returned by AgentExecutor.ainvoke(), containing
                'intermediate_steps' and 'output'.
        user_input: The original user request.
        previous_skills: List of skill names the previous executor had.
        new_skills: List of new skill names being added.
        max_observation_chars: Max chars per tool observation (default 4000).

    Returns:
        Formatted handoff text string. Contains:
        - Header identifying this as a handoff
        - The original user request
        - Numbered list of all completed tool calls with results
        - The previous agent's final response
        - Clear instruction not to repeat completed work
    """
    # Tools whose results should be preserved with higher truncation limits
    # to avoid the new agent re-fetching the same content.
    CONTENT_FETCH_TOOLS = {"fetch_url_content", "web_search"}
    steps = result.get("intermediate_steps", [])
    agent_output = result.get("output", "")

    parts = []

    # Header
    prev_skills_str = ", ".join(previous_skills) if previous_skills else "unknown"
    new_skills_str = ", ".join(new_skills) if new_skills else "unknown"
    parts.append(
        f"ESCALATION HANDOFF \u2014 CONTEXT FROM PREVIOUS AGENT\n"
        f"Previous skills: {prev_skills_str}\n"
        f"New skills added: {new_skills_str}\n"
        f"Original user request: {user_input}"
    )

    # Tool calls
    if steps:
        parts.append("\nCOMPLETED ACTIONS (do NOT repeat these):")
        for i, step in enumerate(steps, 1):
            try:
                action = step[0]
                observation = step[1] if len(step) > 1 else ""
                tool_name = getattr(action, "tool", "unknown")
                tool_input = getattr(action, "tool_input", {})

                # Format tool input compactly
                if isinstance(tool_input, dict):
                    # Remove very long values from display
                    compact_input = {}
                    for k, v in tool_input.items():
                        if isinstance(v, str) and len(v) > 200:
                            compact_input[k] = v[:200] + "..."
                        else:
                            compact_input[k] = v
                    input_str = str(compact_input)
                else:
                    input_str = str(tool_input)
                    if len(input_str) > 200:
                        input_str = input_str[:200] + "..."

                # Format observation with truncation
                # Content-fetching tools get 2x limit to preserve
                # fetched data and prevent duplicate fetches after escalation.
                obs_str = str(observation)
                effective_max = (
                    max_observation_chars * 2
                    if tool_name in CONTENT_FETCH_TOOLS
                    else max_observation_chars
                )
                obs_str = _truncate_observation(obs_str, effective_max)

                parts.append(
                    f"\n{i}. Tool: {tool_name}\n"
                    f"   Input: {input_str}\n"
                    f"   Result: {obs_str}"
                )
            except (IndexError, TypeError, AttributeError):
                parts.append(f"\n{i}. [malformed step — skipped]")
    else:
        parts.append("\nNo tool calls were made by the previous agent.")

    # Previous agent's final response
    if agent_output and agent_output.strip():
        if len(agent_output) > 8000:
            from core.context_compactor import _fallback_truncate
            agent_output_display = _fallback_truncate(agent_output, 8000)
        else:
            agent_output_display = agent_output
        parts.append(
            f"\nPREVIOUS AGENT'S RESPONSE TO USER:\n{agent_output_display}"
        )

    # Instructions for new agent
    parts.append(
        "\nINSTRUCTIONS FOR YOU (the new agent):\n"
        "- The actions listed above are ALREADY COMPLETED. Do NOT repeat them.\n"
        "- You now have additional tools from the newly loaded skill(s).\n"
        "- Use the results above as context and CONTINUE where the previous agent left off.\n"
        "- Focus on completing the parts of the user's request that still need work.\n"
        "- If the previous agent exported files or produced paths, use those directly."
    )

    return "\n".join(parts)


async def async_format_intermediate_steps_for_handoff(
    result: dict,
    user_input: str,
    previous_skills: Optional[List[str]] = None,
    new_skills: Optional[List[str]] = None,
    max_observation_chars: int = 8000,
) -> str:
    """Async version: uses Haiku to intelligently compact large observations."""
    import asyncio
    CONTENT_FETCH_TOOLS = {"fetch_url_content", "web_search"}
    steps = result.get("intermediate_steps", [])
    agent_output = result.get("output", "")

    parts = []
    prev_skills_str = ", ".join(previous_skills) if previous_skills else "unknown"
    new_skills_str = ", ".join(new_skills) if new_skills else "unknown"
    parts.append(
        f"ESCALATION HANDOFF — CONTEXT FROM PREVIOUS AGENT\n"
        f"Previous skills: {prev_skills_str}\n"
        f"New skills added: {new_skills_str}\n"
        f"Original user request: {user_input}"
    )

    if steps:
        # Collect compaction tasks for parallel execution
        compact_tasks = []
        step_metadata = []

        for i, step in enumerate(steps, 1):
            try:
                action = step[0]
                observation = step[1] if len(step) > 1 else ""
                tool_name = getattr(action, "tool", "unknown")
                tool_input = getattr(action, "tool_input", {})

                if isinstance(tool_input, dict):
                    compact_input = {}
                    for k, v in tool_input.items():
                        if isinstance(v, str) and len(v) > 200:
                            compact_input[k] = v[:200] + "..."
                        else:
                            compact_input[k] = v
                    input_str = str(compact_input)
                else:
                    input_str = str(tool_input)
                    if len(input_str) > 200:
                        input_str = input_str[:200] + "..."

                obs_str = str(observation)
                effective_max = (
                    max_observation_chars * 2
                    if tool_name in CONTENT_FETCH_TOOLS
                    else max_observation_chars
                )
                compact_tasks.append(_async_truncate_observation(obs_str, effective_max))
                step_metadata.append((i, tool_name, input_str))
            except (IndexError, TypeError, AttributeError):
                async def _empty():
                    return ""
                compact_tasks.append(_empty())
                step_metadata.append((i, None, None))

        compacted_observations = await asyncio.gather(*compact_tasks)

        parts.append("\nCOMPLETED ACTIONS (do NOT repeat these):")
        for (idx, tool_name, input_str), obs_str in zip(step_metadata, compacted_observations):
            if tool_name is None:
                parts.append(f"\n{idx}. [malformed step — skipped]")
            else:
                parts.append(
                    f"\n{idx}. Tool: {tool_name}\n"
                    f"   Input: {input_str}\n"
                    f"   Result: {obs_str}"
                )
    else:
        parts.append("\nNo tool calls were made by the previous agent.")

    if agent_output and agent_output.strip():
        if len(agent_output) > 8000:
            agent_output_display = await _async_truncate_observation(agent_output, 8000)
        else:
            agent_output_display = agent_output
        parts.append(
            f"\nPREVIOUS AGENT'S RESPONSE TO USER:\n{agent_output_display}"
        )

    parts.append(
        "\nINSTRUCTIONS FOR YOU (the new agent):\n"
        "- The actions listed above are ALREADY COMPLETED. Do NOT repeat them.\n"
        "- You now have additional tools from the newly loaded skill(s).\n"
        "- Use the results above as context and CONTINUE where the previous agent left off.\n"
        "- Focus on completing the parts of the user's request that still need work.\n"
        "- If the previous agent exported files or produced paths, use those directly."
    )

    return "\n".join(parts)


# ── Dynamic skill escalation tool ────────────────────────────────────────
# This is a real LangChain tool that the agent can call when it needs tools
# from a skill that wasn't initially selected. When called, it raises
# SkillEscalationInterrupt — immediately breaking out of the AgentExecutor
# loop. app.py catches this exception and re-creates the executor with the
# expanded skill set. This guarantees no other tool calls in the same batch
# can execute after escalation is requested.

# Global reference to the skill_loader — set by create_request_additional_skill_tool()
_skill_loader_ref: Optional[SkillLoader] = None


class SkillEscalationInterrupt(Exception):
    """Raised by request_additional_skill to immediately break the executor loop.

    app.py catches this and re-creates the executor with the requested skill's
    tools included. This is a hard interrupt — no other tools in the batch
    execute after this.
    """

    def __init__(self, skill_name: str) -> None:
        self.skill_name = skill_name
        super().__init__(f"Skill escalation requested: {skill_name}")


def _request_additional_skill_impl(skill_name: str) -> str:
    """Request tools from an additional skill that wasn't initially selected.

    Call this when you need a tool that is listed in the 'TOOLS IN OTHER SKILLS'
    section of your instructions but is not available in your current toolset.

    This function raises SkillEscalationInterrupt to immediately break out of
    the AgentExecutor loop. app.py catches this exception and handles the
    escalation (re-creating the executor with expanded tools).

    Args:
        skill_name: Name of the skill to request (e.g. 'hpc-submit-job', 'file-operations').

    Raises:
        SkillEscalationInterrupt: Always raised on valid skill names to break
            out of the executor loop immediately.

    Returns:
        Error string only if the skill name is invalid.
    """
    global _skill_loader_ref

    if _skill_loader_ref is None:
        raise SkillEscalationInterrupt(skill_name)

    loader = _skill_loader_ref
    available_skills = loader.list_skill_names()

    if skill_name not in available_skills:
        return (
            f"Error: Skill '{skill_name}' not found. "
            f"Available skills: {', '.join(available_skills)}. "
            "Please check the skill name and try again."
        )

    raise SkillEscalationInterrupt(skill_name)


def create_request_additional_skill_tool(skill_loader: SkillLoader):
    """Create the request_additional_skill LangChain tool.

    This must be called at runtime when a SkillLoader is available.
    The tool is a StructuredTool that wraps _request_additional_skill_impl.

    Args:
        skill_loader: The loaded SkillLoader instance.

    Returns:
        A LangChain StructuredTool instance.
    """
    global _skill_loader_ref
    _skill_loader_ref = skill_loader

    # Lazy import — only needed at runtime
    from langchain_core.tools import StructuredTool

    tool = StructuredTool.from_function(
        func=_request_additional_skill_impl,
        name="request_additional_skill",
        description=(
            "Request tools from an additional skill that wasn't initially selected. "
            "Call this when you need a tool listed in 'TOOLS IN OTHER SKILLS' but "
            "it's not in your current toolset. Pass the skill name (e.g. 'hpc-submit-job', "
            "'file-operations', 'bioinformatics-analysis', 'visualization'). "
            "CRITICAL: This must be your ONLY tool call in the response — do NOT "
            "call any other tools in the same turn. After you call this, the system "
            "will immediately restart you with the new tools available. Any other "
            "tool calls in the same response WILL FAIL because those tools don't "
            "exist yet."
        ),
        return_direct=True,
    )
    return tool


# ── Escalation-aware error handling ─────────────────────────────────────

def _make_escalation_aware_error_handler():
    """Return a handle_parsing_errors callable that injects ESCALATION_MARKER
    when an invalid-tool error occurs.

    When the LLM batches request_additional_skill + an unavailable tool in
    the same parallel call, the unavailable tool fails with 'X is not a valid
    tool'. This handler intercepts that error and returns the ESCALATION_MARKER,
    which causes detect_escalation_in_result() to pick it up.

    Non-escalation errors pass through with the default error message.
    """
    def handler(error) -> str:
        error_str = str(error)
        # Detect the "not a valid tool" pattern from LangChain's AgentExecutor
        if "is not a valid tool" in error_str:
            return (
                f"{ESCALATION_MARKER} skill_escalation_pending\n"
                "Tool not available yet \u2014 skill escalation is in progress. "
                "Do NOT retry with other tools. The system will re-invoke you "
                "with the correct tools automatically."
            )
        # All other errors: default behavior
        return f"Error: {error_str}"
    return handler


# ── LangChain-dependent functions (runtime only) ────────────────────────
# These import langchain_classic at call time, not at module import time,
# so the module can be imported in test environments without langchain.

def create_skill_based_agent(
    llm,
    all_tools: list,
    skill_names: List[str],
    skill_loader: SkillLoader,
    dev_llm=None,
    user_context: Optional[str] = None,
    complexity: str = "standard",
    websearch_enabled: bool = False,
    exclude_tools: set = None,
    include_only_tools: set = None,
    phase_config=None,
    use_opus: bool = False,
    include_memory_check: bool = False,
    model_override: Optional[str] = None,
    active_plan_path: str = "",
    pel=None,
):
    """Create an AgentExecutor configured for the selected skills.

    This is the main entry point called by app.py per turn.
    Imports langchain at call time to avoid import errors in test envs.

    Args:
        llm: Default LLM (Sonnet).
        all_tools: Complete list of tool objects.
        skill_names: List of selected skill names.
        skill_loader: Loaded SkillLoader instance.
        dev_llm: Optional Opus LLM (used only when use_opus=True).
        user_context: Optional pre-built user context block. Injected
            into the system prompt so the agent knows the user's
            environment (work_dir, project, KB) without tool calls.
        complexity: Task complexity from skill selection ('simple',
            'standard', or 'complex'). Used for iteration limits, NOT
            for model selection.
        websearch_enabled: Whether the user has enabled the web search
            globe button. Kept for API compatibility but no longer affects
            tool availability — web_search tools are always injected with
            their own per-search approval gate.
        exclude_tools: Optional set of tool names to forcibly remove.
        include_only_tools: Optional set of tool names — when set, ONLY
            these tools are available. Used for pipeline mode restrictions.
            Superseded by phase_config when both are provided.
        phase_config: Optional PhaseConfig for same-context phase execution.
            When provided, NativeAgentExecutor handles dynamic tool filtering
            per phase (research → plan → execute). Takes precedence over
            include_only_tools.
        use_opus: If True, use Opus (dev_llm). Only set when user
            explicitly requests Opus or advisor escalation recommends it.
        include_memory_check: When True, inject MEMORY_CHECK_PREAMBLE into
            the system prompt so the executor checks memory before acting.
            Only set when no research/planning phase ran for this turn.

    Returns:
        Tuple of (AgentExecutor, int, str) — the executor, the filtered tool
        count, and the display name of the model used (e.g. "opus", "sonnet").
    """
    # Lazy imports — only needed at runtime
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate

    # 1. Build system prompt (now includes user context)
    system_prompt = build_agent_system_prompt(
        skill_names, skill_loader, user_context=user_context,
        include_memory_check=include_memory_check,
    )

    # 1b. Inject web search availability instruction.
    # web_search and fetch_url_content are ALWAYS in the tool pool.
    # The tools have their own approval gate — user is asked per-search.
    system_prompt += (
        "\n\nWEB SEARCH: The web_search and fetch_url_content tools are available. "
        "Use them when you need current, external, or version-specific information "
        "that you cannot reliably answer from training data. Each call will ask "
        "the user for approval before executing. Do NOT make up version numbers, "
        "changelogs, or external documentation — use web_search instead."
    )

    # 1c. Inject iteration-awareness instruction so the LLM gracefully
    # wraps up instead of being hard-stopped by AgentExecutor.
    max_iter = get_agent_config_for_skills(skill_names, skill_loader)["max_iterations"]
    system_prompt += (
        f"\n\nITERATION BUDGET: You have a maximum of {max_iter} tool calls for this turn. "
        f"If you are approaching this limit (~80% used, i.e. ~{int(max_iter * 0.8)} calls), "
        "STOP making new tool calls and instead provide a summary to the user of: "
        "(1) what you accomplished so far, (2) what remains to be done, and "
        "(3) suggest the user ask you to continue in the next turn. "
        "NEVER let yourself be hard-stopped — always leave a graceful summary."
    )

    # 2. Filter tools
    # When phase_config is active, it handles dynamic tool filtering internally
    # via NativeAgentExecutor — don't also apply include_only_tools restriction.
    _effective_include_only = None if phase_config else include_only_tools
    filtered_tools = filter_tools_for_skills(
        all_tools, skill_names, skill_loader,
        exclude_tools=exclude_tools,
        include_only_tools=_effective_include_only,
    )

    # 3. Get config (model, max_iterations)
    config = get_agent_config_for_skills(skill_names, skill_loader)

    # 4. Select LLM based on complexity + skill model override
    # Opus is used whenever complexity='complex' AND dev_llm is available —
    # regardless of which skill is selected. Previously this required
    # config["model"] == "opus" (only true for the dev skill), which meant
    # Opus was NEVER used for hpc_cluster, file_search, etc. even when the
    # user explicitly requested it. Fixed: complexity drives model selection
    # globally; skill's model field is now only used to DOWNGRADE (i.e. a
    # skill can force sonnet even on complex tasks by setting model: sonnet).
    active_llm = llm
    model_display_name = "sonnet"  # default
    _use_opus = (
        dev_llm is not None
        and (use_opus or complexity == "complex")
    )
    if _use_opus:
        active_llm = dev_llm
        model_display_name = "opus"
        logger.info("Using Opus LLM for %s skill (reason=%s)", skill_names[0],
                    "user-requested" if use_opus else "complexity=complex")
    else:
        model_display_name = "sonnet"

    # ── Native executor path (Phase 2) ──────────────────────────────────
    # When IRIS_USE_NATIVE_EXECUTOR=1, use NativeAgentExecutor with direct
    # Anthropic API calls. Gains prompt caching in the tool loop.
    import os as _os
    _use_native_executor = _os.environ.get("IRIS_USE_NATIVE_EXECUTOR", "1") == "1"

    if _use_native_executor:
        from core.native_executor import NativeAgentExecutor
        from core.llm_provider import get_provider, get_provider_for_model, MODEL_REGISTRY
        from core.cost_tracker import CostTracker

        _litellm_url = _os.environ.get("LITELLM_URL", "http://localhost:8080")
        _litellm_key = _os.environ.get("LITELLM_VIRTUAL_KEY", "")

        if model_override and model_override in MODEL_REGISTRY:
            # OpenAI/NVIDIA model override — use get_provider_for_model for auto-detection
            _model_id = model_override
            model_display_name = model_override.split(".")[-1]  # e.g. "nemotron-super-3-120b"
            _provider = get_provider_for_model(
                _model_id,
                base_url=_litellm_url,
                api_key=_litellm_key,
                temperature=0,
                timeout=300,
            )
            _max_tokens = MODEL_REGISTRY[_model_id]["max_tokens"]
            _thinking_budget = 0
        else:
            # Default Anthropic path (unchanged)
            _model_id = (
                "anthropic.claude-opus-4-6-v1" if model_display_name == "opus"
                else "anthropic.claude-sonnet-4-6"
            )
            _thinking_budget = 10000 if model_display_name == "opus" else 5000
            _max_tokens = 32000 if model_display_name == "opus" else 16000

            _provider = get_provider(
                "anthropic",
                model_id=_model_id,
                base_url=_litellm_url,
                api_key=_litellm_key,
                temperature=0,
                max_tokens=_max_tokens,
                thinking_budget=_thinking_budget,
                timeout=300,
            )

        # Build a cheaper Sonnet provider for research phase when Opus is active
        _research_provider = None
        if model_display_name == "opus" and not model_override:
            _research_provider = get_provider(
                "anthropic",
                model_id="anthropic.claude-sonnet-4-6",
                base_url=_litellm_url,
                api_key=_litellm_key,
                temperature=0,
                max_tokens=16000,
                thinking_budget=5000,
                timeout=300,
            )

        executor = NativeAgentExecutor(
            provider=_provider,
            tools=filtered_tools,
            system_prompt=system_prompt,
            max_iterations=config["max_iterations"],
            phase_config=phase_config,
            websearch_enabled=websearch_enabled,
            active_plan_path=active_plan_path,
            research_provider=_research_provider,
            pel=pel,
        )

        logger.info(
            f"Created NATIVE agent for skills={skill_names}, "
            f"tools={len(filtered_tools)}, "
            f"max_iterations={config['max_iterations']}, "
            f"model={model_display_name} ({_model_id}), "
            f"max_tokens={_max_tokens}, "
            f"thinking_budget={_thinking_budget}, "
            f"provider={'openai' if model_override else 'anthropic'}, "
            f"research_model={'sonnet' if _research_provider else 'same'}, "
            f"user_context={'yes' if user_context else 'no'}"
        )

        return executor, len(filtered_tools), model_display_name

    # ── LangChain fallback path ─────────────────────────────────────────

    # 5. Build prompt template
    # CRITICAL: Escape curly braces in system_prompt before passing to
    # ChatPromptTemplate. The system prompt contains dynamic content
    # (user context, knowledge base, skill content) that may include
    # literal curly braces like {name}, {variable}, or {}. LangChain's
    # ChatPromptTemplate interprets these as template variables, causing
    # KeyError: "Input to ChatPromptTemplate is missing variables".
    # Escaping { → {{ and } → }} makes them literal.
    escaped_system_prompt = escape_prompt_braces(system_prompt)

    prompt = ChatPromptTemplate.from_messages([
        ("system", escaped_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 6. Create agent and executor
    agent = create_tool_calling_agent(active_llm, filtered_tools, prompt)

    # Inline observation cap: truncate tool outputs in the scratchpad
    # DURING the current turn. This prevents the LLM from seeing 60K+
    # char outputs in {agent_scratchpad} — the #1 cause of inflated
    # token costs within a single turn. Full output is preserved in
    # intermediate_steps for post-turn compression.
    from core.sub_agent import pre_truncate_large_observation, MAX_TOOL_OBSERVATION_CHARS

    def _trim_intermediate_steps(steps: list) -> list:
        """Trim large observations in the scratchpad to cap context size.

        This fires BEFORE each LLM call within the AgentExecutor loop,
        ensuring the model never sees raw 60K+ char outputs. The full
        output is still in intermediate_steps for post-turn compression.
        """
        trimmed = []
        for step in steps:
            action = step[0]
            observation = step[1] if len(step) > 1 else ""
            obs_str = str(observation) if observation else ""
            if len(obs_str) > MAX_TOOL_OBSERVATION_CHARS:
                tool_name = getattr(action, "tool", "unknown")
                obs_str = pre_truncate_large_observation(
                    obs_str, tool_name=tool_name,
                    max_chars=MAX_TOOL_OBSERVATION_CHARS,
                )
                trimmed.append((action, obs_str))
            else:
                trimmed.append(step)
        return trimmed

    from core.stuck_detection_callback import StuckDetectionCallback
    stuck_cb = StuckDetectionCallback(threshold=3)

    executor = AgentExecutor(
        agent=agent,
        tools=filtered_tools,
        max_iterations=config["max_iterations"],
        handle_parsing_errors=_make_escalation_aware_error_handler(),
        verbose=True,
        return_intermediate_steps=True,
        trim_intermediate_steps=_trim_intermediate_steps,
        callbacks=[stuck_cb],
    )

    logger.info(
        f"Created agent for skills={skill_names}, "
        f"tools={len(filtered_tools)}, "
        f"max_iterations={config['max_iterations']}, "
        f"model={model_display_name}, "
        f"user_context={'yes' if user_context else 'no'}"
    )

    return executor, len(filtered_tools), model_display_name
