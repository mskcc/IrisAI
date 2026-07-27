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

"""core/memory_state.py — Unified memory system for IrisAI.

Single storage location: /home/{USER}/{IRISAI_APP_NAME}/memory/
(IRISAI_APP_NAME is environment-specific: IrisAItest, IrisAIdev, etc.)

Structure:
    session_logs/{session_id}.jsonl  (per-turn, crash-safe via fsync, 30-day retention)
    memory/projects/{name}/
        ├── status.md      (REPLACE — current snapshot, written by deferred curation)
        ├── knowledge.md   (APPEND-ONLY — permanent facts, constraints, paths)
        └── history.md     (APPEND-ONLY — session summaries)
    memory/session/
        └── last_turn.json (overwritten every turn, instant resume)

Persistence model (3 layers):
1. Executor writes directly (update_memory tool) — best-effort during session
2. Per-turn session_log append (session_logs/*.jsonl) — crash-safe, fsync, <1ms
3. Deferred Sonnet curation — runs at NEXT session start (guaranteed time)
   Reads session_log JSONL → writes 3 permanent files → marks log as curated

Markers in session_log JSONL:
- __SESSION_ENDED__ = clean exit (on_chat_end fired), curation pending
- __SESSION_END_CURATED__ = Sonnet curation complete
- Neither = crash (on_chat_end never fired)

Design principles:
- 3 permanent files per project: status.md + knowledge.md + history.md
- Session-end is INSTANT (<10ms) — just appends marker, no LLM call
- Sonnet curation deferred to next session start (has guaranteed time)
- knowledge.md is APPEND-ONLY — never overwrites existing entries
- Storage is per-environment via IRISAI_APP_NAME (not work_dir, which can change)
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Callable, Awaitable, Any

CurationProgressCallback = Callable[..., Awaitable[Any]]

logger = logging.getLogger("core.memory_state")

# ── Configuration ─────────────────────────────────────────────────────────────

IRISAI_APP_NAME = os.environ.get("IRISAI_APP_NAME", "IrisAIdev")


def get_memory_root() -> Path:
    """Return the unified memory root: /home/{USER}/{IRISAI_APP_NAME}/memory/.

    Uses IRISAI_APP_NAME env var (set per deployment: IrisAItest, IrisAIdev, etc.)
    and the current OS user. This path is stable across sessions — unlike work_dir
    which can be changed by the user at runtime.
    """
    username = os.environ.get("USER", "unknown")
    memory_root = Path(f"/home/{username}/{IRISAI_APP_NAME}") / "memory"
    memory_root.mkdir(parents=True, exist_ok=True)
    return memory_root


# ── Phase summary helper ──────────────────────────────────────────────────────


def _build_phase_summary(entries: list) -> str:
    """Build a deterministic phase summary from session log entries.

    Returns a structured string that tells curation LLMs exactly what phase
    the session reached — preventing false claims about work completion.
    """
    markers = [e for e in entries if e.get("type") == "phase_marker"]
    if not markers:
        return (
            "No phase tracking data available — treat all work as DISCUSSED ONLY "
            "unless explicit tool output confirms file creation."
        )

    lines = []
    last_phase = ""
    last_event = ""
    artifacts = []

    for m in markers:
        phase = m.get("phase", "?")
        event = m.get("event", "?")
        meta = m.get("metadata", {})
        ts = m.get("ts", "")[:19]
        lines.append(f"  [{ts}] {phase}: {event}")
        if meta.get("artifact"):
            artifacts.append(meta["artifact"])
        last_phase = phase
        last_event = event

    if last_event == "awaiting_approval":
        lines.append(
            f"\n⚠️ SESSION ENDED WHILE AWAITING APPROVAL for '{last_phase}' phase."
        )
        lines.append("The session NEVER proceeded past this point.")
        lines.append("ANY work described AFTER this phase was only DISCUSSED — NOT EXECUTED.")
    elif last_event == "started" and last_phase != "execute":
        lines.append(f"\n⚠️ SESSION ENDED DURING '{last_phase}' phase (never completed).")
        lines.append("No execution occurred. All file-creation claims are PLANS, not reality.")
    elif last_event == "completed" and last_phase in ("research", "plan"):
        lines.append(f"\n⚠️ '{last_phase}' phase completed but execute phase NEVER started.")
        lines.append("Files were PLANNED but NOT CREATED.")
    elif last_phase == "execute" and last_event in ("started", "completed"):
        lines.append(
            f"\n✅ Execute phase {'completed' if last_event == 'completed' else 'was in progress'}."
        )

    if artifacts:
        lines.append(f"\nArtifacts on disk (verified written): {', '.join(artifacts)}")

    return "\n".join(lines)


# ── Prompts ───────────────────────────────────────────────────────────────────

STATE_CURATION_PROMPT = """You are curating project state for future sessions. Based on the current session's
information, rewrite the project's status.md to reflect what is TRUE NOW.

Project name: {project_name}

Current status.md (may be empty if first session):
{current_state}

Session facts gathered this session:
{session_facts}

Latest findings (if any):
{findings_summary}

Active plan status (if any):
{plan_summary}

Recent conversation context:
{recent_context}

Phase state (AUTHORITATIVE — overrides any conversational claims):
{phase_summary}

---

REWRITE the status.md with these sections (omit empty sections):

## Current Status
- What the user is trying to achieve (their goal/intent)
- Last action and result
- Next step (what's pending or approved)

## Key Paths
- Code root, data dirs, conda envs, configs, output dirs
- Git branch and recent commit (if known)
- Only include paths VERIFIED during this session or confirmed still valid

## Environment & Config
- SLURM partition, GPU count, module loads
- Container paths, Python/CUDA versions
- Any environment-specific constraints

## Blockers
- Current blockers preventing progress (if any)
- Root cause of the MOST RECENT failure only (one paragraph max)
- Do NOT list historical anti-patterns — those belong in knowledge.md

## Active Plan
- Current phase (research/planning/execution/idle)
- Goal summary
- Progress (what's done, what remains)

## History
- Keep the last 3-5 session summaries from the existing History section.
- ADD a one-line summary of THIS session (date, what was accomplished or attempted).
- If there are more than 5 entries, remove the oldest ones.
- Format: "- YYYY-MM-DD: <one sentence summary>"
- NEVER delete this section — it provides continuity across sessions.

Rules:
- Be concise but complete. Target under 5K characters.
- Include EXACT paths, job IDs, commit hashes, branch names, version numbers — never paraphrase.
  A future session must be able to ACT on this without re-discovering these values.
- Remove anything superseded by this session's findings EXCEPT the History section.
- Do NOT include library-internal paths or traceback paths.
- Do NOT include temporary/dynamic paths that won't exist next session.
- Do NOT include "Critical Gotchas", "Lessons Learned", or "Critical Environment Variables"
  sections. Status.md tracks CURRENT STATE only.
  Permanent rules/gotchas live in knowledge.md (project) or global knowledge (_global).
  If status.md grows past 3K chars, you are including too much — be more concise.
- Write as if briefing a colleague who will continue this work tomorrow.
- If a section has no content, omit it entirely (EXCEPT History — always keep it).
- PLANNED vs DONE: The "Phase state" section above is AUTHORITATIVE. If the session is in
  research or plan phase, NO execution has occurred. Do not claim files were "written" or
  "created" unless phase state confirms execute phase ran. Use "planned", "proposed", "drafted".
  Mark such items as ⏳ (pending) in Active Plan — NEVER ✅ (done).
"""

SESSION_FACTS_CURATION_PROMPT = """You are saving session facts to persistent memory. Based on what was discovered
and accomplished in this session, extract the facts worth remembering for future sessions.

Session facts:
{session_facts}

Findings (if any):
{findings_summary}

Plan status (if any):
{plan_summary}

---

Output a clean bullet-point list of facts worth persisting. Include:
- Specific paths, configurations, or environment details discovered
- Errors encountered and their root causes
- Solutions found or workarounds applied
- Key decisions made and their reasoning
- Performance measurements or benchmarks
- Anti-patterns identified ("don't do X because Y")

Exclude:
- Transient information (temp files, job IDs that will expire)
- Facts that are obvious from reading the code/configs directly
- Duplicate information already captured elsewhere

If nothing is worth persisting, output: NOTHING_TO_SAVE

Keep under 3K characters.
"""

FINDINGS_CURATION_PROMPT = """You are deciding whether session findings should be persisted for future reference.

Project: {project_name}

Findings content:
{findings_text}

---

Should these findings be saved to persistent project memory? Answer YES if:
- They contain specific benchmarks, measurements, or performance data
- They identify root causes of issues that might recur
- They document environment-specific behavior or constraints
- They contain verified paths, configs, or settings

Answer NO if:
- The findings are purely about understanding existing code (which can be re-read)
- The findings are transient (build output, temporary errors already fixed)
- The information is already in the project's status.md

If YES, output the findings in a clean format suitable for long-term storage.
If NO, output just: NO_PERSIST

Start your response with either YES_PERSIST or NO_PERSIST on the first line.
"""


# ── Directory Helpers ─────────────────────────────────────────────────────────

def save_last_turn(
    project_name: str,
    user_msg: str,
    assistant_summary: str,
    session_id: str,
    plan_steps_done: Optional[List[str]] = None,
    plan_steps_pending: Optional[List[str]] = None,
    running_jobs: Optional[List[Dict]] = None,
    tool_calls_this_turn: Optional[List[str]] = None,
    active_plan_path: str = "",
    active_findings_path: str = "",
) -> None:
    """Overwritten every turn. Costs <1ms. The guaranteed minimum for resume.

    Even if Haiku curation fails or session is killed, this file always
    reflects the most recently completed turn — including exact plan progress,
    active plan file path, and any running background jobs.
    """
    data = {
        "project": project_name or "",
        "user": (user_msg or "")[:300],
        "assistant": (assistant_summary or "")[:500],
        "session_id": session_id or "",
        "timestamp": datetime.now().isoformat(),
        "plan_steps_done": (plan_steps_done or [])[-10:],
        "plan_steps_pending": (plan_steps_pending or [])[:10],
        "running_jobs": (running_jobs or [])[:5],
        "tool_calls_this_turn": (tool_calls_this_turn or [])[:10],
        "active_plan_path": active_plan_path or "",
        "active_findings_path": active_findings_path or "",
    }
    session_dir = get_memory_root() / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "last_turn.json"
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"[MEMORY_STATE] Failed to write last_turn.json: {e}")


MAX_ATTEMPTS_LOG_ENTRIES = 50
_ATTEMPTS_TRUNCATE_BYTES = 25000


def _truncate_attempts_log(log_path: Path, keep: int = 50) -> None:
    """Keep only the last `keep` entries in the attempts log."""
    try:
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        if len(lines) <= keep:
            return
        kept = lines[-keep:]
        log_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        logger.info(f"[MEMORY_STATE] Truncated attempts_log: {len(lines)} → {keep}")
    except Exception as e:
        logger.warning(f"[MEMORY_STATE] Failed to truncate attempts_log: {e}")


def append_attempt(project_name: str, action: str, result: str, error: str = "") -> None:
    """Append an entry to the project's attempts_log.jsonl.

    Tracks what was tried and whether it succeeded or failed. Auto-truncates
    to the most recent MAX_ATTEMPTS_LOG_ENTRIES when file exceeds size threshold.
    """
    if not project_name or not action:
        return
    project_dir = get_memory_root() / "projects" / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    _ensure_project_files(project_dir, project_name)
    log_path = project_dir / "attempts_log.jsonl"

    entry = json.dumps({
        "action": (action or "")[:200],
        "result": (result or "")[:200],
        "error": (error or "")[:200],
        "timestamp": datetime.now().isoformat(),
    })

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception as e:
        logger.warning(f"[MEMORY_STATE] Failed to append attempts_log: {e}")
        return

    try:
        if log_path.stat().st_size > _ATTEMPTS_TRUNCATE_BYTES:
            _truncate_attempts_log(log_path, MAX_ATTEMPTS_LOG_ENTRIES)
    except OSError:
        pass


def load_attempts_log(project_name: str, limit: int = 20) -> List[Dict]:
    """Load the last N entries from a project's attempts_log.jsonl.

    Uses a seek-from-end approach to avoid reading the entire file into memory
    when the log grows large over months of use.
    """
    if not project_name:
        return []
    log_path = get_memory_root() / "projects" / project_name / "attempts_log.jsonl"
    if not log_path.exists():
        return []
    try:
        file_size = log_path.stat().st_size
        if file_size == 0:
            return []
        # Each entry is ~250 bytes max (action:200 + result:200 + error:200 + overhead)
        # Read more than needed to account for shorter entries
        read_size = min(file_size, limit * 500)
        with open(log_path, "r", encoding="utf-8") as f:
            if read_size < file_size:
                f.seek(file_size - read_size)
                f.readline()  # skip partial first line
            lines = f.readlines()
        entries = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries[-limit:]
    except Exception:
        return []


# ── Work Timeline ────────────────────────────────────────────────────────────



def load_last_turn() -> Optional[Dict]:
    """Load last_turn.json. Returns None if not found."""
    path = get_memory_root() / "session" / "last_turn.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _ensure_project_files(project_dir: Path, project_name: str) -> None:
    """Create the 3 memory scaffold files if they don't exist."""
    initial_files = {
        "status.md": f"# {project_name} — Status\n\nNew project. No work done yet.\n",
        "knowledge.md": f"# {project_name} — Knowledge\n\n",
        "history.md": f"# {project_name} — History\n\n",
    }
    for fname, content in initial_files.items():
        fpath = project_dir / fname
        if not fpath.exists():
            try:
                fpath.write_text(content, encoding="utf-8")
            except Exception:
                pass


def get_project_dir(project_name: str) -> Path:
    """Return the project memory directory, creating it if needed.

    Resolves aliases/typos to canonical name before creating the directory,
    preventing re-creation of merged/renamed project dirs.
    """
    resolved = resolve_project_name(project_name)
    project_dir = get_memory_root() / "projects" / resolved
    project_dir.mkdir(parents=True, exist_ok=True)
    _ensure_project_files(project_dir, resolved)
    return project_dir


def get_session_dir() -> Path:
    """Return the session memory directory, creating it if needed."""
    session_dir = get_memory_root() / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def get_project_state_dir(work_dir: str, project_name: str) -> Path:
    """Return the project state directory. Uses unified memory root.

    The work_dir parameter is kept for backward compatibility but the actual
    storage is always in ~/.iris/memory/projects/{project_name}/.
    """
    return get_project_dir(project_name)


# ── Core State Functions ──────────────────────────────────────────────────────

def load_project_state(work_dir: str, project_name: str) -> str:
    """Load a project's status.md from unified memory. Returns empty string if not found.

    Args:
        work_dir: Kept for API compat — not used for storage location.
        project_name: The project whose state to load.
    """
    if not project_name:
        return ""
    state_path = get_project_dir(project_name) / "status.md"
    if state_path.exists():
        try:
            return state_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"[MEMORY_STATE] Could not read {state_path}: {e}")
    return ""


def write_project_state(work_dir: str, project_name: str, content: str) -> bool:
    """Write curated state to project's status.md in unified memory.

    Args:
        work_dir: Kept for API compat — not used for storage location.
        project_name: The project whose state to write.
        content: The curated state content.
    """
    if not project_name or not content:
        return False
    state_path = get_project_dir(project_name) / "status.md"
    try:
        state_path.write_text(content, encoding="utf-8")
        logger.info(f"[MEMORY_STATE] Saved status.md for '{project_name}' ({len(content)} chars)")
        return True
    except Exception as e:
        logger.error(f"[MEMORY_STATE] Failed to write {state_path}: {e}")
        return False




async def save_project_state(
    work_dir: str,
    project_name: str,
    session_facts: str = "",
    findings_summary: str = "",
    plan_summary: str = "",
    recent_context: str = "",
    phase_summary: str = "",
) -> bool:
    """Curate and save project state using Haiku.

    Called at milestones, session end, context switches, and user explicit request.
    ALSO saves session facts to disk regardless of project.

    Args:
        work_dir: User's work directory (for backward compat, not storage).
        project_name: Active project name (can be empty — session facts still save).
        session_facts: Accumulated session facts.
        findings_summary: Latest findings text.
        plan_summary: Current plan status text.
        recent_context: Recent conversation messages for context.
        phase_summary: Phase state summary from session log markers.

    Returns:
        True if state was successfully saved.
    """
    # NOTE: session facts are already persisted by the caller (app.py post-turn block).
    # No need to duplicate that write here.

    # If no project (or "general" catch-all), nothing more to do
    if not project_name or project_name == "general":
        logger.debug("[MEMORY_STATE] No project_name — skipping project state curation")
        return True

    from core.sub_agent import _call_sub_agent_llm

    current_state = load_project_state(work_dir, project_name)

    prompt = STATE_CURATION_PROMPT.format(
        project_name=project_name,
        current_state=current_state or "(No existing state — this is the first session)",
        session_facts=session_facts or "(No session facts available)",
        findings_summary=findings_summary or "(No findings this session)",
        plan_summary=plan_summary or "(No active plan)",
        recent_context=recent_context or "(No recent context available)",
        phase_summary=phase_summary or "(No phase tracking data — session predates markers)",
    )

    result = await _call_sub_agent_llm(prompt)

    if not result or result.startswith("[Error:"):
        logger.warning(f"[MEMORY_STATE] Haiku curation failed: {result[:200] if result else 'empty'}")
        return False

    return write_project_state(work_dir, project_name, result)


KNOWLEDGE_APPEND_PROMPT = """You are extracting DURABLE knowledge from a session's raw facts.

Project: {project_name}

Session facts (raw assistant responses from this session):
{session_facts}

Existing knowledge.md (DO NOT duplicate entries already here):
{existing_knowledge}

---

Extract NEW permanent facts that a future session needs. Only include entries NOT already
present in existing knowledge.md (check semantics, not exact wording).

For each entry, use this format:
- [TYPE]: <exact fact with real values — paths, versions, IDs, commands>
  Why: <root cause or reasoning — so a future session can judge if this still applies>
  Applies when: <trigger condition — when should a future session act on this>

TYPE must be one of: CONSTRAINT, VALIDATED_APPROACH, FAILED_ATTEMPT, REFERENCE_PATH, CONFIGURATION

Rules:
- Include EXACT paths, job IDs, commit hashes, version numbers — never paraphrase
- Only extract facts that will still be useful in future sessions (not transient state)
- If nothing new was learned this session, respond with exactly: NOTHING_NEW
- Do NOT include conversational content, greetings, or status updates
- Do NOT include facts that are only relevant to the current in-progress task
- Maximum 10 entries per session (prioritize by importance)
"""


async def append_session_knowledge(
    project_name: str,
    session_facts: str,
) -> bool:
    """Append durable knowledge from session facts to project's knowledge.md.

    Called ONLY at session-end. Appends new entries — never replaces existing content.
    Uses Haiku to extract structured knowledge with dedup against existing entries.

    Returns:
        True if knowledge was appended (or nothing new to append).
    """
    if not project_name or project_name == "general":
        logger.debug("[KNOWLEDGE_APPEND] No project — skipping")
        return True

    if not session_facts or len(session_facts) < 100:
        logger.debug("[KNOWLEDGE_APPEND] Session facts too short — skipping")
        return True

    from core.sub_agent import _call_sub_agent_llm

    knowledge_path = get_project_dir(project_name) / "knowledge.md"
    existing_knowledge = ""
    if knowledge_path.exists():
        existing_knowledge = knowledge_path.read_text(encoding="utf-8")

    prompt = KNOWLEDGE_APPEND_PROMPT.format(
        project_name=project_name,
        session_facts=session_facts[-8000:],
        existing_knowledge=existing_knowledge[-4000:] if existing_knowledge else "(empty — first session)",
    )

    result = await _call_sub_agent_llm(prompt)

    if not result or result.startswith("[Error:"):
        logger.warning(f"[KNOWLEDGE_APPEND] Haiku failed: {result[:200] if result else 'empty'}")
        return False

    if "NOTHING_NEW" in result.strip()[:20]:
        logger.info(f"[KNOWLEDGE_APPEND] Nothing new for '{project_name}'")
        return True

    # Append to knowledge.md (never replace)
    try:
        new_section = f"\n\n## Session {__import__('datetime').date.today().isoformat()}\n\n{result.strip()}\n"
        with open(knowledge_path, "a", encoding="utf-8") as f:
            f.write(new_section)
        logger.info(f"[KNOWLEDGE_APPEND] Appended {len(result)} chars to '{project_name}/knowledge.md'")
        return True
    except Exception as e:
        logger.error(f"[KNOWLEDGE_APPEND] Failed to write: {e}")
        return False


def extract_knowledge_from_compaction_summary(summary_text: str) -> str:
    """Extract the ===KNOWLEDGE_EXTRACT=== section from a compaction summary.

    Returns the knowledge content (without delimiters), or empty string if
    not present or contains NOTHING_NEW sentinel.
    """
    if "===KNOWLEDGE_EXTRACT===" not in summary_text:
        return ""

    parts = summary_text.split("===KNOWLEDGE_EXTRACT===", 1)
    if len(parts) < 2:
        return ""

    knowledge_section = parts[1]
    if "===END_KNOWLEDGE_EXTRACT===" in knowledge_section:
        knowledge_section = knowledge_section.split("===END_KNOWLEDGE_EXTRACT===", 1)[0]

    knowledge_section = knowledge_section.strip()

    if not knowledge_section or knowledge_section.startswith("NOTHING_NEW"):
        return ""

    return knowledge_section


def append_compaction_knowledge(project_name: str, knowledge_content: str) -> bool:
    """Append knowledge extracted during compaction to project's knowledge.md.

    Follows the same append-only pattern as session_end_curate. Uses a
    'Compaction' header to distinguish from session-end entries.

    Args:
        project_name: Active project name.
        knowledge_content: Pre-extracted knowledge text (already deduped by LLM).

    Returns:
        True if written successfully (or nothing to write).
    """
    if not project_name or project_name == "general":
        return True

    if not knowledge_content or len(knowledge_content) < 10:
        return True

    from datetime import datetime

    knowledge_path = get_project_dir(project_name) / "knowledge.md"
    try:
        session_header = f"\n\n## Compaction {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        with open(knowledge_path, "a", encoding="utf-8") as f:
            f.write(session_header + knowledge_content.strip() + "\n")
        logger.info(
            f"[COMPACTION_KNOWLEDGE] Appended {len(knowledge_content)} chars "
            f"to '{project_name}/knowledge.md'"
        )
        return True
    except Exception as e:
        logger.error(f"[COMPACTION_KNOWLEDGE] Failed to write: {e}")
        return False


GLOBAL_KNOWLEDGE_PROMPT = """You are extracting DURABLE cross-project knowledge from a general session.
This session had no specific project context — the user was exploring, asking questions, or
doing general work. Extract facts useful across ALL future sessions.

== CONVERSATION CONTEXT ==
{recent_context}

== SESSION FACTS ==
{session_facts}

== EXISTING global knowledge.md (DO NOT duplicate) ==
{existing_knowledge}

---

Extract NEW permanent facts. Only include entries NOT already in existing knowledge.

For each entry, use this format:
- [TYPE]: <exact fact with real values — paths, versions, IDs, commands>
  Why: <root cause or reasoning>
  Applies when: <trigger condition>

TYPE must be one of: CONSTRAINT, VALIDATED_APPROACH, FAILED_ATTEMPT, REFERENCE_PATH, CONFIGURATION, USER_PREFERENCE

Rules:
- Only extract facts useful in FUTURE sessions (not transient state)
- Include EXACT paths, versions, commands — never paraphrase
- If nothing durable was learned, respond with exactly: NOTHING_NEW
- Do NOT include conversational content or status updates
- Maximum 5 entries (these are cross-project, so be selective)
"""


async def curate_global_knowledge(
    username: str,
    session_facts: str = "",
    recent_context: str = "",
) -> bool:
    """Extract cross-project knowledge from a general (no-project) session.

    Lighter than full project curation — only appends to global knowledge.md.
    No status, no history (those only make sense per-project).

    Returns:
        True if knowledge was appended (or nothing new to append).
    """
    combined = (session_facts or "") + (recent_context or "")
    if len(combined) < 100:
        logger.debug("[GLOBAL_KNOWLEDGE] Session too short — skipping")
        return True

    from core.sub_agent import _call_sub_agent_llm
    from core.memory import get_global_memory_dir
    from core.persistence import get_user_data_dir

    memory_dir = get_user_data_dir(username) / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    knowledge_path = memory_dir / "knowledge.md"

    existing_knowledge = ""
    if knowledge_path.exists():
        existing_knowledge = knowledge_path.read_text(encoding="utf-8")

    prompt = GLOBAL_KNOWLEDGE_PROMPT.format(
        recent_context=recent_context[-6000:] if recent_context else "(no context)",
        session_facts=session_facts[-4000:] if session_facts else "(no facts)",
        existing_knowledge=existing_knowledge[-4000:] if existing_knowledge else "(empty — first session)",
    )

    result = await _call_sub_agent_llm(prompt)

    if not result or result.startswith("[Error:"):
        logger.warning(f"[GLOBAL_KNOWLEDGE] Haiku failed: {result[:200] if result else 'empty'}")
        return False

    if "NOTHING_NEW" in result.strip()[:20]:
        logger.info("[GLOBAL_KNOWLEDGE] Nothing new to save globally")
        return True

    try:
        import datetime
        new_section = f"\n\n## Session {datetime.date.today().isoformat()}\n\n{result.strip()}\n"
        with open(knowledge_path, "a", encoding="utf-8") as f:
            f.write(new_section)
        logger.info(f"[GLOBAL_KNOWLEDGE] Appended {len(result)} chars to global knowledge.md")
        return True
    except Exception as e:
        logger.error(f"[GLOBAL_KNOWLEDGE] Failed to write: {e}")
        return False


# ── Session-end Sonnet curation ──────────────────────────────────────────────

CURATION_MODEL = os.environ.get(
    "CURATION_MODEL", "anthropic.claude-sonnet-4-6"
)

SESSION_END_CURATION_PROMPT = """You are curating project memory at session end. Read the full conversation below
and write updates to 3 memory files for the project "{project_name}".

== CONVERSATION THIS SESSION ==
{conversation_text}

== CURRENT status.md ==
{current_status}

== CURRENT knowledge.md (DO NOT duplicate existing entries) ==
{current_knowledge}

== Active plan (if any) ==
{plan_summary}

== PHASE STATE (AUTHORITATIVE — overrides any conversational claims) ==
{phase_summary}

---

Produce output in EXACTLY this format (3 sections separated by the markers shown):

===STATUS_MD===
[Write the FULL new status.md content. This REPLACES the old file entirely.]

Sections to include (omit if empty):
## Current Status
- User's goal/intent this session
- Last action and result
- Next step pending

## Key Paths
- Code root, data dirs, conda envs, output dirs
- Git branch and recent commit (if known)

## Environment & Config
- SLURM partition, GPU, container, Python/CUDA versions

## Blockers
- Current blockers (if any)

## Active Plan
- Phase, goal, progress

## History
- Keep last 3-5 entries from existing History section
- ADD one line for THIS session: "- YYYY-MM-DD: <what was accomplished or attempted>"

===KNOWLEDGE_MD===
[Write ONLY NEW entries to append. If nothing new was learned, write: NOTHING_NEW]
[If a fact applies to ALL projects (not just this one), prefix the entry with [GLOBAL] — it will be
saved to global knowledge via update_memory('knowledge.md', content, project='_global').]

Format each entry as:
- [TYPE]: <exact fact — real paths, versions, IDs, commands>
  Why: <root cause or reasoning>
  Applies when: <trigger condition for future sessions>

TYPE = CONSTRAINT | VALIDATED_APPROACH | FAILED_ATTEMPT | REFERENCE_PATH | CONFIGURATION | USER_PREFERENCE

USER PREFERENCES — PRIORITY EXTRACTION:
If the user expressed dissatisfaction, preferences, or style guidance during this session,
you MUST extract them as [USER_PREFERENCE] entries. Look for:
- Explicit complaints ("not good enough", "that's wrong", "try again", "I don't like this")
- Tool/library preferences ("don't use matplotlib", "I prefer SVG", "use Opus for this")
- Quality standards ("needs better colors", "not publication-grade", "too simple")
- Workflow preferences ("always search the web first", "keep both WT and mutant")
- Scope preferences ("make it simultaneous", "include all conditions")

Format for preferences:
- [USER_PREFERENCE]: <interpreted preference — what the user wants>
  Raw: "<exact user quote, up to 200 chars>"
  Applies when: <context where this preference should be recalled>

===HISTORY_MD===
[Write ONE line to append to history.md:]
- YYYY-MM-DD: <one sentence summary of what was accomplished or attempted this session>

---

RULES:
- Include EXACT paths, job IDs, commit hashes, branch names, version numbers — NEVER paraphrase
- A future session must be able to ACT on status.md without re-discovering values
- knowledge.md entries must NOT duplicate what's already in existing knowledge.md
- Be concise. status.md target: under 3K chars. knowledge.md: max 10 new entries.
- Write as if briefing a colleague who will continue this work tomorrow.
- CRITICAL — PLANNED vs DONE: The PHASE STATE section above is AUTHORITATIVE.
  If the session ended at "awaiting_approval" or in research/plan phase, then ANY file
  creation, code writing, or execution described in conversation is a PLAN or INTENT —
  NOT something that was done. You MUST reflect this accurately:
  * Use "planned" / "drafted" / "proposed" — NEVER "written" / "created" / "completed"
  * In History: write "researched X" or "planned X" — NEVER "built X" or "wrote X"
  * In Active Plan progress: mark such items as ⏳ (pending) — NEVER ✅ (done)
  * The ONLY way something is "done" is if phase state shows execute:started or execute:completed
"""


async def session_end_curate(
    project_name: str,
    work_dir: str,
    conversation_text: str,
    plan_summary: str = "",
    skip_history: bool = False,
    entries: list = None,
) -> bool:
    """Sonnet-level curation — writes status.md + knowledge.md (+ history.md unless skipped).

    Called at session end (deferred to next start) or mid-session (on_chat_resume).
    For mid-session curation, pass skip_history=True to avoid premature history entries.

    Args:
        entries: Raw session log entries (including phase_markers) for phase awareness.

    Returns:
        True if curation succeeded and files were written.
    """
    if not project_name or project_name == "general":
        logger.debug("[SESSION_END_CURATE] No project — skipping")
        return True

    from core.sub_agent import _call_sub_agent_llm

    project_dir = get_project_dir(project_name)
    project_dir.mkdir(parents=True, exist_ok=True)

    current_status = load_project_state(work_dir, project_name) or "(first session)"
    knowledge_path = project_dir / "knowledge.md"
    current_knowledge = ""
    if knowledge_path.exists():
        current_knowledge = knowledge_path.read_text(encoding="utf-8")

    # Cap conversation to last 30K chars for Sonnet input
    _conv_capped = conversation_text[-30000:] if len(conversation_text) > 30000 else conversation_text

    # Enrich plan_summary with on-disk findings/plan state for disruption awareness
    _enriched_plan_summary = plan_summary or "(no active plan)"
    try:
        _proj_findings_dir = project_dir / "findings"
        _proj_plans_dir = project_dir / "plans"
        if _proj_plans_dir.exists():
            _plan_files = sorted(_proj_plans_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
            if _plan_files:
                _latest_plan = _plan_files[0].read_text(encoding="utf-8")[:2000]
                _has_unchecked = "- [ ]" in _latest_plan
                _enriched_plan_summary += (
                    f"\n\n[ON-DISK PLAN: {_plan_files[0].name}]\n{_latest_plan}"
                )
                if _has_unchecked:
                    _enriched_plan_summary += "\n(Plan has unchecked steps — may have been interrupted)"
        if _proj_findings_dir.exists():
            _findings_files = sorted(_proj_findings_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
            if _findings_files:
                _enriched_plan_summary += f"\n\n[ON-DISK FINDINGS: {', '.join(f.name for f in _findings_files[:3])}]"
    except Exception as _e:
        logger.debug(f"[SESSION_END_CURATE] Could not scan findings/plans: {_e}")

    _phase_summary = _build_phase_summary(entries) if entries else (
        "No phase tracking data — session predates phase markers. "
        "Treat all work as TENTATIVE unless file paths are confirmed to exist on disk."
    )

    prompt = SESSION_END_CURATION_PROMPT.format(
        project_name=project_name,
        conversation_text=_conv_capped,
        current_status=current_status,
        current_knowledge=current_knowledge[-4000:] if current_knowledge else "(empty — first session)",
        plan_summary=_enriched_plan_summary,
        phase_summary=_phase_summary,
    )

    result = await _call_sub_agent_llm(prompt, model=CURATION_MODEL)

    if not result or result.startswith("[Error:"):
        logger.warning(f"[SESSION_END_CURATE] Sonnet curation failed: {result[:200] if result else 'empty'}")
        return False

    _ok = _parse_and_write_curation(project_name, work_dir, result, skip_history=skip_history)
    return _ok


def _parse_and_write_curation(project_name: str, work_dir: str, result: str, skip_history: bool = False) -> bool:
    """Parse Sonnet's structured output and write to the 3 files (or 2 if skip_history)."""
    project_dir = get_project_dir(project_name)

    # Extract sections
    status_content = ""
    knowledge_content = ""
    history_content = ""

    if "===STATUS_MD===" in result:
        parts = result.split("===STATUS_MD===", 1)
        after_status = parts[1] if len(parts) > 1 else ""
        if "===KNOWLEDGE_MD===" in after_status:
            status_content = after_status.split("===KNOWLEDGE_MD===", 1)[0].strip()
            remainder = after_status.split("===KNOWLEDGE_MD===", 1)[1]
            if "===HISTORY_MD===" in remainder:
                knowledge_content = remainder.split("===HISTORY_MD===", 1)[0].strip()
                history_content = remainder.split("===HISTORY_MD===", 1)[1].strip()
            else:
                knowledge_content = remainder.strip()
        else:
            status_content = after_status.strip()

    if not status_content:
        logger.warning("[SESSION_END_CURATE] Could not parse status section from Sonnet output")
        return False

    # 1. Write status.md (REPLACE)
    try:
        write_project_state(work_dir, project_name, status_content)
        logger.info(f"[SESSION_END_CURATE] status.md replaced ({len(status_content)} chars)")
    except Exception as e:
        logger.error(f"[SESSION_END_CURATE] Failed to write status.md: {e}")
        return False

    # 2. Append to knowledge.md (APPEND-ONLY)
    if knowledge_content and "NOTHING_NEW" not in knowledge_content[:20]:
        try:
            knowledge_path = project_dir / "knowledge.md"
            session_header = f"\n\n## Session {datetime.now().strftime('%Y-%m-%d')}\n\n"
            with open(knowledge_path, "a", encoding="utf-8") as f:
                f.write(session_header + knowledge_content.strip() + "\n")
            logger.info(f"[SESSION_END_CURATE] knowledge.md appended ({len(knowledge_content)} chars)")
        except Exception as e:
            logger.error(f"[SESSION_END_CURATE] Failed to append knowledge.md: {e}")

    # 3. Append to history.md (APPEND-ONLY) — skip for mid-session curation
    if history_content and not skip_history:
        try:
            history_path = project_dir / "history.md"
            if not history_path.exists():
                history_path.write_text("# Project History\n\n", encoding="utf-8")
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(history_content.strip() + "\n")
            logger.info(f"[SESSION_END_CURATE] history.md appended")
        except Exception as e:
            logger.error(f"[SESSION_END_CURATE] Failed to append history.md: {e}")

    return True


def _parse_session_log_state(log_path) -> dict:
    """Parse a session log JSONL to determine its curation state and group entries by project.

    Returns dict with keys:
      - curated (bool): whether __SESSION_END_CURATED__ marker is present
      - ended (bool): whether __SESSION_ENDED__ marker is present
      - projects: dict of {project_name: {"work_dir": str, "entries": [...]}}
      - phase_markers: list of all phase_marker entries (global, ordered)
    """
    from pathlib import Path as _Path
    log_file = _Path(log_path)
    result = {"curated": False, "ended": False, "projects": {}, "phase_markers": []}

    if not log_file.exists():
        return result

    current_project = ""
    current_work_dir = ""

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = entry.get("content", "")
                entry_type = entry.get("type", "")

                if "__SESSION_END_CURATED__" in content or entry_type == "session_end_curated":
                    result["curated"] = True
                    break

                if "__SESSION_ENDED__" in content or entry_type == "session_ended":
                    result["ended"] = True
                    meta = entry.get("metadata", {})
                    _proj = meta.get("project") or entry.get("project")
                    _wdir = meta.get("work_dir") or entry.get("work_dir")
                    if _proj:
                        current_project = _proj
                    if _wdir:
                        current_work_dir = _wdir
                    continue

                if entry_type == "session_start":
                    current_project = entry.get("project_name", "")
                    current_work_dir = entry.get("work_dir", "")
                    continue

                if entry_type == "project_switch" or "__PROJECT_SWITCH__" in content:
                    meta = entry.get("metadata", {})
                    current_project = meta.get("project", "") or entry.get("project", "")
                    current_work_dir = meta.get("work_dir", "") or current_work_dir
                    continue

                if entry_type == "phase_marker":
                    result["phase_markers"].append(entry)
                    _proj_key = current_project or "_no_project"
                    if _proj_key not in result["projects"]:
                        result["projects"][_proj_key] = {
                            "work_dir": current_work_dir,
                            "entries": [],
                        }
                    result["projects"][_proj_key]["entries"].append(entry)
                    continue

                if entry_type == "message":
                    _proj_key = current_project or "_no_project"
                    if _proj_key not in result["projects"]:
                        result["projects"][_proj_key] = {
                            "work_dir": current_work_dir,
                            "entries": [],
                        }
                    result["projects"][_proj_key]["entries"].append(entry)
    except Exception as e:
        logger.warning(f"[DEFERRED_CURATE] Failed to read session log {log_path}: {e}")

    return result


async def curate_pending_session_logs(
    username: str, current_session_id: str, max_sessions: int = 10,
    progress_cb: Optional[CurationProgressCallback] = None,
) -> str:
    """Curate pending session logs (deferred curation + crash recovery).

    Called at session start (in background). Scans previous session logs for
    those that haven't been curated yet:
    - Has __SESSION_ENDED__ but not __SESSION_END_CURATED__ → clean exit, curation deferred
    - Has neither marker → session crashed
    Both get Haiku curation → global knowledge / project files → marker appended.

    Caps at max_sessions per startup to avoid blocking for minutes.

    Returns user-facing summary message (or empty string if nothing to curate).
    """
    from core.session_log import list_session_logs

    all_logs = list_session_logs(username)
    prev_logs = [l for l in all_logs if l["session_id"] != current_session_id]

    if not prev_logs:
        return ""

    curated_count = 0
    recovered_count = 0
    skipped_count = 0
    messages = []

    # Process oldest first (list is sorted by mtime desc, so reverse)
    pending_logs = []
    for log_info in reversed(prev_logs):
        state = _parse_session_log_state(log_info["path"])

        if state["curated"]:
            continue

        projects = state["projects"]
        if not projects:
            continue

        pending_logs.append((log_info, state))

    if not pending_logs:
        return ""

    total_pending = len(pending_logs)
    logger.info(f"[DEFERRED_CURATE] Found {total_pending} uncurated sessions (processing up to {max_sessions})")

    if progress_cb:
        try:
            items = []
            for i, (li, st) in enumerate(pending_logs[:max_sessions]):
                _label = "deferred" if st["ended"] else "crash recovery"
                _projects = ", ".join(st["projects"].keys()) or "global"
                items.append({"idx": i, "title": f"[{_label}] {_projects}"})
            await progress_cb("found", items=items)
        except Exception:
            pass

    for idx, (log_info, state) in enumerate(pending_logs[:max_sessions]):
        was_clean_exit = state["ended"]
        label = "deferred curation" if was_clean_exit else "crash recovery"
        session_success = False

        if progress_cb:
            try:
                await progress_cb("start", idx=idx)
            except Exception:
                pass

        for project_name, project_data in state["projects"].items():
            entries = project_data["entries"]
            if len(entries) < 3:
                continue

            if not project_name or project_name in ("general", "_no_project"):
                if project_name == "_no_project" and len(entries) > 5:
                    logger.info(f"[DEFERRED_CURATE] {label} [{idx+1}/{min(total_pending, max_sessions)}] — skipping {len(entries)} messages (no project marker, likely legacy session)")
                    session_success = True
                    if progress_cb:
                        try:
                            await progress_cb("skip", idx=idx, project="_no_project", entry_count=len(entries))
                        except Exception:
                            pass
                    continue
                conversation_text = "\n".join(
                    f"[{e.get('role', 'unknown')}]: {e.get('content', '')[:300]}"
                    for e in entries[-10:]
                )
                logger.info(f"[DEFERRED_CURATE] {label} [{idx+1}/{min(total_pending, max_sessions)}] — {len(entries)} messages (global knowledge)")
                success = await curate_global_knowledge(
                    username=username,
                    recent_context=conversation_text,
                )
                if success:
                    session_success = True
                    if progress_cb:
                        try:
                            await progress_cb("project_done", idx=idx, project=project_name or "general", entry_count=len(entries))
                        except Exception:
                            pass
                continue

            conversation_text = "\n".join(
                f"[{e.get('role', 'unknown')}]: {e.get('content', '')}"
                for e in entries
            )

            logger.info(
                f"[DEFERRED_CURATE] {label} [{idx+1}/{min(total_pending, max_sessions)}] — "
                f"{len(entries)} messages for project '{project_name}'"
            )

            success = await session_end_curate(
                project_name=project_name,
                work_dir=project_data["work_dir"],
                conversation_text=conversation_text,
                entries=entries,
            )

            if success:
                session_success = True
                if progress_cb:
                    try:
                        await progress_cb("project_done", idx=idx, project=project_name, entry_count=len(entries))
                    except Exception:
                        pass
                if not was_clean_exit:
                    _last = entries[-1].get("content", "")[:150] if entries else ""
                    messages.append(
                        f"Recovered crashed session for '{project_name}'. Last: {_last}"
                    )
            else:
                logger.warning(f"[DEFERRED_CURATE] {label} failed for '{project_name}' — will retry")
                if progress_cb:
                    try:
                        await progress_cb("project_failed", idx=idx, project=project_name, entry_count=len(entries))
                    except Exception:
                        pass

        if session_success:
            try:
                with open(log_info["path"], "a", encoding="utf-8") as f:
                    marker = json.dumps({
                        "type": "session_end_curated",
                        "ts": datetime.now().isoformat(),
                        "content": "__SESSION_END_CURATED__",
                        "deferred": was_clean_exit,
                    })
                    f.write(marker + "\n")
            except Exception:
                pass

            if was_clean_exit:
                curated_count += 1
            else:
                recovered_count += 1

            if progress_cb:
                try:
                    await progress_cb("done", idx=idx)
                except Exception:
                    pass
        else:
            if progress_cb:
                try:
                    await progress_cb("failed", idx=idx)
                except Exception:
                    pass

    skipped_count = max(0, total_pending - max_sessions)

    if progress_cb:
        try:
            await progress_cb("complete")
        except Exception:
            pass

    # Build user-facing summary
    parts = []
    if curated_count:
        parts.append(f"Curated {curated_count} session(s)")
    if recovered_count:
        parts.append(f"Recovered {recovered_count} crashed session(s)")
    if skipped_count:
        parts.append(f"{skipped_count} more pending (will process next startup)")
    if messages:
        parts.extend(messages)

    summary = ". ".join(parts)
    if summary:
        print(f"[DEFERRED_CURATE] {summary}")
    return summary


def get_state_for_prompt(work_dir: str, project_name: str) -> str:
    """Return project state (status.md + knowledge.md + registry) for agent context.

    The 3-file model (status.md, knowledge.md, history.md) provides all
    cross-session memory. Registry provides software discovery context.
    """
    parts = []

    if project_name:
        project_dir = get_project_dir(project_name)
        status_content = load_project_state(work_dir, project_name)
        if status_content:
            parts.append(
                f"═══ PROJECT STATUS: {project_name} ═══\n"
                f"{status_content}\n"
                f"═══ END PROJECT STATUS ═══"
            )

        knowledge_path = project_dir / "knowledge.md"
        if knowledge_path.exists():
            knowledge_content = knowledge_path.read_text(encoding="utf-8")
            if knowledge_content.strip():
                parts.append(
                    f"═══ PROJECT KNOWLEDGE: {project_name} ═══\n"
                    f"{knowledge_content}\n"
                    f"═══ END PROJECT KNOWLEDGE ═══"
                )

    # Include software registry summary
    if work_dir:
        registry_summary = _get_registry_summary_for_prompt(work_dir)
        if registry_summary:
            parts.append(
                f"═══ SOFTWARE REGISTRY (installed tools) ═══\n"
                f"{registry_summary}\n"
                f"═══ END SOFTWARE REGISTRY ═══"
            )

    return "\n\n".join(parts) if parts else ""


def _get_registry_summary_for_prompt(work_dir: str, max_entries: int = 15, max_chars: int = 2000) -> str:
    """Load registry.yaml and return a compact summary for prompt injection."""
    from pathlib import Path
    registry_path = Path(work_dir) / "software" / "registry.yaml"
    if not registry_path.exists():
        return ""
    try:
        import yaml
        with open(registry_path, "r") as f:
            data = yaml.safe_load(f)
        packages = data.get("packages", []) if data else []
        if not packages:
            return ""
        packages.sort(key=lambda p: p.get("registered", ""), reverse=True)
        lines = [f"Registered software ({len(packages)} total):"]
        for pkg in packages[:max_entries]:
            name = pkg.get("name", "?")
            version = pkg.get("version", "?")
            prefix = pkg.get("prefix", "?")
            if prefix and not prefix.startswith("/"):
                prefix = f"{work_dir}/software/{prefix}"
            purpose = pkg.get("purpose", "")
            line = f"  - {name} {version} at {prefix}"
            if purpose:
                line += f" ({purpose})"
            lines.append(line)
        if len(packages) > max_entries:
            lines.append(f"  ... and {len(packages) - max_entries} more (call query_software for full list)")
        summary = "\n".join(lines)
        return summary[:max_chars]
    except Exception:
        return ""


async def save_findings_to_memory(
    findings_path: str,
    project_name: str = "",
) -> Optional[str]:
    """Persist findings to the unified memory system.

    If project_name is set, saves to projects/{name}/findings_*.md
    Otherwise saves to session/findings_*.md

    Haiku decides whether findings are worth persisting long-term.
    """
    if not findings_path:
        return None

    from core.researcher import read_findings_file
    findings_text = read_findings_file(findings_path)
    if not findings_text:
        return None

    from core.sub_agent import _call_sub_agent_llm

    prompt = FINDINGS_CURATION_PROMPT.format(
        project_name=project_name or "(no specific project)",
        findings_text=findings_text,
    )

    result = await _call_sub_agent_llm(prompt)

    if not result or "NO_PERSIST" in result.split("\n")[0]:
        logger.info(f"[MEMORY_STATE] Findings not persisted (Haiku said no): {findings_path}")
        return None

    # Extract the curated content (after the YES_PERSIST line)
    lines = result.split("\n", 1)
    curated_content = lines[1].strip() if len(lines) > 1 else findings_text

    # Decide where to save
    if project_name:
        target_dir = get_project_dir(project_name)
    else:
        target_dir = get_session_dir()

    findings_filename = Path(findings_path).name
    persist_path = target_dir / f"findings_{findings_filename}"
    try:
        persist_path.write_text(curated_content, encoding="utf-8")
        logger.info(f"[MEMORY_STATE] Persisted findings to {persist_path}")
        return str(persist_path)
    except Exception as e:
        logger.error(f"[MEMORY_STATE] Failed to persist findings: {e}")
        return None


# ── Project Registry Utilities ────────────────────────────────────────────────

PROJECTS_INDEX_FILENAME = "projects_index.json"


def _load_projects_index() -> Dict:
    """Load the projects index from the unified memory system.

    The index lives at {memory_root}/projects_index.json and contains
    metadata about all known projects (name, description, dates).
    This is the SINGLE source of truth for project discovery.
    """
    index_path = get_memory_root() / PROJECTS_INDEX_FILENAME
    try:
        with open(index_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"projects": []}


def _save_projects_index(data: Dict) -> None:
    """Write the projects index to the unified memory system."""
    index_path = get_memory_root() / PROJECTS_INDEX_FILENAME
    try:
        with open(index_path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        logger.error(f"[MEMORY_STATE] Failed to write projects index: {e}")


def _fuzzy_match_project(name: str, existing_names: List[str], threshold: int = 2) -> Optional[str]:
    """Find an existing project name that's within `threshold` edit distance of `name`.

    Returns the existing name if a close match is found, None otherwise.
    Uses Levenshtein distance (insertions, deletions, substitutions).
    """
    name_lower = name.lower().strip()
    for existing in existing_names:
        existing_lower = existing.lower().strip()
        if name_lower == existing_lower:
            return existing
        # Simple Levenshtein distance
        if abs(len(name_lower) - len(existing_lower)) > threshold:
            continue
        d = _levenshtein(name_lower, existing_lower)
        if d <= threshold and d > 0:
            return existing
    return None


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost,
            ))
        prev_row = curr_row
    return prev_row[-1]


def register_project(project_name: str, description: str = "", work_dir: str = "") -> None:
    """Register a new project in the unified memory index.

    Called when Haiku assigns a project_context that doesn't exist yet.
    Idempotent — if the project already exists, only updates date_last_active.
    Uses fuzzy matching to prevent near-duplicate project names (e.g. "seggar" vs "segger").
    If work_dir is provided, also creates the project work directory.
    """
    if not project_name or project_name == "general":
        return
    data = _load_projects_index()
    projects = data.get("projects", [])

    today = datetime.now().strftime("%Y-%m-%d")

    # Exact match — just update last_active
    for p in projects:
        if p.get("name") == project_name:
            p["date_last_active"] = today
            _save_projects_index(data)
            if work_dir:
                wd_dir = Path(work_dir, "projects", project_name)
                wd_dir.mkdir(parents=True, exist_ok=True)
                _ensure_project_files(wd_dir, project_name)
            return

    # Fuzzy match — if a near-duplicate exists, use it instead of creating new
    existing_names = [p.get("name", "") for p in projects]
    match = _fuzzy_match_project(project_name, existing_names)
    if match:
        for p in projects:
            if p.get("name") == match:
                p["date_last_active"] = today
                if not p.get("aliases"):
                    p["aliases"] = []
                if project_name not in p["aliases"] and project_name != match:
                    p["aliases"].append(project_name)
                _save_projects_index(data)
                logger.info(
                    f"[MEMORY_STATE] Fuzzy matched '{project_name}' → existing project '{match}'"
                )
                if work_dir:
                    wd_dir = Path(work_dir, "projects", match)
                    wd_dir.mkdir(parents=True, exist_ok=True)
                    _ensure_project_files(wd_dir, match)
                return

    projects.append({
        "name": project_name,
        "description": description,
        "date_created": today,
        "date_last_active": today,
    })
    data["projects"] = projects
    _save_projects_index(data)
    if work_dir:
        wd_dir = Path(work_dir, "projects", project_name)
        wd_dir.mkdir(parents=True, exist_ok=True)
        _ensure_project_files(wd_dir, project_name)
    logger.info(f"[MEMORY_STATE] Registered new project: '{project_name}'")


def resolve_project_name(name: str) -> str:
    """Resolve a project name to its canonical form.

    Checks aliases and fuzzy matches. Returns the canonical name if found,
    or the input name unchanged if no match exists.
    """
    if not name or name == "general":
        return name
    data = _load_projects_index()
    projects = data.get("projects", [])

    # Check exact match first
    for p in projects:
        if p.get("name") == name:
            return name

    # Check aliases
    for p in projects:
        aliases = p.get("aliases", [])
        if name in aliases:
            canonical = p.get("name", name)
            logger.info(f"[MEMORY_STATE] Resolved alias '{name}' → '{canonical}'")
            return canonical

    # Check fuzzy match
    existing_names = [p.get("name", "") for p in projects]
    match = _fuzzy_match_project(name, existing_names)
    if match:
        logger.info(f"[MEMORY_STATE] Fuzzy resolved '{name}' → '{match}'")
        return match

    return name


def get_known_project_names() -> List[str]:
    """Return the list of known project names from the unified memory system.

    Sources (single source of truth):
    1. projects_index.json in {memory_root}/ (authoritative, has metadata)
    2. {memory_root}/projects/ directory (catches any projects that have
       status.md saved but weren't registered — reconciles automatically)

    Always includes "general" as the default for non-project-specific work.
    Returns max 20 names to keep the Haiku prompt concise.
    """
    # Source 1: Projects index (primary)
    data = _load_projects_index()
    indexed_names: List[str] = []
    for p in data.get("projects", []):
        name = p.get("name", "").strip()
        if name and name != "general":
            indexed_names.append(name)

    # Source 2: Memory filesystem (reconciliation — catches unregistered)
    fs_names: set = set()
    memory_projects = get_memory_root() / "projects"
    if memory_projects.is_dir():
        for entry in memory_projects.iterdir():
            if entry.is_dir() and not entry.name.startswith("."):
                fs_names.add(entry.name)
    fs_names.discard("general")
    for name in indexed_names:
        fs_names.discard(name)

    # Build final list: "general" first, then indexed, then unregistered FS
    result = ["general"] + indexed_names
    remaining_slots = 20 - len(result)
    if remaining_slots > 0 and fs_names:
        result.extend(sorted(fs_names)[:remaining_slots])

    return result


def get_known_projects_with_descriptions() -> str:
    """Return a formatted string of known projects for the skill selection prompt.

    Format: "general (default for non-project queries), vjepa2 (V-JEPA benchmarking), ..."
    Includes descriptions when available for better Haiku disambiguation.
    """
    data = _load_projects_index()
    parts = ["general (default — non-project-specific queries)"]

    for p in data.get("projects", []):
        name = p.get("name", "").strip()
        desc = p.get("description", "").strip()
        if name:
            if desc:
                parts.append(f"{name} ({desc})")
            else:
                parts.append(name)

    # Also pick up unregistered memory-only projects
    memory_projects = get_memory_root() / "projects"
    registered = {p.get("name") for p in data.get("projects", [])}
    if memory_projects.is_dir():
        for entry in memory_projects.iterdir():
            if entry.is_dir() and not entry.name.startswith(".") and entry.name not in registered:
                parts.append(entry.name)

    # Cap at 20 entries
    return ", ".join(parts[:20])


# ── Summarize Utility ─────────────────────────────────────────────────────────

async def summarize_for_context(
    content: str,
    purpose: str,
    max_chars: int = 4000,
) -> str:
    """Use Haiku to summarize content for a specific purpose.

    This replaces all hard [:N] truncation in the codebase.
    Only call when content genuinely exceeds what can fit in context.

    Args:
        content: The text to summarize.
        purpose: What the summary will be used for.
        max_chars: Target maximum character count.

    Returns:
        Summarized content, or original if short enough.
    """
    if not content or len(content) <= max_chars:
        return content

    from core.sub_agent import _call_sub_agent_llm

    prompt = (
        f"Summarize the following content for this purpose: {purpose}\n"
        f"Keep under {max_chars} characters. Preserve all actionable information, "
        f"paths, commands, decisions, and specific facts. Remove redundancy and "
        f"boilerplate.\n\n"
        f"Content to summarize:\n{content}"
    )

    result = await _call_sub_agent_llm(prompt)
    if not result or result.startswith("[Error:"):
        return content[:max_chars]
    return result
