"""Fresh Start button handler logic.

Handles the /fresh command — curates project memory then clears conversation
context for a fresh start. Session logs already record the full conversation
(crash-safe JSONL), so no separate conversation_history save is needed.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def handle_fresh_start(
    history: List[Any],
    username: str,
    session_id: str,
    work_dir: Optional[str] = None,
    project_name: Optional[str] = None,
    session_facts: str = "",
    plan_summary: str = "",
    protocol_recorder=None,
    protocol_player=None,
) -> Dict[str, Any]:
    """Process a Fresh Start button click.

    Curates project memory (like a mini session-end) before clearing,
    so that learnings from the conversation are preserved in the 3-file
    memory system.

    Returns a result dict describing what happened. The caller (app.py)
    uses this to send appropriate toasts and clear session state.
    """
    if not history:
        return {
            "action": "skip",
            "message_count": 0,
            "cleared_project": "",
            "toast_message": "No conversation to save — already fresh!",
            "toast_type": "info",
            "should_clear_history": False,
        }

    _protocol_note = ""
    if protocol_recorder and getattr(protocol_recorder, "is_active", False):
        try:
            protocol_recorder.save_draft(reason="fresh_start")
            _protocol_note = " Protocol draft saved."
        except Exception:
            _protocol_note = " (Protocol draft save failed.)"

    if protocol_player and hasattr(protocol_player, "abort"):
        try:
            protocol_player.abort()
            protocol_player.save_report()
            _protocol_note += " Play mode aborted."
        except Exception:
            pass

    # Curate memory before clearing (acts as mini session-end)
    _memory_note = ""
    recent_context = "\n".join(
        f"[{getattr(m, 'type', 'unknown')}]: {getattr(m, 'content', '')[:300]}"
        for m in history[-6:]
    )

    if project_name and project_name != "general":
        try:
            from core.memory_state import save_project_state

            saved = await save_project_state(
                work_dir=work_dir or "",
                project_name=project_name,
                session_facts=session_facts,
                plan_summary=plan_summary,
                recent_context=recent_context,
            )
            if saved:
                _memory_note = f" Memory for '{project_name}' updated."
            else:
                _memory_note = f" Memory update for '{project_name}' skipped."
        except Exception as e:
            logger.warning(f"[FRESH_START] Project memory curation failed: {e}")
            _memory_note = " (Memory update failed.)"
    else:
        # No project — extract cross-project knowledge to global
        try:
            from core.memory_state import curate_global_knowledge

            saved = await curate_global_knowledge(
                username=username,
                session_facts=session_facts,
                recent_context=recent_context,
            )
            if saved:
                _memory_note = " Global knowledge updated."
        except Exception as e:
            logger.warning(f"[FRESH_START] Global knowledge curation failed: {e}")
            _memory_note = " (Global knowledge update failed.)"

    msg_count = len(history)
    return {
        "action": "saved",
        "message_count": msg_count,
        "cleared_project": project_name or "",
        "toast_message": f"✅ {msg_count} messages archived — starting fresh!{_memory_note}{_protocol_note}",
        "toast_type": "success",
        "should_clear_history": True,
    }
