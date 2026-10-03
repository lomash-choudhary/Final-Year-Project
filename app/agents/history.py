"""
Conversation history as the nodes put it into a prompt.

One helper instead of four copies (planner, clarifier, responder, advisor) so the
memory policy is set in one place: the rolling summary of older turns first, then
the recent window verbatim, each message capped at MEMORY_MSG_MAX_CHARS. The
caller (`main.py`) has already bounded the window, so prompt size stays flat
however long the conversation runs — see `app/memory/store.py`.
"""

from __future__ import annotations

from app.agents.state import AgentState
from app.config import settings


def format_history(
    state: AgentState,
    *,
    user_label: str = "User",
    empty: str = "(no earlier turns)",
) -> str:
    """Summary + earlier turns, excluding the message currently being handled."""
    messages = state.get("messages", [])
    prior = messages[:-1][-settings.MEMORY_WINDOW_TURNS * 2:]
    cap = settings.MEMORY_MSG_MAX_CHARS

    lines: list[str] = []
    summary = (state.get("memory_summary") or "").strip()
    if summary:
        lines.append(f"Summary of the earlier conversation: {summary}")
    lines += [
        f"{user_label if m.get('role') == 'user' else 'Assistant'}: {str(m.get('content', ''))[:cap]}"
        for m in prior
    ]
    return "\n".join(lines) or empty


def has_history(state: AgentState) -> bool:
    return len(state.get("messages", [])) > 1 or bool(state.get("memory_summary"))
