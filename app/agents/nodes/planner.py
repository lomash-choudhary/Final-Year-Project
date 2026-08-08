"""
Planner node — intent classification + query rewriting.

Two jobs, one cheap model call:

1. Decide whether this turn needs the corpus at all. "Thanks, that helps" does
   not, and running retrieval on it wastes a vector search and an LLM call.

2. Rewrite the message into a standalone search query. Vector search has no
   memory: "and in buffalo?" embeds to nothing useful. Resolved against the
   history it becomes "prevalence of theileriosis in buffalo in India", which
   retrieves correctly. This single step is responsible for most of the quality
   difference between a one-shot RAG demo and a multi-turn assistant.

Runs on the fast tier — the 70B quota is reserved for answer synthesis.
"""

from __future__ import annotations

import re

import logfire

from app.agents.state import AgentState
from app.llm import AllTargetsFailed, router

_PROMPT = """You are the planning step of a veterinary research assistant. Its knowledge base is a \
corpus of peer-reviewed papers on cattle and buffalo disease (haemoprotozoal diseases, brucellosis, \
lumpy skin disease, foot and eye disorders, genetic disorders, E. coli, dairy herd health).

CONVERSATION SO FAR:
{history}

LATEST USER MESSAGE:
"{message}"

Decide which applies:

- CONVERSATIONAL — the message is small talk, a thank-you, a meta-question about the conversation, \
or can be answered entirely from the conversation above without consulting any paper.
- RESEARCH — answering needs evidence from the papers.

If RESEARCH, also write a self-contained search query: resolve every pronoun and ellipsis against \
the conversation, keep the technical terms the papers would actually use, and drop conversational \
filler. Do not invent details that were never mentioned.

Reply in exactly this format and nothing else:
INTENT: <CONVERSATIONAL or RESEARCH>
QUERY: <the search query, or NONE for CONVERSATIONAL>"""


def _format_history(messages: list[dict], limit: int = 6) -> str:
    """Last few turns, excluding the message being planned for."""
    prior = messages[:-1][-limit:]
    if not prior:
        return "(this is the first message)"
    return "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {str(m.get('content', ''))[:600]}"
        for m in prior
    )


def _parse(raw: str, fallback_query: str) -> tuple[str, str]:
    """Pull INTENT/QUERY out of the model output, tolerating format drift."""
    intent_match = re.search(r"INTENT:\s*(CONVERSATIONAL|RESEARCH)", raw, re.IGNORECASE)
    query_match = re.search(r"QUERY:\s*(.+)", raw, re.IGNORECASE)

    intent = (intent_match.group(1).lower() if intent_match else "")
    query = query_match.group(1).strip().strip('"') if query_match else ""

    if not intent:
        # No parseable intent: default to research. A needless retrieval is a
        # far cheaper mistake than answering a factual question with no evidence.
        intent = "research"

    if intent == "research" and (not query or query.upper() == "NONE"):
        query = fallback_query

    return ("conversational" if intent == "conversational" else "research"), query


def planner_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    user_message = str(messages[-1]["content"]) if messages else state.get("original_query", "")
    history = _format_history(messages)

    with logfire.span("Planner", query=user_message[:120]):
        try:
            response = router.invoke(
                _PROMPT.format(history=history, message=user_message),
                tier="fast",
                temperature=0.0,
                max_tokens=180,
                feature="planner",
            )
            intent, search_query = _parse(response.content, user_message)
            logfire.info("Intent classified", intent=intent, search_query=search_query[:120])

        except AllTargetsFailed as exc:
            # Degrade rather than fail: treat it as a research question and search
            # with the raw message. Retrieval still works without the planner.
            logfire.warning("Planner unavailable ({err}) — using the raw query", err=str(exc)[:200])
            intent, search_query = "research", user_message

        if intent == "conversational":
            return {
                "intent": "conversational",
                "search_query": "",
                "documents": [],
                "status": "Answering from conversation memory",
                "plan": ["Intent: conversational — no retrieval needed"],
            }

        return {
            "intent": "research",
            "search_query": search_query,
            "status": f"Searching the literature for: {search_query}",
            "plan": [f"Intent: research", f"Search query: {search_query}"],
        }
