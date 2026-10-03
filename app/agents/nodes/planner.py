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

from app.agents.history import format_history, has_history
from app.agents.state import AgentState
from app.llm import AllTargetsFailed, router

_PROMPT = """You are the planning step of a cattle and buffalo health assistant. Its knowledge base \
is a corpus of peer-reviewed papers on bovine disease (haemoprotozoal diseases, brucellosis, lumpy \
skin disease, foot and eye disorders, genetic disorders, E. coli, dairy herd health, mastitis, \
lameness, respiratory disease, antimicrobial treatment).

CONVERSATION SO FAR:
{history}

LATEST USER MESSAGE:
"{message}"

Classify the message as exactly one of:

- CONVERSATIONAL — small talk, a thank-you, or a question answerable entirely from the conversation \
above without consulting any paper. Never a new question about an animal's health.
- SYMPTOM — a practical question from someone who keeps animals: a problem with their own animal, \
or what to give, apply, feed, clean or vaccinate with, or how to prevent a disease. Examples: \
"my cow has stopped eating", "what footbath should I use", "how do I clean my shed", "what should \
I give for worms". This includes short replies that answer earlier follow-up questions.
- RESEARCH — a question about the studies themselves: prevalence, incidence, seasonality, study \
findings, methods, comparisons. Examples: "what is the prevalence of theileriosis in India", \
"which season has the highest incidence", "what did the study find".

Then write a self-contained search query for the knowledge base:
- Resolve every pronoun and ellipsis against the conversation.
- Use plain descriptive terms a veterinary paper would use.
- Write it as natural language, NOT as a boolean expression. Do not use AND, OR, quotes or brackets \
— the search is semantic, so operators only add noise.
- For SYMPTOM, name the signs in veterinary terms AND keep what the farmer is asking for \
(what to apply, what to give, how to prevent). Example: "my cow's udder is swollen and the milk has \
clots, what medicine?" becomes "swollen udder clots in milk clinical mastitis dairy cow treatment".
- For SYMPTOM, also give the single most likely disease name, or NONE if unclear.
- Keep the farmer's body-part words; do not move them (a cow's left side is the belly/rumen side, \
not the chest).
- Do not invent details that were never mentioned.

Reply in exactly this format and nothing else:
INTENT: <CONVERSATIONAL or SYMPTOM or RESEARCH>
QUERY: <the search query, or NONE for CONVERSATIONAL>
DISEASE: <disease name, or NONE>"""


# Boolean operators are meaningless to a vector search and dilute the embedding,
# but models reach for PubMed syntax the moment they see a research corpus.
_BOOLEAN_NOISE = re.compile(r"\b(AND|OR|NOT)\b|[()\[\]\"]")


def _clean_query(query: str) -> str:
    cleaned = _BOOLEAN_NOISE.sub(" ", query)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" -:") or query


def _parse_disease(raw: str) -> str:
    """The planner's disease guess, or "" — it only ever widens retrieval, so a miss is harmless."""
    match = re.search(r"DISEASE:\s*(.+)", raw, re.IGNORECASE)
    disease = _clean_query(match.group(1).strip().strip('"')) if match else ""
    return "" if disease.upper().startswith(("NONE", "UNKNOWN", "UNCLEAR")) else disease[:80]


def _parse(raw: str, fallback_query: str, has_history: bool = True) -> tuple[str, str]:
    """Pull INTENT/QUERY out of the model output, tolerating format drift."""
    intent_match = re.search(r"INTENT:\s*(CONVERSATIONAL|SYMPTOM|RESEARCH)", raw, re.IGNORECASE)
    query_match = re.search(r"QUERY:\s*(.+)", raw, re.IGNORECASE)

    intent = (intent_match.group(1).lower() if intent_match else "")
    query = query_match.group(1).strip().strip('"') if query_match else ""

    if intent == "conversational" and not has_history:
        # Nothing to answer from: a first message labelled conversational was a
        # health question the small model misread ("bachhde ko khansi hai" came
        # back as a refusal). Small talk never gets here — the guardrails answer it.
        intent = "symptom"

    if intent not in ("conversational", "symptom", "research"):
        # No parseable intent: default to research. A needless retrieval is a
        # far cheaper mistake than answering a factual question with no evidence.
        intent = "research"

    if intent != "conversational":
        if not query or query.upper() == "NONE":
            query = fallback_query
        query = _clean_query(query)

    return intent, query


def planner_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    # The translator ran first, so this is English regardless of what was typed.
    user_message = state.get("query_en") or (
        str(messages[-1]["content"]) if messages else state.get("original_query", "")
    )
    history = format_history(state, empty="(this is the first message)")

    with logfire.span("Planner", query=user_message[:120]):
        try:
            response = router.invoke(
                _PROMPT.format(history=history, message=user_message),
                tier="fast",
                temperature=0.0,
                max_tokens=220,
                feature="planner",
            )
            intent, search_query = _parse(response.content, user_message, has_history(state))
            disease = _parse_disease(response.content) if intent == "symptom" else ""
            logfire.info("Intent classified", intent=intent, search_query=search_query[:120])

        except AllTargetsFailed as exc:
            # Degrade rather than fail: treat it as a research question and search
            # with the raw message. Retrieval still works without the planner.
            logfire.warning("Planner unavailable ({err}) — using the raw query", err=str(exc)[:200])
            intent, search_query, disease = "research", user_message, ""

        base_plan = state.get("plan", [])

        if intent == "conversational":
            return {
                "intent": "conversational",
                "search_query": "",
                "likely_disease": "",
                "documents": [],
                "status": "Answering from conversation memory",
                "plan": base_plan + ["Intent: conversational — no retrieval needed"],
            }

        plan = [f"Intent: {intent}", f"Search query: {search_query}"]
        if disease:
            plan.append(f"Likely disease: {disease}")
        return {
            "intent": intent,
            "search_query": search_query,
            "likely_disease": disease,
            "status": f"Looking this up: {search_query}",
            "plan": base_plan + plan,
        }
