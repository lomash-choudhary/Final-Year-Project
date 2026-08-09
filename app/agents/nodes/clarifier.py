"""
Clarifier node — ask before answering, but only when it changes the answer.

A farmer types "my cow has stopped eating". That single sentence is consistent
with a dozen conditions ranging from mild indigestion to something that needs a
vet within hours. Answering it directly means either a vague list of
possibilities (useless) or a confident guess (dangerous).

A real vet asks first: how long, any fever, drinking water, ruminating,
pregnant, dung normal. Two or three answers usually narrow it enormously.

So this node decides whether to answer or to ask.

Three rules keep it from becoming annoying:

1. **It never asks twice in a row.** `awaiting_clarification` is carried across
   turns by the checkpointer, so when the user replies to the questions, the
   node knows this message is an answer and moves straight to advice.
2. **It only asks when the answer would actually change.** A question already
   containing duration, temperature and appetite does not need follow-ups.
3. **It is bounded** by MAX_CLARIFICATION_ROUNDS.

Asking is also the cheapest possible turn: no retrieval, no big model, no
translation of retrieved passages. One small model call and we are done.
"""

from __future__ import annotations

import re

import logfire

from app.agents.state import AgentState
from app.config import settings
from app.llm import AllTargetsFailed, router

_PROMPT = """You are a veterinary assistant triaging a livestock owner's problem before giving advice.

CONVERSATION SO FAR:
{history}

FARMER'S LATEST MESSAGE:
"{message}"

Decide whether you need more information before you can give safe, specific advice.

Ask for more ONLY if the missing details would genuinely change your advice — for example how long \
the problem has lasted, whether there is fever, whether the animal is eating, drinking or \
ruminating, whether it is pregnant or recently calved, or what the dung and milk look like.

Do NOT ask if:
- the farmer has already given those details
- the question is general knowledge rather than a sick animal
- the signs described are already severe enough to need a vet regardless of the answers

Ask at most {max_questions} questions. Keep each one short, plain, and answerable by someone \
standing next to the animal. No medical jargon.

Reply in exactly this format and nothing else:
NEED_MORE: <YES or NO>
QUESTIONS:
- <question 1>
- <question 2>"""


def _format_history(messages: list[dict], limit: int = 6) -> str:
    prior = messages[:-1][-limit:]
    if not prior:
        return "(this is the first message)"
    return "\n".join(
        f"{'Farmer' if m.get('role') == 'user' else 'Assistant'}: {str(m.get('content', ''))[:500]}"
        for m in prior
    )


def _parse(raw: str, limit: int) -> tuple[bool, list[str]]:
    need_more = False
    questions: list[str] = []

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("NEED_MORE:"):
            need_more = "YES" in stripped.upper()
        elif stripped.startswith(("-", "•", "*")) or re.match(r"^\d+[.)]", stripped):
            question = re.sub(r"^([-•*]|\d+[.)])\s*", "", stripped).strip()
            if question and question.upper() != "NONE":
                questions.append(question)

    questions = questions[:limit]
    # "Yes I need more" with no questions attached is a malformed answer, not a
    # reason to stall the conversation.
    if need_more and not questions:
        need_more = False

    return need_more, questions


def _compose_message(questions: list[str]) -> str:
    lines = [
        "I can help with that. A few quick questions so I can give you the right advice:",
        "",
    ]
    lines += [f"{i}. {q}" for i, q in enumerate(questions, start=1)]
    lines += ["", "You can answer in one message — even short answers help."]
    return "\n".join(lines)


def clarify_node(state: AgentState) -> dict:
    query = state.get("query_en") or state.get("original_query", "")
    messages = state.get("messages", [])
    rounds = state.get("clarification_rounds", 0)

    # Rule 1: the user is answering our previous questions — do not ask again.
    if state.get("awaiting_clarification"):
        logfire.info("User is answering previous follow-ups — proceeding to advice")
        return {
            "awaiting_clarification": False,
            "follow_up_questions": [],
            "plan": state.get("plan", []) + ["Clarifier: follow-up answers received → proceeding"],
        }

    # Rule 3: bounded.
    if not settings.ENABLE_CLARIFICATION or rounds >= settings.MAX_CLARIFICATION_ROUNDS:
        return {
            "awaiting_clarification": False,
            "follow_up_questions": [],
            "plan": state.get("plan", []) + ["Clarifier: skipped (disabled or budget spent)"],
        }

    with logfire.span("Clarifier", query=query[:120], rounds=rounds):
        try:
            response = router.invoke(
                _PROMPT.format(
                    history=_format_history(messages),
                    message=query,
                    max_questions=settings.MAX_FOLLOW_UP_QUESTIONS,
                ),
                tier="fast",
                temperature=0.2,
                max_tokens=300,
                feature="clarifier",
            )
            need_more, questions = _parse(response.content, settings.MAX_FOLLOW_UP_QUESTIONS)
        except AllTargetsFailed as exc:
            # Rule 2 fails safe: if we cannot decide, answer rather than stall.
            logfire.warning("Clarifier unavailable ({err}) — answering directly", err=str(exc)[:200])
            return {
                "awaiting_clarification": False,
                "follow_up_questions": [],
                "plan": state.get("plan", []) + ["Clarifier: unavailable, answering directly"],
            }

        if not need_more:
            logfire.info("Enough detail given — no follow-ups needed")
            return {
                "awaiting_clarification": False,
                "follow_up_questions": [],
                "plan": state.get("plan", []) + ["Clarifier: enough detail, no follow-ups needed"],
            }

        logfire.info("Asking follow-up questions", count=len(questions), questions=questions)
        return {
            "awaiting_clarification": True,
            "clarification_rounds": rounds + 1,
            "follow_up_questions": questions,
            "final_answer": _compose_message(questions),
            "care_level": "info",
            "status": "Asking follow-up questions",
            "plan": state.get("plan", []) + [f"Clarifier: asked {len(questions)} follow-up question(s)"],
            "messages": [{"role": "assistant", "content": _compose_message(questions)}],
        }
