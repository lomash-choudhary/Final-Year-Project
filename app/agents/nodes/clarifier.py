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
2. **It only asks when the answer would actually change.** The default is to
   answer. A specific problem ("limping, wound between the hooves") is answered
   directly; routine checklist questions (duration, appetite, pregnancy) are
   never asked just to be thorough — an early prompt that listed them as
   examples made the model ask all four on every turn.
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

_PROMPT = """You are a veterinary assistant deciding whether to answer a livestock owner's problem \
now, or ask a follow-up question first.

CONVERSATION SO FAR:
{history}

FARMER'S LATEST MESSAGE:
"{message}"

The default is NO — answer directly. Most messages already contain enough to give useful advice, \
and every question you ask delays help for the animal.

Answer YES only if BOTH are true:
1. You genuinely cannot tell from the message whether this is a home-care problem or a vet problem, \
or which of two clearly different treatments applies.
2. One or two specific answers would settle that.

Always answer NO when:
- the problem is specific enough to advise on (e.g. "limping with a wound between the hooves", \
"swollen udder with clots in the milk", "calf with diarrhoea", "round bald patches on the skin")
- the farmer asks what medicine or treatment to use for a sign they named
- the signs are already serious enough to need a vet regardless of the answers
- the question is general knowledge rather than about one sick animal
- the farmer has already given the key details

Never ask routine checklist questions (duration, appetite, pregnancy, dung) just to be thorough. \
Ask only the question whose answer would change your advice. At most {max_questions}, fewer is \
better. Short, plain, answerable by someone standing next to the animal.

Reply in exactly this format and nothing else:
NEED_MORE: <YES or NO>
QUESTIONS:
- <question, only if YES>"""


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
    if len(questions) == 1:
        return f"Quick question before I advise: {questions[0]}"
    lines = ["Quick questions before I advise:", ""]
    lines += [f"{i}. {q}" for i, q in enumerate(questions, start=1)]
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
