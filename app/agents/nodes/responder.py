"""
Responder node — grounded answer synthesis.

Three behaviours, deliberately separated:

  conversational  answer from history only, no corpus, no citations
  no evidence     say so deterministically, with zero LLM calls. An LLM asked to
                  "admit you don't know" will sometimes answer anyway from
                  parametric memory, which is exactly the failure a grounded
                  system exists to prevent — and it costs quota to get wrong.
  grounded        numbered context + a prompt that requires inline [n] citations

The context budget (MAX_CONTEXT_CHARS) is enforced by dropping whole passages
from the end rather than truncating mid-passage: half a passage is a half-truth
the model will happily complete.
"""

from __future__ import annotations

import logfire

from app.agents.state import AgentState
from app.config import settings
from app.llm import AllTargetsFailed, router

_RESEARCH_PROMPT = """You are a veterinary research assistant. You answer strictly from the \
peer-reviewed passages provided below — never from prior knowledge.

SOURCE PASSAGES:
{context}

CONVERSATION SO FAR:
{history}

QUESTION:
"{question}"

Rules:
1. Use only the passages above. If they do not contain the answer, say so plainly and state what \
the corpus does cover on this topic.
2. Cite every factual claim inline with the passage number, like [1] or [2][4].
3. Quote figures (percentages, sample sizes, years, regions) exactly as written. Never round or \
extrapolate a number that is not in the text.
4. When passages disagree, present both findings and attribute each one.
5. Be direct and specific. Lead with the answer, then the supporting detail.
6. Write in clear prose or short bullets. No preamble about being an AI."""

_CONVERSATIONAL_PROMPT = """You are a veterinary research assistant for cattle and buffalo disease \
literature. Answer the user's latest message using the conversation below.

CONVERSATION SO FAR:
{history}

LATEST MESSAGE:
"{message}"

Keep it brief and natural. If answering properly would need evidence from the papers, say you will \
look it up rather than guessing at findings."""

_NO_EVIDENCE = (
    "I could not find anything in the indexed papers that answers this.\n\n"
    "The corpus covers cattle and buffalo disease research — haemoprotozoal infections "
    "(theileriosis, babesiosis, anaplasmosis), brucellosis, lumpy skin disease, foot and eye "
    "disorders, genetic disorders, *E. coli*, and dairy-herd health management.\n\n"
    "Rephrasing with the terminology a paper would use, or naming the disease and region "
    "explicitly, usually helps."
)


def _format_history(messages: list[dict], limit: int = 6) -> str:
    prior = messages[:-1][-limit:]
    if not prior:
        return "(no earlier turns)"
    return "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {str(m.get('content', ''))[:800]}"
        for m in prior
    )


def _build_context(documents: list[dict], budget: int) -> tuple[str, int]:
    """Numbered passages that fit the budget. Returns (text, passages_used)."""
    blocks: list[str] = []
    used = 0

    for i, doc in enumerate(documents, start=1):
        block = (
            f"[{i}] Source: {doc['source']} — page {doc['page_label']} "
            f"(relevance {doc['score']:.3f})\n{doc['content']}"
        )
        if used + len(block) > budget and blocks:
            logfire.info(
                "Context budget reached — dropped {n} lower-ranked passage(s)",
                n=len(documents) - len(blocks), budget=budget,
            )
            break
        blocks.append(block)
        used += len(block)

    return "\n\n---\n\n".join(blocks), len(blocks)


def generate_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    # Always the English form — the translator node ran before the planner.
    user_message = state.get("query_en") or (
        str(messages[-1]["content"]) if messages else state.get("original_query", "")
    )
    history = _format_history(messages)
    documents = state.get("documents", [])
    intent = state.get("intent", "research")

    # ── no evidence: answer deterministically, spend nothing ──────────────────
    if intent == "research" and not documents:
        logfire.info("Answering with the no-evidence response (0 LLM calls)")
        return {
            "final_answer": _NO_EVIDENCE,
            "status": "No supporting evidence found",
            "plan": state.get("plan", []) + ["Responder: no evidence in corpus — answered without an LLM call"],
            "messages": [{"role": "assistant", "content": _NO_EVIDENCE}],
            "llm_meta": {"target": "none", "cached": False, "fallback_used": False},
        }

    if intent == "conversational":
        prompt = _CONVERSATIONAL_PROMPT.format(history=history, message=user_message)
        passages_used = 0
    else:
        context, passages_used = _build_context(documents, settings.MAX_CONTEXT_CHARS)
        prompt = _RESEARCH_PROMPT.format(
            context=context, history=history, question=user_message
        )

    with logfire.span("Answer synthesis", intent=intent, passages=passages_used):
        try:
            response = router.invoke(
                prompt,
                tier="quality",
                temperature=0.1,
                feature="responder",
            )
        except AllTargetsFailed as exc:
            logfire.error("Answer synthesis failed: {err}", err=str(exc)[:400])
            message = (
                "Every configured language model is currently unavailable or rate-limited. "
                "Check your Groq and Gemini keys, or wait for the quota window to reset."
            )
            return {
                "final_answer": message,
                "status": "LLM unavailable",
                "plan": state.get("plan", []) + ["Responder: all LLM targets exhausted"],
                "messages": [{"role": "assistant", "content": message}],
                "llm_meta": {"target": "none", "error": str(exc)[:300]},
            }

        plan: list[str] = []
        if response.cached:
            plan.append("Responder: gateway cache hit — no tokens spent")
        elif response.fallback_used:
            plan.append(f"Responder: answered by fallback target ({response.target_label})")
        else:
            plan.append(f"Responder: answered by {response.target_label} ({response.model})")

        return {
            "final_answer": response.content,
            "status": "Answer generated",
            "plan": state.get("plan", []) + plan,
            "messages": [{"role": "assistant", "content": response.content}],
            "llm_meta": {
                "target": response.target_label,
                "model": response.model,
                "provider": response.provider,
                "cached": response.cached,
                "fallback_used": response.fallback_used,
                "latency_ms": response.latency_ms,
                "passages_used": passages_used,
            },
        }
