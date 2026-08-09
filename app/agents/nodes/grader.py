"""
Grader node — the self-correction loop.

A plain retrieve-then-answer pipeline has no idea whether what it retrieved is
any good. When the first query phrasing misses, it answers from irrelevant
passages, and that is precisely when RAG systems hallucinate most confidently.

The grader closes the loop: judge the context, and if it is too weak, rewrite
the query and send it back to the retriever. This cycle is the reason the agent
is a LangGraph state machine rather than a linear chain.

Quota discipline — the LLM is only consulted when the cheap signals are
ambiguous:
  - zero documents            -> weak, no model call
  - strong top rerank score   -> sufficient, no model call
  - anything in between       -> one fast-tier call

Bounded by MAX_REFINEMENTS (default 1) so a bad query can never loop forever.
"""

from __future__ import annotations

import re

import logfire

from app.agents.state import AgentState
from app.config import settings
from app.llm import AllTargetsFailed, router

# Above this cross-encoder score the top passage is almost always on-topic,
# so the grading model call is skipped entirely.
_CONFIDENT_SCORE = 0.5

_GRADE_PROMPT = """You are grading retrieved evidence for a veterinary research assistant.

QUESTION:
"{question}"

RETRIEVED PASSAGES:
{context}

Can these passages support a factual answer to the question? Judge only relevance and coverage, \
not writing quality. Partial coverage counts as YES.

If NO, propose a better search query using the terminology a veterinary research paper would use \
(species, disease name, region, study measure).

Reply in exactly this format and nothing else:
VERDICT: <YES or NO>
QUERY: <improved search query, or NONE if the verdict is YES>"""


_BOOLEAN_NOISE = re.compile(r"\b(AND|OR|NOT)\b|[()\[\]]")

# Small models often answer "QUERY: x" and then keep talking — '... can be
# improved to "y" or "z" to better match terminology'. Taking the whole line
# verbatim turns that commentary into the next search query.
_EXPLANATION = re.compile(
    r"\b(can be improved|could be|would be|better match|to better|instead of|i suggest|"
    r"this (query|search)|note:|explanation:)",
    re.IGNORECASE,
)


def _clean_query(query: str, fallback: str, max_words: int = 16) -> str:
    """Reduce the model's QUERY line to something a vector search can use."""
    text = query.strip().strip('"').strip("'")

    # Cut at the first sign the model started explaining itself.
    match = _EXPLANATION.search(text)
    if match:
        text = text[: match.start()]

    # A stray closing quote marks the end of the intended query.
    for quote in ('"', "'"):
        if text.count(quote) == 1:
            text = text.split(quote)[0]

    text = _BOOLEAN_NOISE.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" .,:;-")

    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])

    # Too short to be a real query means the parse failed; keep what we had.
    return text if len(text.split()) >= 2 else fallback


def _parse(raw: str, fallback: str = "") -> tuple[bool, str]:
    verdict_line = ""
    query = ""
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("VERDICT:"):
            verdict_line = stripped.split(":", 1)[1].strip().upper()
        elif stripped.upper().startswith("QUERY:") and not query:
            # `not query` keeps the first QUERY line; later lines are commentary.
            query = stripped.split(":", 1)[1].strip()

    sufficient = not verdict_line.startswith("NO")
    if query.upper().startswith("NONE"):
        query = ""
    elif query:
        query = _clean_query(query, fallback)

    return sufficient, query


def grade_node(state: AgentState) -> dict:
    documents = state.get("documents", [])
    question = state.get("search_query") or state.get("original_query", "")
    used = state.get("refinements", 0)
    budget_left = used < settings.MAX_REFINEMENTS

    with logfire.span("Grading context", docs=len(documents), refinements_used=used):

        # ── cheap signal 1: nothing came back ─────────────────────────────────
        if not documents:
            if budget_left:
                # Broaden rather than rephrase — an empty result usually means the
                # query was too specific for the corpus.
                broadened = " ".join(question.split()[:8]) or question
                logfire.info("Empty retrieval — broadening query", new_query=broadened)
                return {
                    "context_quality": "weak",
                    "refinements": used + 1,
                    "search_query": broadened,
                    "status": "No results — retrying with a broader query",
                    "plan": state.get("plan", []) + [f"Grader: no passages found → retrying with '{broadened}'"],
                }
            return {
                "context_quality": "empty",
                "status": "No supporting evidence in the corpus",
                "plan": state.get("plan", []) + ["Grader: no passages found, refinement budget spent"],
            }

        top_score = float(documents[0].get("score", 0.0))

        # ── cheap signal 2: clearly good ──────────────────────────────────────
        if top_score >= _CONFIDENT_SCORE:
            logfire.info("Context accepted on score alone", top_score=round(top_score, 4))
            return {
                "context_quality": "sufficient",
                "plan": state.get("plan", []) + [f"Grader: context accepted (top score {top_score:.3f})"],
            }

        # ── no budget left: use what we have ──────────────────────────────────
        if not budget_left:
            return {
                "context_quality": "sufficient",
                "plan": state.get("plan", []) + ["Grader: refinement budget spent, answering from best available context"],
            }

        # ── ambiguous band: ask the cheap model ───────────────────────────────
        context = "\n\n".join(
            f"[{i + 1}] {d['source']} p.{d['page_label']}\n{d['content'][:700]}"
            for i, d in enumerate(documents[:3])
        )

        try:
            response = router.invoke(
                _GRADE_PROMPT.format(question=question, context=context),
                tier="fast",
                temperature=0.0,
                max_tokens=160,
                feature="grader",
            )
            sufficient, better_query = _parse(response.content, fallback=question)
        except AllTargetsFailed as exc:
            logfire.warning("Grader unavailable ({err}) — accepting context", err=str(exc)[:200])
            return {
                "context_quality": "sufficient",
                "plan": state.get("plan", []) + ["Grader: unavailable, accepting retrieved context"],
            }

        if sufficient:
            logfire.info("Context judged sufficient", top_score=round(top_score, 4))
            return {
                "context_quality": "sufficient",
                "plan": state.get("plan", []) + [f"Grader: context judged relevant (top score {top_score:.3f})"],
            }

        new_query = better_query or question
        logfire.info("Context judged weak — refining", old=question[:80], new=new_query[:80])
        return {
            "context_quality": "weak",
            "refinements": used + 1,
            "search_query": new_query,
            "status": f"Refining search: {new_query}",
            "plan": state.get("plan", []) + [f"Grader: context too weak → re-searching as '{new_query}'"],
        }
