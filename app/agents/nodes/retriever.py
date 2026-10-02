"""
Retriever node — dense search then cross-encoder rerank.

Retrieve wide (RETRIEVAL_TOP_K, default 20), rerank precisely, keep
RERANK_TOP_N (default 5). The wide first pass is what gives the reranker a
chance to find the right chunk at position 14; the narrow second pass is what
keeps the LLM's context short enough to stay grounded.

On the farmer (symptom) path, reference-list chunks are dropped before the
rerank. "Petersen GC: Foot Diseases in Cattle, Part II Treatment" scores highly
against a treatment question but carries no treatment, and it costs one of only
five context slots. The filter is not applied to research questions: the same
year/number density also marks prevalence tables, which researchers need.

This node is re-entered on a self-correction loop, so it appends to the plan
with a pass number instead of overwriting it.
"""

from __future__ import annotations

import re

import logfire

from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search
from app.services.retrieval.ranking_service import rerank


# Citation-shaped tokens: "et al", years, "13:53-58" page ranges, journal abbreviations, DOIs.
_CITATION = re.compile(
    r"et al\.?|\b(?:19|20)\d{2}\b|\d+\s*:\s*\d+\s*[-–]\s*\d+|\bJ\.\s|\bVet\.|\bDairy Sci|\bdoi\b|\[CrossRef\]",
    re.IGNORECASE,
)
# Per 1000 chars. Calibrated on the corpus: prose sits at 0-3, reference lists at 6-15.
_REFERENCE_DENSITY = 6.0


def _is_reference_list(text: str) -> bool:
    return len(_CITATION.findall(text)) * 1000 / max(len(text), 1) >= _REFERENCE_DENSITY


def retrieve_node(state: AgentState) -> dict:
    query = state.get("search_query") or state.get("original_query", "")
    attempt = state.get("refinements", 0) + 1

    with logfire.span("Retrieval", query=query[:120], attempt=attempt):
        candidates = search(query)

        if not candidates:
            logfire.warning("No candidates returned for query", query=query[:120])
            return {
                "documents": [],
                "context_quality": "empty",
                "status": "No matching passages found in the corpus",
                "plan": state.get("plan", []) + [f"Retrieval pass {attempt}: 0 candidates"],
            }

        if state.get("intent") == "symptom":
            prose = [c for c in candidates if not _is_reference_list(c.content)]
            candidates = prose or candidates

        top = rerank(query, candidates)
        documents = [chunk.to_dict() for chunk in top]

        sources = sorted({d["source"] for d in documents})
        logfire.info(
            "Context assembled",
            kept=len(documents), from_candidates=len(candidates),
            sources=sources, top_score=documents[0]["score"] if documents else None,
        )

        return {
            "documents": documents,
            "status": f"Retrieved {len(documents)} passages from {len(sources)} paper(s)",
            "plan": state.get("plan", []) + [
                f"Retrieval pass {attempt}: {len(candidates)} candidates → reranked to {len(documents)}",
                f"Sources: {', '.join(sources[:4])}{'...' if len(sources) > 4 else ''}",
            ],
        }
