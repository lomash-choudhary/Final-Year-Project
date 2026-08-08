"""
Retriever node — dense search then cross-encoder rerank.

Retrieve wide (RETRIEVAL_TOP_K, default 20), rerank precisely, keep
RERANK_TOP_N (default 5). The wide first pass is what gives the reranker a
chance to find the right chunk at position 14; the narrow second pass is what
keeps the LLM's context short enough to stay grounded.

This node is re-entered on a self-correction loop, so it appends to the plan
with a pass number instead of overwriting it.
"""

from __future__ import annotations

import logfire

from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search
from app.services.retrieval.ranking_service import rerank


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
