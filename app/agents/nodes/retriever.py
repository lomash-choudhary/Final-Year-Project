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
from app.config import settings
from app.services.retrieval.qdrant_service import search
from app.services.retrieval.ranking_service import rerank


# Citation-shaped tokens: "et al", years, "13:53-58" page ranges, journal abbreviations, DOIs.
_CITATION = re.compile(
    r"et al\.?|\b(?:19|20)\d{2}\b|\d+\s*:\s*\d+\s*[-–]\s*\d+|\bJ\.\s|\bVet\.|\bDairy Sci|\bdoi\b|\[CrossRef\]",
    re.IGNORECASE,
)
# Per 1000 chars. Calibrated on the corpus: prose sits at 0-3, reference lists at 6-15.
_REFERENCE_DENSITY = 6.0


# Research-stage treatments. Farmer advice names established medicines only, but the mastitis
# review (vetsci-12) and a secretome paper filled four of five slots with phage, propolis and
# secretome passages, so the advisor had nothing usable to name. Dropping them before the rerank
# lets "antibiotics such as penicillin and cephalosporins" back in.
_EXPERIMENTAL = re.compile(
    r"phage|secretome|conditioned medium|propolis|nanoparticle|stem cell|in vitro|baicalin",
    re.IGNORECASE,
)


def _is_reference_list(text: str) -> bool:
    return len(_CITATION.findall(text)) * 1000 / max(len(text), 1) >= _REFERENCE_DENSITY


def retrieve_node(state: AgentState) -> dict:
    query = state.get("search_query") or state.get("original_query", "")
    attempt = state.get("refinements", 0) + 1

    with logfire.span("Retrieval", query=query[:120], attempt=attempt):
        candidates = search(query)
        disease = state.get("likely_disease", "") if attempt == 1 else ""
        if disease and state.get("intent") == "symptom":
            # Farmers describe signs; treatment passages are written under the disease
            # name. "calf cough runny nose" alone found only decongestant passages, while
            # leading the query with a guessed disease turned round bald patches into
            # lumpy skin disease. Searching both, then reranking the union, keeps the
            # sign matches when the guess is wrong.
            seen = {(c.source, c.chunk_index) for c in candidates}
            candidates += [
                c for c in search(f"{disease} treatment") if (c.source, c.chunk_index) not in seen
            ]
            query = f"{query} {disease}"

        if not candidates:
            logfire.warning("No candidates returned for query", query=query[:120])
            return {
                "documents": [],
                "context_quality": "empty",
                "status": "No matching passages found in the corpus",
                "plan": state.get("plan", []) + [f"Retrieval pass {attempt}: 0 candidates"],
            }

        if state.get("intent") == "symptom":
            prose = [
                c for c in candidates
                if not _is_reference_list(c.content) and not _EXPERIMENTAL.search(c.content)
            ]
            candidates = prose or candidates

        if state.get("intent") == "symptom":
            # At most 3 passages per paper. The footbath question filled all 5 slots with one
            # lameness paper (two of them its introduction) and never saw the copper-sulfate /
            # formalin table in animals-14 that answers it.
            ranked = rerank(query, candidates, top_n=len(candidates))
            historical = settings.historical_sources
            if any(c.source not in historical and c.score >= 0.3 for c in ranked):
                # The advisor ignores the old book when modern evidence exists, so its passages
                # must not take slots either: for foot rot it held 3 of 5, the advisor dropped
                # them, and the treatment passage ranked 6th never reached the model.
                ranked = [c for c in ranked if c.source not in historical]
            per_source: dict[str, int] = {}
            top = []
            for chunk in ranked:
                if per_source.get(chunk.source, 0) < 3:
                    top.append(chunk)
                    per_source[chunk.source] = per_source.get(chunk.source, 0) + 1
                if len(top) == settings.RERANK_TOP_N:
                    break
        else:
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
