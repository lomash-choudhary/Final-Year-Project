"""
Cross-encoder reranking with FlashRank (local ONNX, zero API cost).

Why a second ranking stage exists
---------------------------------
Cosine similarity over independently-embedded text is a bi-encoder: query and
document never see each other, so scoring is approximate. It is fast enough to
scan the whole collection, which is exactly what you want for a first pass.

A cross-encoder reads query and document *together* and is much more accurate,
but far too slow to run over an entire index. FlashRank makes it practical by
running a quantised ONNX model on CPU.

So: retrieve 20 cheaply, rerank those 20 accurately, keep the best 5. The LLM
then gets a short, dense context instead of a long, noisy one — which is the
difference between a grounded answer and a hallucinated one.
"""

from __future__ import annotations

import threading
import time

import logfire

from app.config import settings
from app.services.retrieval.qdrant_service import RetrievedChunk

_ranker = None
_lock = threading.Lock()
_unavailable = False  # set once if FlashRank cannot load, so we stop retrying


def _get_ranker():
    """Lazy singleton — model load is slow and must not happen at import time."""
    global _ranker, _unavailable
    if _ranker is not None or _unavailable:
        return _ranker

    with _lock:
        if _ranker is not None or _unavailable:
            return _ranker
        try:
            from flashrank import Ranker
            logfire.info("Loading FlashRank cross-encoder (first run downloads the ONNX model)")
            try:
                _ranker = Ranker(cache_dir="/tmp/flashrank")
            except Exception:
                _ranker = Ranker()  # default cache dir
            logfire.info("FlashRank ready")
        except Exception as exc:
            _unavailable = True
            logfire.warning(
                "FlashRank unavailable ({err}) — falling back to raw vector order",
                err=str(exc)[:200],
            )
    return _ranker


def rerank(query: str, chunks: list[RetrievedChunk], top_n: int | None = None) -> list[RetrievedChunk]:
    """
    Re-score chunks against the query and return the best `top_n`.
    Falls back to the incoming (vector-similarity) order if the reranker is down —
    a slightly worse ordering is always better than a failed request.
    """
    top_n = top_n or settings.RERANK_TOP_N
    if not chunks:
        return []

    ranker = _get_ranker()
    if ranker is None:
        return chunks[:top_n]

    started = time.time()
    try:
        from flashrank import RerankRequest

        passages = [{"id": i, "text": c.content} for i, c in enumerate(chunks) if c.content.strip()]
        if not passages:
            return chunks[:top_n]

        results = ranker.rerank(RerankRequest(query=query, passages=passages))

        ordered: list[RetrievedChunk] = []
        for result in results[:top_n]:
            original = chunks[int(result["id"])]
            # Overwrite the cosine score with the cross-encoder score: it is the
            # number that actually determined the final ordering, so it is the
            # honest one to show in the UI.
            ordered.append(
                RetrievedChunk(
                    content=original.content,
                    source=original.source,
                    page_label=original.page_label,
                    score=float(result.get("score", original.score)),
                    chunk_index=original.chunk_index,
                )
            )

        logfire.info(
            "Reranked {n} -> {k} in {ms}ms",
            n=len(passages), k=len(ordered),
            ms=int((time.time() - started) * 1000),
            top_score=round(ordered[0].score, 4) if ordered else None,
        )
        return ordered

    except Exception as exc:
        logfire.warning("Reranking failed ({err}) — keeping vector order", err=str(exc)[:200])
        return chunks[:top_n]
