"""
Qdrant access layer — collection lifecycle, indexing and search.

Two decisions here matter more than the rest:

1. Deterministic point IDs. A point's ID is uuid5("<filename>:<chunk_index>"),
   so re-ingesting a file overwrites its own points instead of duplicating them.
   Combined with delete-by-source before upsert, re-running ingestion on an
   edited document leaves no orphaned chunks behind.

2. Dimension guarding. `ensure_collection` compares the live collection's vector
   size against the active embedding backend and refuses to proceed on a
   mismatch. Without this you get a stream of opaque upsert errors, or worse,
   a collection half-filled with vectors from two different models.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_query, get_embedding_dim

# Stable namespace so IDs are reproducible across machines and runs.
_ID_NAMESPACE = uuid.UUID("6f1f9b6e-2a3d-4d8f-9a1c-0d7b5e4c3a21")


class QdrantUnavailable(RuntimeError):
    pass


class DimensionMismatch(RuntimeError):
    pass


@dataclass
class RetrievedChunk:
    content: str
    source: str
    page_label: str
    score: float
    chunk_index: int

    @property
    def citation(self) -> str:
        return f"{self.source} (p. {self.page_label})" if self.page_label != "n/a" else self.source

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "source": self.source,
            "page_label": self.page_label,
            "score": round(self.score, 4),
            "chunk_index": self.chunk_index,
            "citation": self.citation,
        }


_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
            timeout=settings.QDRANT_TIMEOUT,
        )
    return _client


def point_id(source: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"{source}:{chunk_index}"))


# ── collection lifecycle ───────────────────────────────────────────────────────

def ensure_collection(wipe: bool = False) -> int:
    """
    Create the collection if absent, validate its dimension if present.
    Returns the vector dimension in use.
    """
    client = get_client()
    name = settings.QDRANT_COLLECTION

    with logfire.span("Ensure collection", collection=name, wipe=wipe):
        try:
            exists = client.collection_exists(name)
        except Exception as exc:
            raise QdrantUnavailable(
                f"Cannot reach Qdrant at {settings.QDRANT_URL}: {exc}\n"
                "Local setup: `docker compose up -d`. Cloud: check QDRANT_CLUSTER_ENDPOINT and QDRANT_API_KEY."
            ) from exc

        if exists and wipe:
            client.delete_collection(name)
            logfire.warning("Collection deleted (--wipe)", collection=name)
            exists = False

        # Probing the backend can be slow (model download) — do it only when needed.
        dim = get_embedding_dim()

        if exists:
            info = client.get_collection(name)
            existing_dim = _vector_size(info)
            if existing_dim is not None and existing_dim != dim:
                raise DimensionMismatch(
                    f"Collection '{name}' stores {existing_dim}-dim vectors but the active "
                    f"embedding backend produces {dim}-dim. Mixing them would corrupt retrieval.\n"
                    f"Fix: re-ingest with --wipe, or point QDRANT_COLLECTION at a different name."
                )
            logfire.info("Collection ready", collection=name, dim=dim, points=info.points_count)
            return dim

        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            # Research corpora are small; indexing from the first vector keeps
            # search quality consistent instead of waiting for the 20k default.
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1000),
        )
        logfire.info("Collection created", collection=name, dim=dim, distance="cosine")

        # Payload index makes filter-by-document searches and per-source deletes
        # fast rather than full scans.
        try:
            client.create_payload_index(
                collection_name=name,
                field_name="source",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            logfire.warning("Could not create payload index on 'source' ({err})", err=str(exc))

        return dim


def _vector_size(info) -> int | None:
    """Read the configured vector size, tolerating named-vector configurations."""
    try:
        params = info.config.params.vectors
    except Exception:
        return None
    if params is None:
        return None
    size = getattr(params, "size", None)
    if size is not None:
        return size
    if isinstance(params, dict):  # named vectors
        first = next(iter(params.values()), None)
        return getattr(first, "size", None)
    return None


def delete_by_source(source: str) -> None:
    """Remove every chunk belonging to one document — run before re-upserting it."""
    client = get_client()
    try:
        client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
                )
            ),
            wait=True,
        )
        logfire.info("Cleared previous chunks", source=source)
    except Exception as exc:
        logfire.warning("delete_by_source failed for {source}: {err}", source=source, err=str(exc))


_TIMEOUT_MARKERS = ("timed out", "timeout", "deadline", "connection", "reset by peer", "502", "503", "504")


def _is_transient(exc: Exception) -> bool:
    return any(m in str(exc).lower() for m in _TIMEOUT_MARKERS)


def _upsert_window(client: QdrantClient, window: list[models.PointStruct], depth: int = 0) -> int:
    """
    Write one window, retrying transient failures and halving on timeout.

    A free-tier cloud cluster is throttled: the same 24-point write can take 2s
    or 30s depending on what else the shard is doing. Halving on timeout finds a
    size the cluster will actually accept instead of failing the whole document —
    which is what a single 64-point, 800 KB write does at 3072 dimensions.
    """
    for attempt in range(1, 4):
        try:
            client.upsert(collection_name=settings.QDRANT_COLLECTION, points=window, wait=True)
            return len(window)
        except Exception as exc:
            transient = _is_transient(exc)

            if transient and len(window) > 4 and depth < 3:
                mid = len(window) // 2
                logfire.warning(
                    "Qdrant write timed out on {size} points — splitting into {a}+{b}",
                    size=len(window), a=mid, b=len(window) - mid,
                )
                return (
                    _upsert_window(client, window[:mid], depth + 1)
                    + _upsert_window(client, window[mid:], depth + 1)
                )

            if transient and attempt < 3:
                wait = 2.0 * attempt
                logfire.warning(
                    "Qdrant upsert retry {n}/3 in {wait}s ({err})",
                    n=attempt, wait=wait, err=str(exc)[:200],
                )
                time.sleep(wait)
                continue

            logfire.error("Qdrant upsert failed permanently: {err}", err=str(exc)[:300])
            raise

    raise RuntimeError("Qdrant upsert retries exhausted")


def upsert_points(points: list[models.PointStruct]) -> int:
    """Upsert in sub-batches — one large request is far more fragile than several small ones."""
    if not points:
        return 0

    client = get_client()
    written = 0
    batch = max(1, settings.QDRANT_UPSERT_BATCH)

    for start in range(0, len(points), batch):
        written += _upsert_window(client, points[start : start + batch])

    return written


def collection_stats() -> dict:
    try:
        info = get_client().get_collection(settings.QDRANT_COLLECTION)
        return {
            "collection": settings.QDRANT_COLLECTION,
            "points": info.points_count,
            "vectors_dim": _vector_size(info),
            "status": str(info.status),
        }
    except Exception as exc:
        return {"collection": settings.QDRANT_COLLECTION, "error": str(exc)[:200]}


def list_sources(limit: int = 500) -> list[str]:
    """Distinct document names currently indexed — used by /sources and the UI."""
    try:
        client = get_client()
        sources: set[str] = set()
        offset = None
        for _ in range(20):  # hard page cap; the corpus is not unbounded
            records, offset = client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                limit=limit,
                offset=offset,
                with_payload=["source"],
                with_vectors=False,
            )
            for record in records:
                src = (record.payload or {}).get("source")
                if src:
                    sources.add(src)
            if offset is None:
                break
        return sorted(sources)
    except Exception as exc:
        logfire.warning("list_sources failed: {err}", err=str(exc))
        return []


# ── search ─────────────────────────────────────────────────────────────────────

def search(query: str, limit: int | None = None, source_filter: str | None = None) -> list[RetrievedChunk]:
    """Dense vector search. Returns [] on failure so the agent can still respond."""
    limit = limit or settings.RETRIEVAL_TOP_K

    with logfire.span("Vector search", query=query[:120], limit=limit):
        try:
            vector = embed_query(query)
        except Exception as exc:
            logfire.error("Could not embed query: {err}", err=str(exc)[:300])
            return []

        query_filter = None
        if source_filter:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchValue(value=source_filter))]
            )

        try:
            response = get_client().query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                score_threshold=settings.MIN_RELEVANCE_SCORE or None,
            )
        except Exception as exc:
            logfire.error("Qdrant search failed: {err}", err=str(exc)[:300])
            return []

        results = [
            RetrievedChunk(
                content=(point.payload or {}).get("text", ""),
                source=(point.payload or {}).get("source", "unknown"),
                page_label=(point.payload or {}).get("page_label", "n/a"),
                score=float(point.score),
                chunk_index=int((point.payload or {}).get("chunk_index", -1)),
            )
            for point in response.points
        ]

        logfire.info(
            "Retrieved {n} candidates",
            n=len(results),
            top_score=round(results[0].score, 4) if results else None,
        )
        return results
