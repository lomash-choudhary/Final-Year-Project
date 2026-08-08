"""
Embedding service — Gemini primary, local sentence-transformers fallback.

Why Groq is not in this chain
-----------------------------
Groq serves chat/completion models only; it has no embeddings endpoint. The
Groq primary/fallback key pair protects the *reasoning* side of the system
(app/llm/router.py). On the embedding side the free-tier ladder is:

    Gemini (free API quota)  ->  sentence-transformers (local, unlimited, offline)

Dimension safety
----------------
The vector dimension is *probed*, never hardcoded: we embed one short string and
measure the result. Different Google accounts get different models, and models
change dimension between versions — a hardcoded 3072 is how a collection ends up
silently rejecting every upsert.

Because Qdrant fixes the dimension at collection creation, the backend is chosen
once per process and then locked. If Gemini dies mid-run we raise instead of
quietly switching to a 768-dim local model, which would corrupt the index.
"""

from __future__ import annotations

import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

import logfire

from app.config import settings
from app.services.retrieval.embedding_cache import EmbeddingCache


class EmbeddingError(RuntimeError):
    pass


# Substrings that mean "slow down and retry", as opposed to "this will never work".
_RETRYABLE_MARKERS = (
    "429", "rate limit", "ratelimit", "quota", "resource_exhausted",
    "resource exhausted", "too many requests", "503", "502", "504",
    "unavailable", "deadline", "timeout", "internal error", "overloaded",
)

# Substrings that mean the batch itself is the problem — split it, do not retry.
_BATCH_SIZE_MARKERS = (
    "batch size", "too many requests in batch", "request payload size",
    "exceeds the maximum", "invalid_argument", "400",
)


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _RETRYABLE_MARKERS)


def _is_batch_problem(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _BATCH_SIZE_MARKERS)


# Gemini reports how long to wait, in two different shapes depending on which
# layer produced the error. Both are worth honouring: the computed exponential
# backoff tops out around 17s, which never clears a per-minute quota window.
_RETRY_AFTER_PATTERNS = (
    re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", re.IGNORECASE),
    re.compile(r"retry\s+in\s+(\d+(?:\.\d+)?)\s*s", re.IGNORECASE),
    re.compile(r"retry[-_ ]?after['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)", re.IGNORECASE),
)


def _retry_after(exc: Exception) -> float | None:
    """Seconds the provider asked us to wait, if it said so."""
    message = str(exc)
    delays = []
    for pattern in _RETRY_AFTER_PATTERNS:
        match = pattern.search(message)
        if match:
            try:
                delays.append(float(match.group(1)))
            except ValueError:
                pass
    if not delays:
        return None
    # Cap at two minutes — a longer instruction means a daily quota, and sleeping
    # through that is worse than failing with a clear message.
    return min(max(delays), 120.0)


class _RateLimiter:
    """
    Client-side ceiling on **texts embedded per minute**.

    This counts texts, not requests, because that is what Gemini charges:

        Quota exceeded for metric: .../embed_content_free_tier_requests, limit: 100

    is 100 *texts* per minute per project per model. A batch of 16 costs 16
    units. Throttling batches instead would permit ~14x the real ceiling.

    Staying under the limit deliberately is much cheaper than discovering it by
    failing — a rejected request still burns quota, and the mandated cooldown is
    close to a full minute.
    """

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max(1, max_per_minute)
        self._events: list[tuple[float, int]] = []   # (timestamp, units)
        self._blocked_until = 0.0
        self._lock = threading.Lock()

    def _prune(self, now: float) -> int:
        self._events = [(t, n) for t, n in self._events if now - t < 60.0]
        return sum(n for _, n in self._events)

    def acquire(self, units: int = 1) -> None:
        """Block until `units` more texts can be sent inside the sliding window."""
        units = max(1, units)

        while True:
            with self._lock:
                now = time.monotonic()

                # A provider-mandated cooldown outranks our own accounting.
                if now < self._blocked_until:
                    sleep_for = self._blocked_until - now
                else:
                    used = self._prune(now)

                    if used + units <= self.max_per_minute or not self._events:
                        # `not self._events` lets an oversized batch through
                        # rather than deadlocking on a limit it can never satisfy.
                        self._events.append((now, units))
                        return

                    # Wait for the oldest event to age out of the window.
                    sleep_for = 60.0 - (now - self._events[0][0]) + 0.05

            if sleep_for > 0:
                logfire.info(
                    "Embedding rate limiter pausing {secs}s (ceiling {cap} texts/min)",
                    secs=round(sleep_for, 1), cap=self.max_per_minute, requested=units,
                )
                time.sleep(sleep_for)

    def penalize(self, seconds: float) -> None:
        """
        Provider said 429 — hold every caller off for exactly this long.

        The event window is cleared rather than marked full: the provider's
        cooldown *is* the window reset, and keeping stale events would stack a
        second ~60s wait on top of a delay the provider said was enough.
        """
        with self._lock:
            self._blocked_until = max(self._blocked_until, time.monotonic() + seconds)
            self._events = []


@dataclass
class EmbeddingBackend:
    name: str                                        # "gemini" | "local"
    model: str
    dim: int
    embed_documents: Callable[[list[str]], list[list[float]]]
    embed_single: Callable[[str], list[float]]


# ── module state (initialised once, then locked) ──────────────────────────────
_backend: EmbeddingBackend | None = None
_init_lock = threading.Lock()
_limiter = _RateLimiter(settings.EMBED_MAX_RPM)
_cache = EmbeddingCache(settings.resolve_path(settings.EMBEDDING_CACHE_PATH), settings.EMBEDDING_CACHE_ENABLED)


# ── Gemini backend ─────────────────────────────────────────────────────────────

def _build_gemini() -> EmbeddingBackend | None:
    """Probe Gemini embedding models in order; return the first that responds."""
    if not settings.GEMINI_API_KEY:
        logfire.info("No GEMINI_API_KEY — skipping Gemini embedding tier")
        return None

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as exc:
        logfire.warning("langchain-google-genai not installed ({err})", err=str(exc))
        return None

    candidates = (
        [settings.GEMINI_EMBEDDING_MODEL]
        if settings.GEMINI_EMBEDDING_MODEL
        else list(settings.GEMINI_EMBEDDING_CANDIDATES)
    )

    for model_name in candidates:
        try:
            # task_type is a real quality lever: asymmetric embeddings score
            # documents and queries in different spaces. Older versions of the
            # package do not accept the kwarg, hence the TypeError branch.
            try:
                doc_model = GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=settings.GEMINI_API_KEY,
                    task_type="retrieval_document",
                )
                query_model = GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=settings.GEMINI_API_KEY,
                    task_type="retrieval_query",
                )
                asymmetric = True
            except TypeError:
                doc_model = query_model = GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=settings.GEMINI_API_KEY,
                )
                asymmetric = False

            _limiter.acquire()
            probe = query_model.embed_query("bovine theileriosis prevalence")
            dim = len(probe)
            if dim == 0:
                raise EmbeddingError("probe returned an empty vector")

            logfire.info(
                "Gemini embeddings ready",
                model=model_name, dim=dim, asymmetric_task_types=asymmetric,
            )
            return EmbeddingBackend(
                name="gemini",
                model=model_name,
                dim=dim,
                embed_documents=doc_model.embed_documents,
                embed_single=query_model.embed_query,
            )

        except Exception as exc:
            logfire.warning(
                "Gemini model '{model}' unavailable ({err}) — trying next candidate",
                model=model_name, err=str(exc)[:300],
            )

    logfire.warning("No Gemini embedding model responded to the probe")
    return None


# ── local backend ──────────────────────────────────────────────────────────────

def _build_local() -> EmbeddingBackend:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError(
            "No embedding backend available: Gemini is unreachable and "
            "sentence-transformers is not installed. Run `pip install sentence-transformers` "
            "or set a working GEMINI_API_KEY."
        ) from exc

    name = settings.LOCAL_EMBEDDING_MODEL
    logfire.info("Loading local embedding model (first run downloads weights)", model=name)
    model = SentenceTransformer(name)

    def _docs(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

    def _one(text: str) -> list[float]:
        return model.encode([text], show_progress_bar=False, normalize_embeddings=True)[0].tolist()

    dim = len(_one("probe"))
    logfire.info("Local embeddings ready", model=name, dim=dim)
    return EmbeddingBackend(name="local", model=name, dim=dim, embed_documents=_docs, embed_single=_one)


def _init() -> EmbeddingBackend:
    global _backend
    if _backend is not None:
        return _backend

    with _init_lock:
        if _backend is not None:
            return _backend

        provider = settings.EMBEDDING_PROVIDER
        with logfire.span("Embedding backend init", requested=provider):
            if provider == "local":
                _backend = _build_local()
            elif provider == "gemini":
                built = _build_gemini()
                if built is None:
                    raise EmbeddingError(
                        "EMBEDDING_PROVIDER=gemini but no Gemini model could be reached. "
                        "Check GEMINI_API_KEY, or set EMBEDDING_PROVIDER=auto to allow the local fallback."
                    )
                _backend = built
            else:  # auto
                _backend = _build_gemini() or _build_local()

        logfire.info("Embedding backend locked", provider=_backend.name, model=_backend.model, dim=_backend.dim)
        return _backend


# ── retry / batching ───────────────────────────────────────────────────────────

def _embed_batch_with_retry(backend: EmbeddingBackend, batch: list[str], depth: int = 0) -> list[list[float]]:
    """Embed one batch, retrying on transient errors and splitting on batch errors."""
    max_attempts = max(1, settings.EMBED_MAX_RETRIES)

    for attempt in range(1, max_attempts + 1):
        try:
            _limiter.acquire(len(batch))
            return backend.embed_documents(batch)

        except Exception as exc:
            # A batch that is structurally too large will fail identically on
            # every retry. Halve it instead of burning the retry budget.
            if _is_batch_problem(exc) and len(batch) > 1 and depth < 4:
                mid = len(batch) // 2
                logfire.warning(
                    "Batch rejected ({err}) — splitting {size} -> {a}+{b}",
                    err=str(exc)[:160], size=len(batch), a=mid, b=len(batch) - mid,
                )
                return (
                    _embed_batch_with_retry(backend, batch[:mid], depth + 1)
                    + _embed_batch_with_retry(backend, batch[mid:], depth + 1)
                )

            if _is_retryable(exc) and attempt < max_attempts:
                # Prefer the provider's own instruction. A quota window is
                # ~60s wide, while exponential backoff tops out near 17s — so
                # computed backoff alone just retries inside the same blocked
                # minute and burns all five attempts for nothing.
                mandated = _retry_after(exc)
                computed = min(60.0, 2.0 ** attempt) + random.uniform(0, 1.5)
                wait = max(mandated + 1.0, computed) if mandated else computed

                if mandated:
                    _limiter.penalize(mandated)

                logfire.warning(
                    "Embedding call failed ({err}) — retry {n}/{max} in {wait}s{src}",
                    err=str(exc)[:200], n=attempt, max=max_attempts, wait=round(wait, 1),
                    src=" (provider-specified)" if mandated else "",
                )
                time.sleep(wait)
                continue

            logfire.error("Embedding failed permanently: {err}", err=str(exc)[:400])
            raise EmbeddingError(
                f"{backend.name}/{backend.model} embedding failed after {attempt} attempt(s): {exc}"
            ) from exc

    raise EmbeddingError(f"Embedding retries exhausted for {backend.name}/{backend.model}")


# ── public API ─────────────────────────────────────────────────────────────────

def get_backend() -> EmbeddingBackend:
    return _init()


def get_embedding_dim() -> int:
    """Vector dimension of the active backend. Probed, never hardcoded."""
    return _init().dim


def describe() -> dict:
    """Backend + cache snapshot for /health and ingestion reports."""
    try:
        backend = _init()
        info = {"provider": backend.name, "model": backend.model, "dim": backend.dim}
    except Exception as exc:
        info = {"provider": "unavailable", "error": str(exc)[:200]}
    info["cache"] = _cache.stats()
    return info


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed many documents. Cache-aware: only uncached texts hit the API.
    Returns vectors in the same order as `texts`.
    """
    if not texts:
        return []

    backend = _init()

    cached = _cache.get_many(texts, backend.name, backend.model, backend.dim)
    if cached:
        logfire.info(
            "Embedding cache served {hit}/{total} texts",
            hit=len(cached), total=len(texts),
        )

    results: list[list[float] | None] = [None] * len(texts)
    for i, vec in cached.items():
        results[i] = vec

    # Collapse repeated text to a single API call. Identical passages do occur —
    # a duplicated table, a repeated abstract — and embedding the same string
    # twice buys an identical vector for twice the quota.
    first_seen: dict[str, int] = {}
    pending_idx: list[int] = []
    aliases: dict[int, int] = {}   # duplicate position -> position that will be embedded

    for i in range(len(texts)):
        if i in cached:
            continue
        anchor = first_seen.get(texts[i])
        if anchor is None:
            first_seen[texts[i]] = i
            pending_idx.append(i)
        else:
            aliases[i] = anchor

    if aliases:
        logfire.info(
            "Collapsed {n} repeated text(s) — embedded once, reused in place",
            n=len(aliases), unique=len(pending_idx),
        )

    batch_size = max(1, settings.EMBED_BATCH_SIZE)
    for start in range(0, len(pending_idx), batch_size):
        window = pending_idx[start : start + batch_size]
        batch = [texts[i] for i in window]

        with logfire.span(
            "Embed batch",
            provider=backend.name, size=len(batch),
            progress=f"{start + len(window)}/{len(pending_idx)}",
        ):
            vectors = _embed_batch_with_retry(backend, batch)

        if len(vectors) != len(batch):
            raise EmbeddingError(
                f"Backend returned {len(vectors)} vectors for {len(batch)} inputs — refusing to misalign the index."
            )

        for i, vec in zip(window, vectors):
            if len(vec) != backend.dim:
                raise EmbeddingError(
                    f"Dimension drift: expected {backend.dim}, got {len(vec)}. "
                    "The embedding model changed mid-run; re-ingest with --wipe."
                )
            results[i] = vec

        _cache.put_many(batch, vectors, backend.name, backend.model, backend.dim)

    # Fan the deduplicated vectors back out to every position that shared the text.
    for duplicate_pos, anchor_pos in aliases.items():
        results[duplicate_pos] = results[anchor_pos]

    missing = [i for i, v in enumerate(results) if v is None]
    if missing:
        raise EmbeddingError(f"{len(missing)} texts produced no vector (indices {missing[:5]}...)")

    return results  # type: ignore[return-value]


def embed_query(query: str) -> list[float]:
    """Embed a single search query. Uses the retrieval_query task type on Gemini."""
    backend = _init()

    cached = _cache.get_many([query], backend.name, backend.model, backend.dim)
    if 0 in cached:
        return cached[0]

    max_attempts = max(1, settings.EMBED_MAX_RETRIES)
    for attempt in range(1, max_attempts + 1):
        try:
            _limiter.acquire(1)
            vector = backend.embed_single(query)
            _cache.put_many([query], [vector], backend.name, backend.model, backend.dim)
            return vector
        except Exception as exc:
            if _is_retryable(exc) and attempt < max_attempts:
                mandated = _retry_after(exc)
                computed = min(30.0, 2.0 ** attempt) + random.uniform(0, 1.0)
                # A live query cannot sit for a full minute waiting on quota, so
                # the mandated delay is capped here even though ingestion honours
                # it fully. Better a fast failure the caller can report.
                wait = min(max(mandated, computed), 20.0) if mandated else computed
                logfire.warning(
                    "Query embedding retry {n}/{max} in {wait}s ({err})",
                    n=attempt, max=max_attempts, wait=round(wait, 1), err=str(exc)[:200],
                )
                time.sleep(wait)
                continue
            raise EmbeddingError(f"Query embedding failed: {exc}") from exc

    raise EmbeddingError("Query embedding retries exhausted")
