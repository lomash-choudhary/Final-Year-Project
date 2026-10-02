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
The embedding model is a single pinned name from .env (GEMINI_EMBEDDING_MODEL) —
there is no probe list. The Gemini embedding family mixes dimensions across
generations (3072 for gemini-embedding-001, 768 for text-embedding-004 and
embedding-001), so a fallback chain here does not degrade gracefully: it writes a
second vector space into the same collection. One model, or the tier is off.

The *dimension* is still measured rather than hardcoded — we embed one short
string and take len() of the result — because a model can change its output width
between versions, and a hardcoded 3072 is how a collection ends up silently
rejecting every upsert.

Because Qdrant fixes the dimension at collection creation, the backend is chosen
once per process and then locked. If Gemini dies mid-run we raise instead of
quietly switching to a 768-dim local model, which would corrupt the index.

Key-level failover (GEMINI_FALLBACK_API_KEY)
--------------------------------------------
What *can* change safely mid-run is the API key. The free quota (~100 texts/min,
and a daily cap) belongs to a Google Cloud project, so a key from a second
project is a genuine second quota for the *same* model — same vector space,
same dimension, same cache entries. Each key gets its own rate limiter, because
a cooldown the primary was told to observe says nothing about the fallback's
window. A batch spends EMBED_MAX_RETRIES attempts on the active key; when those
are exhausted on rate-limit errors, the same batch moves to the next key with a
fresh budget, and that key stays active for the rest of the run (a key that just
failed five times is usually out for the day, not the minute). Only when every
key has exhausted its budget on one batch does the run raise — and the manifest
resumes from that file next time.
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
class _KeySlot:
    """One credential's clients and its own quota accounting."""
    label: str                                       # "primary" | "fallback" | "local"
    limiter: _RateLimiter
    embed_documents: Callable[[list[str]], list[list[float]]]
    embed_single: Callable[[str], list[float]]


@dataclass
class EmbeddingBackend:
    name: str                                        # "gemini" | "local"
    model: str
    dim: int
    slots: list[_KeySlot]                            # tried in order; same model on every slot
    active: int = 0                                  # sticky: index of the slot currently in use

    @property
    def key_label(self) -> str:
        return self.slots[self.active].label


# ── module state (initialised once, then locked) ──────────────────────────────
_backend: EmbeddingBackend | None = None
_init_lock = threading.Lock()
_cache = EmbeddingCache(settings.resolve_path(settings.EMBEDDING_CACHE_PATH), settings.EMBEDDING_CACHE_ENABLED)


# ── Gemini backend ─────────────────────────────────────────────────────────────

def _gemini_keys() -> list[tuple[str, str]]:
    """(label, key) pairs in failover order. A fallback equal to the primary shares its quota."""
    keys = [("primary", settings.GEMINI_API_KEY)] if settings.GEMINI_API_KEY else []
    fallback = settings.GEMINI_FALLBACK_API_KEY
    if fallback and fallback != settings.GEMINI_API_KEY:
        keys.append(("fallback", fallback))
    elif fallback:
        logfire.warning("GEMINI_FALLBACK_API_KEY equals GEMINI_API_KEY — it shares the same quota; ignored")
    return keys


def _build_gemini() -> EmbeddingBackend | None:
    """Build the Gemini backend from the one configured model, or return None."""
    keys = _gemini_keys()
    if not keys:
        logfire.info("No GEMINI_API_KEY — skipping Gemini embedding tier")
        return None

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as exc:
        logfire.warning("langchain-google-genai not installed ({err})", err=str(exc))
        return None

    # One name, straight from .env. There is no candidate list to fall through
    # to: the alternatives embed at a different width, so "try the next one" would
    # mean writing 768-dim vectors into a 3072-dim collection. Blank means this
    # tier is unconfigured, not "use the usual one". Every key uses this same name.
    model_name = settings.gemini_embedding_model
    if not model_name:
        logfire.warning(
            "GEMINI_EMBEDDING_MODEL is empty — set it in .env (see .env.example). "
            "Skipping the Gemini embedding tier."
        )
        return None

    slots: list[_KeySlot] = []
    asymmetric = True
    for label, api_key in keys:
        # task_type is a real quality lever: asymmetric embeddings score
        # documents and queries in different spaces. Older versions of the
        # package do not accept the kwarg, hence the TypeError branch.
        try:
            doc_model = GoogleGenerativeAIEmbeddings(
                model=model_name, google_api_key=api_key, task_type="retrieval_document",
            )
            query_model = GoogleGenerativeAIEmbeddings(
                model=model_name, google_api_key=api_key, task_type="retrieval_query",
            )
        except TypeError:
            doc_model = query_model = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
            asymmetric = False
        slots.append(_KeySlot(
            label=label,
            limiter=_RateLimiter(settings.EMBED_MAX_RPM),
            embed_documents=doc_model.embed_documents,
            embed_single=query_model.embed_query,
        ))

    # Probe each key in order until one answers. A primary that is already out of
    # daily quota must not fail the probe and push `auto` onto the 768-dim local
    # model — that ends in DimensionMismatch against the 3072-dim collection.
    for index, slot in enumerate(slots):
        try:
            slot.limiter.acquire()
            probe = slot.embed_single("bovine theileriosis prevalence")
            dim = len(probe)
            if dim == 0:
                raise EmbeddingError("probe returned an empty vector")
        except Exception as exc:
            logfire.warning(
                "Gemini embedding probe failed on {key} key ({err})",
                key=slot.label, model=model_name, err=str(exc)[:300],
            )
            continue

        logfire.info(
            "Gemini embeddings ready",
            model=model_name, dim=dim, asymmetric_task_types=asymmetric,
            active_key=slot.label, keys=[s.label for s in slots],
        )
        return EmbeddingBackend(name="gemini", model=model_name, dim=dim, slots=slots, active=index)

    # Do not substitute another Gemini model here. Returning None hands the
    # decision to _init_backend(), which either falls back to the local model
    # (a deliberate, logged, dimension-checked switch) or raises.
    logfire.warning(
        "Gemini embedding model '{model}' unavailable on every key — Gemini tier disabled",
        model=model_name, keys=[s.label for s in slots],
    )
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
    if not name:
        raise EmbeddingError(
            "LOCAL_EMBEDDING_MODEL is empty — the offline embedding backend has no model to "
            "load. Set it in .env (see .env.example); model names are never defaulted in code."
        )
    logfire.info("Loading local embedding model (first run downloads weights)", model=name)
    model = SentenceTransformer(name)

    def _docs(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

    def _one(text: str) -> list[float]:
        return model.encode([text], show_progress_bar=False, normalize_embeddings=True)[0].tolist()

    dim = len(_one("probe"))
    logfire.info("Local embeddings ready", model=name, dim=dim)
    slot = _KeySlot(label="local", limiter=_RateLimiter(settings.EMBED_MAX_RPM), embed_documents=_docs, embed_single=_one)
    return EmbeddingBackend(name="local", model=name, dim=dim, slots=[slot])


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

        logfire.info(
            "Embedding backend locked", provider=_backend.name, model=_backend.model, dim=_backend.dim,
            active_key=_backend.key_label,
        )
        return _backend


# ── retry / batching ───────────────────────────────────────────────────────────

class _KeyExhausted(Exception):
    """The active key spent its whole retry budget on rate-limit / transient errors."""


def _fail_over(backend: EmbeddingBackend, tried: int, last_exc: Exception) -> None:
    """Advance to the next key, or raise if every key has had its budget for this call."""
    if tried >= len(backend.slots):
        raise EmbeddingError(
            f"{backend.name}/{backend.model} embedding failed on every key "
            f"({', '.join(s.label for s in backend.slots)}): {last_exc}"
        ) from last_exc
    previous = backend.key_label
    backend.active = (backend.active + 1) % len(backend.slots)
    logfire.warning(
        "Embedding key '{old}' exhausted after {n} retries — switching to '{new}' key (same model)",
        old=previous, new=backend.key_label, n=settings.EMBED_MAX_RETRIES, model=backend.model,
    )


def _embed_batch_on_slot(slot: _KeySlot, backend: EmbeddingBackend, batch: list[str], depth: int) -> list[list[float]]:
    """Embed one batch on one key, retrying transient errors and splitting on batch errors."""
    # EMBED_MAX_RETRIES counts retries, so the first try is extra: 5 -> 6 calls.
    max_attempts = max(0, settings.EMBED_MAX_RETRIES) + 1

    for attempt in range(1, max_attempts + 1):
        try:
            slot.limiter.acquire(len(batch))
            return slot.embed_documents(batch)

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
                    _embed_batch_on_slot(slot, backend, batch[:mid], depth + 1)
                    + _embed_batch_on_slot(slot, backend, batch[mid:], depth + 1)
                )

            if not _is_retryable(exc):
                logfire.error("Embedding failed permanently: {err}", err=str(exc)[:400], key=slot.label)
                raise EmbeddingError(
                    f"{backend.name}/{backend.model} embedding failed on '{slot.label}' key: {exc}"
                ) from exc

            if attempt == max_attempts:
                raise _KeyExhausted(str(exc)[:400]) from exc

            # Prefer the provider's own instruction. A quota window is
            # ~60s wide, while exponential backoff tops out near 17s — so
            # computed backoff alone just retries inside the same blocked
            # minute and burns all five attempts for nothing.
            mandated = _retry_after(exc)
            computed = min(60.0, 2.0 ** attempt) + random.uniform(0, 1.5)
            wait = max(mandated + 1.0, computed) if mandated else computed

            if mandated:
                slot.limiter.penalize(mandated)

            logfire.warning(
                "Embedding call failed ({err}) — retry {n}/{max} on {key} key in {wait}s{src}",
                err=str(exc)[:200], n=attempt, max=max_attempts - 1, key=slot.label, wait=round(wait, 1),
                src=" (provider-specified)" if mandated else "",
            )
            time.sleep(wait)

    raise _KeyExhausted("retry loop ended without a result")


def _embed_batch_with_retry(backend: EmbeddingBackend, batch: list[str]) -> list[list[float]]:
    """Embed one batch, failing over across keys when one exhausts its retry budget."""
    tried = 0
    while True:
        tried += 1
        try:
            return _embed_batch_on_slot(backend.slots[backend.active], backend, batch, depth=0)
        except _KeyExhausted as exc:
            _fail_over(backend, tried, exc)


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
        info = {
            "provider": backend.name, "model": backend.model, "dim": backend.dim,
            "active_key": backend.key_label, "keys": [s.label for s in backend.slots],
        }
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
            provider=backend.name, key=backend.key_label, size=len(batch),
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

    # EMBED_MAX_RETRIES counts retries, so the first try is extra: 5 -> 6 calls.
    max_attempts = max(0, settings.EMBED_MAX_RETRIES) + 1
    tried = 0
    while True:
        tried += 1
        slot = backend.slots[backend.active]
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                slot.limiter.acquire(1)
                vector = slot.embed_single(query)
                _cache.put_many([query], [vector], backend.name, backend.model, backend.dim)
                return vector
            except Exception as exc:
                if not _is_retryable(exc):
                    raise EmbeddingError(f"Query embedding failed on '{slot.label}' key: {exc}") from exc
                last_exc = exc
                if attempt == max_attempts:
                    break
                mandated = _retry_after(exc)
                computed = min(30.0, 2.0 ** attempt) + random.uniform(0, 1.0)
                # A live query cannot sit for a full minute waiting on quota, so
                # the mandated delay is capped here even though ingestion honours
                # it fully. Better a fast failure the caller can report.
                wait = min(max(mandated, computed), 20.0) if mandated else computed
                logfire.warning(
                    "Query embedding retry {n}/{max} on {key} key in {wait}s ({err})",
                    n=attempt, max=max_attempts - 1, key=slot.label, wait=round(wait, 1), err=str(exc)[:200],
                )
                time.sleep(wait)
        _fail_over(backend, tried, last_exc or EmbeddingError("query retries exhausted"))
