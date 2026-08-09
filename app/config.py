"""
Centralised configuration.

Every tunable lives here and is sourced from `.env`. Nothing else in the codebase
calls os.getenv() directly, so there is exactly one place to look when a knob
misbehaves.

`validate(scope)` is the important part: it returns human-readable problems for a
given entry point (ingestion / api / evals) instead of letting the app die three
layers deep with an opaque SDK error.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Project root = the directory containing this file's parent (app/ -> repo root).
ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(dotenv_path=ROOT_DIR / ".env")


# ── env coercion helpers ───────────────────────────────────────────────────────
# The .env in this project is written as `KEY = "value"`. python-dotenv already
# handles that, but we strip defensively so a stray quote never becomes part of
# an API key.

def _str(name: str, default: str = "") -> str:
    # A variable that is present but blank (`BACKEND_URL = ""`) must fall back to
    # the default, not override it with an empty string. Half-filled .env files
    # are the norm, and an empty BACKEND_URL produces a request to "/query".
    raw = os.getenv(name)
    if raw is None:
        return default
    cleaned = raw.strip().strip('"').strip("'").strip()
    return cleaned or default


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name) or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name) or default)
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = _str(name).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # ── identity ──────────────────────────────────────────────────────────────
    APP_NAME: str = "Bovine Disease Research Assistant"
    ENVIRONMENT: str = field(default_factory=lambda: _str("LOGFIRE_ENVIRONMENT", "dev"))

    # ── Gemini (embeddings primary, chat last-resort) ──────────────────────────
    GEMINI_API_KEY: str = field(default_factory=lambda: _str("GEMINI_API_KEY"))
    GEMINI_EMBEDDING_MODEL: str = field(default_factory=lambda: _str("GEMINI_EMBEDDING_MODEL"))
    GEMINI_CHAT_MODEL: str = field(default_factory=lambda: _str("GEMINI_CHAT_MODEL", "gemini-2.5-flash"))

    # Probed in order when GEMINI_EMBEDDING_MODEL is blank. The first model the
    # key can actually reach wins — different Google accounts have different
    # model availability, and hardcoding one is the #1 cause of a dead ingest.
    GEMINI_EMBEDDING_CANDIDATES: tuple[str, ...] = (
        "models/gemini-embedding-001",
        "models/text-embedding-004",
        "models/embedding-001",
    )

    # ── Groq (reasoning) ──────────────────────────────────────────────────────
    GROQ_API_KEY: str = field(default_factory=lambda: _str("GROQ_API_KEY"))
    GROQ_FALLBACK_API_KEY: str = field(default_factory=lambda: _str("GROQ_FALLBACK_API_KEY"))
    GROQ_PRIMARY_MODEL: str = field(default_factory=lambda: _str("GROQ_PRIMARY_MODEL", "llama-3.3-70b-versatile"))
    GROQ_FAST_MODEL: str = field(default_factory=lambda: _str("GROQ_FAST_MODEL", "llama-3.1-8b-instant"))

    # ── Per-feature Groq keys ─────────────────────────────────────────────────
    # Each pipeline stage can own a key so one busy stage cannot rate-limit the
    # others. Any of these left blank simply falls back to GROQ_API_KEY, so the
    # system works with one key and gets more headroom with four.
    GROQ_TRANSLATE_API_KEY: str = field(default_factory=lambda: _str("GROQ_TRANSLATE_API_KEY"))
    GROQ_CLARIFIER_API_KEY: str = field(default_factory=lambda: _str("GROQ_CLARIFIER_API_KEY"))
    GROQ_ADVISOR_API_KEY: str = field(default_factory=lambda: _str("GROQ_ADVISOR_API_KEY"))
    # Translation is mechanical; a small model does it well and costs far less.
    GROQ_TRANSLATE_MODEL: str = field(default_factory=lambda: _str("GROQ_TRANSLATE_MODEL", "llama-3.1-8b-instant"))

    # ── Qdrant ────────────────────────────────────────────────────────────────
    QDRANT_URL: str = field(
        default_factory=lambda: _str("QDRANT_CLUSTER_ENDPOINT") or _str("QDRANT_URL", "http://localhost:6333")
    )
    QDRANT_API_KEY: str = field(default_factory=lambda: _str("QDRANT_API_KEY"))
    QDRANT_COLLECTION: str = field(default_factory=lambda: _str("QDRANT_COLLECTION", "bovine_disease_rag"))
    # Points per upsert request. A 3072-dim vector is ~12 KB before payload, so
    # 64 points is an ~800 KB write — enough to time out a throttled free-tier
    # cloud cluster. Smaller writes are slower in aggregate but far more reliable.
    QDRANT_UPSERT_BATCH: int = field(default_factory=lambda: _int("QDRANT_UPSERT_BATCH", 24))
    QDRANT_TIMEOUT: int = field(default_factory=lambda: _int("QDRANT_TIMEOUT", 120))

    # ── embedding pipeline ────────────────────────────────────────────────────
    EMBEDDING_PROVIDER: str = field(default_factory=lambda: _str("EMBEDDING_PROVIDER", "auto").lower())
    LOCAL_EMBEDDING_MODEL: str = field(
        default_factory=lambda: _str("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    )
    EMBED_BATCH_SIZE: int = field(default_factory=lambda: _int("EMBED_BATCH_SIZE", 16))
    # TEXTS per minute, not requests per minute. Gemini's free embedding quota is
    # `embed_content_free_tier_requests: limit 100`, and it is charged per text
    # embedded — a batch of 16 costs 16 units, not 1. Throttling batches instead
    # of texts is how a run sails past the ceiling and 429s on the fifth file.
    EMBED_MAX_RPM: int = field(default_factory=lambda: _int("EMBED_MAX_RPM", 90))
    EMBED_MAX_RETRIES: int = field(default_factory=lambda: _int("EMBED_MAX_RETRIES", 5))
    EMBEDDING_CACHE_ENABLED: bool = field(default_factory=lambda: _bool("EMBEDDING_CACHE_ENABLED", True))
    EMBEDDING_CACHE_PATH: str = field(
        default_factory=lambda: _str("EMBEDDING_CACHE_PATH", ".cache/embeddings.sqlite3")
    )

    # ── chunking ──────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = field(default_factory=lambda: _int("CHUNK_SIZE", 1400))
    CHUNK_OVERLAP: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 200))
    MIN_CHUNK_CHARS: int = field(default_factory=lambda: _int("MIN_CHUNK_CHARS", 120))

    # ── retrieval ─────────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = field(default_factory=lambda: _int("RETRIEVAL_TOP_K", 20))
    RERANK_TOP_N: int = field(default_factory=lambda: _int("RERANK_TOP_N", 5))
    MIN_RELEVANCE_SCORE: float = field(default_factory=lambda: _float("MIN_RELEVANCE_SCORE", 0.0))

    # ── agent ─────────────────────────────────────────────────────────────────
    ENABLE_SELF_CORRECTION: bool = field(default_factory=lambda: _bool("ENABLE_SELF_CORRECTION", True))
    MAX_REFINEMENTS: int = field(default_factory=lambda: _int("MAX_REFINEMENTS", 1))
    MAX_CONTEXT_CHARS: int = field(default_factory=lambda: _int("MAX_CONTEXT_CHARS", 18000))
    GUARDRAILS_MODE: str = field(default_factory=lambda: _str("GUARDRAILS_MODE", "fast").lower())

    # ── Consumer assistant behaviour ──────────────────────────────────────────
    ENABLE_TRANSLATION: bool = field(default_factory=lambda: _bool("ENABLE_TRANSLATION", True))
    ENABLE_CLARIFICATION: bool = field(default_factory=lambda: _bool("ENABLE_CLARIFICATION", True))
    # How many times in a row the assistant may ask follow-up questions before it
    # must answer with whatever it has. 1 keeps the conversation moving.
    MAX_CLARIFICATION_ROUNDS: int = field(default_factory=lambda: _int("MAX_CLARIFICATION_ROUNDS", 1))
    MAX_FOLLOW_UP_QUESTIONS: int = field(default_factory=lambda: _int("MAX_FOLLOW_UP_QUESTIONS", 4))
    # Farmers get plain advice; researchers get [n] citations. Turning this on
    # puts citation markers into consumer answers too.
    SHOW_CITATIONS_IN_ADVICE: bool = field(default_factory=lambda: _bool("SHOW_CITATIONS_IN_ADVICE", False))
    LLM_CACHE_ENABLED: bool = field(default_factory=lambda: _bool("LLM_CACHE_ENABLED", True))
    LLM_CACHE_TTL: int = field(default_factory=lambda: _int("LLM_CACHE_TTL", 900))

    # ── observability ─────────────────────────────────────────────────────────
    LOGFIRE_TOKEN: str = field(default_factory=lambda: _str("LOGFIRE_TOKEN"))
    LANGSMITH_TRACING: bool = field(default_factory=lambda: _bool("LANGSMITH_TRACING", False))
    LANGSMITH_API_KEY: str = field(default_factory=lambda: _str("LANGSMITH_API_KEY"))
    LANGSMITH_PROJECT: str = field(default_factory=lambda: _str("LANGSMITH_PROJECT", "bovine-disease-rag"))
    LANGSMITH_ENDPOINT: str = field(
        default_factory=lambda: _str("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    )

    # ── UI / evals ────────────────────────────────────────────────────────────
    BACKEND_URL: str = field(default_factory=lambda: _str("BACKEND_URL", "http://localhost:8000"))
    JUDGE_GROQ: str = field(default_factory=lambda: _str("JUDGE_GROQ"))

    # Browser origins allowed to call this API. Comma-separated, or "*" for any.
    # Defaults to "*" so a new frontend works without configuration; set it to
    # your deployed domain once you know it, so only your app can spend your
    # free-tier quota.
    CORS_ORIGINS: str = field(default_factory=lambda: _str("CORS_ORIGINS", "*"))

    # ── optional gateway ──────────────────────────────────────────────────────
    ENABLE_PORTKEY: bool = field(default_factory=lambda: _bool("ENABLE_PORTKEY", False))
    PORTKEY_API_KEY: str = field(default_factory=lambda: _str("PORTKEY_API_KEY"))

    # ── paths ─────────────────────────────────────────────────────────────────
    DATA_DIR: str = field(default_factory=lambda: str(ROOT_DIR / "DATA"))
    PROCESSED_DATA_DIR: str = field(default_factory=lambda: str(ROOT_DIR / "processed_data"))
    MANIFEST_PATH: str = field(default_factory=lambda: str(ROOT_DIR / "ingestion_manifest.json"))

    # ── derived helpers ───────────────────────────────────────────────────────

    @property
    def judge_api_key(self) -> str:
        """Eval judge key, falling back to the main Groq key."""
        return self.JUDGE_GROQ or self.GROQ_API_KEY

    @property
    def qdrant_is_local(self) -> bool:
        return "localhost" in self.QDRANT_URL or "127.0.0.1" in self.QDRANT_URL

    def feature_key(self, feature: str) -> str:
        """
        Dedicated Groq key for a pipeline stage, or "" if it has none.

        Returning "" (rather than the shared key) matters: the router uses this
        only to decide whether to *prepend* a dedicated target. The shared keys
        are always appended after it, so a stage with no key of its own still
        gets the full fallback chain.
        """
        return {
            "translate": self.GROQ_TRANSLATE_API_KEY,
            "clarifier": self.GROQ_CLARIFIER_API_KEY,
            "advisor": self.GROQ_ADVISOR_API_KEY,
        }.get(feature, "")

    @property
    def cors_origins(self) -> list[str]:
        """CORS_ORIGINS parsed into the list FastAPI's middleware expects."""
        raw = self.CORS_ORIGINS.strip()
        if not raw or raw == "*":
            return ["*"]
        return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]

    def resolve_path(self, value: str) -> Path:
        """Turn a possibly-relative config path into an absolute one under ROOT_DIR."""
        p = Path(value)
        return p if p.is_absolute() else ROOT_DIR / p

    # ── validation ────────────────────────────────────────────────────────────

    def validate(self, scope: str = "api") -> list[str]:
        """
        Return a list of configuration problems for the given entry point.
        Empty list means good to go. Callers decide whether to warn or abort.

        scope: "ingestion" | "api" | "evals"
        """
        problems: list[str] = []

        if not self.QDRANT_URL:
            problems.append("QDRANT_CLUSTER_ENDPOINT is empty — set it to http://localhost:6333 or your Qdrant Cloud URL.")
        elif not self.QDRANT_URL.startswith(("http://", "https://")):
            problems.append(f"QDRANT_CLUSTER_ENDPOINT must start with http:// or https:// (got '{self.QDRANT_URL}').")

        if not self.qdrant_is_local and not self.QDRANT_API_KEY:
            problems.append("QDRANT_API_KEY is empty but QDRANT_CLUSTER_ENDPOINT points at a remote cluster.")

        if scope == "ingestion":
            if self.EMBEDDING_PROVIDER in ("auto", "gemini") and not self.GEMINI_API_KEY:
                if self.EMBEDDING_PROVIDER == "gemini":
                    problems.append("EMBEDDING_PROVIDER=gemini but GEMINI_API_KEY is empty.")
                else:
                    problems.append(
                        "GEMINI_API_KEY is empty — ingestion will fall back to the local embedding model "
                        "(slower first run, but free and offline)."
                    )
            if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
                problems.append(f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be smaller than CHUNK_SIZE ({self.CHUNK_SIZE}).")
            if self.EMBED_BATCH_SIZE > self.EMBED_MAX_RPM:
                problems.append(
                    f"EMBED_BATCH_SIZE ({self.EMBED_BATCH_SIZE}) exceeds EMBED_MAX_RPM "
                    f"({self.EMBED_MAX_RPM} texts/min) — a single batch cannot fit inside the rate limit."
                )

        if scope in ("api", "evals"):
            if not self.GROQ_API_KEY and not self.GEMINI_API_KEY:
                problems.append("No LLM key found — set GROQ_API_KEY (preferred) or GEMINI_API_KEY.")
            if self.GUARDRAILS_MODE not in ("off", "fast", "full"):
                problems.append(f"GUARDRAILS_MODE must be one of off|fast|full (got '{self.GUARDRAILS_MODE}').")
            if self.RERANK_TOP_N > self.RETRIEVAL_TOP_K:
                problems.append(
                    f"RERANK_TOP_N ({self.RERANK_TOP_N}) exceeds RETRIEVAL_TOP_K ({self.RETRIEVAL_TOP_K}) — "
                    "the reranker cannot return more documents than were retrieved."
                )

        if scope == "evals" and not self.judge_api_key:
            problems.append("Neither JUDGE_GROQ nor GROQ_API_KEY is set — the RAGAS judge has no LLM.")

        return problems

    def summary(self) -> dict:
        """Non-secret snapshot, safe to log or expose on /health."""
        return {
            "environment": self.ENVIRONMENT,
            "qdrant_url": self.QDRANT_URL,
            "qdrant_collection": self.QDRANT_COLLECTION,
            "embedding_provider": self.EMBEDDING_PROVIDER,
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP,
            "retrieval_top_k": self.RETRIEVAL_TOP_K,
            "rerank_top_n": self.RERANK_TOP_N,
            "guardrails_mode": self.GUARDRAILS_MODE,
            "self_correction": self.ENABLE_SELF_CORRECTION,
            "translation": self.ENABLE_TRANSLATION,
            "clarification": self.ENABLE_CLARIFICATION,
            "groq_keys_configured": sum(bool(k) for k in (self.GROQ_API_KEY, self.GROQ_FALLBACK_API_KEY)),
            "dedicated_feature_keys": [
                name for name in ("translate", "clarifier", "advisor") if self.feature_key(name)
            ],
            "gemini_configured": bool(self.GEMINI_API_KEY),
        }


settings = Settings()
