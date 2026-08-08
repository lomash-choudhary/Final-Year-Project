"""
On-disk embedding cache (sqlite3, stdlib only).

The single most expensive resource in this project is free-tier Gemini quota.
Without a cache, every `--wipe` re-ingest spends it again on text that has not
changed. With it, a re-index of an unchanged corpus costs zero API calls.

The cache key includes provider, model *and* dimension. A vector produced by
gemini-embedding-001 is meaningless to all-mpnet-base-v2, so entries must never
collide across backends.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import logfire

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    key       TEXT PRIMARY KEY,
    provider  TEXT NOT NULL,
    model     TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vector    TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_provider_model ON embeddings(provider, model);
"""


class EmbeddingCache:
    def __init__(self, path: str | Path, enabled: bool = True):
        self.enabled = enabled
        self.path = Path(path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self.hits = 0
        self.misses = 0

        if not self.enabled:
            return

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False: FastAPI serves requests on a thread pool.
            # All access is serialised through self._lock anyway.
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except Exception as exc:
            # A broken cache must never break ingestion — degrade to no-cache.
            logfire.warning("Embedding cache disabled ({err})", err=str(exc), path=str(self.path))
            self.enabled = False
            self._conn = None

    @staticmethod
    def _key(text: str, provider: str, model: str, dim: int) -> str:
        payload = f"{provider}|{model}|{dim}|{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get_many(self, texts: list[str], provider: str, model: str, dim: int) -> dict[int, list[float]]:
        """Return {position_in_texts: vector} for the entries already cached."""
        if not self.enabled or not self._conn or not texts:
            return {}

        keys = [self._key(t, provider, model, dim) for t in texts]
        found: dict[str, list[float]] = {}

        try:
            with self._lock:
                # Chunked IN clause — sqlite caps host parameters at 999.
                for start in range(0, len(keys), 500):
                    window = keys[start : start + 500]
                    placeholders = ",".join("?" * len(window))
                    rows = self._conn.execute(
                        f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})",
                        window,
                    ).fetchall()
                    for key, vector in rows:
                        found[key] = json.loads(vector)
        except Exception as exc:
            logfire.warning("Embedding cache read failed ({err})", err=str(exc))
            return {}

        result = {i: found[k] for i, k in enumerate(keys) if k in found}
        self.hits += len(result)
        self.misses += len(texts) - len(result)
        return result

    def put_many(self, texts: list[str], vectors: list[list[float]], provider: str, model: str, dim: int) -> None:
        if not self.enabled or not self._conn or not texts:
            return
        try:
            rows = [
                (self._key(t, provider, model, dim), provider, model, dim, json.dumps(v))
                for t, v in zip(texts, vectors)
            ]
            with self._lock:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO embeddings (key, provider, model, dim, vector) VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
                self._conn.commit()
        except Exception as exc:
            logfire.warning("Embedding cache write failed ({err})", err=str(exc))

    def stats(self) -> dict:
        total = self.hits + self.misses
        rows = 0
        if self.enabled and self._conn:
            try:
                with self._lock:
                    rows = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            except Exception:
                pass
        return {
            "enabled": self.enabled,
            "path": str(self.path),
            "rows": rows,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }

    def close(self) -> None:
        if self._conn:
            with self._lock:
                self._conn.close()
            self._conn = None
