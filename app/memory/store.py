"""
Conversation memory — one store for the agent's memory and the UI's chat history.

Why this exists
---------------
The graph used to remember conversations through LangGraph's `MemorySaver`, which
lives in process RAM. Every restart — and every Render free-tier sleep — wiped it,
so in practice the assistant had no memory, and the frontend could not show past
chats because nothing outlived the request. What it did keep, it kept badly: the
`messages` reducer grew without bound and every node's full state (retrieved
passages included) was checkpointed on every step.

What this does instead
----------------------
Postgres (the frontend's Neon database, schema `rag`) holds two tables:
`conversations` (title, rolling summary, clarifier state) and `messages`. The
agent never sees the raw history. It sees a **bounded** context:

    rolling summary of older turns  +  last MEMORY_WINDOW_TURNS turns verbatim

so prompt size stays flat whether a chat has 3 turns or 300. That is the
anti-rot design, in four parts:

1. **Window + rolling summary.** Once MEMORY_SUMMARY_BATCH_TURNS turns have spilled
   out of the window, one fast-tier call folds them into the summary, capped at
   MEMORY_SUMMARY_MAX_CHARS. It runs as a background task after the response is
   sent, so the user never waits for it. A failed summary keeps the window and
   is retried on the next turn.
2. **Store only what memory needs.** Memory reads the English text of each turn.
   Retrieved passages and the plan are never fed back; sources are kept per
   message, trimmed, for display only.
3. **Retention.** Idle conversations past MEMORY_RETENTION_DAYS (or
   MEMORY_ANON_RETENTION_DAYS without a user) are deleted, and each user keeps at
   most MEMORY_MAX_CONVERSATIONS_PER_USER. Cleanup runs at most once a day.
4. **Degrades, never fails.** No DATABASE_URL, or the database is unreachable →
   an in-process fallback keeps the conversation going (lost on restart, not
   listed in the UI). The question is still answered either way.

Ownership: every conversation carries the `user_id` that created it, and every
read and write filters on it. The frontend's auth is still a mock, so this keeps
users apart but is not a security boundary — see MEMORY.md open threads.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import logfire

from app.config import settings
from app.llm.router import AllTargetsFailed, router

_TITLE_MAX = 60
_SOURCE_PREVIEW = 300
_LOCAL_MAX_CONVERSATIONS = 500
_CLEANUP_EVERY_S = 24 * 3600

_SCHEMA = """
CREATE SCHEMA IF NOT EXISTS rag;
CREATE TABLE IF NOT EXISTS rag.conversations (
    id                     text PRIMARY KEY,
    user_id                text,
    title                  text NOT NULL DEFAULT 'New chat',
    summary                text NOT NULL DEFAULT '',
    summarized_upto        bigint NOT NULL DEFAULT 0,
    awaiting_clarification boolean NOT NULL DEFAULT false,
    clarification_rounds   integer NOT NULL DEFAULT 0,
    language               text NOT NULL DEFAULT 'en',
    message_count          integer NOT NULL DEFAULT 0,
    created_at             timestamptz NOT NULL DEFAULT now(),
    updated_at             timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS conversations_user_recent
    ON rag.conversations (user_id, updated_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS conversations_updated ON rag.conversations (updated_at);
CREATE TABLE IF NOT EXISTS rag.messages (
    id              bigserial PRIMARY KEY,
    conversation_id text NOT NULL REFERENCES rag.conversations(id) ON DELETE CASCADE,
    role            text NOT NULL,
    content         text NOT NULL,
    content_en      text,
    in_memory       boolean NOT NULL DEFAULT true,
    care_level      text,
    sources         jsonb NOT NULL DEFAULT '[]'::jsonb,
    meta            jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_conversation ON rag.messages (conversation_id, id DESC);
"""

_SUMMARY_PROMPT = """You keep the running memory of a chat between a cattle farmer or researcher \
and a veterinary assistant.

CURRENT SUMMARY:
{summary}

NEW TURNS:
{turns}

Write the updated summary in at most {words} words. Keep only what is needed to continue the chat: \
the animal (species, breed, age, sex, pregnancy), symptoms and since when, what was already tried, \
the disease or topic discussed, advice given, and questions still open. Drop greetings and repetition. \
Plain sentences, no headings. Reply with the summary only."""

_SUMMARY_PREFIX = re.compile(r"^\s*(updated\s+)?summary\s*:\s*", re.IGNORECASE)


class ConversationForbidden(Exception):
    """The conversation id exists but belongs to another user."""


@dataclass
class MemoryContext:
    """What one turn of the agent gets to remember."""

    conversation_id: str
    summary: str = ""
    window: list[dict] = field(default_factory=list)   # [{"role", "content"}], English
    awaiting_clarification: bool = False
    clarification_rounds: int = 0
    is_new: bool = True


@dataclass
class Turn:
    """One user message and the reply to it, ready to persist."""

    user_text: str
    user_text_en: str
    answer: str
    answer_en: str
    in_memory: bool = True              # False for guardrail replies: shown, never remembered
    care_level: str | None = None
    sources: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    awaiting_clarification: bool = False
    clarification_rounds: int = 0
    language: str = "en"


def make_title(text: str) -> str:
    title = " ".join(text.split())
    return title if len(title) <= _TITLE_MAX else title[: _TITLE_MAX - 1].rstrip() + "…"


def slim_sources(sources: list[dict]) -> list[dict]:
    """Keep what a sources panel shows; drop the passage bodies that bloat storage."""
    return [
        {
            "source": s.get("source", ""),
            "page_label": s.get("page_label", ""),
            "citation": s.get("citation", ""),
            "score": s.get("score", 0),
            "chunk_index": s.get("chunk_index", 0),
            "content": str(s.get("content", ""))[:_SOURCE_PREVIEW],
        }
        for s in sources
    ]


def _window_size() -> int:
    return max(1, settings.MEMORY_WINDOW_TURNS) * 2


def _summarize(summary: str, turns: list[dict]) -> str:
    """Fold turns into the summary. Raises AllTargetsFailed; the caller retries next turn."""
    cap = settings.MEMORY_MSG_MAX_CHARS
    lines = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {str(m['content'])[:cap]}" for m in turns
    )
    response = router.invoke(
        _SUMMARY_PROMPT.format(
            summary=summary or "(none yet)",
            turns=lines,
            words=max(40, settings.MEMORY_SUMMARY_MAX_CHARS // 6),
        ),
        tier="fast",
        temperature=0.0,
        # Generous on purpose: a reasoning model on the fast tier spends its hidden
        # channel out of this cap and would return empty content (DOCS/06 §22).
        max_tokens=800,
        feature="memory",
    )
    text = _SUMMARY_PREFIX.sub("", response.content.strip())
    return text[: settings.MEMORY_SUMMARY_MAX_CHARS].strip()


# ── Postgres backend ───────────────────────────────────────────────────────────


class _PostgresBackend:
    name = "postgres"

    def __init__(self, url: str):
        from psycopg_pool import ConnectionPool

        # Every call is ONE statement and ONE round trip: Neon can be 100–250 ms
        # away, so a per-checkout health check or a BEGIN/COMMIT pair would double
        # or triple the cost. Instead idle connections are closed before Neon's
        # 5-minute auto-suspend kills them (max_idle), and a call that still hits
        # a dead connection is retried once (`_run`).
        # prepare_threshold=None: Neon's pooled endpoint is PgBouncer in transaction
        # mode, where server-side prepared statements can land on another backend.
        self._pool = ConnectionPool(
            url,
            min_size=1,
            max_size=4,
            open=False,
            timeout=10,
            max_idle=240,
            kwargs={"autocommit": True, "prepare_threshold": None},
        )
        self._pool.open(wait=False)
        self._schema_ready = False
        self._lock = threading.Lock()

    def _run(self, sql: str, params: tuple = (), *, fetch: str = "all"):
        """One statement, one round trip; retried once if the pooled connection was dead."""
        import psycopg

        if not self._schema_ready:
            with self._lock:
                if not self._schema_ready:
                    with self._pool.connection() as conn:
                        conn.execute(_SCHEMA)
                    self._schema_ready = True
                    logfire.info("Memory schema ready")
        for attempt in (1, 2):
            try:
                with self._pool.connection() as conn:
                    cur = conn.execute(sql, params)
                    if fetch == "one":
                        return cur.fetchone()
                    if fetch == "rowcount":
                        return cur.rowcount
                    return cur.fetchall()
            except psycopg.OperationalError:
                # Neon suspended the compute or dropped the socket; the pool has
                # discarded that connection, so the retry gets a fresh one.
                if attempt == 2:
                    raise

    def ping(self) -> dict:
        row = self._run(
            "SELECT (SELECT count(*) FROM rag.conversations), (SELECT count(*) FROM rag.messages)",
            fetch="one",
        )
        return {"conversations": row[0], "messages": row[1]}

    def load(self, conversation_id: str, user_id: str | None) -> MemoryContext:
        # Conversation row + its recent window in one round trip. A new
        # conversation returns no rows; a conversation with no window yet returns
        # one row whose message columns are NULL.
        rows = self._run(
            "SELECT c.user_id, c.summary, c.awaiting_clarification, c.clarification_rounds, m.role, m.text "
            "FROM rag.conversations c LEFT JOIN LATERAL ("
            "  SELECT id, role, coalesce(content_en, content) AS text FROM rag.messages "
            "  WHERE conversation_id = c.id AND in_memory AND id > c.summarized_upto "
            "  ORDER BY id DESC LIMIT %s"
            ") m ON true WHERE c.id = %s ORDER BY m.id",
            (_window_size(), conversation_id),
        )
        if not rows:
            return MemoryContext(conversation_id=conversation_id)
        head = rows[0]
        if head[0] != user_id:
            raise ConversationForbidden(conversation_id)
        return MemoryContext(
            conversation_id=conversation_id,
            summary=head[1],
            window=[{"role": r[4], "content": r[5]} for r in rows if r[4] is not None],
            awaiting_clarification=head[2],
            clarification_rounds=head[3],
            is_new=False,
        )

    def save(self, conversation_id: str, user_id: str | None, turn: Turn) -> bool:
        from psycopg.types.json import Jsonb

        # Upsert the conversation and insert both messages in ONE statement — atomic
        # without a transaction block, and one round trip. The ownership guard in
        # the upsert's WHERE makes `owned` empty for someone else's id, which in
        # turn inserts no messages.
        row = self._run(
            "WITH owned AS ("
            "  INSERT INTO rag.conversations AS c "
            "  (id, user_id, title, awaiting_clarification, clarification_rounds, language, message_count) "
            "  VALUES (%(cid)s, %(uid)s, %(title)s, %(awaiting)s, %(rounds)s, %(lang)s, 2) "
            "  ON CONFLICT (id) DO UPDATE SET updated_at = now(), "
            "  awaiting_clarification = EXCLUDED.awaiting_clarification, "
            "  clarification_rounds = EXCLUDED.clarification_rounds, "
            "  language = EXCLUDED.language, message_count = c.message_count + 2 "
            "  WHERE c.user_id IS NOT DISTINCT FROM EXCLUDED.user_id RETURNING id"
            "), added AS ("
            "  INSERT INTO rag.messages "
            "  (conversation_id, role, content, content_en, in_memory, care_level, sources, meta) "
            "  SELECT owned.id, v.role, v.content, v.content_en, %(mem)s, v.care_level, v.sources, v.meta "
            "  FROM owned, (VALUES "
            "    (1, 'user', %(q)s, %(q_en)s, NULL::text, '[]'::jsonb, '{}'::jsonb), "
            "    (2, 'assistant', %(a)s, %(a_en)s, %(care)s, %(sources)s, %(meta)s)"
            "  ) AS v(ord, role, content, content_en, care_level, sources, meta) ORDER BY v.ord "
            "  RETURNING 1"
            ") SELECT count(*) FROM added",
            {
                "cid": conversation_id, "uid": user_id, "title": make_title(turn.user_text),
                "awaiting": turn.awaiting_clarification, "rounds": turn.clarification_rounds,
                "lang": turn.language, "mem": turn.in_memory,
                "q": turn.user_text, "q_en": turn.user_text_en,
                "a": turn.answer, "a_en": turn.answer_en, "care": turn.care_level,
                "sources": Jsonb(turn.sources), "meta": Jsonb(turn.meta),
            },
            fetch="one",
        )
        return bool(row and row[0])

    def pending_summary(self, conversation_id: str) -> tuple[str, int, list[dict]]:
        """(summary, summarized_upto, every in-memory message not yet summarised)."""
        rows = self._run(
            "SELECT c.summary, c.summarized_upto, m.id, m.role, coalesce(m.content_en, m.content) "
            "FROM rag.conversations c LEFT JOIN rag.messages m "
            "ON m.conversation_id = c.id AND m.in_memory AND m.id > c.summarized_upto "
            "WHERE c.id = %s ORDER BY m.id",
            (conversation_id,),
        )
        if not rows:
            return "", 0, []
        return rows[0][0], rows[0][1], [
            {"id": r[2], "role": r[3], "content": r[4]} for r in rows if r[2] is not None
        ]

    def set_summary(self, conversation_id: str, expected_upto: int, upto: int, summary: str) -> bool:
        # Compare-and-set: two overlapping background jobs cannot both fold the same turns.
        row = self._run(
            "UPDATE rag.conversations SET summary = %s, summarized_upto = %s "
            "WHERE id = %s AND summarized_upto = %s RETURNING id",
            (summary, upto, conversation_id, expected_upto),
            fetch="one",
        )
        return row is not None

    def list(self, user_id: str, cursor: str | None, limit: int) -> tuple[list[dict], str | None]:
        params: list = [user_id]
        where = "user_id = %s"
        if cursor and "|" in cursor:
            stamp, last_id = cursor.split("|", 1)
            where += " AND (updated_at, id) < (%s::timestamptz, %s)"
            params += [stamp, last_id]
        rows = self._run(
            f"SELECT id, title, updated_at, message_count FROM rag.conversations WHERE {where} "
            "ORDER BY updated_at DESC, id DESC LIMIT %s",
            (*params, limit + 1),
        )
        items = [
            {"id": r[0], "title": r[1], "updated_at": r[2].isoformat(), "message_count": r[3]}
            for r in rows[:limit]
        ]
        next_cursor = f"{items[-1]['updated_at']}|{items[-1]['id']}" if len(rows) > limit else None
        return items, next_cursor

    def messages(
        self, conversation_id: str, user_id: str, before: int | None, limit: int
    ) -> tuple[list[dict], int | None] | None:
        # Ownership check and page in one round trip: a conversation that is not
        # this user's yields no rows, and every real conversation has messages.
        rows = self._run(
            "SELECT m.id, m.role, m.content, m.care_level, m.sources, m.meta, m.created_at "
            "FROM rag.messages m JOIN rag.conversations c ON c.id = m.conversation_id "
            "WHERE m.conversation_id = %s AND c.user_id = %s AND (%s::bigint IS NULL OR m.id < %s) "
            "ORDER BY m.id DESC LIMIT %s",
            (conversation_id, user_id, before, before, limit + 1),
        )
        if not rows and before is None:
            return None
        page = rows[:limit]
        items = [
            {
                "id": r[0], "role": r[1], "content": r[2], "care_level": r[3],
                "sources": r[4], "meta": r[5], "created_at": r[6].isoformat(),
            }
            for r in reversed(page)
        ]
        return items, (page[-1][0] if len(rows) > limit else None)

    def rename(self, conversation_id: str, user_id: str, title: str) -> bool:
        row = self._run(
            "UPDATE rag.conversations SET title = %s WHERE id = %s AND user_id = %s RETURNING id",
            (make_title(title), conversation_id, user_id),
            fetch="one",
        )
        return row is not None

    def delete(self, conversation_id: str, user_id: str) -> bool:
        row = self._run(
            "DELETE FROM rag.conversations WHERE id = %s AND user_id = %s RETURNING id",
            (conversation_id, user_id),
            fetch="one",
        )
        return row is not None

    def cleanup(self) -> dict:
        expired = self._run(
            "DELETE FROM rag.conversations WHERE "
            "(user_id IS NOT NULL AND updated_at < now() - make_interval(days => %s)) OR "
            "(user_id IS NULL AND updated_at < now() - make_interval(days => %s))",
            (settings.MEMORY_RETENTION_DAYS, settings.MEMORY_ANON_RETENTION_DAYS),
            fetch="rowcount",
        )
        over_cap = self._run(
            "DELETE FROM rag.conversations c USING ("
            "  SELECT id FROM (SELECT id, row_number() OVER "
            "    (PARTITION BY user_id ORDER BY updated_at DESC) AS rn "
            "    FROM rag.conversations WHERE user_id IS NOT NULL) ranked WHERE rn > %s"
            ") old WHERE c.id = old.id",
            (settings.MEMORY_MAX_CONVERSATIONS_PER_USER,),
            fetch="rowcount",
        )
        return {"expired": expired, "over_cap": over_cap}


# ── in-process fallback ────────────────────────────────────────────────────────


class _LocalBackend:
    """Bounded RAM store with the same contract — used when Postgres is absent or down."""

    name = "in-process"

    def __init__(self):
        self._data: OrderedDict[str, dict] = OrderedDict()
        self._next_id = 0
        self._lock = threading.Lock()

    def ping(self) -> dict:
        return {"conversations": len(self._data)}

    def load(self, conversation_id: str, user_id: str | None) -> MemoryContext:
        with self._lock:
            convo = self._data.get(conversation_id)
            if convo is None:
                return MemoryContext(conversation_id=conversation_id)
            if convo["user_id"] != user_id:
                raise ConversationForbidden(conversation_id)
            recent = [
                m for m in convo["messages"] if m["in_memory"] and m["id"] > convo["summarized_upto"]
            ][-_window_size():]
            return MemoryContext(
                conversation_id=conversation_id,
                summary=convo["summary"],
                window=[{"role": m["role"], "content": m["content"]} for m in recent],
                awaiting_clarification=convo["awaiting_clarification"],
                clarification_rounds=convo["clarification_rounds"],
                is_new=False,
            )

    def save(self, conversation_id: str, user_id: str | None, turn: Turn) -> bool:
        with self._lock:
            convo = self._data.get(conversation_id)
            if convo is None:
                convo = {"user_id": user_id, "summary": "", "summarized_upto": 0, "messages": []}
                self._data[conversation_id] = convo
            elif convo["user_id"] != user_id:
                return False
            self._data.move_to_end(conversation_id)
            convo["awaiting_clarification"] = turn.awaiting_clarification
            convo["clarification_rounds"] = turn.clarification_rounds
            for role, text in (("user", turn.user_text_en), ("assistant", turn.answer_en)):
                self._next_id += 1
                convo["messages"].append(
                    {"id": self._next_id, "role": role, "content": text, "in_memory": turn.in_memory}
                )
            # Only what the summary has not absorbed needs to stay in RAM.
            keep = _window_size() * 3
            if len(convo["messages"]) > keep:
                convo["messages"] = convo["messages"][-keep:]
            while len(self._data) > _LOCAL_MAX_CONVERSATIONS:
                self._data.popitem(last=False)
        return True

    def pending_summary(self, conversation_id: str) -> tuple[str, int, list[dict]]:
        with self._lock:
            convo = self._data.get(conversation_id)
            if convo is None:
                return "", 0, []
            upto = convo["summarized_upto"]
            return convo["summary"], upto, [
                dict(m) for m in convo["messages"] if m["in_memory"] and m["id"] > upto
            ]

    def set_summary(self, conversation_id: str, expected_upto: int, upto: int, summary: str) -> bool:
        with self._lock:
            convo = self._data.get(conversation_id)
            if convo is None or convo["summarized_upto"] != expected_upto:
                return False
            convo["summary"], convo["summarized_upto"] = summary, upto
        return True

    def cleanup(self) -> dict:
        return {}


# ── public facade ──────────────────────────────────────────────────────────────


class ConversationStore:
    """Picks the backend, and falls back to RAM per call when Postgres is down."""

    def __init__(self):
        self._local = _LocalBackend()
        self._pg: _PostgresBackend | None = None
        self._last_cleanup = 0.0
        self._init_lock = threading.Lock()
        self._initialised = False

    def _ensure(self) -> None:
        if self._initialised:
            return
        with self._init_lock:
            if self._initialised:
                return
            if settings.DATABASE_URL:
                try:
                    self._pg = _PostgresBackend(settings.DATABASE_URL)
                except Exception as exc:
                    logfire.error("Memory database unavailable ({err}) — using RAM", err=str(exc)[:300])
            self._initialised = True

    @property
    def persistent(self) -> bool:
        self._ensure()
        return self._pg is not None

    def initialize(self) -> str:
        """Called at startup so a bad DATABASE_URL shows up in the boot log."""
        self._ensure()
        if self._pg is None:
            logfire.warning("Conversation memory is in-process only — set DATABASE_URL to persist it")
            return self._local.name
        try:
            stats = self._pg.ping()
            logfire.info("Conversation memory ready", backend="postgres", **stats)
        except Exception as exc:
            logfire.error("Memory database unreachable at startup ({err})", err=str(exc)[:300])
        return self._pg.name

    def status(self) -> dict:
        self._ensure()
        if self._pg is None:
            return {"backend": self._local.name, **self._local.ping()}
        try:
            return {"backend": "postgres", "status": "ok", **self._pg.ping()}
        except Exception as exc:
            return {"backend": "postgres", "status": "error", "error": str(exc)[:200]}

    def load(self, conversation_id: str, user_id: str | None) -> tuple[MemoryContext, str]:
        """Context for one turn and the backend that served it. Raises ConversationForbidden."""
        self._ensure()
        if self._pg is not None:
            try:
                return self._pg.load(conversation_id, user_id), self._pg.name
            except ConversationForbidden:
                raise
            except Exception as exc:
                logfire.warning("Memory load failed ({err}) — using RAM", err=str(exc)[:300])
        return self._local.load(conversation_id, user_id), self._local.name

    def save(self, conversation_id: str, user_id: str | None, turn: Turn) -> bool:
        """Persist a turn. False when it could not be stored durably (the answer is unaffected)."""
        self._ensure()
        if self._pg is not None:
            try:
                return self._pg.save(conversation_id, user_id, turn)
            except Exception as exc:
                logfire.warning("Memory save failed ({err}) — kept in RAM only", err=str(exc)[:300])
        self._local.save(conversation_id, user_id, turn)
        return False

    def maybe_summarize(self, conversation_id: str) -> None:
        """Fold turns that left the window into the summary. Background task; never raises."""
        self._ensure()
        backend = self._pg or self._local
        try:
            summary, upto, pending = backend.pending_summary(conversation_id)
            window = _window_size()
            if len(pending) < window + max(1, settings.MEMORY_SUMMARY_BATCH_TURNS) * 2:
                return
            fold = pending[:-window]
            with logfire.span("Memory summary", conversation=conversation_id, turns=len(fold) // 2):
                new_summary = _summarize(summary, fold)
                if not new_summary:
                    logfire.warning("Memory summary came back empty — retrying next turn")
                    return
                stored = backend.set_summary(conversation_id, upto, fold[-1]["id"], new_summary)
                logfire.info("Memory summarised", stored=stored, chars=len(new_summary))
        except AllTargetsFailed as exc:
            logfire.warning("Memory summary skipped ({err}) — retrying next turn", err=str(exc)[:200])
        except Exception as exc:
            logfire.warning("Memory summary failed ({err})", err=str(exc)[:300])

    def maybe_cleanup(self) -> None:
        """Retention sweep, at most once a day per process. Background task; never raises."""
        now = time.monotonic()
        if self._pg is None or (self._last_cleanup and now - self._last_cleanup < _CLEANUP_EVERY_S):
            return
        self._last_cleanup = now
        try:
            with logfire.span("Memory cleanup"):
                removed = self._pg.cleanup()
                logfire.info("Memory cleanup done", **removed)
        except Exception as exc:
            logfire.warning("Memory cleanup failed ({err})", err=str(exc)[:300])

    # ── history API (Postgres only: RAM memory is not a history) ──────────────

    def _require_pg(self) -> _PostgresBackend:
        self._ensure()
        if self._pg is None:
            raise RuntimeError("Chat history needs DATABASE_URL")
        return self._pg

    def list_conversations(self, user_id: str, cursor: str | None, limit: int):
        return self._require_pg().list(user_id, cursor, limit)

    def get_messages(self, conversation_id: str, user_id: str, before: int | None, limit: int):
        return self._require_pg().messages(conversation_id, user_id, before, limit)

    def rename(self, conversation_id: str, user_id: str, title: str) -> bool:
        return self._require_pg().rename(conversation_id, user_id, title)

    def delete(self, conversation_id: str, user_id: str) -> bool:
        return self._require_pg().delete(conversation_id, user_id)


store = ConversationStore()
