# 04 · Observability

## Two tools, different jobs

| Tool | Answers |
|---|---|
| **Pydantic Logfire** | "What did the system do, and how long did each part take?" — spans across ingestion, retrieval, the gateway and the API |
| **LangSmith** | "What exactly did the LLM see and return at each node?" — prompt/response pairs inside the agent |

Both are optional. With no `LOGFIRE_TOKEN` the system traces to console; with no
`LANGSMITH_API_KEY`, LangChain tracing is switched off entirely rather than left pointing at a dead
endpoint (which would add a failed HTTP call to every node).

---

## The ordering rule

`configure_observability()` must run **before** importing anything that emits spans. Every
executable does this:

```python
from app.observability import configure_observability
configure_observability("bovine-rag-api")

import logfire            # noqa: E402
from app.agents.graph import rag_agent   # noqa: E402
```

The `noqa: E402` comments are not sloppiness — they mark a deliberate ordering constraint. Import
the agent first and every span emitted at module-import time is lost.

`configure_observability()` is idempotent. Streamlit re-executes its whole script on every
interaction, and calling `logfire.configure()` repeatedly leaks OpenTelemetry providers until the
process slows to a crawl.

---

## Span structure

A query produces roughly this tree:

```
Query (question, thread_id)
├── Guardrails (query)
├── Planner (query)
│   └── LLM call (feature=planner, tier=fast)
├── Retrieval (query, attempt)
│   ├── Vector search (query, limit)
│   └── [reranking logged inline]
├── Grading context (docs, refinements_used)
│   └── LLM call (feature=grader, tier=fast)      ← absent when the score short-circuits
└── Answer synthesis (intent, passages)
    └── LLM call (feature=responder, tier=quality)
```

Ingestion:

```
Ingestion run (files, dry_run)
└── Ingest file (filename)
    ├── PDF extraction (filename)
    ├── Chunking (chars, pages)
    └── Embed batch (provider, size, progress)   × N
```

`rag_agent.invoke()` is called **synchronously** in `main.py`. LangGraph's async path runs nodes in
a different context, which detaches the span tree and makes the trace unreadable.

---

## Reading the traces

| Question | What to look for |
|---|---|
| Why was this answer wrong? | The `Retrieval` span's `sources` and `top_score`. A low top score means retrieval failed, not generation |
| Why is it slow? | Span durations. A first query includes the one-time FlashRank model download |
| Am I burning quota? | Count `LLM call` spans per query. Expect 2 |
| Is fallback firing? | `Answered by fallback target '...'` warnings. Frequent ones mean your primary key is saturated |
| Is the cache working? | `Gateway cache hit` and `Embedding cache served N/M texts` |

---

## Structured logging conventions

Logfire templates use named placeholders so values stay queryable rather than being baked into a
string:

```python
logfire.info("Retrieved {n} candidates", n=len(results), top_score=0.84)   # queryable
logfire.info(f"Retrieved {len(results)} candidates")                        # not queryable
```

The first form lets you filter on `top_score < 0.3` across every query ever made. The second gives
you text to grep.

---

## Health endpoint

`GET /health` is the fastest diagnosis path — it reports each dependency separately, so a failure
is locatable rather than just "degraded":

```json
{
  "status": "healthy",
  "qdrant":     { "collection": "bovine_disease_rag", "points": 812, "vectors_dim": 3072 },
  "embeddings": { "provider": "gemini", "model": "models/gemini-embedding-001", "dim": 3072,
                  "cache": { "rows": 812, "hits": 40, "hit_rate": 0.83 } },
  "guardrails": { "mode": "fast", "nemo_tier": "disabled" },
  "llm":        { "cache": { "hits": 3, "hit_rate": 0.21 },
                  "chains": { "quality": ["groq-primary/quality", "groq-fallback/quality", ...] } }
}
```

`status: degraded` with a `hint` field means the collection is empty or unreachable — the hint
tells you the command to run.
