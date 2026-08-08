# 05 · Environment Variables

Every variable is read in exactly one place: `app/config.py`. Nothing else in the codebase calls
`os.getenv()` directly, so there is one place to look when a knob misbehaves.

`Settings.validate(scope)` checks them per entry point (`ingestion` / `api` / `evals`) and returns
readable problems instead of letting the app die three layers deep in an SDK.

---

## Required

| Variable | Purpose | Where to get it |
|---|---|---|
| `GEMINI_API_KEY` | Embeddings (and last-resort chat fallback) | https://aistudio.google.com/apikey |
| `GROQ_API_KEY` | Primary reasoning model | https://console.groq.com/keys |
| `QDRANT_CLUSTER_ENDPOINT` | Vector DB URL | `http://localhost:6333` for Docker |

## Strongly recommended

| Variable | Purpose |
|---|---|
| `GROQ_FALLBACK_API_KEY` | A second Groq account. This is a genuine second free quota — the router falls back to it when the first is rate-limited. Setting it to the same value as `GROQ_API_KEY` buys nothing and the router detects and skips it |
| `QDRANT_API_KEY` | Required for Qdrant Cloud; leave blank for local Docker |
| `LOGFIRE_TOKEN` | Distributed tracing. Without it, tracing goes to console only |

---

## Embedding pipeline

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_PROVIDER` | `auto` | `auto` tries Gemini then falls back to a local model. `gemini` fails loudly instead. `local` never touches an API |
| `GEMINI_EMBEDDING_MODEL` | *(blank)* | Blank means probe `gemini-embedding-001` → `text-embedding-004` → `embedding-001` and use the first your key can reach. Model availability differs per account, which is why this is probed rather than assumed |
| `LOCAL_EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | Downloads ~420 MB on first use, then runs offline forever |
| `EMBED_BATCH_SIZE` | `16` | Texts per Gemini request. Auto-halves on a batch-size rejection |
| `EMBED_MAX_RPM` | `90` | Client-side ceiling. Free tier allows ~100 — staying under deliberately is cheaper than discovering the limit by failing |
| `EMBED_MAX_RETRIES` | `5` | Exponential backoff with jitter on 429/quota errors |
| `EMBEDDING_CACHE_ENABLED` | `true` | **Leave this on.** It is what makes re-ingestion free |
| `EMBEDDING_CACHE_PATH` | `.cache/embeddings.sqlite3` | Keyed by provider + model + dimension + text |

> Changing `GEMINI_EMBEDDING_MODEL` or `EMBEDDING_PROVIDER` after ingesting requires
> `--wipe`. The vector dimension is baked into the Qdrant collection, and the code refuses to mix
> dimensions rather than corrupting the index.

---

## Chunking

| Variable | Default | Notes |
|---|---|---|
| `CHUNK_SIZE` | `1400` | Characters. Roughly one to two paragraphs of a research paper |
| `CHUNK_OVERLAP` | `200` | Must be less than `CHUNK_SIZE`; clamped to `CHUNK_SIZE // 5` if not |
| `MIN_CHUNK_CHARS` | `120` | Fragments below this are dropped — they are page numbers and footnote markers |

Changing any of these requires a `--wipe` re-ingest to take effect on existing documents.

---

## Retrieval

| Variable | Default | Notes |
|---|---|---|
| `RETRIEVAL_TOP_K` | `20` | Candidates from Qdrant. Raise for recall, at reranking cost |
| `RERANK_TOP_N` | `5` | Kept after the cross-encoder. Must be ≤ `RETRIEVAL_TOP_K` (validated) |
| `MIN_RELEVANCE_SCORE` | `0.0` | Cosine floor. `0.0` disables it. Try `0.3` if you see obviously unrelated passages |

---

## Agent

| Variable | Default | Notes |
|---|---|---|
| `ENABLE_SELF_CORRECTION` | `true` | Adds the grader node and the retrieval cycle. `false` compiles the linear graph — useful for A/B measuring the loop's value |
| `MAX_REFINEMENTS` | `1` | Retrieval passes beyond the first. `1` means at most two |
| `MAX_CONTEXT_CHARS` | `18000` | Context budget. Lower it if you hit Groq TPM limits on long answers |
| `GUARDRAILS_MODE` | `fast` | `off` / `fast` / `full`. See [08](08_GUARDRAILS.md) |
| `LLM_CACHE_ENABLED` | `true` | In-process response cache |
| `LLM_CACHE_TTL` | `900` | Seconds |

---

## Models

| Variable | Default |
|---|---|
| `GROQ_PRIMARY_MODEL` | `llama-3.3-70b-versatile` |
| `GROQ_FAST_MODEL` | `llama-3.1-8b-instant` |
| `GEMINI_CHAT_MODEL` | `gemini-2.5-flash` |

Groq deprecates models periodically. If you see `model_not_found` or `decommissioned`, check
https://console.groq.com/docs/models and update these — the router treats such errors as fatal for
that target and moves straight to the next one, so the system keeps working while you fix it.

---

## Observability

| Variable | Default | Notes |
|---|---|---|
| `LOGFIRE_TOKEN` | *(blank)* | Blank = console tracing only |
| `LOGFIRE_ENVIRONMENT` | `dev` | Tag for separating dev from demo traces |
| `LANGSMITH_TRACING` | `true` | Only takes effect when `LANGSMITH_API_KEY` is also set |
| `LANGSMITH_API_KEY` | *(blank)* | |
| `LANGSMITH_PROJECT` | `bovine-disease-rag` | |
| `LANGSMITH_ENDPOINT` | `https://api.smith.langchain.com` | |

---

## UI and evals

| Variable | Default | Notes |
|---|---|---|
| `BACKEND_URL` | `http://localhost:8000` | Used by the UI and both eval harnesses |
| `JUDGE_GROQ` | *(blank)* | A third Groq key for the RAGAS judge. Scoring 16 samples across 5 metrics is ~80 model calls — enough to exhaust the quota your live app is using. Falls back to `GROQ_API_KEY` |

---

## Optional — Portkey

| Variable | Default |
|---|---|
| `ENABLE_PORTKEY` | `false` |
| `PORTKEY_API_KEY` | *(blank)* |

**Not required.** This project ships its own gateway with equivalent behaviour. See
[09_LLM_GATEWAY.md](09_LLM_GATEWAY.md).

---

## Notes on format

The `.env` uses `KEY = "value"` with spaces and quotes. `python-dotenv` handles that, and
`config.py` strips whitespace and quotes again defensively — a stray quote character silently
included in an API key produces a 401 that looks like a wrong key.
