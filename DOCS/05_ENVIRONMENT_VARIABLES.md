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

## Qdrant write tuning

| Variable | Default | Notes |
|---|---|---|
| `QDRANT_UPSERT_BATCH` | `24` | Points per upsert. A 3072-dim vector is ~12 KB before payload, so 64 points is an ~800 KB write — enough to time out a throttled free cloud cluster. Raise it for local Docker |
| `QDRANT_TIMEOUT` | `120` | Seconds. Writes also retry 3× and halve the batch on timeout before giving up |

---

## Embedding pipeline

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_PROVIDER` | `auto` | `auto` tries Gemini then falls back to a local model. `gemini` fails loudly instead. `local` never touches an API |
| `GEMINI_EMBEDDING_MODEL` | *(blank)* | Pins one embedding model. Blank means probe `GEMINI_EMBEDDING_CANDIDATES` in order and use the first your key can reach — model availability differs per account, which is why this is probed rather than assumed |
| `GEMINI_EMBEDDING_CANDIDATES` | *(from `.env`)* | Comma-separated, ordered probe list. Only consulted when `GEMINI_EMBEDDING_MODEL` is blank. Both blank = the Gemini embedding tier is skipped entirely |
| `LOCAL_EMBEDDING_MODEL` | *(from `.env`)* | Offline `sentence-transformers` fallback. Downloads its weights on first use, then runs offline forever. Required unless `EMBEDDING_PROVIDER = gemini` |
| `EMBED_BATCH_SIZE` | `16` | Texts per Gemini request. Auto-halves on a batch-size rejection |
| `EMBED_MAX_RPM` | `90` | **Texts per minute, not requests per minute.** Gemini charges `embed_content_free_tier_requests` per *text*: a batch of 16 costs 16 units, not 1. The free ceiling is 100. Keep this below it |
| `EMBED_MAX_RETRIES` | `5` | Retries on 429. The provider's own `retryDelay` (typically ~55s) is parsed from the error and honoured — computed backoff alone tops out near 17s and just retries inside the same blocked minute |
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

**Every model name in this project comes from `.env`. None is hardcoded anywhere in the source —
not as a constant, not as a default, not as a fallback.** Providers rename and decommission
checkpoints on their own schedule; a literal baked into the code turns that into a 404 at request
time instead of one line to edit. `.env.example` carries working values to copy.

The trade-off is that a blank variable genuinely disables that tier rather than silently
substituting something. `python -m scripts.doctor` prints the resolved names and flags the blanks
before you spend a single call, and `/health` reports them under `config.models`.

| Variable | Used by | Blank means |
|---|---|---|
| `GROQ_PRIMARY_MODEL` | responder, advisor (`tier="quality"`) | quality tier dropped from the ladder |
| `GROQ_FAST_MODEL` | planner, grader, clarifier, translator (`tier="fast"`) | fast tier dropped from the ladder |
| `GROQ_TRANSLATE_MODEL` | the dedicated translate target | falls back to `GROQ_FAST_MODEL` |
| `GEMINI_CHAT_MODEL` | last-resort chat fallback | no cross-provider fallback |
| `GEMINI_EMBEDDING_MODEL` / `GEMINI_EMBEDDING_CANDIDATES` | ingestion + query vectors | Gemini embedding tier skipped |
| `LOCAL_EMBEDDING_MODEL` | offline embedding fallback | offline fallback unavailable |
| `RERANKER_MODEL` | FlashRank cross-encoder | FlashRank's own default checkpoint (the library owns that name) |
| `JUDGE_MODEL` | RAGAS judge | falls back to `GROQ_FAST_MODEL` |
| `EVAL_EMBEDDING_MODEL` | RAGAS embedding metrics | `python -m evals.metrics` refuses to run |

Groq deprecates models periodically. If you see `model_not_found` or `decommissioned`, check
https://console.groq.com/docs/models and update `.env` — the router treats such errors as fatal for
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
