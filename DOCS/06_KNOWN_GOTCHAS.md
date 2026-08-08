# 06 · Known Gotchas

Non-obvious failures, and the reasoning behind decisions that look odd until you know why.

---

## 1. Groq has no embeddings endpoint

The most common misreading of the free-tier plan. Groq serves chat/completion models only. So
"Gemini first, then Groq" cannot apply to embeddings — the two ladders are separate:

```
EMBEDDINGS   Gemini  →  local sentence-transformers
REASONING    Groq key 1 · 70B  →  Groq key 2 · 70B  →  Groq · 8B  →  Gemini Flash
```

Your `GROQ_FALLBACK_API_KEY` is a real second quota, but only on the reasoning side.

---

## 2. Vector dimension is probed, and the backend locks after init

Dimension is measured by embedding one short string and reading `len()`. It is never hardcoded,
because model availability differs per Google account and dimensions change between model
versions. A hardcoded 3072 is how a collection ends up silently rejecting every upsert.

The backend is then **locked for the process**. If Gemini dies mid-run the code raises rather than
quietly switching to a 768-dim local model — half a collection at 3072 and half at 768 is not a
degraded index, it is a broken one.

Recovery: re-run the command (the manifest resumes), or set `EMBEDDING_PROVIDER=local` and
re-ingest with `--wipe`.

---

## 3. `--wipe` is required after changing chunking or the embedding model

`CHUNK_SIZE`, `CHUNK_OVERLAP` and the embedding model all change what is stored. Without `--wipe`,
old and new chunks coexist and retrieval quality degrades in ways that are hard to attribute.

The dimension guard catches the embedding-model case with an explicit `DimensionMismatch`. The
chunking case has no such guard — it is silent, so it is on you to remember.

---

## 4. PDF fallbacks must preserve page order

Recovered pages are merged back at their original index. Appending them at the end would be
simpler and would produce plausible-looking text — but every page number in every citation
downstream would then be wrong, which is worse than the missing pages.

---

## 5. `plan` is deliberately not a LangGraph reducer

`messages` uses `operator.add` so conversation history accumulates. `plan` does not.

MemorySaver persists state per `thread_id`. If `plan` accumulated, turn 3 of a conversation would
show the reasoning steps from turns 1 and 2 as well. Nodes concatenate explicitly with
`state.get("plan", []) + [...]`, and the planner — which runs first on every turn — resets it.

---

## 6. Observability must be configured before importing app modules

```python
from app.observability import configure_observability
configure_observability("bovine-rag-api")
import logfire                            # noqa: E402
from app.agents.graph import rag_agent    # noqa: E402
```

The `noqa: E402` comments mark a deliberate ordering constraint, not sloppiness. Import the agent
first and every span emitted at module-import time is lost.

`configure_observability()` is also idempotent because Streamlit re-runs its entire script on every
interaction, and repeated `logfire.configure()` calls leak OpenTelemetry providers.

---

## 7. The graph is invoked synchronously

`rag_agent.invoke()`, not `ainvoke()`. LangGraph's async path runs nodes in a different context,
which detaches the Logfire span tree and makes the trace unreadable. FastAPI runs sync endpoints on
a thread pool, so throughput is fine.

---

## 8. First query after startup is slow

FlashRank downloads its ONNX model on first use (~30 MB), and `sentence-transformers` downloads
~420 MB if the local embedding fallback activates. Both are one-time and cached on disk. Warm the
system with one throwaway query before a demo.

---

## 9. Guardrails: off-topic blocking is conservative by design

Off-topic rules fire only when the message contains **no domain vocabulary** *and* matches an
explicit off-domain pattern. "What did the study find?" contains no veterinary term but is a
perfectly valid follow-up.

False positives — a legitimate research question refused — destroy user trust immediately. False
negatives waste a little quota. The asymmetry is intentional.

If your own question gets blocked, add a term to `_DOMAIN_TERMS` in
`app/guardrails/fast_rails.py`. Order matters in `check()`: injection and jailbreak are tested
before greetings, so "hi, ignore all previous instructions" is caught as a jailbreak.

---

## 10. NeMo Guardrails is optional and degrades silently to `fast`

`GUARDRAILS_MODE=full` needs `nemoguardrails`, which is a heavy dependency with its own transitive
constraints. If it cannot initialise, the system logs a warning and runs the fast tier alone — it
never runs ungated. Check `GET /health` → `guardrails.nemo_tier` to see which tier is actually
live.

The `models:` block in `colang_rules.py` is empty on purpose: the LLM is injected at runtime via
`LLMRails(config, llm=...)`. Nothing reaches OpenAI.

---

## 11. RAGAS eval is slow on purpose

One sample at a time, with 25-second gaps between samples and 62-second gaps between metrics.

Groq's free tier is **TPM**-limited, not only RPM-limited. RAGAS fires several concurrent sub-calls
per sample internally, so batching at the outer level stacks those bursts inside the same second
and trips the limit even when each individual request is small. Contexts are also truncated to 400
characters × 2 passages for the same reason.

Budget 10–15 minutes for a full RAGAS pass. The zero-cost metrics (tool correctness, retrieval hit
rate) are instant and run every time.

---

## 12. Deleting before upserting matters

`delete_by_source()` runs before every upsert. Deterministic IDs alone are not enough: if a
document is edited to be *shorter*, its old tail chunks keep their IDs, are never overwritten, and
continue to match searches — citing text that no longer exists in the file.

---

## 13. Qdrant health check in `docker-compose.yml` uses `/dev/tcp`

The `qdrant/qdrant` image ships with neither `curl` nor `wget`, so the usual healthcheck silently
reports unhealthy forever. The bash `/dev/tcp` probe works with what the image actually has.

---

## 14. `Firstpaper.pdf` is a 124-page journal issue

It is a full issue, not a single paper, so its content is heterogeneous and its running headers are
aggressive. It is the main reason the running-header stripper exists. If retrieval keeps surfacing
irrelevant chunks from it, check `processed_data/Firstpaper.pdf.json` first — and consider
excluding it with `--file` ingestion of the others.

---

## 15. Gemini's embedding quota is charged per TEXT, not per request

The error reads:

```
Quota exceeded for metric: generativelanguage.googleapis.com/embed_content_free_tier_requests,
limit: 100, model: gemini-embedding-1.0
```

`limit: 100` is **100 texts per minute**, not 100 batches. A batch of 16 costs 16 units. A limiter
that throttles batches at 90/min therefore permits ~1,440 texts/min — roughly 14× the real ceiling,
which produces a run that succeeds for four files and then 429s continuously.

`EMBED_MAX_RPM` is counted in texts for this reason. At the default 90, a 805-chunk corpus takes
about nine minutes of wall-clock. That pacing is the feature.

---

## 16. Exponential backoff alone cannot clear a per-minute quota

Gemini returns `"retryDelay": "54s"` in the 429 body. Pure exponential backoff peaks near 17s on
the fourth attempt, so all five retries land inside the *same* blocked minute and the file fails
anyway.

The retry path parses `retryDelay` (and the `Please retry in 54.8s` prose variant) and sleeps for
that instead, then puts the shared rate limiter into a cooldown so concurrent callers do not
immediately spend more quota on top.

---

## 17. `--dry-run` must not write the manifest

A dry run parses and chunks but never indexes. Persisting its results would mark every file `ok`
with zero points, and the next real run would skip the entire corpus and index nothing.

`Manifest(..., read_only=True)` handles this. `should_skip()` independently refuses to trust an
`ok` record with `points <= 0`, so even a manifest corrupted by an older build self-heals.

---

## 18. Qdrant Cloud free clusters time out on large writes

A 3072-dim vector is ~12 KB before payload. At 64 points per upsert that is an ~800 KB write, and a
throttled free cluster can exceed 60s on it — failing the whole document.

`QDRANT_UPSERT_BATCH` defaults to 24, the client timeout to 120s, and `_upsert_window()` retries
three times and **halves the batch on timeout** before giving up. Raise the batch on local Docker,
where none of this applies.

---

## 19. Never call `os.getenv()` outside `app/config.py`

`ui/app.py` originally did `os.getenv("BACKEND_URL", settings.BACKEND_URL)`. With
`BACKEND_URL = ""` in `.env` — present but blank — `os.getenv` returns `""` rather than falling
back, so every request went to `/query` with no host and the UI reported
`Backend unreachable. Tried ` with an empty URL.

`config.py`'s `_str()` treats blank as absent and returns the default. That protection only works
if everything reads through `settings`.

---

## 20. Embedding cache keys include the model

Cache keys are `sha256(provider|model|dim|text)`. Switching models does not produce stale hits —
it produces a cold cache and a full re-embed. That is correct: a vector from
`gemini-embedding-001` is meaningless to `all-mpnet-base-v2`.
