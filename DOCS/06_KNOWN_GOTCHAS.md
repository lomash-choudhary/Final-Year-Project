# 06 · Known Gotchas

Non-obvious failures, and the reasoning behind decisions that look odd until you know why.

---

## 1. Groq has no embeddings endpoint

The most common misreading of the free-tier plan. Groq serves chat/completion models only. So
"Gemini first, then Groq" cannot apply to embeddings — the two ladders are separate:

```
EMBEDDINGS   Gemini key 1  →  Gemini key 2 (same model)  →  local sentence-transformers
REASONING    Groq key 1 · GROQ_PRIMARY_MODEL  →  Groq key 2 · GROQ_PRIMARY_MODEL
          →  Groq key 1 · GROQ_FAST_MODEL     →  Groq key 2 · GROQ_FAST_MODEL
          →  Gemini · GEMINI_CHAT_MODEL
```

Your `GROQ_FALLBACK_API_KEY` is a real second quota, but only on the reasoning side. The
embedding side has its own: `GEMINI_FALLBACK_API_KEY` runs the **same** `GEMINI_EMBEDDING_MODEL`
(same vector space, same dimension, same cache entries), so swapping keys mid-run is safe in a way
swapping models never is. A batch spends `EMBED_MAX_RETRIES` retries on the active key, then moves
to the next key, which stays active for the rest of the run. Each key has its own rate limiter.
The free quota belongs to a Google Cloud *project* — a second key from the same project shares it
and buys nothing.

---

## 2. One embedding model, dimension probed, backend locked after init

`GEMINI_EMBEDDING_MODEL` is a single pinned name with **no fallback list**. That is deliberate:
the Gemini embedding family mixes widths across generations — 3072 for `gemini-embedding-001`,
768 for `text-embedding-004` and `embedding-001` — so "try the next model" is not graceful
degradation, it is a second vector space written into the same collection. If the configured
model is unreachable, the Gemini tier is disabled and the decision passes to `_init_backend()`,
which either switches to the local model (logged, dimension-checked) or raises.

The *dimension* is still measured rather than hardcoded — embed one short string, read `len()` —
because a model can change its output width between versions, and a hardcoded 3072 is how a
collection ends up silently rejecting every upsert.

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

`messages` uses `operator.add` so nodes can append replies. `plan` does not: it is one turn's
reasoning. When the graph had a `MemorySaver` checkpointer, an accumulating plan would have shown
turn 1 and 2's steps on turn 3. The graph now has no checkpointer (memory is seeded per request —
§24), but the rule stands: nodes concatenate explicitly with `state.get("plan", []) + [...]`, and
`translate_in` resets it.

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

There is also a **daily** cap of roughly 1000 texts per Google Cloud project. The ~2200-chunk corpus
cannot be embedded from scratch on one key in one day — that is what `GEMINI_FALLBACK_API_KEY` and
the embedding cache are for. Adding 1474 chunks on 2026-10-01 exhausted the primary key's day after
~950 texts; the run failed over to the fallback key and finished without intervention.

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
it produces a cold cache and a full re-embed. That is correct: a vector from the hosted embedding
model is meaningless to the local one. The `model` component is whatever name config resolved to,
which is why changing `GEMINI_EMBEDDING_MODEL` in `.env` invalidates the cache automatically.

---

## 21. No model name is hardcoded anywhere — `.env` is the only source

Not as a constant, not as a `_str(..., "default")`, not as a fallback in an `except` branch. Every
model identifier in the system — `GROQ_PRIMARY_MODEL`, `GROQ_FAST_MODEL`, `GROQ_TRANSLATE_MODEL`,
`GEMINI_CHAT_MODEL`, `GEMINI_EMBEDDING_MODEL`,
`LOCAL_EMBEDDING_MODEL`, `RERANKER_MODEL`, `JUDGE_MODEL`, `EVAL_EMBEDDING_MODEL` — is read from
`.env` through `settings`.

**Why.** Model names are provider inventory, not application logic. Groq decommissions checkpoints
on a few weeks' notice; Google renames embedding models between releases. A literal baked into the
source turns that announcement into a 404 at request time, discovered by a user, in a file nobody
thought to grep. Worse, a *default* hides the problem: the variable looks configurable, but a typo
in `.env` silently falls back to a name that may itself be dead.

**The trade-off, stated plainly.** A blank variable now genuinely disables that tier instead of
substituting something. That is the point — it fails visibly:

- `_build_chain()` drops any target whose model is blank, and logs an error if that empties the
  ladder; `chain()` then raises `AllTargetsFailed`, which every node already handles by degrading.
- `_build_gemini()` returns `None` (skipping the tier) when no embedding candidate is configured;
  `_build_local()` raises `EmbeddingError`.
- `evals.metrics` refuses to start without a judge model and an eval embedding model.
- `settings.validate()` reports every blank name for the relevant scope, so
  `python -m scripts.doctor` catches it before an API call is spent. The doctor prints the full
  resolved set under **Models**, and `/health` reports it as `config.models`.

Two names fall back to *another configured value*, never to a literal: `GROQ_TRANSLATE_MODEL` and
`JUDGE_MODEL` both fall back to `GROQ_FAST_MODEL`. One name is deliberately absent:
`RERANKER_MODEL` left blank hands the choice to FlashRank's own default, because that checkpoint
belongs to the library, not to this project.

When a model dies, the fix is one line in `.env` and a restart — no code change, no redeploy of
logic, no grep.

---

## 22. `GROQ_FAST_MODEL` must not be a reasoning model

The fast tier runs the mechanical nodes — planner, grader, clarifier, translator — and each caps
`max_tokens` tightly, because these nodes emit two or three lines of structured text and fire on
every single query:

| Node | `max_tokens` | Expected reply |
|---|---|---|
| grader | 160 | `VERDICT: …` / `QUERY: …` |
| planner | 180 | `INTENT: …` / `QUERY: …` |
| clarifier | 300 | follow-up questions |
| translator | 400 | translated text |

A reasoning model spends that budget on its hidden channel *before* emitting any content. Measured
on the real planner prompt at its real cap of 180:

```
openai/gpt-oss-20b    finish_reason="length"  completion_tokens=180  content=""
qwen/qwen3.8-27b      finish_reason="stop"    completion_tokens=19   content="INTENT: RESEARCH\nQUERY: …"
```

**Why this is worse than a crash.** The `_parse` helpers are deliberately tolerant (gotcha 5 in
`AGENTS.md §7`): given an empty string the planner returns `("research", raw_query)` and the grader
returns its default verdict. Nothing raises, nothing logs an error, and `AllTargetsFailed` never
fires because the call *succeeded*. The graph keeps answering — with the planner and grader
silently reduced to constants on every request. There is no signal in `/health`, in
`thought_process`, or in the Logfire span tree.

`GROQ_PRIMARY_MODEL` has no such constraint: the responder and advisor pass `max_tokens=None`, so a
reasoning model is a good choice there and is what the quality tier uses.

**Before setting a new `GROQ_FAST_MODEL`**, confirm it returns a parseable body within 160 tokens:

```bash
curl -s https://api.groq.com/openai/v1/chat/completions \
  -H "Authorization: Bearer $GROQ_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"<candidate>","messages":[{"role":"user","content":"Reply with the single word: ok"}],"max_tokens":160}' \
  | jq '.choices[0] | {finish_reason, content: .message.content}'
```

`finish_reason: "length"` with an empty `content` disqualifies the model for the fast tier.

Note that Groq's `GET /v1/models` and `/chat/completions` sit behind Cloudflare, which returns
`403 error code: 1010` to clients sending no `User-Agent` (Python's bare `urllib` among them). That
403 is a client-fingerprint rejection, **not** an invalid key — resend with a normal `User-Agent`
before concluding anything about your credentials.

---

## 23. A non-empty page can still be unusable text

The PDF cascade originally fell through to the next tier only on an **empty** page. An OCR'd
archive.org scan extracted through pypdf with every space missing — 72 of 82 pages arrived as one
token each — and passed every check, because it was not empty. It would have embedded 122 chunks
of noise. The loader now scores each page's "glued" share (characters inside >25-char tokens) and
re-extracts pages above 0.5 with PyMuPDF. Run `--dry-run` on new PDFs and look at the text: the
check is cheap, and this class of defect is invisible in the run summary.

---

## 24. Conversation memory: bounded, persistent, and never the whole history

The graph used to remember conversations with LangGraph's `MemorySaver`. Two problems:

- **It lived in RAM.** Every restart — and every Render free-tier sleep — wiped it, so in
  production the assistant effectively had no memory, and there was nothing for the frontend to
  show as history.
- **It rotted.** `messages` grew without bound, and every node's full state (retrieved passages
  included) was checkpointed on every step. Nodes only read the last six messages, so older turns
  were silently dropped rather than summarised.

Memory is now Postgres (`app/memory/store.py`) and the graph has no checkpointer. The model sees
`rolling summary + last MEMORY_WINDOW_TURNS turns`, so prompt size is flat. Things that look like
optimisations but break it:

- **Loading the full history into the prompt.** Long chats would blow the Groq TPM budget and
  dilute the current question. The window is the cap; the summary carries the rest.
- **Storing passages or the plan as memory.** That is the old bloat. Sources are kept per message,
  trimmed to 300 characters, for display only.
- **Summarising on the request path.** It is a background task; a failed summary keeps the window
  and retries next turn.
- **Health-checking pooled connections on checkout.** Neon can be 100–250 ms away; a check per
  checkout doubles every call. The pool closes idle connections before Neon's 5-minute suspend
  (`max_idle=240`) and retries once on `OperationalError` instead. Every store call is one
  statement — the turn upsert + both message inserts are a single CTE.
- **Server-side prepared statements on Neon's pooled endpoint.** It is PgBouncer in transaction
  mode; `prepare_threshold=None` keeps psycopg from preparing.
- **Fixed eval thread ids.** Memory now survives across runs, so `eval-<id>` would replay the
  previous run's turns. The eval harnesses append a per-run tag.

Ownership is by `user_id` on every read and write. The frontend's auth is still a mock token, so
this separates users but is not a security boundary.
