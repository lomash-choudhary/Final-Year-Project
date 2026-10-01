# AGENTS.md

Working guide for AI coding agents (and humans) on this repository.
`CLAUDE.md` contains only `@AGENTS.md` — never add content there; this file is the single source of truth for *how to work*.
`MEMORY.md` holds *project state* — read it at the start of every session and update it at the
end (see §12).

There is exactly **one** `AGENTS.md`, **one** `CLAUDE.md` and **one** `MEMORY.md`, all at the repo
root. Do not create per-directory copies or tool-specific variants (`.cursorrules`, `GEMINI.md`,
…); point them here instead. (`.kilo/worktrees/*` are tool-managed git worktrees — their copies are
checkouts of these same files, not separate sources.)

---

## 1. What this project is

**Bovine Disease Research Assistant** — an agentic RAG system over 16 peer-reviewed papers on
cattle and buffalo disease (haemoprotozoal infections, brucellosis, lumpy skin disease, foot and
eye disorders, genetic disorders, *E. coli*, dairy-herd health).

It serves **two audiences through one LangGraph graph**:

| Audience | Example input | Path | Output shape |
|---|---|---|---|
| Researcher | "prevalence of theileriosis in India" | planner → retriever → grader → **responder** | grounded prose with inline `[n]` citations + page-accurate sources |
| Farmer | "meri gaay khana nahi kha rahi" | planner → clarifier → retriever → grader → **advisor** | plain-language *What this looks like / What to do now / Watch for* + `care_level` badge |

Hard constraint that explains most design decisions: **everything must run on free tiers.**
Gemini for embeddings, Groq for reasoning (multiple keys chained), Qdrant local/free-cloud,
FlashRank + sentence-transformers locally. No paid API anywhere.

Input may be English, Devanagari Hindi, or Roman-script Hindi ("Hinglish"). The graph is
bracketed by translation nodes so **everything between `translate_in` and `translate_out` is
English only**.

---

## 2. Repository layout

```text
app/
├── main.py                       FastAPI: /query /health /sources /stats /graph /graph/mermaid
├── config.py                     EVERY tunable. The only place os.getenv() is allowed.
├── observability.py              Logfire + LangSmith bootstrap (must run before other imports)
├── agents/
│   ├── graph.py                  LangGraph wiring, routers, MemorySaver checkpointer
│   ├── state.py                  AgentState TypedDict + reducer decisions
│   └── nodes/
│       ├── translator.py         translate_in / translate_out + free language detection
│       ├── planner.py            intent classification (conversational|symptom|research) + query rewrite
│       ├── clarifier.py          farmer follow-up questions, cross-turn wait
│       ├── retriever.py          Qdrant search → FlashRank rerank
│       ├── grader.py             self-correction loop (re-search on weak context)
│       ├── responder.py          researcher answer, numbered context, [n] citations
│       └── advisor.py            farmer advice + care_level extraction
├── guardrails/
│   ├── fast_rails.py             deterministic regex tier — zero API cost
│   ├── colang_rules.py           NeMo Colang intents (optional tier)
│   └── rails.py                  tier orchestration + graceful degradation
├── llm/router.py                 LLM gateway: fallback ladder, retries, TTL response cache
├── ingestion/
│   ├── processor.py              CLI: load → clean → chunk → JSON → embed → index
│   ├── manifest.py               incremental re-ingestion, resume, dedup bookkeeping
│   ├── cleaning.py               ligatures, de-hyphenation, running-header removal
│   ├── chunking/splitter.py      3-tier chunker + page attribution by offset mapping
│   └── loaders/                  pdf.py (3 tiers, per page) · html · text · office · base
└── services/retrieval/
    ├── embedding.py              Gemini → local fallback, rate limiter, dimension probe
    ├── embedding_cache.py        sqlite3 on-disk vector cache
    ├── qdrant_service.py         collection lifecycle, deterministic IDs, search
    └── ranking_service.py        FlashRank cross-encoder singleton

evals/                            golden_dataset.json · pipeline.py · metrics.py · guardrails_eval.py · app.py
ui/app.py                         Streamlit chat UI
scripts/doctor.py                 preflight check (no API calls by default)
DOCS/01..10                       deep-dive docs — 06_KNOWN_GOTCHAS.md is the important one
DATA/                             the corpus (16 PDFs)
processed_data/                   generated: parsed+chunked JSON per document (gitignored)
ingestion_manifest.json           generated: ingestion state (gitignored)
.cache/embeddings.sqlite3         generated: embedding cache (gitignored)
```

---

## 3. Commands

Always work inside the venv: `source .venv/bin/activate`. `make help` lists shortcuts.

```bash
# setup
make setup                                   # venv + pip install -r requirements.txt
cp .env.example .env                         # then fill keys

# infrastructure
docker compose up -d                         # local Qdrant on :6333 (make qdrant)
docker compose down                          # stop, keep data;  down -v deletes vectors

# preflight
python -m scripts.doctor                     # config + Qdrant + corpus + parsers, 0 API calls
python -m scripts.doctor --live              # + probes keys (spends ~2 calls)

# ingestion
python -m app.ingestion.processor --dry-run  # parse + chunk only, 0 quota, writes processed_data/
python -m app.ingestion.processor --wipe     # drop collection and rebuild (first run / config change)
python -m app.ingestion.processor            # skip unchanged files (safe to re-run, resumes)
python -m app.ingestion.processor --force    # re-ingest everything, keep collection
python -m app.ingestion.processor --limit 2
python -m app.ingestion.processor --file "fnx050.pdf"

# run (three terminals)
uvicorn app.main:app --reload --port 8000    # make api
streamlit run ui/app.py                      # make ui         → :8501
streamlit run evals/app.py --server.port 8502 # make evals      → :8502

# evaluation
python -m evals.pipeline                     # phase 1: replay golden set against the LIVE API
python -m evals.metrics                      # phase 2: zero-cost metrics + RAGAS (10-15 min)
python -m evals.guardrails_eval              # guardrail confusion matrix

# cleanup
make clean-index                             # rm processed_data/ manifest .cache (Qdrant untouched)
```

There is **no test suite**. Verification is: `scripts/doctor.py`, `--dry-run`, `GET /health`,
and the eval suite. Do not claim tests pass — there are none to run.

---

## 4. The request path

```
POST /query
  └─ guardrails gate (fast regex; blocked path spends ZERO model calls)
  └─ rag_agent.invoke(state, thread_id)        ← synchronous, deliberately
       translate_in    detect language; translate to English only if not English
       planner         intent + standalone search query        (fast tier)
       ├─ conversational → responder
       ├─ symptom        → clarifier → (ask & wait) or retriever
       └─ research       → retriever
       retriever       Qdrant top-20 → FlashRank top-5
       grader          cheap signals first; LLM only in the ambiguous band; may loop back once
       responder/advisor
       translate_out   translate answer + follow-ups back to the user's language
  └─ QueryResponse: answer, thought_process, sources, llm meta, language, care_level,
                    follow_up_questions, awaiting_answer
```

Everything the agent decided is returned to the caller — the plan, the passages, which LLM target
answered, whether a fallback or cache was hit. The eval harness reads `thought_process` to infer
which tool path was taken, so **changing plan strings can break `evals/pipeline.py:detect_tool`.**

---

## 5. Architectural invariants — do not break these

These are load-bearing. Each one exists because the obvious alternative failed.
Full reasoning in `DOCS/06_KNOWN_GOTCHAS.md`.

1. **`os.getenv()` only in `app/config.py`.** Everything else reads `settings`. `config._str()`
   treats a present-but-blank var as absent; `os.getenv` does not, which is how `BACKEND_URL = ""`
   once produced requests to `/query` with no host.
2. **Observability is configured before importing app modules.** The `# noqa: E402` comments in
   `main.py`, `processor.py`, `doctor.py`, `ui/app.py` mark a deliberate ordering constraint.
3. **The graph is invoked synchronously** (`invoke`, not `ainvoke`). LangGraph's async path runs
   nodes in a different context and detaches the Logfire span tree.
4. **`messages` is a reducer (`operator.add`); `plan` is not.** MemorySaver persists per
   `thread_id`, so an accumulating plan would replay every earlier turn's reasoning. Nodes
   concatenate explicitly: `state.get("plan", []) + [...]`. The planner resets it.
5. **Vector dimension is probed, never hardcoded**, and the embedding backend is **locked after
   init**. Mid-run switching from 3072-dim Gemini to 768-dim local would corrupt the collection,
   so it raises instead.
6. **Deterministic point IDs (`uuid5("<file>:<chunk_index>")`) + `delete_by_source()` before every
   upsert.** IDs alone are not enough: a document edited shorter leaves orphan tail chunks.
7. **PDF fallbacks are per page and recovered pages return to their original index.** Appending
   them would silently corrupt every page citation.
8. **Chunk strategies are validated before acceptance.** A splitter returning one 200 KB "chunk"
   embeds fine, retrieves for everything, and blows the context budget.
9. **`EMBED_MAX_RPM` counts TEXTS per minute, not requests.** Gemini's free embedding quota is
   charged per text; a batch of 16 costs 16 units.
10. **Exponential backoff alone cannot clear a per-minute quota.** The retry path parses Gemini's
    `retryDelay` (and the "Please retry in 54.8s" prose variant) and honours it, then cools down
    the shared limiter.
11. **Guardrails block off-topic only on explicit off-domain signals**, and are suppressed whenever
    any `_DOMAIN_TERMS` pattern matches. "What did the study find?" has no veterinary term but is a
    valid follow-up. False positives cost trust; false negatives cost a little quota.
12. **Rail order matters**: injection → jailbreak → greeting → farewell → capabilities → off-topic.
    "hi, ignore all previous instructions" must be caught as a jailbreak.
13. **"No evidence" is answered deterministically with zero LLM calls.** A model asked to admit
    ignorance will sometimes answer from parametric memory instead.
14. **Degradation is never silent.** `fallback_used` flows from the router through `llm_meta` to
    the UI; NeMo failing to load logs a warning and degrades to `fast` — it never runs ungated.
15. **`--dry-run` must not write the manifest** (`Manifest(..., read_only=True)`), or the next real
    run skips the whole corpus.
16. **Evals hit the live API, not the graph.** What is measured is the system as deployed.
17. **The advisor never names a prescription medicine or dose**, and red-flag signs force
    `care_level = vet_now` regardless of what the passages say. Default on a missing/unparsable
    care level is the conservative `vet_soon`.
18. **Farmer answers carry no citation markers and no sources panel** (`main.py` suppresses
    `sources` when `intent == "symptom"`); the advisor's context is deliberately unnumbered so the
    model cannot cite. `SHOW_CITATIONS_IN_ADVICE=true` puts them back.
19. **No model name is hardcoded anywhere — not even as a fallback.** Every model identifier comes
    from `.env` via `settings` (`GROQ_PRIMARY_MODEL`, `GROQ_FAST_MODEL`, `GROQ_TRANSLATE_MODEL`,
    `GEMINI_CHAT_MODEL`, `GEMINI_EMBEDDING_MODEL`,
    `LOCAL_EMBEDDING_MODEL`, `RERANKER_MODEL`, `JUDGE_MODEL`, `EVAL_EMBEDDING_MODEL`). No
    `_str("...", "some-model")` default, no literal in an `except` branch, no constant in a node
    or eval module. A blank name disables that tier **loudly**: the router drops the target and
    logs, the embedding builder skips or raises, `validate()` reports it and `scripts/doctor.py`
    prints the resolved set. The only two fallbacks allowed are onto *other configured values* —
    `GROQ_TRANSLATE_MODEL` and `JUDGE_MODEL` fall back to `GROQ_FAST_MODEL`. Adding a new model
    call means adding a new variable to `app/config.py` **and** `.env.example`.

---

## 6. LLM gateway (`app/llm/router.py`)

In-process equivalent of Portkey/LiteLLM. Do not add a hosted gateway; `ENABLE_PORTKEY` /
`PORTKEY_API_KEY` exist only as inert config.

Ladder for `tier="quality"` (reversed for `tier="fast"`, the fast model first):

```
GROQ_API_KEY + GROQ_PRIMARY_MODEL → GROQ_FALLBACK_API_KEY + GROQ_PRIMARY_MODEL
  → GROQ_API_KEY + GROQ_FAST_MODEL → GROQ_FALLBACK_API_KEY + GROQ_FAST_MODEL
  → GEMINI_API_KEY + GEMINI_CHAT_MODEL → AllTargetsFailed
```

- Groq rate-limits per **(key, model)**, so the chain exploits both axes.
- **Model names come from `.env` only** (invariant 19). A target whose model is blank is dropped
  from the ladder; if that empties it, `chain()` raises `AllTargetsFailed` after logging why.
- `feature="translate"|"clarifier"|"advisor"` prepends that stage's dedicated key
  (`settings.feature_key`) so one busy stage cannot rate-limit the others; the shared pool is still
  appended behind it.
- A `GROQ_FALLBACK_API_KEY` identical to the primary is dropped — it shares the quota.
- 2 attempts per target; `_FATAL_MARKERS` (401/404/decommissioned) skip retry entirely.
- TTL response cache keyed on `(tier, feature, temperature, messages)`.
- **Planner, grader, clarifier and translator run on `tier="fast"`.** Spending 70B quota there is
  what exhausts it before the answer is generated. Keep new cheap/mechanical nodes on `fast`.
- Every node wraps `router.invoke` in `try/except AllTargetsFailed` and **degrades rather than
  fails**. Preserve that pattern in new nodes.

---

## 7. Conventions to follow when editing

- **Style**: `from __future__ import annotations`, dataclasses, type hints, `snake_case`, 4-space
  indent, ~100–110 char lines. No formatter/linter is configured — match the surrounding file.
- **Docstrings are design documents.** Every module opens with *why it exists and what breaks
  without it*. New modules must do the same; keep existing ones updated when behaviour changes.
- **Logging is `logfire` only** — `logfire.span` around any unit of work, `logfire.info/warning/
  error` with **structured kwargs and `{placeholder}` templates**, never f-strings in the message.
  No `print()` outside `scripts/doctor.py` (which is a CLI report) and `observability.py`'s
  last-resort fallback.
- **Truncate everything that goes into a log or a prompt** (`str(exc)[:300]`, `query[:120]`).
- **New tunables go in `app/config.py` + `.env.example`**, with a comment explaining the unit and
  why the default is what it is, and a `validate(scope)` rule if a bad value is silently harmful.
- **Node contract**: a node takes `AgentState` and returns a *partial* dict of updates. Append to
  `plan`, never overwrite. If a node can emit a user-visible answer, also append to `messages`.
- **Model output is parsed defensively.** Every prompt specifies an exact reply format, and every
  parser tolerates drift (`_parse` helpers in planner/grader/clarifier/advisor, plus
  `_clean_query` stripping boolean operators and the grader's `_EXPLANATION` cut). Copy that
  posture rather than trusting the format.
- **Failure is isolated.** Ingestion isolates per file; retrieval returns `[]` rather than raising;
  guardrail and reranker outages degrade instead of taking the request down.

---

## 8. Configuration quick reference

Everything lives in `.env` (see `.env.example`, documented in `DOCS/05_ENVIRONMENT_VARIABLES.md`).

| Needed | Variable |
|---|---|
| Required | `GEMINI_API_KEY` (embeddings), `GROQ_API_KEY` (reasoning), `QDRANT_CLUSTER_ENDPOINT` |
| Strongly recommended | `GROQ_FALLBACK_API_KEY` (a real second free quota) |
| Required (models) | `GROQ_PRIMARY_MODEL`, `GROQ_FAST_MODEL`, `GEMINI_EMBEDDING_MODEL` — nothing is defaulted in code |
| Optional | `GROQ_TRANSLATE_API_KEY`, `GROQ_CLARIFIER_API_KEY`, `GROQ_ADVISOR_API_KEY`, `QDRANT_API_KEY` (cloud only), `LOGFIRE_TOKEN`, `LANGSMITH_API_KEY`, `JUDGE_GROQ` |
| Optional (models) | `GROQ_TRANSLATE_MODEL`, `GEMINI_CHAT_MODEL`, `LOCAL_EMBEDDING_MODEL`, `RERANKER_MODEL`, `JUDGE_MODEL`, `EVAL_EMBEDDING_MODEL` |
| Inert | `PORTKEY_API_KEY`, `ENABLE_PORTKEY` |

Defaults worth knowing: `CHUNK_SIZE=1400` / `CHUNK_OVERLAP=200`, `RETRIEVAL_TOP_K=20` /
`RERANK_TOP_N=5`, `MAX_REFINEMENTS=1`, `MAX_CONTEXT_CHARS=18000`, `GUARDRAILS_MODE=fast`,
`EMBED_MAX_RPM=90` (texts/min), `EMBED_BATCH_SIZE=16`, `QDRANT_UPSERT_BATCH=24`,
`MAX_CLARIFICATION_ROUNDS=1`, `LLM_CACHE_TTL=900`.

Feature flags that change the compiled graph shape — `ENABLE_SELF_CORRECTION`,
`ENABLE_CLARIFICATION`, `ENABLE_TRANSLATION`. The first two add/remove nodes at build time, so
changing them requires a process restart.

`.env.prod` holds the deployed backend's values. `requirements-prod.txt` is the API-only install
(~¼ the size) — it deliberately excludes `sentence-transformers`/PyTorch and the eval stack,
because the deployed server never ingests and never runs RAGAS. **Ingestion runs on a laptop
against the same Qdrant cluster.**

---

## 9. Changing things safely

| If you change… | Then you must… |
|---|---|
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, chunker logic | re-ingest with `--wipe` (no guard catches this — it is silent) |
| the embedding model or provider | re-ingest with `--wipe`; `DimensionMismatch` will otherwise stop you |
| loaders or cleaning | `--dry-run` first, then read `processed_data/<file>.json` before spending quota |
| plan strings in nodes | check `evals/pipeline.py:detect_tool` still classifies correctly |
| a model that has been decommissioned | edit `.env` only — no source change; `python -m scripts.doctor` prints the resolved names |
| guardrail regexes | run `python -m evals.guardrails_eval` — it reports FP/FN separately |
| prompts in nodes | re-check the corresponding `_parse` helper still matches the required format |
| anything in the graph | `GET /graph/mermaid` renders the compiled shape without network access |

**Before any real ingestion run**: `python -m scripts.doctor` → `--dry-run` → inspect one
`processed_data/*.json`. Reading the parsed text is the fastest way to catch a badly-extracted PDF,
and it costs nothing.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Cannot reach Qdrant` | `docker compose up -d`, or check `QDRANT_CLUSTER_ENDPOINT` |
| `/health` shows 0 points | not ingested: `python -m app.ingestion.processor --wipe` |
| `DimensionMismatch` | embedding model changed → re-ingest with `--wipe` |
| `All LLM targets exhausted` | every Groq key rate-limited; wait for the window or add `GROQ_FALLBACK_API_KEY` |
| Planner/grader always take the default path | `GROQ_FAST_MODEL` is a reasoning model: it burns the node's `max_tokens` cap on its hidden channel and returns empty content, which the tolerant `_parse` helpers absorb silently. See `DOCS/06_KNOWN_GOTCHAS.md` §22 |
| `No usable LLM target` / `model_not_found` | a model name in `.env` is blank or decommissioned — model names are never defaulted in code. Fix `.env`, restart, confirm with `python -m scripts.doctor` |
| Ingestion stops on a 429 | re-run the same command (the manifest resumes); lower `EMBED_MAX_RPM` |
| `The write operation timed out` | Qdrant Cloud throttling → lower `QDRANT_UPSERT_BATCH`, raise `QDRANT_TIMEOUT` |
| UI says "Backend unreachable. Tried " | `BACKEND_URL` present but blank in `.env` |
| Answers cite the wrong page | inspect `processed_data/<file>.json`; the PDF may need a different extractor tier |
| First query very slow | FlashRank downloads its ONNX model once (~30 MB); warm with a throwaway query before a demo |
| Legitimate question blocked | add a term to `_DOMAIN_TERMS` in `app/guardrails/fast_rails.py` |
| Hindi input retrieves nothing | check `detect_language` — Roman-script Hindi needs ≥2 marker words (≥1 if ≤4 words) |

---

## 11. Documentation map

`DOCS/01_SYSTEM_OVERVIEW` · `02_INGESTION_ENGINE` · `03_AGENT_NODES` · `04_OBSERVABILITY` ·
`05_ENVIRONMENT_VARIABLES` · **`06_KNOWN_GOTCHAS`** · `07_RERANKING` · `08_GUARDRAILS` ·
`09_LLM_GATEWAY` · `10_EVALS`. Mermaid diagrams in `ARCHITECTURE.md`; setup narrative in
`README.md`. `Understading_the_project/` is personal study notes — gitignored, not a deliverable.

**Keep docs in sync.** This project's documentation is part of the deliverable (final-year
project). A behaviour change that contradicts `README.md`, `ARCHITECTURE.md` or a `DOCS/` file
should update that file in the same change.

---

## 12. Session protocol — update MEMORY.md every time

**Every working session ends with an update to `MEMORY.md`**, even if the session changed no
code. This is a standing instruction from the project owner, not a suggestion.

1. **Start**: read `MEMORY.md` before touching anything. Treat its "Current state" as a claim to
   verify (`git log`, `GET /health`, `scripts/doctor`), not as fact.
2. **End**: before your final reply, edit `MEMORY.md`:
   - bump `Last reviewed:` to today's date (absolute, `YYYY-MM-DD`);
   - correct "Current state" if anything changed (commits, corpus, index, deployment, models);
   - add new settled decisions, non-obvious facts and open threads; delete ones that are resolved
     or turned out wrong;
   - append one dated line to the **Session log** at the bottom: what was done and what is left.
3. **Scope**: `MEMORY.md` records what is *not* derivable from the code or git history — state,
   decisions, reasons, open threads. Rules for *how to work* belong in this file instead; do not
   duplicate them there.
