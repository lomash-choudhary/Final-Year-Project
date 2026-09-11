# MEMORY.md

Durable project context that is **not** derivable from the code: current state, settled decisions,
and open threads. Working instructions live in [AGENTS.md](AGENTS.md).

Last reviewed: 2026-09-11.

---

## Project identity

- Final-year project: **Bovine Disease Research Assistant** — agentic RAG over veterinary
  literature on cattle and buffalo disease.
- Two audiences, one graph: **researchers** (cited, grounded answers) and **Indian smallholder
  farmers** (plain-language advice with an urgency level, in Hindi or Hinglish if they write that
  way). The farmer path is the newer half and the one that differentiates the project.
- The documentation (`README.md`, `ARCHITECTURE.md`, `DOCS/01..10`) is **part of the deliverable**,
  not incidental. It is written to be defended in a viva — hence the "why, not what" docstrings
  and the `DOCS/06_KNOWN_GOTCHAS.md` catalogue.
- Hard constraint set at the start and never relaxed: **zero paid APIs.** Every fallback ladder,
  cache, rate limiter and manifest exists to survive on free tiers.

---

## Current state (verified 2026-09-11)

- Branch `main`, clean tree. History: `init` → ingestion runs → deployment-ready → monitoring fixes
  → translation/clarification nodes (`ce1d768`, the most recent work).
- **Corpus ingested**: 16 PDFs in `DATA/` → 15 indexed `ok`, 1 detected as a content-level
  duplicate. **805 chunks / 805 points** in Qdrant, `recursive` chunk strategy,
  `models/gemini-embedding-001` at **3072 dimensions**. Manifest timestamps are 2026-08-08.
- Deployment: the API is deployment-ready for a PaaS (Render-shaped) using `.env.prod` and
  `requirements-prod.txt`. **The deployed server never ingests** — ingestion runs locally against
  the same Qdrant cluster, which is why the prod requirements exclude PyTorch and the eval stack.
- No test suite exists, by choice. Verification is `scripts/doctor.py`, `--dry-run` +
  `processed_data/*.json` inspection, `GET /health`, and the `evals/` suite.

---

## Settled decisions (do not re-litigate without a reason)

- **Own LLM gateway instead of Portkey/LiteLLM.** `app/llm/router.py` implements routing,
  failover, retries, backoff and a TTL cache in-process. `PORTKEY_API_KEY` / `ENABLE_PORTKEY`
  remain in config as inert placeholders; `DOCS/09` explains what a hosted gateway would add.
- **Groq for reasoning, Gemini for embeddings — the two ladders are separate.** Groq has no
  embeddings endpoint. This was the single most common misreading of the plan.
- **Qdrant over pgvector/Chroma**, cosine distance, page-tagged payloads.
- **FlashRank (local ONNX cross-encoder) over a hosted reranker** — accuracy without API cost.
- **Deterministic guardrails first, NeMo optional.** An LLM rail spends a model call on every
  greeting it exists to reject cheaply.
- **Evals hit the live API rather than importing the graph**, so guardrails, gateway fallback and
  the self-correction loop are inside what is measured.
- **Safety posture of the farmer path is deliberately conservative**: red-flag signs force
  `vet_now`, no prescription medicines or doses are ever named, and an unparsable care level
  defaults to `vet_soon`. A false "see a vet" costs a consultation fee; a false "treat at home"
  can cost the animal.

---

## Non-obvious facts worth remembering

- `Firstpaper.pdf` is a **124-page journal issue**, not a single paper. Its aggressive running
  headers are the reason the header stripper exists; it is the usual suspect when retrieval
  surfaces irrelevant chunks.
- Gemini's free embedding quota (`limit: 100`) is charged **per text, not per request** — hence
  `EMBED_MAX_RPM` being counted in texts. A full 805-chunk ingest takes ~9 minutes of deliberate
  pacing.
- Gemini returns a `retryDelay` (e.g. `54s`) that exponential backoff alone cannot outlast; the
  retry path parses and honours it.
- Roman-script Hindi ("meri gaay khana nahi kha rahi") is how most Indian users actually type.
  A Devanagari-only detector would misclassify the majority of real Hindi input as English and
  feed nonsense to the retriever — `translator.detect_language` uses a marker-word list for this,
  and `fast_rails._DOMAIN_TERMS` carries Hinglish vocabulary because guardrails run *before*
  translation.
- The typo'd directory name `Understading_the_project/` is personal study notes and is gitignored
  on purpose — not a deliverable, do not tidy it into the docs.
- Live artefacts (`processed_data/`, `ingestion_manifest.json`, `.cache/`, `evals/results/`) are
  gitignored; a fresh clone must re-run ingestion.

---

## Open threads / known limitations

- **Image-only (scanned) PDF pages have no OCR tier.** After all three extractors they are
  reported as warnings and skipped. A tier-4 OCR path is the obvious next extension.
- **`MemorySaver` is in-process.** Conversation state does not survive an API restart and is not
  shared across replicas — fine for a demo, wrong for real multi-instance deployment.
- **The gateway response cache is per-process and unbounded-ish** (crude 512-entry trim). Not a
  datastore.
- **`GUARDRAILS_MODE=full` (NeMo) is rarely exercised**; it degrades to `fast` if the dependency
  cannot initialise. Check `GET /health` → `guardrails.nemo_tier` to see what is actually live.
- **RAGAS runs take 10–15 minutes** by design (TPM pacing). Budget for it before a demo; the
  zero-cost metrics are instant.
- **No automated tests and no CI.** Worth adding if the project continues past submission —
  guardrail regexes and the `_parse` helpers are the highest-value targets.
