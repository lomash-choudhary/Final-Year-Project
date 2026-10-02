# MEMORY.md

Durable project context that is **not** derivable from the code: current state, settled decisions,
and open threads. Working instructions live in [AGENTS.md](AGENTS.md).

Last reviewed: 2026-10-03.

**Every session must update this file before it ends** — protocol in AGENTS.md §12.

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

## Current state (verified 2026-10-01)

- Branch `main`, HEAD `fcf4e7d` (30 new PDFs added) + uncommitted 2026-10-01 changes: Gemini
  embedding key failover, soft-hyphen cleaning, glued-page PyMuPDF re-extraction (+ docs).
  Earlier: HEAD `250125f`. History: `init` → ingestion runs → deployment-ready →
  monitoring fixes → translation/clarification nodes (`ce1d768`) → two "code refactor, model
  update" commits on 2026-09-12 (`9133c21`, `250125f`) that removed every hardcoded model name
  (AGENTS.md invariant 19) and added `AGENTS.md` / `CLAUDE.md` / `MEMORY.md`.
- **Corpus ingested (verified 2026-10-01)**: 46 PDFs in `DATA/` → 44 indexed `ok`, 1 content-level
  duplicate (`Theileriosis_prevalence_status_in_cattle-2.pdf`), 1 skipped as non-English (`11.pdf`,
  Russian — owner wants **English PDFs only**). **2175 chunks / 2175 points** in
  Qdrant Cloud, `models/gemini-embedding-001` at **3072 dimensions**. The original 15 files
  (805 points, 2026-08-08) were skipped unchanged; the 30 new ones added 1474, minus 104 removed
  for `11.pdf`.
- Corpus now goes beyond the original haemoprotozoa/brucellosis/LSD focus: AABP proceedings on
  antimicrobial use/resistance, mastitis, lameness, BRD, mycoplasma, plus a 1900s archive.org
  book (`notesondiseaseof00kori.pdf`).
- Deployment: the API is deployment-ready for a PaaS (Render-shaped) using `.env.prod` and
  `requirements-prod.txt`. **The deployed server never ingests** — ingestion runs locally against
  the same Qdrant cluster, which is why the prod requirements exclude PyTorch and the eval stack.
- No test suite exists, by choice. Verification is `scripts/doctor.py`, `--dry-run` +
  `processed_data/*.json` inspection, `GET /health`, and the `evals/` suite.

---

## Settled decisions (do not re-litigate without a reason)

- **English-only corpus** (owner decision 2026-10-01). Enforced by the ingestion language gate
  `processor._english_check` (`INGEST_ENGLISH_ONLY`); a non-English PDF stays in `DATA/` but is
  never indexed.
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
  `vet_now` (answer collapses to "Contact a vet now" + ≤2 safe steps), and an unparsable care level
  defaults to `vet_soon`. **Changed 2026-10-02 (owner request):** the advisor now names medicines,
  but only ones the retrieved passages name, and prescription drugs get no dose. **Changed again
  2026-10-03:** **no disclaimers or prescription tags of any kind in answers** — owner will handle
  disclaimers separately later; do not add them back. The model's own notes are stripped in code;
  doses are still cut and "give X injection" steps still rewritten to "ask your vet". Audience is farmers with little
  schooling: plain words, no jargon. Prompts must stay concise and contain no tables (owner rule). Answers are capped at ~100 words in four sections (Likely cause / Medicine / What to
  do / Call the vet if); the owner wants short, on-point answers. A false "see a vet" costs a consultation fee; a false "treat at home"
  can cost the animal.

---

## Non-obvious facts worth remembering

- `Firstpaper.pdf` is a **124-page journal issue**, not a single paper. Its aggressive running
  headers are the reason the header stripper exists; it is the usual suspect when retrieval
  surfaces irrelevant chunks.
- **Gemini free embedding quota also has a daily cap of ~1000 texts per project.** On
  2026-10-01 the primary key 429'd after ~900–1000 texts that day (retryDelay ~59s, never clears)
  and the run failed over to `GEMINI_FALLBACK_API_KEY` for the last ~10 files. A full re-ingest
  (~2200 texts) needs ~3 project-days of quota — **never `--wipe` casually**; the embedding cache
  (`.cache/embeddings.sqlite3`) is what makes a rebuild free, so do not delete it either.
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
- `notesondiseaseof00kori.pdf` (1900s) contains archaic remedies (turpentine, udder inflation for
  milk fever, …). The advisor prompt explicitly forbids passing on outdated or unsafe remedies.
- Clarifier default is now **answer directly** (owner: "only ask if really needed"). The old prompt
  listed duration/fever/appetite/pregnancy as examples and the model asked all four every turn.
  `MAX_FOLLOW_UP_QUESTIONS` 4 → 2 (code default, `.env`, `.env.example`). Live check 2026-10-02:
  hoof wound / mild fever / red urine → no questions; "my cow is not well" → 1 question.
- `LANGSMITH_API_KEY` in `.env` returns 401 (seen 2026-10-02) — tracing to LangSmith is not
  working; Logfire unaffected.
- Planner SYMPTOM rewrite must keep **what the farmer asks for** ("what to apply/give"). The
  "…causes and treatment" rewrite (2026-10-02, reverted same day) dropped it, and the hoof-wound
  query then missed both the oxytetracycline-spray and copper-sulfate-dressing passages.
- Farmer path drops reference-list chunks before rerank (`retriever._is_reference_list`, citation
  density ≥ 6/1000 chars). Not applied to research: prevalence tables score the same.
- Advisor medicine rules are enforced in code (`_enforce_medicine_rules`) — gpt-oss-120b ignored
  the prompt for brand names, clenbuterol and phenothiazine. Prompt has no persona and no disclaimer
  (owner: disclaimer to be added later, keep the prompt lean).
- The typo'd directory name `Understading_the_project/` is personal study notes and is gitignored
  on purpose — not a deliverable, do not tidy it into the docs.
- Live artefacts (`processed_data/`, `ingestion_manifest.json`, `.cache/`, `evals/results/`) are
  gitignored; a fresh clone must re-run ingestion.

---

## Open threads / known limitations

- The new embedding failover/cleaning/loader changes are **uncommitted** as of 2026-10-01 —
  owner to review and commit.
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
- **Corpus gaps the advisor cannot fix** (2026-10-03): foot rot topical = only the digital-dermatitis
  oxytetracycline gauze (aabp_1998 p5) is retrieved, so the hoof answer uses it; teat warts = only
  tarantula-extract study → "Ask your vet"; milk fever and FMD vaccine have no modern treatment →
  "Ask your vet"; shed cleaning gets generic steps (kori copper-sulphate passage rarely retrieved).
  `questions.md` expects levamisole for warts — the corpus does not support that.
- **Translation runs on `qwen/qwen3.8-27b` (fast tier)**; Hindi wording is the weakest part (a farm
  glossary in the prompt fixed "bachhde" → "child" and mastitis → "मामा"). Moving `translate_out` to
  the quality tier is the next lever if quality matters more than 120B quota.
- **No automated tests and no CI.** Worth adding if the project continues past submission —
  guardrail regexes and the `_parse` helpers are the highest-value targets.

---

## Session log

One dated line per session, newest last: what was done, what is left.

- 2026-09-12 — Model names moved entirely to `.env` (invariant 19); agent docs created.
- 2026-10-01 — Confirmed single root `AGENTS.md` / `CLAUDE.md` → AGENTS.md / `MEMORY.md`; added the
  end-of-session MEMORY.md protocol (AGENTS.md §12) and this log. No code changes. Corpus/index
  facts above not re-verified this session (Qdrant not queried).
- 2026-10-01 — `CLAUDE.md` reduced to the single line `@AGENTS.md`; all guidance lives in
  `AGENTS.md` only (owner's preference — never add content to `CLAUDE.md`).
- 2026-10-01 — Ingested the 30 PDFs added in `fcf4e7d` (805 → 2279 points, 0 failures, ~33 min at
  `EMBED_MAX_RPM=60` via CLI env override). Added `GEMINI_FALLBACK_API_KEY` (per-key limiter,
  failover after `EMBED_MAX_RETRIES` retries, sticky) — it fired live once. Fixed soft hyphens
  (AABP scans) and glued-word pages (archive.org book → PyMuPDF). Left: commit; evals not re-run
  on the larger corpus; golden set does not cover the new topics.
- 2026-10-01 — Owner asked for English-only: added ingestion language gate, removed `11.pdf`
  (Russian, 104 points) → 2175 points. Only non-English file in the corpus. Still uncommitted.
- 2026-10-01 — Audited Qdrant directly: 2175 points / 44 sources, all English (per-document and
  per-chunk check), exact match with manifest; only 11.pdf (non-English) and the Theileriosis
  duplicate are absent by design. Incremental run embedded nothing — corpus fully ingested.
- 2026-10-02 — Advisor reworked per owner: short four-section format, names corpus-grounded medicines
  (prescription drugs: no dose), vet_now → "Contact a vet now". Planner symptom queries now add
  "causes and treatment". AGENTS.md invariant 17 updated. Added `questions.md` (28 test questions:
  medicine / vet-now / edge / Hindi). Left: not live-tested (API was down); commit; evals not re-run.
- 2026-10-02 — Clarifier reworked to default to answering directly; max 2 questions, shorter
  wording. Live-tested on 4 messages (see facts). Left: full graph not run end-to-end; commit.
- 2026-10-02 — Verified hoof-wound answer ("Naxcel (Ceftiofur)") is grounded in
  aabp_1998 LamenessOfDairyCattle p6. Prompt tweaked: generic names only (corpus is US-heavy, brand
  names useless in India), up to 3 options, topical treatment first when the farmer asks what to apply.
- 2026-10-02 — Hoof-wound answer now correct (ceftiofur/penicillin/sulfadimethoxine + antiseptic,
  matches aabp_1998 p6) but leaked "(Appendix I)" from passage text → prompt rule + regex strip
  (`advisor._strip_doc_references`). Open: retrieval top-5 for hoof wound is 4× aabp_1998 (one a
  reference list) + kori book; the copper-sulfate/formalin dressing passage in
  animals-14-01836-v2.pdf is not retrieved. Reference-list chunks wasting rerank slots is worth fixing.
- 2026-10-02 — Owner unhappy with answer quality. Root cause was retrieval (planner rewrite lost the
  question). Fixed planner rewrite, reference-list filter, lean advisor prompt (no persona/disclaimer,
  fallback never mentions "sources"), code-enforced medicine rules. Full graph run on 6 questions:
  hoof → oxytetracycline/lincomycin topical, mastitis → penicillin/ceftiofur, theileriosis →
  buparvaquone, red urine → vet_now. Open: calf cough retrieves the 1987 bronchodilator paper rather
  than the BRD antibiotic passages (tilmicosin/florfenicol); evals not re-run.
- 2026-10-02 — Owner asked whether "(vet must prescribe)" came from the data: it did NOT — it is a
  blanket code rule (allowlist of farm remedies; everything else tagged). Replaced per-item tags
  with one line under Medicine; "(topical)" → "(put on the wound)"; Unicode normalised before the
  regexes. Verified hoof-answer drugs in corpus: oxytetracycline/lincomycin gauze (aabp_1998 p5,
  digital dermatitis), silver sulfadiazine (animals-14 p10, sole ulcer), foot-rot injection drugs
  (aabp_1998 p6). Open: owner may want the prescription line removed or reworded.
- 2026-10-02 — Owner: no "(vet must prescribe)" or prescription notes in answers — the frontend
  will carry the disclaimer. Removed the note line, the farm-remedy allowlist and the prompt rule;
  the model's own vet tags are stripped. No doses for prescription drugs is unchanged.

- 2026-10-02 — Q&A only: explained the INGEST_ENGLISH_ONLY language gate to the owner. No code change.

- 2026-10-02 — Reviewed a farmer foot-rot answer against DATA/: diagnosis right, topical drugs taken
  from the digital-dermatitis section, jargon, self-injection advice. Logged as open thread; no code change.
- 2026-10-03 — Live-tested all 28 `questions.md` + 5 edge cases (clarify, greeting, off-topic,
  jailbreak, dose request) via POST /query, 4 rounds. Fixed: planner routed practical farmer
  questions to RESEARCH (shed cleaning, FMD vaccine got tables) and a first-turn Hinglish question to
  CONVERSATIONAL (refusal); planner now returns DISEASE and the retriever unions a "<disease>
  treatment" search (calf cough → tilmicosin/tulathromycin); advisor: canonical prescription tag, dose
  stripping, "give X injection" → "ask your vet", [OLD BOOK] label (`ADVICE_HISTORICAL_SOURCES`),
  established treatments only, more jargon rewrites; Hinglish answers forced to Latin script;
  translator farm glossary; farmer-friendly guardrail replies; gpt-oss "【1†L1】" citations → [n] in
  responder. Final run: all 33 acceptable. Left: commit; evals/guardrails_eval not re-run; Hindi quality.
- 2026-10-03 — Owner asked if the prescription tag comes from DATA/: it does not, `advisor._RX` adds it
  to every antibiotic. Corpus: 1982 AABP lists oxytetracycline as non-Rx (US, 1982); aabp_1998 p6 says
  tetracyclines are extra-label in dairy cattle and its appendix says consult a vet.
- 2026-10-03 — Owner: remove the prescription tag, no disclaimers at all for now. Removed `RX_TAG`,
  the prompt line and translator line; AGENTS.md #17 updated. Live-checked Q1/7/28: no tags, no doses.
- 2026-10-03 — Discussed a 30-query farmer eval (10 medicine / 10 vet-now / 10 mixed): deterministic
  checks first (care level, medicine recall, unsupported drugs, doses, format, language), judge optional.
  Blocker noted: /query hides sources for symptom intent, so evals need a debug flag. No code change.
- 2026-10-03 — Eval metrics agreed in discussion: judge-based Correctness + Helpfulness, deterministic
  source recall, safety gates (care level, dose) in code. No code change.
