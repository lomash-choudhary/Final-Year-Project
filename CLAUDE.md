# CLAUDE.md

All guidance for working in this repository lives in **[AGENTS.md](AGENTS.md)** — read it first.

@AGENTS.md

Project state, decisions and open threads: **[MEMORY.md](MEMORY.md)**.

## The 60-second version

Agentic RAG over 16 peer-reviewed cattle/buffalo disease papers. FastAPI + LangGraph + Qdrant +
FlashRank, with an in-process multi-key LLM gateway. Serves **researchers** (cited, grounded
answers) and **farmers** (plain-language advice with an urgency level) through one graph, in
English or Hindi (Devanagari or Roman script).

Non-negotiable constraint: **everything runs on free tiers.** Most of the odd-looking code exists
to protect quota or to fail loudly instead of silently corrupting the index.

Before changing anything, know these:

- `os.getenv()` is allowed **only** in `app/config.py`; everything else reads `settings`.
- `app.observability.configure_observability()` runs **before** other app imports (`# noqa: E402`
  marks it deliberate).
- The graph is invoked **synchronously**; `messages` is a reducer, `plan` is not.
- Embedding dimension is **probed and then locked**; changing chunking or the embedding model
  requires re-ingesting with `--wipe`.
- Cheap nodes (planner, grader, clarifier, translator) use `tier="fast"`; every node degrades on
  `AllTargetsFailed` rather than failing the request.
- **No model name is hardcoded** — not even as a fallback. Every one comes from `.env` via
  `settings`; a blank one disables that tier loudly. See AGENTS.md invariant 19.
- There is **no test suite** — verify with `python -m scripts.doctor`,
  `python -m app.ingestion.processor --dry-run`, `GET /health`, and `evals/`.

AGENTS.md §5 lists the full set of invariants; `DOCS/06_KNOWN_GOTCHAS.md` explains why each exists.
