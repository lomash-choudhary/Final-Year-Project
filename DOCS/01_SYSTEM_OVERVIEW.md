# 01 · System Overview

## What this system is

A question-answering system over a fixed corpus of 16 peer-reviewed papers on cattle and buffalo
disease. It answers only from those papers, cites the source and page for every claim, and says so
when the corpus does not cover something.

"Agentic" here is not decoration. The agent makes three decisions per query that a linear
retrieve-then-answer chain cannot:

1. **Is retrieval needed at all?** (planner)
2. **What should I actually search for?** (planner — query rewriting)
3. **Is what I retrieved good enough to answer from?** (grader — and if not, search again)

## The two flows

### Ingestion (offline, run once per corpus change)

```
DATA/*.pdf
  → loader cascade      pypdf → pdfplumber → PyMuPDF, per page
  → cleaning            ligatures, hyphenation, running headers
  → chunker cascade     recursive → paragraph → window, validated
  → page attribution    each chunk knows which pages it spans
  → processed_data/     JSON snapshot for debugging
  → embeddings          Gemini, cached on disk, rate-limited
  → Qdrant              deterministic IDs, page-tagged payloads
  → manifest            what was ingested, from which bytes
```

Entry point: `python -m app.ingestion.processor`

### Query (online, per request)

```
POST /query
  → guardrails          deterministic rails; blocked requests cost 0 model calls
  → planner             intent + standalone query rewrite     (fast tier LLM)
  → retriever           Qdrant top-20 → FlashRank top-5       (0 LLM calls)
  → grader              is this enough? if not, re-search      (fast tier, often skipped)
  → responder           grounded synthesis with [n] citations  (quality tier LLM)
  → response            answer + reasoning trace + sources + gateway metadata
```

Entry point: `uvicorn app.main:app`

## Cost profile per query

| Stage | LLM calls | Notes |
|---|---|---|
| Guardrails | 0 | Regex. One call only if `GUARDRAILS_MODE=full` and the fast tier passed |
| Planner | 1 | Fast tier (8B) |
| Retrieval | 0 | One embedding call for the query, usually a cache hit |
| Reranking | 0 | Local ONNX cross-encoder |
| Grader | 0 or 1 | Skipped when the top rerank score is confident |
| Responder | 1 | Quality tier (70B) |

**Typical: 2 model calls per query, 1 on the cheap model.** A blocked query costs 0. A repeated
query costs 0 (gateway cache).

## Where to look when something is wrong

| Question | File |
|---|---|
| What text actually got indexed? | `processed_data/<filename>.json` |
| Which files were ingested and when? | `ingestion_manifest.json` |
| Is everything reachable? | `GET /health` |
| Which papers are indexed? | `GET /sources` |
| Why did it answer that way? | The `thought_process` array in the `/query` response |
| What did each stage do? | Logfire trace, or console output if no token is set |

## Component map

| Concern | Module |
|---|---|
| Configuration | `app/config.py` |
| Tracing setup | `app/observability.py` |
| Document loading | `app/ingestion/loaders/` |
| Text repair | `app/ingestion/cleaning.py` |
| Chunking | `app/ingestion/chunking/splitter.py` |
| Incremental ingest | `app/ingestion/manifest.py` |
| Embeddings + cache | `app/services/retrieval/embedding*.py` |
| Vector store | `app/services/retrieval/qdrant_service.py` |
| Reranking | `app/services/retrieval/ranking_service.py` |
| LLM routing | `app/llm/router.py` |
| Safety | `app/guardrails/` |
| Agent | `app/agents/` |
| HTTP API | `app/main.py` |
| Chat UI | `ui/app.py` |
| Evaluation | `evals/` |
