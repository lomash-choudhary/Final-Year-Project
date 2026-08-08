# 03 · Agent Nodes

Four nodes, wired in `app/agents/graph.py`. State lives in `app/agents/state.py`.

---

## Planner (`nodes/planner.py`)

Runs on the **fast tier** (8B). Two jobs in one call.

### 1. Intent classification

`CONVERSATIONAL` — small talk, thanks, or a question answerable from the conversation alone.
`RESEARCH` — needs evidence from the papers.

Getting this right saves a vector search and an LLM call on every "thanks, that helps".

### 2. Query rewriting

This is the step that separates a multi-turn assistant from a one-shot demo.

```
User turn 1:  "What is the prevalence of theileriosis in Indian cattle?"
User turn 2:  "And in buffalo?"
```

Turn 2 embedded literally retrieves nothing useful — it has no content. Resolved against history
it becomes `prevalence of theileriosis in buffalo in India`, which retrieves correctly.

### Failure handling

If the planner cannot be reached, the node degrades to `intent=research` with the raw message as
the query. Retrieval still works without a planner; refusing to answer would be worse. An
unnecessary retrieval is a far cheaper mistake than answering a factual question with no evidence,
so an unparseable response also defaults to `research`.

---

## Retriever (`nodes/retriever.py`)

Zero LLM calls.

```
Qdrant cosine search  → RETRIEVAL_TOP_K (20) candidates
FlashRank cross-encoder → RERANK_TOP_N (5) kept
```

Wide first pass, narrow second pass. The wide pass is what gives the reranker a chance to find the
right chunk sitting at position 14; the narrow pass is what keeps the LLM's context short enough
to stay grounded. See [07_RERANKING.md](07_RERANKING.md).

Re-entered on a self-correction loop, so it labels its plan entries with a pass number.

---

## Grader (`nodes/grader.py`)

The self-correction loop, and the reason this is a state machine rather than a chain.

A plain retrieve-then-answer pipeline has no idea whether what it retrieved is any good. When the
first query phrasing misses, it answers from irrelevant passages — and that is exactly when RAG
systems hallucinate most confidently.

### Decision ladder (cheapest signal first)

| Situation | Action | LLM calls |
|---|---|---|
| Zero documents, budget left | Broaden the query (first 8 words) and retry | 0 |
| Zero documents, no budget | Return `empty` → responder answers "not in corpus" | 0 |
| Top rerank score ≥ 0.5 | Accept | 0 |
| No refinement budget left | Accept whatever we have | 0 |
| Anything else | Ask the fast model: is this enough? If not, rewrite the query | 1 |

Bounded by `MAX_REFINEMENTS` (default 1), so at most two retrieval passes per query. Disable the
whole node with `ENABLE_SELF_CORRECTION=false` — the graph then compiles the linear version, which
is useful for measuring exactly what the loop buys you in the eval suite.

Empty retrieval is handled by **broadening** rather than rephrasing: nothing coming back usually
means the query was too specific for the corpus, not that it was worded badly.

---

## Responder (`nodes/responder.py`)

Runs on the **quality tier** (70B). Three distinct behaviours.

### 1. No evidence — answered with zero LLM calls

When intent is `research` and no documents survived, the node returns a fixed message.

This is deliberate. An LLM instructed to "say you don't know" will sometimes answer anyway from
parametric memory — it *has* read veterinary literature during training. That is precisely the
failure a grounded system exists to prevent, and it costs quota to get wrong.

### 2. Conversational

History only. No context, no citations.

### 3. Grounded

Passages are numbered and labelled with source and page:

```
[1] Source: Theileriosis_prevalence_status_in_cattle.pdf — page 1 (relevance 0.847)
The theileriosis prevalence was 20% [95% level, CI 16-25%, PI 2-74%]...
```

The prompt requires inline `[n]` citations, exact reproduction of figures, and explicit
attribution when passages disagree — which matters in this corpus, where several papers are
meta-analyses over overlapping study sets.

### Context budget

`MAX_CONTEXT_CHARS` (18000) is enforced by dropping **whole passages** from the lowest-ranked end,
never by truncating mid-passage. Half a passage is a half-truth the model will happily complete —
a sentence cut before "…in crossbred cattle only" changes the finding.

---

## State (`state.py`)

| Field | Reducer | Why |
|---|---|---|
| `messages` | `operator.add` | Accumulates across turns — this is the conversation memory |
| `plan` | none (overwrite) | MemorySaver persists per thread; an accumulating plan would replay every previous turn's reasoning. Nodes concatenate explicitly, and the planner resets it |
| `documents` | none | Must be replaced on a retry, not appended |
| `refinements` | none | A counter |

## Memory

`MemorySaver` keyed by `thread_id`. The Streamlit UI generates one per session and resets it with
"Clear conversation". Memory is in-process — restarting uvicorn clears it. For persistence across
restarts, swap in `SqliteSaver` in `graph.py`; the rest of the system is unaffected.

---

## Routing

```python
planner  → responder   if intent == "conversational"
         → retriever   otherwise

grader   → retriever   if context_quality == "weak" and refinements <= MAX_REFINEMENTS
         → responder   otherwise
```

`GET /graph/mermaid` returns the compiled graph as Mermaid source — useful for a report, and it
proves the diagram matches what actually compiled.
