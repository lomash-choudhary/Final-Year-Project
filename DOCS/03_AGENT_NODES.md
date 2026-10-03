# 03 · Agent Nodes

Four nodes, wired in `app/agents/graph.py`. State lives in `app/agents/state.py`.

---

## Planner (`nodes/planner.py`)

Runs on the **fast tier** (8B). Two jobs in one call.

### 1. Intent classification

`CONVERSATIONAL` — small talk, thanks, or a question answerable from the conversation alone.
`SYMPTOM` — any practical question from someone who keeps animals: a sick animal, or what to give,
apply, clean or vaccinate with. Goes to the clarifier and the farmer advisor.
`RESEARCH` — a question about the studies themselves (prevalence, findings, methods).

Getting this right saves a vector search and an LLM call on every "thanks, that helps". Two
corrections live in `_parse`: a first message can never be `CONVERSATIONAL` (there is nothing to
answer from — "bachhde ko khansi hai" once came back as a refusal), and "how do I clean my shed"
is `SYMPTOM`, not `RESEARCH` — it used to get a cited research answer with a table.

For `SYMPTOM` the planner also returns `DISEASE:` — its single best guess, stored as
`likely_disease` and shown in the plan as `Likely disease: …`.

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

On the first pass of a `SYMPTOM` turn with a `likely_disease`, it runs a second search for
`"<disease> treatment"`, merges the candidates and reranks the union against query + disease.
Farmers describe signs while treatment passages are written under the disease name ("calf cough
runny nose" alone found only decongestant passages; adding "pneumonia" found tilmicosin and
tulathromycin). Putting the guess *into* the main query was worse: round bald patches became
lumpy skin disease. The union keeps the sign matches when the guess is wrong.

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
| `messages` | `operator.add` | Seeded per request with the memory window + the new message; nodes append replies |
| `memory_summary` | none | Rolling summary of turns older than the window |
| `plan` | none (overwrite) | This turn's reasoning only. Nodes concatenate explicitly, and `translate_in` resets it |
| `documents` | none | Must be replaced on a retry, not appended |
| `refinements` | none | A counter |

## Memory

Conversation memory lives in Postgres (`app/memory/store.py`, schema `rag` in the frontend's Neon
database), keyed by `thread_id` and owned by `user_id`. The graph has **no checkpointer**.

Per request, `main.py`:

1. loads the conversation's **rolling summary** and its **last `MEMORY_WINDOW_TURNS` turns**
   (English text only) in one round trip;
2. seeds `messages` with that window + the new message, and `memory_summary` with the summary;
   the clarifier's `awaiting_clarification` / `clarification_rounds` come from the same row;
3. after the graph runs, saves the user message and the answer (one statement);
4. schedules two background tasks: fold turns that left the window into the summary (one
   fast-tier call per `MEMORY_SUMMARY_BATCH_TURNS` turns, capped at `MEMORY_SUMMARY_MAX_CHARS`),
   and a daily retention sweep (`MEMORY_RETENTION_DAYS`, `MEMORY_MAX_CONVERSATIONS_PER_USER`).

Every node builds its history through `app/agents/history.format_history`, so the prompt holds
`summary + window`, each message capped at `MEMORY_MSG_MAX_CHARS` — constant size however long
the chat is. Retrieved passages and the plan are never stored as memory; guardrail replies are
stored for the history view but never fed back to the model.

Without `DATABASE_URL`, or with the database down, a bounded in-process store takes over: the
conversation still has memory until restart, and the history endpoints return 503.

The same rows feed the frontend's sidebar through `GET /conversations` (list, no messages) and
`GET /conversations/{id}/messages` (newest page first), both keyset-paginated.

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
