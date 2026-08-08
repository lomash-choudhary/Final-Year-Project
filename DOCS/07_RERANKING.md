# 07 · Reranking

## Why retrieval has two stages

### Stage 1 — bi-encoder (vector search)

Query and document are embedded **independently**, then compared by cosine similarity.

```
embed(query)     → [0.12, -0.45, ...]
embed(document)  → [0.11, -0.43, ...]      computed at ingestion time
similarity        = cosine(a, b)
```

Because document vectors are precomputed, this scales to millions of documents. But the two texts
never see each other — the model compresses each into a fixed vector before comparison, and that
compression is lossy in ways that matter:

- *"prevalence of theileriosis in cattle"* and *"prevalence of theileriosis in buffalo"* land very
  close together. The distinction is one word, and it is the word the question was about.
- A passage that merely mentions theileriosis scores similarly to one reporting its prevalence.

### Stage 2 — cross-encoder (reranking)

Query and document are fed to the model **together**, as one sequence:

```
[CLS] prevalence of theileriosis in cattle [SEP] The theileriosis prevalence was 20%... [SEP]
                                    → relevance score
```

Attention runs across both, so the model can weigh "cattle" against "buffalo" directly. This is
substantially more accurate — and far too slow to run over an entire index, because every
document must be re-encoded against every query.

### The combination

```
Qdrant  : 800 chunks → 20 candidates    fast, approximate
FlashRank:  20       →  5 passages      slow per item, accurate, only 20 items
```

Cheap wide net, expensive fine filter. Neither works alone: vector search alone hands the LLM
noise, and a cross-encoder alone cannot scan the corpus.

---

## FlashRank

A quantised ONNX cross-encoder (`ms-marco-MiniLM-L-6-v2` by default) running on CPU.

- No API key, no network at query time, no cost
- ~10-50 ms for 20 passages on a laptop
- ~30 MB model, downloaded once on first use and cached

For a project that must stay free, this is the highest-value component in the stack: it is the one
place where quality improves at zero marginal cost.

---

## Implementation notes (`app/services/retrieval/ranking_service.py`)

**Lazy singleton.** The model loads on first use, not at import. Loading at import would make
`uvicorn --reload` painful and would break the ingestion CLI, which never reranks anything.

**Graceful degradation.** If FlashRank cannot load, the module sets an `_unavailable` flag, logs
once, and returns the vector-similarity order truncated to `top_n`. A slightly worse ordering is
always better than a failed request — and the flag stops it retrying a load that will not succeed.

**Score replacement.** The returned chunks carry the cross-encoder score, not the original cosine
score. The cross-encoder score is what actually determined the final ordering, so it is the honest
number to show in the UI and to threshold on in the grader.

---

## Tuning

| Setting | Effect |
|---|---|
| `RETRIEVAL_TOP_K` ↑ (20 → 40) | Better recall — the reranker gets more chances to find a buried passage. Costs reranking latency, which is small |
| `RERANK_TOP_N` ↑ (5 → 8) | More context for the LLM. Costs tokens and dilutes attention. Watch faithfulness in the evals |
| `MIN_RELEVANCE_SCORE` ↑ | Drops weak candidates before reranking. Try `0.3` if obviously unrelated passages keep appearing |

The grader's `_CONFIDENT_SCORE = 0.5` threshold is calibrated against FlashRank's output range. If
you swap the reranker model, re-check that constant — it decides when the grader skips its LLM call.

---

## Measuring whether it helps

`retrieval_hit_rate` in the eval suite is the metric to watch: it asks whether the passage set
contained the paper the reference answer came from, with no LLM in the loop. Run it with reranking
on and off (comment out the `rerank()` call in `nodes/retriever.py`) and compare. On a corpus this
size the difference shows up most on questions where several papers discuss the same disease.
