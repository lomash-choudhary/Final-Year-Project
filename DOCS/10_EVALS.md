# 10 · Evaluation

"It seems to work" is not a result. This suite produces numbers you can put in a report and
defend.

---

## Two phases

### Phase 1 — replay (`evals/pipeline.py`)

Sends all 16 golden questions to the **live** `POST /query` endpoint and records the answer, the
retrieved passages, the citations, and the tool path taken.

It deliberately hits the running API rather than importing the graph. What gets measured is the
system as deployed — guardrails, gateway fallback and the self-correction loop included. Importing
the graph directly would measure a different system from the one being demoed.

```bash
python -m evals.pipeline          # ~2 minutes; results saved to evals/results/
```

### Phase 2 — scoring (`evals/metrics.py`)

```bash
python -m evals.metrics
```

---

## Zero-cost metrics

No LLM involved. Run these as often as you like.

### `tool_correctness`

Jaccard overlap between expected and actual tool paths:

```
score = |expected ∩ actual| / |expected ∪ actual|
```

The three paths are `retrieve_documents`, `direct_answer` and `guardrails`. This measures the
planner's routing decisions — a low score means the agent is retrieving for conversational turns
(wasted quota) or answering factual questions without evidence (hallucination risk).

### `retrieval_hit_rate`

Did the retrieved passage set include the paper the reference answer came from?

This is the most diagnostically useful metric in the suite, because it **separates retrieval
failures from generation failures**. A bad answer with a hit means the LLM mishandled good
context. A bad answer with a miss means retrieval never gave it a chance — and those need
completely different fixes.

Samples where `source_doc` is `none` (conversational turns, the deliberate out-of-corpus question)
score a hit when nothing was retrieved, since retrieving nothing is the correct behaviour there.

---

## RAGAS metrics (LLM-judged)

| Metric | Question it answers | Low score means |
|---|---|---|
| **Faithfulness** | Is every claim in the answer supported by the retrieved context? | Hallucination — the model is drawing on training data |
| **Answer relevancy** | Does the answer actually address the question? | Evasive or padded generation |
| **Context precision** | Is the retrieved context free of noise? | Reranking is not filtering enough; lower `RERANK_TOP_N` |
| **Context recall** | Does the context cover what the reference answer needs? | Retrieval is missing passages; raise `RETRIEVAL_TOP_K` or re-chunk |
| **Answer correctness** | Does the answer match the reference? | Composite — read the four above first |

**Faithfulness and context recall are the pair to watch.** High faithfulness with low recall means
an honest system with a retrieval problem, which is the good failure mode. The reverse means the
model is inventing things, which is the one that matters.

---

## The free-tier token budget

Groq's free tier is **TPM**-limited, not only RPM-limited. The runner is paced accordingly:

| Guard | Value | Why |
|---|---|---|
| Samples per call | 1 | RAGAS fires several concurrent sub-calls per sample internally; batching stacks those bursts into one second |
| Gap between samples | 25 s | Lets the sliding TPM window recover |
| Gap between metrics | 62 s | Full window reset |
| Context truncation | 400 chars | An untruncated Faithfulness call over five 1500-char passages is ~8,000 tokens and is rejected outright |
| Passages per sample | 2 | Same reason |
| Judge model | `llama-3.1-8b-instant` | Cheaper per token than the 70B, and adequate for judging |
| Judge key | `JUDGE_GROQ` | ~80 calls per full pass — enough to exhaust the quota your live app is using |

Budget **10–15 minutes** for a full RAGAS pass. That is not slowness to fix; it is the cost of
running a proper evaluation without paying for API access.

A metric that fails does not discard the ones that succeeded — each is scored and stored
independently.

---

## Guardrail evaluation

```bash
python -m evals.guardrails_eval
```

10 test cases: jailbreaks, prompt injections, off-topic requests, and legitimate research
questions that must **not** be blocked. Scored as a confusion matrix, because a single accuracy
number hides which failure mode is happening.

See [08_GUARDRAILS.md](08_GUARDRAILS.md) for how to read it.

---

## The golden dataset

`evals/golden_dataset.json` — 16 RAG samples and 10 guardrail cases.

Reference answers are **quoted from the abstracts of the indexed papers**, not written from
memory. That matters: a reference answer invented by a model would make context recall and answer
correctness measure agreement with a hallucination.

Two samples exist to test honesty rather than knowledge:

- **#15** asks about African swine fever, which is not in the corpus. The correct behaviour is to
  say so, not to answer from general veterinary knowledge.
- **#16** is "Thanks, that was helpful" — the correct tool path is `direct_answer`, with no
  retrieval.

### Adding samples

```json
{
  "id": 17,
  "domain": "your_topic",
  "source_doc": "the_paper.pdf",
  "question": "...",
  "reference": "Quoted or closely paraphrased from the paper.",
  "relevant_contexts": ["the passage the answer comes from"],
  "expected_tools": ["retrieve_documents"],
  "actual_response": "",
  "actual_contexts": [],
  "actual_tools_called": []
}
```

`source_doc` must match the filename exactly — `retrieval_hit_rate` matches against it.

---

## Using the results

A defensible evaluation section compares configurations rather than reporting one number. Cheap
comparisons this system supports directly:

| Comparison | How |
|---|---|
| Self-correction on vs off | `ENABLE_SELF_CORRECTION=false`, restart, re-run |
| Reranking on vs off | Comment out `rerank()` in `nodes/retriever.py` |
| Chunk size 800 vs 1400 vs 2000 | Re-ingest with `--wipe` for each |
| `RERANK_TOP_N` 3 vs 5 vs 8 | No re-ingest needed, just restart |
| Guardrail tiers `fast` vs `full` | `GUARDRAILS_MODE`, then `evals.guardrails_eval` |

Run the zero-cost metrics for every configuration and a full RAGAS pass only on the two or three
that look most promising. That is how you get a comparison table without spending a week of free
quota.
