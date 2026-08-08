"""
Eval phase 2 — scoring.

Two families of metric, and the split matters on a free tier:

  zero-cost (run always)
    tool_correctness    did the agent pick the right path (retrieve / answer
                        directly / block)? Jaccard over expected vs actual.
    retrieval_hit_rate  did the passage set include the paper the answer should
                        have come from? This is pure retrieval quality with no
                        judge in the loop, and it is the metric that tells you
                        whether a bad answer was a retrieval failure or a
                        generation failure.

  LLM-judged (RAGAS, costs quota)
    faithfulness        is every claim supported by the retrieved context?
    answer_relevancy    does the answer address the question?
    context_precision   is the retrieved context free of noise?
    context_recall      does it cover what the reference answer needs?
    answer_correctness  does it match the reference?

Groq's free tier is TPM-limited, not just RPM-limited, so the RAGAS pass runs
one sample at a time with cooldowns between metrics and truncates contexts.
Without that, a Faithfulness request over five 1500-char passages is a single
~8,000-token call that the free tier rejects outright.
"""

from __future__ import annotations

import asyncio

import logfire
import pandas as pd

from app.config import settings

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
JUDGE_MODEL = settings.GROQ_FAST_MODEL

# Calibrated for Groq's free on-demand tier (~6,000 TPM).
COOLDOWN_BETWEEN_METRICS = 62
COOLDOWN_BETWEEN_SAMPLES = 25
CONTEXT_TRUNCATE = 400   # chars per passage handed to the judge
CONTEXT_LIMIT = 2        # passages per sample handed to the judge


# ── zero-cost metrics ──────────────────────────────────────────────────────────

def tool_correctness(samples: list[dict]) -> pd.DataFrame:
    """Jaccard overlap between expected and actual tool paths. No LLM involved."""
    rows = []
    for sample in samples:
        called = set(sample.get("actual_tools_called") or [])
        expected = set(sample.get("expected_tools") or [])
        union = called | expected
        score = len(called & expected) / len(union) if union else 0.0
        rows.append({
            "question": sample["question"][:70],
            "expected": ", ".join(sorted(expected)) or "-",
            "actual": ", ".join(sorted(called)) or "-",
            "tool_correctness": round(score, 3),
        })
    return pd.DataFrame(rows)


def retrieval_hit_rate(samples: list[dict]) -> pd.DataFrame:
    """
    Did retrieval surface the paper the reference answer came from?

    Samples whose source_doc is "none" (conversational turns, deliberate
    out-of-corpus questions) are scored as hits when nothing was retrieved,
    because retrieving nothing is the correct behaviour there.
    """
    rows = []
    for sample in samples:
        expected_doc = (sample.get("source_doc") or "").strip()
        citations = " ".join(sample.get("actual_citations") or [])

        if not expected_doc or expected_doc == "none":
            hit = 1 if not citations.strip() else 0
            note = "expected no retrieval"
        else:
            hit = 1 if expected_doc.lower() in citations.lower() else 0
            note = expected_doc[:45]

        rows.append({
            "question": sample["question"][:70],
            "expected_source": note,
            "hit": hit,
        })
    return pd.DataFrame(rows)


# ── RAGAS judge ────────────────────────────────────────────────────────────────

def _build_judge():
    """
    A judge on a separate key. Scoring 16 samples across 5 metrics is ~80 model
    calls — enough to exhaust the quota the live app is using, which is why
    JUDGE_GROQ exists as its own variable.
    """
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise RuntimeError(
            f"RAGAS is not installed ({exc}). Run `pip install ragas` or use the "
            "zero-cost metrics only."
        ) from exc

    api_key = settings.judge_api_key
    if not api_key:
        raise RuntimeError("No judge key. Set JUDGE_GROQ (preferred) or GROQ_API_KEY in .env.")

    # Groq exposes an OpenAI-compatible endpoint, so the OpenAI client works
    # against it directly with a different base_url.
    client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    judge_llm = llm_factory(JUDGE_MODEL, provider="openai", client=client)

    # Local embeddings for the metrics that need them — no API cost.
    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        use_api=False,
    )
    return judge_llm, embeddings


def _prepare(golden_dataset: dict) -> list[dict]:
    """Keep answered samples, and shrink contexts to fit the free-tier TPM ceiling."""
    prepared = []
    for sample in golden_dataset["rag_samples"]:
        if not (sample.get("actual_response") or "").strip():
            continue
        raw = sample.get("actual_contexts") or sample.get("relevant_contexts") or []
        prepared.append({
            **sample,
            "actual_contexts": [c[:CONTEXT_TRUNCATE] for c in raw[:CONTEXT_LIMIT]],
        })
    return prepared


async def _cooldown(seconds: int, label: str, status_cb=None) -> None:
    if status_cb:
        status_cb(f"Cooling down {seconds}s after {label} (free-tier TPM buffer)...")
    for _ in range(max(1, seconds // 5)):
        await asyncio.sleep(5)


async def _score_one_by_one(metric, inputs: list[dict], label: str, status_cb=None) -> list:
    """
    One sample per call with a pause between them.

    RAGAS fires several concurrent sub-calls per sample internally, so batching
    at this level stacks those bursts inside the same second and trips the TPM
    limit even when the per-sample size is fine.
    """
    scores = []
    for i, item in enumerate(inputs):
        if i > 0:
            await _cooldown(COOLDOWN_BETWEEN_SAMPLES, f"{label} sample {i}", status_cb)
        if status_cb:
            status_cb(f"{label}: sample {i + 1}/{len(inputs)}")
        result = await metric.abatch_score([item])
        scores.extend(result)
    return scores


def _to_frame(key: str, samples: list[dict], scores) -> pd.DataFrame:
    return pd.DataFrame([
        {"question": s["question"][:70], key: round(float(getattr(r, "value", r)), 3)}
        for s, r in zip(samples, scores)
    ])


async def run_ragas_metrics(golden_dataset: dict, status_cb=None) -> dict[str, pd.DataFrame]:
    """Run the five LLM-judged RAGAS metrics. Returns {metric_name: DataFrame}."""
    from ragas.metrics.collections import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    judge_llm, embeddings = _build_judge()
    samples = _prepare(golden_dataset)
    if not samples:
        raise ValueError("No samples have an actual_response — run phase 1 first.")

    results: dict[str, pd.DataFrame] = {}

    experiments = [
        (
            "faithfulness", "Faithfulness",
            lambda: Faithfulness(llm=judge_llm),
            lambda s: {
                "user_input": s["question"],
                "response": s["actual_response"],
                "retrieved_contexts": s["actual_contexts"],
            },
        ),
        (
            "answer_relevancy", "Answer Relevancy",
            lambda: AnswerRelevancy(llm=judge_llm, embeddings=embeddings),
            lambda s: {"user_input": s["question"], "response": s["actual_response"]},
        ),
        (
            "context_precision", "Context Precision",
            lambda: ContextPrecision(llm=judge_llm),
            lambda s: {
                "user_input": s["question"],
                "reference": s["reference"],
                "retrieved_contexts": s["actual_contexts"],
            },
        ),
        (
            "context_recall", "Context Recall",
            lambda: ContextRecall(llm=judge_llm),
            lambda s: {
                "user_input": s["question"],
                "reference": s["reference"],
                "retrieved_contexts": s["actual_contexts"],
            },
        ),
        (
            "answer_correctness", "Answer Correctness",
            lambda: AnswerCorrectness(llm=judge_llm, embeddings=embeddings),
            lambda s: {
                "user_input": s["question"],
                "response": s["actual_response"],
                "reference": s["reference"],
            },
        ),
    ]

    with logfire.span("Eval phase 2 — RAGAS", samples=len(samples)):
        for position, (key, label, build_metric, build_input) in enumerate(experiments):
            if position > 0:
                await _cooldown(COOLDOWN_BETWEEN_METRICS, label, status_cb)

            if status_cb:
                status_cb(f"Metric {position + 1}/{len(experiments)} — {label} ({len(samples)} samples)")

            with logfire.span(f"RAGAS {label}"):
                try:
                    inputs = [build_input(s) for s in samples]
                    scores = await _score_one_by_one(build_metric(), inputs, label, status_cb)
                    frame = _to_frame(key, samples, scores)
                    results[key] = frame
                    logfire.info(f"{label} complete", mean=round(frame[key].mean(), 3))
                except Exception as exc:
                    # One metric failing must not discard the ones that succeeded.
                    logfire.error(f"{label} failed: {{err}}", err=str(exc)[:300])
                    if status_cb:
                        status_cb(f"{label} failed: {str(exc)[:160]}")

    if status_cb:
        status_cb("RAGAS scoring complete.")
    return results


def run_all_metrics(golden_dataset: dict, status_cb=None, include_ragas: bool = True) -> dict:
    """Zero-cost metrics always; RAGAS only when asked for."""
    samples = [s for s in golden_dataset["rag_samples"] if (s.get("actual_response") or "").strip()]

    results: dict = {
        "tool_correctness": tool_correctness(golden_dataset["rag_samples"]),
        "retrieval_hit_rate": retrieval_hit_rate(golden_dataset["rag_samples"]),
    }

    if include_ragas and samples:
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except Exception:
            pass
        results.update(asyncio.run(run_ragas_metrics(golden_dataset, status_cb)))

    return results


if __name__ == "__main__":
    from app.observability import configure_observability
    from evals.pipeline import load_results

    configure_observability("bovine-rag-evals")

    dataset = load_results()
    if dataset is None:
        raise SystemExit("No phase 1 results found. Run: python -m evals.pipeline")

    scores = run_all_metrics(dataset, status_cb=print, include_ragas=True)

    print("\n" + "=" * 60)
    for name, frame in scores.items():
        numeric = frame.select_dtypes("number")
        if not numeric.empty:
            print(f"  {name:<22} mean {numeric.iloc[:, -1].mean():.3f}")
    print("=" * 60)
