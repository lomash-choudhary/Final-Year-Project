"""
Evaluation dashboard.

    streamlit run evals/app.py     (requires uvicorn running on :8000)

Three tabs mirroring the three things worth demonstrating:
    1. Live replay      — run the golden dataset through the deployed system
    2. Quality metrics  — zero-cost metrics always, RAGAS on request
    3. Guardrails       — confusion matrix over the safety test cases
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from evals.guardrails_eval import run_guardrails_eval, summarise  # noqa: E402
from evals.metrics import retrieval_hit_rate, run_all_metrics, tool_correctness  # noqa: E402
from evals.pipeline import load_golden_dataset, load_results, run_pipeline, save_results  # noqa: E402

st.set_page_config(page_title="RAG Evaluation Suite", page_icon="📊", layout="wide")

st.title("📊 Evaluation Suite")
st.caption(f"Scoring the live system at `{settings.BACKEND_URL}` against the golden dataset.")

golden = load_golden_dataset()

if "phase1" not in st.session_state:
    st.session_state.phase1 = load_results()

tab_replay, tab_metrics, tab_guardrails = st.tabs(
    ["1 · Live replay", "2 · Quality metrics", "3 · Guardrails"]
)


# ── tab 1 ──────────────────────────────────────────────────────────────────────
with tab_replay:
    st.subheader("Replay the golden dataset")
    st.markdown(
        f"Sends all **{len(golden['rag_samples'])}** questions to `POST /query` and records the "
        "answer, the retrieved passages and the tool path taken. Calls are spaced out to stay "
        "within the free Groq rate limit, so expect a couple of minutes."
    )

    if st.session_state.phase1:
        answered = sum(1 for s in st.session_state.phase1["rag_samples"] if s.get("actual_response"))
        st.success(f"Previous run loaded — {answered}/{len(st.session_state.phase1['rag_samples'])} answered.")

    if st.button("Run replay", type="primary"):
        progress = st.progress(0.0)
        log = st.empty()

        def _callback(i, total, question, stage, preview=""):
            if stage == "calling":
                log.info(f"[{i + 1}/{total}] {question}")
            else:
                progress.progress((i + 1) / total)
                log.success(f"[{i + 1}/{total}] {(preview or '(no answer)')[:160]}")

        with st.spinner("Replaying..."):
            st.session_state.phase1 = run_pipeline(golden, _callback)
            dest = save_results(st.session_state.phase1)
        st.success(f"Done — saved to `{dest}`")

    if st.session_state.phase1:
        st.divider()
        rows = [
            {
                "id": s["id"],
                "question": s["question"][:70],
                "tool": ", ".join(s.get("actual_tools_called") or []),
                "answer": (s.get("actual_response") or "")[:110],
                "sources": len(s.get("actual_contexts") or []),
                "ms": s.get("elapsed_ms", 0),
            }
            for s in st.session_state.phase1["rag_samples"]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── tab 2 ──────────────────────────────────────────────────────────────────────
with tab_metrics:
    st.subheader("Quality metrics")

    if not st.session_state.phase1:
        st.warning("Run the live replay in tab 1 first.")
    else:
        dataset = st.session_state.phase1
        samples = dataset["rag_samples"]

        st.markdown("#### Zero-cost metrics")
        st.caption("Computed locally — no LLM calls, no quota spent.")

        tools = tool_correctness(samples)
        hits = retrieval_hit_rate(samples)

        col_a, col_b = st.columns(2)
        col_a.metric("Tool correctness", f"{tools['tool_correctness'].mean():.3f}")
        col_b.metric("Retrieval hit rate", f"{hits['hit'].mean():.3f}")

        with st.expander("Per-sample breakdown"):
            st.dataframe(tools, use_container_width=True, hide_index=True)
            st.dataframe(hits, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### RAGAS metrics (LLM-judged)")
        st.caption(
            "Five metrics scored by a judge model on the `JUDGE_GROQ` key. Runs one sample at a "
            "time with cooldowns to respect the free tier — budget roughly 10-15 minutes."
        )
        if not settings.JUDGE_GROQ:
            st.info("`JUDGE_GROQ` is not set — the judge will fall back to `GROQ_API_KEY`.")

        if st.button("Run RAGAS scoring"):
            status = st.empty()
            with st.spinner("Scoring..."):
                try:
                    results = run_all_metrics(dataset, status_cb=status.info, include_ragas=True)
                    st.session_state.metrics = results
                    status.success("Scoring complete.")
                except Exception as exc:
                    status.error(f"Scoring failed: {exc}")

        if "metrics" in st.session_state:
            ragas_keys = [
                k for k in st.session_state.metrics
                if k not in ("tool_correctness", "retrieval_hit_rate")
            ]
            if ragas_keys:
                cols = st.columns(len(ragas_keys))
                for col, key in zip(cols, ragas_keys):
                    frame = st.session_state.metrics[key]
                    col.metric(key.replace("_", " ").title(), f"{frame[key].mean():.3f}")

                for key in ragas_keys:
                    with st.expander(key.replace("_", " ").title()):
                        st.dataframe(st.session_state.metrics[key], use_container_width=True, hide_index=True)


# ── tab 3 ──────────────────────────────────────────────────────────────────────
with tab_guardrails:
    st.subheader("Guardrail accuracy")
    st.markdown(
        f"Runs **{len(golden['guardrails_samples'])}** safety test cases — jailbreaks, prompt "
        "injections, off-topic requests, and legitimate research questions that must *not* be "
        "blocked."
    )

    if st.button("Run guardrail tests", type="primary"):
        progress = st.progress(0.0)
        log = st.empty()

        def _callback(i, total, text):
            progress.progress((i + 1) / total)
            log.info(f"[{i + 1}/{total}] {text[:80]}")

        with st.spinner("Testing..."):
            st.session_state.guardrails = run_guardrails_eval(golden["guardrails_samples"], _callback)
        log.success("Complete.")

    if "guardrails" in st.session_state:
        report = summarise(st.session_state.guardrails)

        cols = st.columns(4)
        cols[0].metric("Accuracy", report["accuracy"])
        cols[1].metric("Precision", report["precision"])
        cols[2].metric("Recall", report["recall"])
        cols[3].metric("F1", report["f1"])

        matrix = pd.DataFrame(
            [[report["TP"], report["FN"]], [report["FP"], report["TN"]]],
            index=["Should block", "Should allow"],
            columns=["Blocked", "Allowed"],
        )
        st.markdown("**Confusion matrix**")
        st.dataframe(matrix, use_container_width=True)

        if report["false_positives"]:
            st.error("Over-blocked — legitimate questions were refused:")
            for text in report["false_positives"]:
                st.write(f"- {text}")
        if report["false_negatives"]:
            st.warning("Under-blocked — these should have been refused:")
            for text in report["false_negatives"]:
                st.write(f"- {text}")
        if not report["false_positives"] and not report["false_negatives"]:
            st.success("Every case classified correctly.")

        st.dataframe(
            pd.DataFrame([
                {
                    "id": s["id"],
                    "category": s.get("category"),
                    "input": s["input"][:70],
                    "expected_blocked": s["expected_blocked"],
                    "actual_blocked": s.get("actual_blocked"),
                    "result": s.get("result"),
                }
                for s in st.session_state.guardrails
            ]),
            use_container_width=True,
            hide_index=True,
        )
