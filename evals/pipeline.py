"""
Eval phase 1 — replay the golden dataset against the live API.

Deliberately hits the running FastAPI service rather than importing the graph:
what gets measured is the system as deployed, including guardrails, the gateway
fallback chain and the self-correction loop. Importing the graph directly would
measure a different system from the one being demoed.

Captures per sample: the answer, the retrieved contexts, and which tool path the
agent took (retrieval / direct answer / guardrail), inferred from the reasoning
trace the API returns.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import logfire
import requests

from app.config import settings

API_URL = f"{settings.BACKEND_URL.rstrip('/')}/query"
REQUEST_TIMEOUT = 180
DELAY_BETWEEN_CALLS = 6      # seconds — keeps the replay inside Groq's free RPM
RESPONSE_TRUNCATE = 1200     # chars kept for judging; full text stays in the trace
MAX_CONTEXTS = 4

RESULTS_DIR = Path(__file__).parent / "results"


def detect_tool(thought_process: list[str], blocked: bool) -> str:
    """Map the agent's reasoning trace onto the tool it effectively used."""
    if blocked:
        return "guardrails"

    joined = " ".join(thought_process).lower()
    if "guardrails fired" in joined:
        return "guardrails"
    if "retrieval pass" in joined or "intent: research" in joined or "sources:" in joined:
        return "retrieve_documents"
    if "conversational" in joined:
        return "direct_answer"
    return "unknown"


def load_golden_dataset(path: str | Path | None = None) -> dict:
    target = Path(path) if path else Path(__file__).parent / "golden_dataset.json"
    return json.loads(target.read_text(encoding="utf-8"))


def run_pipeline(golden_dataset: dict, progress_callback=None) -> dict:
    """
    Enrich every rag_sample with live API output.
    progress_callback(index, total, question, stage, preview="") drives the UI.
    """
    dataset = copy.deepcopy(golden_dataset)
    samples = dataset["rag_samples"]
    total = len(samples)

    with logfire.span("Eval phase 1 — live replay", samples=total, api=API_URL):
        for i, sample in enumerate(samples):
            question = sample["question"]

            if progress_callback:
                progress_callback(i, total, question, "calling")

            with logfire.span(f"Replay {i + 1}/{total}", question=question[:100]):
                try:
                    response = requests.post(
                        API_URL,
                        json={"q": question, "thread_id": f"eval-{sample['id']}"},
                        timeout=REQUEST_TIMEOUT,
                    )
                    response.raise_for_status()
                    data = response.json()

                    answer = data.get("answer") or ""
                    sources = data.get("sources") or []

                    sample["actual_response"] = answer[:RESPONSE_TRUNCATE]
                    sample["actual_contexts"] = [
                        s.get("content", "") for s in sources[:MAX_CONTEXTS] if s.get("content")
                    ]
                    sample["actual_tools_called"] = [
                        detect_tool(data.get("thought_process") or [], bool(data.get("blocked")))
                    ]
                    sample["actual_citations"] = [s.get("citation", "") for s in sources[:MAX_CONTEXTS]]
                    sample["elapsed_ms"] = data.get("elapsed_ms", 0)
                    sample["llm_target"] = (data.get("llm") or {}).get("target", "")

                    logfire.info(
                        "Captured", tool=sample["actual_tools_called"][0],
                        chars=len(answer), contexts=len(sample["actual_contexts"]),
                    )

                except requests.exceptions.ConnectionError:
                    logfire.error("Cannot reach the API at {url}", url=API_URL)
                    _mark_failed(sample, "backend unreachable")
                except Exception as exc:
                    logfire.error("Replay failed: {err}", err=str(exc)[:300])
                    _mark_failed(sample, str(exc)[:200])

            if progress_callback:
                progress_callback(i, total, question, "done", sample.get("actual_response", ""))

            if i < total - 1:
                time.sleep(DELAY_BETWEEN_CALLS)

    return dataset


def _mark_failed(sample: dict, reason: str) -> None:
    sample["actual_response"] = ""
    # Fall back to the reference contexts so downstream metrics do not crash on
    # an empty list; the empty actual_response already marks the sample as failed.
    sample["actual_contexts"] = sample.get("relevant_contexts", [])
    sample["actual_tools_called"] = ["unknown"]
    sample["error"] = reason


def save_results(dataset: dict, name: str = "phase1_results.json") -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = RESULTS_DIR / name
    dest.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    return dest


def load_results(name: str = "phase1_results.json") -> dict | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    from app.observability import configure_observability

    configure_observability("bovine-rag-evals")

    print(f"Replaying golden dataset against {API_URL}\n")
    golden = load_golden_dataset()

    def _progress(i, total, question, stage, preview=""):
        if stage == "calling":
            print(f"[{i + 1}/{total}] {question[:70]}...")
        else:
            print(f"          -> {(preview or '(empty)')[:90]}\n")

    enriched = run_pipeline(golden, _progress)
    out = save_results(enriched)

    answered = sum(1 for s in enriched["rag_samples"] if s.get("actual_response"))
    print(f"Phase 1 complete: {answered}/{len(enriched['rag_samples'])} answered")
    print(f"Saved to {out}")
    print("\nNext: streamlit run evals/app.py  (or python -m evals.metrics)")
