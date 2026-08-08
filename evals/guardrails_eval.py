"""
Guardrail evaluation — binary classification scored as precision / recall.

A guardrail has two ways to be wrong and they are not symmetric:

  false positive  a legitimate research question gets blocked. Users lose trust
                  in the system immediately.
  false negative  a jailbreak or off-topic query gets through. Wasted quota, and
                  an answer outside the system's competence.

Reporting a single "accuracy" number hides which one is happening, so the
confusion matrix is reported in full. The golden set deliberately includes
near-miss legitimate queries ("What is the economic cost of mastitis") that a
naive keyword blocklist would reject.
"""

from __future__ import annotations

import copy
import time

import logfire
import requests

from app.config import settings

API_URL = f"{settings.BACKEND_URL.rstrip('/')}/query"
DELAY = 1.0  # the fast rail tier makes no API calls, so this can stay short


def _was_blocked(payload: dict) -> bool:
    if payload.get("blocked"):
        return True
    steps = payload.get("thought_process") or []
    return any("guardrails fired" in str(step).lower() for step in steps)


def run_guardrails_eval(samples: list[dict], progress_callback=None) -> list[dict]:
    """Run each test case against the live API, labelling results TP/TN/FP/FN."""
    results = copy.deepcopy(samples)
    total = len(results)

    with logfire.span("Eval — guardrails", total=total):
        for i, sample in enumerate(results):
            if progress_callback:
                progress_callback(i, total, sample["input"])

            try:
                response = requests.post(
                    API_URL,
                    json={"q": sample["input"], "thread_id": f"guardrail-eval-{sample['id']}"},
                    timeout=90,
                )
                response.raise_for_status()
                payload = response.json()
                blocked = _was_blocked(payload)
                sample["block_reason"] = payload.get("block_reason")
            except requests.exceptions.ConnectionError:
                logfire.error("Cannot reach the API at {url}", url=API_URL)
                blocked = False
                sample["error"] = "backend unreachable"
            except Exception as exc:
                logfire.error("Guardrail test failed: {err}", err=str(exc)[:200])
                blocked = False
                sample["error"] = str(exc)[:200]

            expected = bool(sample["expected_blocked"])
            sample["actual_blocked"] = blocked

            if expected and blocked:
                sample["result"] = "TP"
            elif not expected and not blocked:
                sample["result"] = "TN"
            elif not expected and blocked:
                sample["result"] = "FP"   # over-blocking: a real question refused
            else:
                sample["result"] = "FN"   # under-blocking: an attack got through

            logfire.info(
                "Guardrail case {id}: {result}",
                id=sample["id"], result=sample["result"],
                category=sample.get("category"), blocked=blocked, expected=expected,
            )

            if i < total - 1:
                time.sleep(DELAY)

    return results


def summarise(results: list[dict]) -> dict:
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for sample in results:
        counts[sample.get("result", "FN")] += 1

    tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total = max(1, sum(counts.values()))

    return {
        **counts,
        "total": sum(counts.values()),
        "accuracy": round((tp + tn) / total, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positives": [s["input"] for s in results if s.get("result") == "FP"],
        "false_negatives": [s["input"] for s in results if s.get("result") == "FN"],
    }


if __name__ == "__main__":
    from app.observability import configure_observability
    from evals.pipeline import load_golden_dataset

    configure_observability("bovine-rag-evals")

    golden = load_golden_dataset()
    outcome = run_guardrails_eval(
        golden["guardrails_samples"],
        lambda i, n, text: print(f"[{i + 1}/{n}] {text[:60]}"),
    )
    report = summarise(outcome)

    print("\n" + "=" * 60)
    print(f"  TP {report['TP']}   TN {report['TN']}   FP {report['FP']}   FN {report['FN']}")
    print(f"  accuracy {report['accuracy']}  precision {report['precision']}  "
          f"recall {report['recall']}  f1 {report['f1']}")
    if report["false_positives"]:
        print("\n  Over-blocked (legitimate queries refused):")
        for text in report["false_positives"]:
            print(f"    - {text}")
    if report["false_negatives"]:
        print("\n  Under-blocked (should have been refused):")
        for text in report["false_negatives"]:
            print(f"    - {text}")
    print("=" * 60)
