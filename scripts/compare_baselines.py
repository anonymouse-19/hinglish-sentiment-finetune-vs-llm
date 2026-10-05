"""Compare every Phase 2 baseline on the SAME examples (the LLM runs only cover stratified subsets, D-023).

Usage:
    python scripts/compare_baselines.py

For each split, every LLM run is restricted to the ids it answered, and TF-IDF is scored on exactly those
ids too, so the rows of the table are directly comparable.

Output: reports/baselines/comparison.json (+ a printed table)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hinglish_sentiment.config import resolve  # noqa: E402
from hinglish_sentiment.eval.metrics import bootstrap_ci, classification_metrics  # noqa: E402

REPORTS = resolve("reports/baselines")


def main() -> None:
    tfidf_metrics = json.loads((REPORTS / "tfidf" / "metrics.json").read_text(encoding="utf-8"))
    rows = []
    for run_dir in sorted((REPORTS / "llm").glob("*_*_*")):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        llm = pd.read_parquet(run_dir / "predictions.parquet")
        tfidf = pd.read_parquet(REPORTS / "tfidf" / f"predictions_{m['split']}.parquet").set_index("id").loc[llm.id]
        for name, pred in [(f"{m['model']} {m['mode']}-shot", llm.pred_id.values), ("tfidf_logreg (same ids)", tfidf.pred_id.values)]:
            scores = classification_metrics(llm.label_id, pred)
            rows.append({
                "split": m["split"], "n": len(llm), "complete": m["complete"], "model": name,
                "macro_f1": scores["macro_f1"], **bootstrap_ci(llm.label_id, pred),
                "accuracy": scores["accuracy"], "n_invalid": scores["n_invalid"],
                "p50_ms": m["latency"]["p50_ms"] if "tfidf" not in name else tfidf_metrics["serving"]["latency_batch1_cpu"]["p50_ms"],
                "usd_per_1k": m["cost"]["per_1k_predictions_usd"] if "tfidf" not in name else None,
            })
    table = pd.DataFrame(rows).drop_duplicates(["split", "model", "n"])
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print(table.drop(columns=["n_resamples"]).to_string(index=False))
    (REPORTS / "comparison.json").write_text(table.to_json(orient="records", indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
