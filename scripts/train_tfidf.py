"""Phase 2 baseline: TF-IDF + logistic regression.

Usage:
    python scripts/train_tfidf.py [--config configs/tfidf.yaml]

1. Grid search on train -> validation macro-F1 (the test set is not looked at).
2. Refit the best config on train, score val / test / YouTube (OOD) once.
3. Time single-example CPU predictions (p50) and measure the serialized model size.

Outputs:
    reports/baselines/tfidf/sweep.csv            every grid point with its val scores
    reports/baselines/tfidf/metrics.json         metrics, robustness by language mix, latency, size
    reports/baselines/tfidf/predictions_<split>.parquet
    outputs/tfidf_lr/model.joblib                (git-ignored)
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hinglish_sentiment import LABELS  # noqa: E402
from hinglish_sentiment.baselines.tfidf import build_pipeline  # noqa: E402
from hinglish_sentiment.config import load_config, resolve  # noqa: E402
from hinglish_sentiment.eval.metrics import (  # noqa: E402
    bootstrap_ci,
    classification_metrics,
    latency_summary,
    metrics_by_group,
)

EVAL_SPLITS = ("val", "test", "test_youtube")


def time_single_predictions(model, texts: list[str], warmup: int) -> list[float]:
    for t in texts[:warmup]:
        model.predict([t])
    out = []
    for t in texts:
        start = time.perf_counter()
        model.predict([t])
        out.append((time.perf_counter() - start) * 1000)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tfidf.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_dir, report_dir = resolve(cfg["paths"]["processed_dir"]), resolve(cfg["paths"]["report_dir"])
    model_path = resolve(cfg["paths"]["model_path"])
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(data_dir / "train.parquet")
    splits = {s: pd.read_parquet(data_dir / f"{s}.parquet") for s in EVAL_SPLITS}

    # 1. Grid search on validation ---------------------------------------------------------------
    rows = []
    grid = cfg["grid"]
    for features, C, cw in itertools.product(grid["features"], grid["C"], grid["class_weight"]):
        model = build_pipeline(features, C, cw, cfg["seed"])
        start = time.perf_counter()
        model.fit(train.text, train.label_id)
        fit_s = time.perf_counter() - start
        m = classification_metrics(splits["val"].label_id, model.predict(splits["val"].text))
        rows.append({"features": features, "C": C, "class_weight": cw or "none", "val_macro_f1": m["macro_f1"],
                     "val_accuracy": m["accuracy"], "fit_seconds": round(fit_s, 2)})
        print(f"{features:>9}  C={C:<5} cw={cw or 'none':<8}  val macro-F1={m['macro_f1']:.4f}  ({fit_s:.1f}s)")
    sweep = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    sweep.to_csv(report_dir / "sweep.csv", index=False)
    best = sweep.iloc[0]
    best_params = {"features": best.features, "C": float(best.C),
                   "class_weight": None if best.class_weight == "none" else best.class_weight}
    print(f"\nBest on val: {best_params} -> macro-F1 {best.val_macro_f1:.4f}")

    # 2. Refit best on train, score every split once ---------------------------------------------
    model = build_pipeline(**best_params, seed=cfg["seed"])
    model.fit(train.text, train.label_id)
    joblib.dump(model, model_path)

    results = {"model": "tfidf_logreg", "best_params": best_params, "train_size": len(train), "splits": {}}
    for name, df in splits.items():
        proba = model.predict_proba(df.text)
        pred = proba.argmax(axis=1)
        has_mix = "mix_bucket" in df  # YouTube comments have no word-level language tags
        out = df[["id", "label_id"] + (["mix_bucket"] if has_mix else [])].assign(pred_id=pred)
        for i, label in enumerate(LABELS):
            out[f"p_{label}"] = proba[:, i]
        out.to_parquet(report_dir / f"predictions_{name}.parquet", index=False)
        results["splits"][name] = {
            **classification_metrics(out.label_id, out.pred_id),
            **bootstrap_ci(out.label_id, out.pred_id, seed=cfg["seed"]),
            **({"by_mix_bucket": metrics_by_group(out, "mix_bucket")} if has_mix else {}),
        }
        s = results["splits"][name]
        print(f"{name:>13}: macro-F1 {s['macro_f1']:.4f} [{s['macro_f1_ci_low']:.3f}, {s['macro_f1_ci_high']:.3f}]"
              f"  acc {s['accuracy']:.4f}")

    # 3. Serving characteristics -----------------------------------------------------------------
    texts = splits["test"].text.tolist()[: cfg["latency"]["n_examples"]]
    lat = time_single_predictions(model, texts, cfg["latency"]["warmup"])
    start = time.perf_counter()
    model.predict(splits["test"].text)
    batch_s = time.perf_counter() - start
    n_features = sum(len(v.vocabulary_) for _, v in model.named_steps["features"].transformer_list)
    results["serving"] = {
        "latency_batch1_cpu": latency_summary(lat),
        "throughput_batch_full_test_per_s": float(len(splits["test"]) / batch_s),
        "model_size_mb": model_path.stat().st_size / 1e6,
        "n_features": int(n_features),
        "n_parameters": int(np.prod(model.named_steps["clf"].coef_.shape) + len(LABELS)),
        "hardware": f"{platform.processor() or platform.machine()} (local CPU, single process)",
    }
    print(f"latency p50 {results['serving']['latency_batch1_cpu']['p50_ms']:.2f} ms · "
          f"size {results['serving']['model_size_mb']:.1f} MB · {n_features:,} features")
    with open(report_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
