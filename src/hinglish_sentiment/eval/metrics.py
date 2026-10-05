"""Metrics shared by every model in the comparison (baselines, LLMs, fine-tuned models).

Predictions are label ids in {0, 1, 2}. An LLM answer that can't be parsed is encoded as INVALID (-1): it
is scored as a wrong prediction for the true class, never silently dropped (DECISION_LOG D-019).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from hinglish_sentiment import LABELS

INVALID = -1
LABEL_IDS = list(range(len(LABELS)))


def classification_metrics(y_true, y_pred) -> dict:
    """Macro-F1 (headline), accuracy, weighted-F1, per-class P/R/F1 and the confusion matrix."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_IDS, zero_division=0)
    # rows = true label, columns = predicted label (+ an "invalid" column for unparseable LLM answers)
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_IDS + [INVALID])[: len(LABELS)]
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABEL_IDS, average="weighted", zero_division=0)),
        "n_invalid": int((y_pred == INVALID).sum()),
        "per_class": {
            label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": {"labels_true": LABELS, "labels_pred": LABELS + ["invalid"], "matrix": cm.tolist()},
    }


def bootstrap_ci(y_true, y_pred, n_resamples: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for macro-F1: how much the score would move on a different test sample."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n_resamples, len(y_true)))
    scores = [f1_score(y_true[i], y_pred[i], labels=LABEL_IDS, average="macro", zero_division=0) for i in idx]
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return {"macro_f1_ci_low": float(lo), "macro_f1_ci_high": float(hi), "n_resamples": n_resamples}


def metrics_by_group(df: pd.DataFrame, group_col: str, true_col: str = "label_id", pred_col: str = "pred_id") -> dict:
    """Macro-F1 / accuracy per subset (e.g. language-mix bucket), with the subset's label distribution.

    The label mix differs a lot between buckets (mostly-Hindi tweets are more often negative), so each
    subset's label distribution is reported next to its score.
    """
    out = {}
    for name, g in df.groupby(group_col):
        out[str(name)] = {
            "n": int(len(g)),
            "macro_f1": float(f1_score(g[true_col], g[pred_col], labels=LABEL_IDS, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(g[true_col], g[pred_col])),
            "label_share": {LABELS[k]: float(v) for k, v in g[true_col].value_counts(normalize=True).sort_index().items()},
        }
    return out


def latency_summary(latencies_ms) -> dict:
    lat = np.asarray(latencies_ms, dtype=float)
    if lat.size == 0:
        return {"n": 0}
    return {
        "n": int(lat.size),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "mean_ms": float(lat.mean()),
    }
