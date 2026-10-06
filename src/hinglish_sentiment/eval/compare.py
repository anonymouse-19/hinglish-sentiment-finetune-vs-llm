"""Phase 4: put every model's results on one footing (same tweets, same metrics, same cost definition).

Each model's predictions and serving numbers are loaded from the files the earlier phases wrote, so this
module never re-runs a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from hinglish_sentiment.eval.metrics import LABEL_IDS


@dataclass
class ModelResult:
    key: str                       # short id, e.g. "tfidf", "llm_few", "encoder", "decoder"
    name: str                      # display name
    family: str                    # classical | llm | fine-tuned
    predictions: dict[str, pd.DataFrame] = field(default_factory=dict)  # split -> id, label_id, pred_id[, mix_bucket]
    p50_ms: float | None = None
    latency_basis: str = ""
    usd_per_1k: float | None = None
    cost_basis: str = ""
    params: str = ""
    size: str = ""
    complete: bool = True          # False while an LLM run is still partial


def cost_per_1k_self_hosted(throughput_per_s: float, usd_per_hour: float) -> float:
    """$ per 1,000 predictions for a machine kept busy at the measured throughput."""
    return usd_per_hour / (throughput_per_s * 3600) * 1000


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0))


def paired_bootstrap(y_true, pred_a, pred_b, n_resamples: int = 2000, seed: int = 42) -> dict:
    """Macro-F1(a) - macro-F1(b) on the same examples, with a 95% CI from resampling *tweets*.

    Pairing (both models scored on each resample) removes the shared difficulty of the sample, so this is
    much more sensitive than comparing two independent confidence intervals (D-024).
    """
    y, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_resamples, len(y)))
    diffs = np.array([macro_f1(y[i], a[i]) - macro_f1(y[i], b[i]) for i in idx])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": macro_f1(y, a) - macro_f1(y, b), "ci_low": float(lo), "ci_high": float(hi),
            "p_a_not_better": float((diffs <= 0).mean()), "n": int(len(y))}


def _fmt_params(n: float) -> str:
    return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.0f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def load_tfidf(root: Path, hw: dict) -> ModelResult | None:
    d = root / "tfidf"
    if not (d / "metrics.json").exists():
        return None
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    sv = m["serving"]
    return ModelResult(
        key="tfidf", name="TF-IDF + LogReg", family="classical",
        predictions={s: pd.read_parquet(d / f"predictions_{s}.parquet") for s in m["splits"]},
        p50_ms=sv["latency_batch1_cpu"]["p50_ms"], latency_basis="laptop CPU, batch 1",
        usd_per_1k=cost_per_1k_self_hosted(sv["throughput_batch_full_test_per_s"], hw["usd_per_hour"]),
        cost_basis=f"{hw['instance']} @ ${hw['usd_per_hour']}/h, laptop-measured throughput",
        params=_fmt_params(sv["n_parameters"]), size=f"{sv['model_size_mb']:.0f} MB")


def load_llm_runs(root: Path) -> list[ModelResult]:
    """One ModelResult per prompting mode; each split's predictions come from its own run directory."""
    out: dict[str, ModelResult] = {}
    for run in sorted((root / "llm").glob("*_*_*")):
        if not (run / "metrics.json").exists():
            continue
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        key = f"llm_{m['mode']}"
        r = out.setdefault(key, ModelResult(
            key=key, name=f"{m['model'].split('/')[-1]} {m['mode']}-shot", family="llm",
            latency_basis=f"{m['provider']} API, incl. network", cost_basis=f"billed tokens x list price ({m['prices']['as_of']})",
            params="117B (MoE, 5.1B active)", size="hosted"))
        r.predictions[m["split"]] = pd.read_parquet(run / "predictions.parquet")
        if m["split"] == "test" or r.p50_ms is None:  # prefer test-set serving numbers
            r.p50_ms = m["latency"]["p50_ms"]
            r.usd_per_1k = m["cost"]["per_1k_predictions_usd"]
        r.complete = r.complete and m["complete"]
    return list(out.values())


def load_finetuned(root: Path, family: str, hw: dict) -> ModelResult | None:
    d = root / family
    if not (d / "metrics.json").exists():
        return None
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    sv, best = m["serving"], m["best_run"]
    tput = next(v for k, v in sv.items() if k.startswith("throughput"))
    if family == "encoder":
        name = {"xlmr": "XLM-R base", "muril": "MuRIL base", "muril_demoji": "MuRIL base (emoji→text)"}[best["variant"]]
        size = f"{sv['saved_model_mb']:.0f} MB"
        params = _fmt_params(sv["params_total"])
    else:
        name = f"Qwen3-1.7B QLoRA (r={best['lora_r']})"
        size = f"{sv['saved_model_mb']:.0f} MB adapter + 4-bit base ({sv['memory_footprint_mb']:.0f} MB in memory)"
        params = f"{_fmt_params(sv['params_total'])} ({_fmt_params(sv['params_trainable'])} trained)"
    return ModelResult(
        key=family, name=name, family="fine-tuned",
        predictions={s: pd.read_parquet(d / f"predictions_{s}.parquet") for s in m["splits"]},
        p50_ms=sv["latency_batch1"]["p50_ms"], latency_basis=f"{sv['device']}, batch 1",
        usd_per_1k=cost_per_1k_self_hosted(tput, hw["usd_per_hour"]),
        cost_basis=f"{hw['instance']} @ ${hw['usd_per_hour']}/h, batch-64 throughput",
        params=params, size=size)


def on_ids(pred: pd.DataFrame, ids) -> pd.DataFrame:
    """A model's predictions restricted to (and ordered by) the given tweet ids."""
    return pred.set_index("id").loc[list(ids)].reset_index()
