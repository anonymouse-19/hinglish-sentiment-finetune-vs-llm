"""Phase 3: hyperparameter sweep for one model family, then a single final evaluation of the winner.

Usage:
    python scripts/finetune_sweep.py --family encoder                  # Option A: XLM-R / MuRIL (9 runs)
    python scripts/finetune_sweep.py --family decoder                  # Option B: Qwen3-1.7B QLoRA (4 runs)
    python scripts/finetune_sweep.py --family encoder --smoke          # tiny models, CPU, ~1 min: pipeline check
    python scripts/finetune_sweep.py --family encoder --output-root /content/drive/MyDrive/hinglish-ft   # Colab

Resumable: a run whose result.json exists is skipped, so a Colab disconnect only loses the run in progress.

Steps:
  1. Train every grid point on train, early-stopping on val macro-F1 (each logged to W&B).
  2. Pick the best run by val macro-F1 -- test is not looked at.
  3. Reload the winner from disk, score val / test / YouTube once, time inference, measure size.

Outputs:
    <output-root>/<family>/<run>/{model/, result.json}
    <report-dir>/<family>/{sweep.csv, metrics.json, predictions_<split>.parquet}
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hinglish_sentiment import LABELS  # noqa: E402
from hinglish_sentiment.config import load_config, resolve  # noqa: E402
from hinglish_sentiment.eval.metrics import bootstrap_ci, classification_metrics, metrics_by_group  # noqa: E402

SPLITS = ("train", "val", "test", "test_youtube")


def grid_runs(cfg: dict, family: str) -> list[dict]:
    grid = cfg[family]["grid"]
    runs = []
    if family == "encoder":
        for variant, lr in itertools.product(grid["model"], grid["learning_rate"]):
            v = cfg["encoder"]["variants"][variant]
            runs.append({"name": f"{variant}_lr{lr:.0e}", "variant": variant, "model_key": v["model"],
                         "demojize": v["demojize"], "learning_rate": lr})
    else:
        for model_key, r, lr in itertools.product(grid["model"], grid["lora_r"], grid["learning_rate"]):
            runs.append({"name": f"{model_key}_r{r}_lr{lr:.0e}", "variant": model_key, "model_key": model_key,
                         "demojize": False, "lora_r": r, "learning_rate": lr})
    return runs


def prune_models(fam_out: Path, results: list[dict]) -> None:
    """Delete the weights of every run except the best so far by val macro-F1 (its result.json is kept)."""
    best = max(results, key=lambda r: r["val_macro_f1"])["name"]
    for r in results:
        if r["name"] != best and (fam_out / r["name"] / "model").exists():
            shutil.rmtree(fam_out / r["name"] / "model")


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["encoder", "decoder"], required=True)
    parser.add_argument("--config", default="configs/finetune.yaml")
    parser.add_argument("--output-root", default=None, help="where models go (default: config paths.output_root)")
    parser.add_argument("--report-dir", default=None, help="where metrics go (default: config paths.report_dir)")
    parser.add_argument("--only", nargs="*", help="run only these run names")
    parser.add_argument("--smoke", action="store_true", help="tiny models + data, CPU-friendly pipeline check")
    parser.add_argument("--skip-final", action="store_true", help="train only; don't evaluate the winner")
    parser.add_argument("--keep-all-models", action="store_true",
                        help="keep every run's weights (default: only the best-so-far, to fit a 15 GB Drive)")
    args = parser.parse_args()

    from hinglish_sentiment.finetune.train import load_trained, predict_proba, time_inference, train_run

    cfg = load_config(args.config)
    family = args.family
    output_root = Path(args.output_root) if args.output_root else resolve(cfg["paths"]["output_root"])
    report_root = Path(args.report_dir) if args.report_dir else resolve(cfg["paths"]["report_dir"])
    report_to = cfg["tracking"]["report_to"]
    data_dir = resolve(cfg["paths"]["processed_dir"])
    splits = {s: pd.read_parquet(data_dir / f"{s}.parquet") for s in SPLITS}

    if args.smoke:
        output_root, report_root, report_to = output_root.with_name(output_root.name + "_smoke"), \
            resolve("outputs/reports_smoke"), "none"
        sm = cfg["smoke"]
        splits = {s: df.sample(min(len(df), sm["n_train"] if s == "train" else sm["n_eval"]), random_state=0)
                  for s, df in splits.items()}
        cfg[family]["train"]["epochs"] = 1
        cfg["latency"].update(n_examples=5, warmup=1)

    def model_source(model_key: str) -> tuple[str, str | None]:
        if args.smoke:
            return cfg["smoke"]["models"][model_key], None
        m = cfg["models"][model_key]
        return m["hf_id"], m["revision"]

    runs = grid_runs(cfg, family)
    if args.only:
        runs = [r for r in runs if r["name"] in args.only]
    fam_out, fam_report = output_root / family, report_root / family
    fam_report.mkdir(parents=True, exist_ok=True)

    # 1. Sweep -----------------------------------------------------------------------------------------
    results = []
    for i, run in enumerate(runs, 1):
        run_dir = fam_out / run["name"]
        done = run_dir / "result.json"
        if done.exists():
            results.append(json.loads(done.read_text(encoding="utf-8")))
            print(f"[{i}/{len(runs)}] {run['name']}: already done (val macro-F1 {results[-1]['val_macro_f1']:.4f})")
            continue
        print(f"\n[{i}/{len(runs)}] {run['name']} ...", flush=True)
        hf_id, revision = model_source(run["model_key"])
        res = train_run(run, cfg, family, hf_id, revision, splits, run_dir, report_to)
        run_dir.mkdir(parents=True, exist_ok=True)
        done.write_text(json.dumps(res, indent=2), encoding="utf-8")
        results.append(res)
        print(f"  -> val macro-F1 {res['val_macro_f1']:.4f} (epoch {res['best_epoch']}, {res['train_minutes']} min)")
        if not args.keep_all_models:
            prune_models(fam_out, results)

    sweep = pd.DataFrame(results).sort_values("val_macro_f1", ascending=False)
    sweep.to_csv(fam_report / "sweep.csv", index=False)
    print("\n" + sweep[["name", "val_macro_f1", "val_accuracy", "best_epoch", "train_minutes"]].to_string(index=False))
    if args.skip_final or len(sweep) == 0:
        return

    # 2-3. Winner, chosen on val only; reload from disk and score every split once ----------------------
    best = sweep.iloc[0].to_dict()
    print(f"\nBest on val: {best['name']} ({best['val_macro_f1']:.4f}). Final evaluation ...")
    hf_id, revision = model_source(best["model_key"])
    model_dir = fam_out / best["name"] / "model"
    model, tokenizer = load_trained(model_dir, family, cfg, hf_id, revision)
    demojize = bool(best["demojize"])

    metrics = {"family": family, "best_run": best, "selected_on": "val macro-F1", "splits": {}}
    for name in ("val", "test", "test_youtube"):
        df = splits[name]
        proba = predict_proba(model, tokenizer, df.text.tolist(), demojize, cfg["max_length"])
        has_mix = "mix_bucket" in df
        out = df[["id", "label_id"] + (["mix_bucket"] if has_mix else [])].assign(pred_id=proba.argmax(1))
        for j, label in enumerate(LABELS):
            out[f"p_{label}"] = proba[:, j]
        out.to_parquet(fam_report / f"predictions_{name}.parquet", index=False)
        metrics["splits"][name] = {
            **classification_metrics(out.label_id, out.pred_id),
            **bootstrap_ci(out.label_id, out.pred_id, seed=cfg["seed"]),
            **({"by_mix_bucket": metrics_by_group(out, "mix_bucket")} if has_mix else {}),
        }
        s = metrics["splits"][name]
        print(f"{name:>13}: macro-F1 {s['macro_f1']:.4f} [{s['macro_f1_ci_low']:.3f}, {s['macro_f1_ci_high']:.3f}]")

    lat = cfg["latency"]
    metrics["serving"] = {
        **time_inference(model, tokenizer, splits["test"].text.tolist(), demojize, cfg["max_length"],
                         lat["n_examples"], lat["warmup"], lat["batch_size"]),
        "saved_model_mb": dir_size_mb(model_dir),
        "params_total": best["params_total"],
        "params_trainable": best["params_trainable"],
        "quantized_4bit": best["quantized_4bit"],
    }
    sv = metrics["serving"]
    print(f"latency p50 {sv['latency_batch1']['p50_ms']:.1f} ms on {sv['device']} · saved {sv['saved_model_mb']:.0f} MB "
          f"· {sv['params_total'] / 1e6:.0f}M params ({sv['params_trainable'] / 1e6:.1f}M trained)")
    (fam_report / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (fam_out / "BEST_RUN.txt").write_text(best["name"], encoding="utf-8")


if __name__ == "__main__":
    main()
