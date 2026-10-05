"""Phase 2 baseline: zero-/few-shot classification with a large LLM behind an OpenAI-compatible API.

Usage:
    python scripts/run_llm_baseline.py --mode zero --split val            # develop prompts on val
    python scripts/run_llm_baseline.py --mode few  --split test           # score test once
    python scripts/run_llm_baseline.py --mode zero --split val --dry-run  # print the prompt, no API call
    python scripts/run_llm_baseline.py --mode zero --split val --n 20     # override subset size (smoke test)

Interrupted or out of daily quota? Re-run the same command: answered requests come from the cache.

Outputs (reports/baselines/llm/<model>_<mode>_<split>/):
    predictions.parquet   one row per example: answer, parsed label, tokens, latency, cost
    metrics.json          metrics, robustness by language mix, latency p50/p95, tokens, cost per 1k
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import openai
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hinglish_sentiment.config import PROJECT_ROOT, load_config, resolve  # noqa: E402
from hinglish_sentiment.data.split import stratified_subset  # noqa: E402
from hinglish_sentiment.eval.metrics import (  # noqa: E402
    bootstrap_ci,
    classification_metrics,
    latency_summary,
    metrics_by_group,
)
from hinglish_sentiment.llm.client import LLMClient, ResponseCache  # noqa: E402
from hinglish_sentiment.llm.prompts import build_messages, parse_label  # noqa: E402


def load_api_key(env_name: str) -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get(env_name)
    if not key:
        sys.exit(f"{env_name} is not set. Put it in a .env file in the project root (see .env.example).")
    return key


def summarize(df: pd.DataFrame, seed: int) -> dict:
    """Cached rows keep the latency measured when they were first requested, so all rows count for latency."""
    n = len(df)
    return {
        **classification_metrics(df.label_id, df.pred_id),
        **bootstrap_ci(df.label_id, df.pred_id, seed=seed),
        **({"by_mix_bucket": metrics_by_group(df, "mix_bucket")} if "mix_bucket" in df else {}),
        "latency": latency_summary(df.latency_ms),
        "tokens": {
            "prompt_mean": float(df.prompt_tokens.mean()),
            "completion_mean": float(df.completion_tokens.mean()),
            "reasoning_mean": float(df.reasoning_tokens.mean()),
            "cached_prompt_share": float(df.cached_tokens.sum() / max(df.prompt_tokens.sum(), 1)),
            "finish_reason_length": int((df.finish_reason == "length").sum()),
        },
        "cost": {
            "total_usd": float(df.cost_usd.sum()),
            "per_1k_predictions_usd": float(df.cost_usd.sum() / n * 1000),
        },
        "requests_answered_from_cache_this_run": int(df.from_cache_at_run.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llm.yaml")
    parser.add_argument("--mode", choices=["zero", "few"], required=True)
    parser.add_argument("--split", choices=["val", "test", "test_youtube"], required=True)
    parser.add_argument("--n", type=int, default=None, help="override the configured subset size")
    parser.add_argument("--dry-run", action="store_true", help="print one prompt and the plan; no API calls")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"]
    data_dir = resolve(cfg["paths"]["processed_dir"])

    df = pd.read_parquet(data_dir / f"{args.split}.parquet")
    strat = ["label", "mix_bucket"] if "mix_bucket" in df else ["label"]
    n = args.n if args.n is not None else cfg["eval"]["n"].get(args.split)
    df = stratified_subset(df, n, seed, strat).reset_index(drop=True)

    examples = None
    if args.mode == "few":
        ids_path = resolve(cfg["few_shot"]["ids_path"])
        if not ids_path.exists():
            sys.exit(f"{ids_path} not found: run `python scripts/select_few_shot.py` first.")
        ids = load_config(ids_path)["ids"]
        train = pd.read_parquet(data_dir / "train.parquet").set_index("id")
        examples = train.loc[ids].reset_index()

    model_slug = cfg["model"].split("/")[-1]
    out_dir = resolve(cfg["paths"]["report_dir"]) / f"{model_slug}_{args.mode}_{args.split}"
    print(f"{cfg['provider']}:{cfg['model']} · {args.mode}-shot · {args.split} · {len(df)} examples -> {out_dir}")

    if args.dry_run:
        for m in build_messages(df.text.iloc[0], examples):
            print(f"\n--- {m['role']} ---\n{m['content']}")
        chars = sum(len(m["content"]) for m in build_messages(df.text.iloc[0], examples))
        print(f"\n~{chars / 3.5:.0f} prompt tokens per request (rough estimate, chars / 3.5)")
        return

    client = LLMClient(cfg, load_api_key(cfg["api_key_env"]), ResponseCache(resolve(cfg["paths"]["cache_path"])))
    rows, stopped_early = [], None
    for i, ex in enumerate(df.itertuples()):
        try:
            r = client.complete(build_messages(ex.text, examples))
        except (openai.RateLimitError, openai.APIError) as exc:
            stopped_early = f"{type(exc).__name__}: {exc}"
            print(f"\nStopped at {i}/{len(df)}: {stopped_early}\nRe-run the same command later to resume.")
            break
        rows.append({"id": ex.id, "label_id": ex.label_id, **({"mix_bucket": ex.mix_bucket} if "mix_bucket" in df else {}),
                     "answer": r.content, "pred_id": parse_label(r.content), "finish_reason": r.finish_reason,
                     "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                     "reasoning_tokens": r.reasoning_tokens, "cached_tokens": r.cached_tokens,
                     "latency_ms": r.latency_ms, "attempts": r.attempts, "cost_usd": r.cost_usd,
                     "from_cache_at_run": r.from_cache})
        if (i + 1) % 50 == 0:
            done = pd.DataFrame(rows)
            print(f"  {i + 1}/{len(df)}  acc so far {(done.pred_id == done.label_id).mean():.3f}  "
                  f"p50 {done.latency_ms.median():.0f} ms  ${done.cost_usd.sum():.4f}")

    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame(rows)
    pred.to_parquet(out_dir / "predictions.parquet", index=False)
    result = {
        "provider": cfg["provider"], "model": cfg["model"], "mode": args.mode, "split": args.split,
        "n_planned": len(df), "complete": stopped_early is None, "stopped_early": stopped_early,
        "generation": cfg["generation"], "prices": cfg["prices"],
        "few_shot_ids": examples.id.tolist() if examples is not None else [],
        **summarize(pred, seed),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"\n{'complete' if result['complete'] else 'PARTIAL'}: n={result['n']}  macro-F1 {result['macro_f1']:.4f} "
          f"[{result['macro_f1_ci_low']:.3f}, {result['macro_f1_ci_high']:.3f}]  invalid {result['n_invalid']}  "
          f"p50 {result['latency']['p50_ms']:.0f} ms  ${result['cost']['per_1k_predictions_usd']:.4f} / 1k")


if __name__ == "__main__":
    main()
