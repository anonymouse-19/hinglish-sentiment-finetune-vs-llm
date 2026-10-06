"""Phase 4: sample one model's mistakes for manual error analysis, with every model's prediction alongside.

Usage:
    python scripts/error_analysis.py                    # the best fine-tuned model (or best available)
    python scripts/error_analysis.py --model tfidf --n 30

Mistakes are spread over the six confusion types (true -> predicted), highest-confidence first, because a
confident mistake says more about the model (or the label) than a coin flip does.

Output: outputs/error_analysis_<model>.md -- git-ignored, since it quotes tweet text (D-004).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hinglish_sentiment import LABELS  # noqa: E402
from hinglish_sentiment.config import load_config, resolve  # noqa: E402
from hinglish_sentiment.eval.compare import load_finetuned, load_llm_runs, load_tfidf  # noqa: E402

SHORT = {0: "neg", 1: "neu", 2: "pos", -1: "invalid"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="tfidf | llm_zero | llm_few | encoder | decoder")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=24, help="mistakes to sample (spread over confusion types)")
    args = parser.parse_args()

    cfg = load_config("configs/cost.yaml")
    hw = cfg["hardware"]
    base, ft = resolve(cfg["paths"]["baselines_dir"]), resolve(cfg["paths"]["finetune_dir"])
    models = {m.key: m for m in [load_tfidf(base, hw["cpu_small"]), *load_llm_runs(base),
                                 load_finetuned(ft, "encoder", hw["gpu_t4"]), load_finetuned(ft, "decoder", hw["gpu_t4"])]
              if m is not None and args.split in m.predictions}
    key = args.model or next(k for k in ("encoder", "decoder", "llm_few", "tfidf") if k in models)
    target = models[key]

    data = pd.read_parquet(resolve("data/processed") / f"{args.split}.parquet")[["id", "text", "label_id", "mix_bucket"]]
    df = data.merge(target.predictions[args.split][["id", "pred_id"] + [c for c in target.predictions[args.split]
                                                                         if c.startswith("p_")]], on="id")
    for k, m in models.items():
        if k != key:
            df = df.merge(m.predictions[args.split][["id", "pred_id"]].rename(columns={"pred_id": f"pred_{k}"}),
                          on="id", how="left")
    wrong = df[df.pred_id != df.label_id].copy()
    if any(c.startswith("p_") for c in wrong):
        wrong["confidence"] = wrong[[f"p_{l}" for l in LABELS]].max(axis=1)
    else:
        wrong["confidence"] = 0.0
    per_type = max(1, args.n // 6)
    picked = (wrong.sort_values("confidence", ascending=False)
              .groupby(["label_id", "pred_id"], group_keys=False).head(per_type))

    others = [k for k in models if k != key]
    lines = [f"# Error analysis: {target.name} on {args.split}",
             f"{len(wrong)} mistakes out of {len(df)} ({len(wrong) / len(df):.1%}). Showing {len(picked)}, "
             f"spread over confusion types, most confident first.", "",
             "Confusion counts (true → predicted):", ""]
    for (t, p), g in wrong.groupby(["label_id", "pred_id"]):
        lines.append(f"- {SHORT[t]} → {SHORT[p]}: {len(g)}")
    lines.append("")
    for i, r in enumerate(picked.itertuples(), 1):
        other = " · ".join(f"{models[k].name}: {SHORT.get(getattr(r, f'pred_{k}'), '–') if pd.notna(getattr(r, f'pred_{k}')) else '–'}"
                           for k in others)
        lines += [f"## {i}. true **{SHORT[r.label_id]}** → predicted **{SHORT[r.pred_id]}** "
                  f"(confidence {r.confidence:.2f}, {r.mix_bucket}, `{r.id}`)",
                  f"> {r.text}", "", f"Other models: {other}", "", "Why: _…_", ""]
    out = resolve("outputs") / f"error_analysis_{key}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(wrong)} mistakes; wrote {len(picked)} examples -> {out}")


if __name__ == "__main__":
    main()
