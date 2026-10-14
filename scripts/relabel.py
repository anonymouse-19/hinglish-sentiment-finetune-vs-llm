"""Optional human check (D-036): when the LLM and the gold label disagree, who is right?

Usage:
    python scripts/relabel.py make     # -> outputs/relabel_sheet.csv  (40 test tweets, no labels shown)
    #   ...fill the `your_label` column with negative / neutral / positive (Excel or VS Code)...
    python scripts/relabel.py score    # -> agreement of your labels with gold vs with the LLM

The sheet mixes 20 tweets where the few-shot LLM disagrees with the gold label and 20 where they agree
(a control), shuffled, so the labeller can't tell which is which. Files stay in git-ignored outputs/ (tweet text, D-004).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hinglish_sentiment import LABEL2ID, LABELS  # noqa: E402
from hinglish_sentiment.config import resolve  # noqa: E402

SHEET = resolve("outputs/relabel_sheet.csv")
KEY = resolve("outputs/relabel_key.json")   # which rows are disagreements; not shown to the labeller
LLM_RUN = resolve("reports/baselines/llm/gpt-oss-120b_few_test/predictions.parquet")


def make(n_each: int = 20, seed: int = 7) -> None:
    llm = pd.read_parquet(LLM_RUN)[["id", "label_id", "pred_id"]]
    text = pd.read_parquet(resolve("data/processed/test.parquet"))[["id", "text"]]
    df = llm.merge(text, on="id")
    dis = df[(df.pred_id != df.label_id) & (df.pred_id >= 0)].sample(n_each, random_state=seed)
    agree = df[df.pred_id == df.label_id].sample(n_each, random_state=seed)
    sheet = pd.concat([dis.assign(group="disagree"), agree.assign(group="agree")]).sample(frac=1, random_state=seed)
    SHEET.parent.mkdir(exist_ok=True)
    sheet[["id", "text"]].assign(your_label="").to_csv(SHEET, index=False, encoding="utf-8-sig")
    KEY.write_text(json.dumps(sheet[["id", "label_id", "pred_id", "group"]].to_dict("records")), encoding="utf-8")
    print(f"Wrote {len(sheet)} tweets to {SHEET}\nFill `your_label` with one of: {', '.join(LABELS)}")


def score() -> None:
    sheet = pd.read_csv(SHEET, encoding="utf-8-sig")
    key = pd.DataFrame(json.loads(KEY.read_text(encoding="utf-8")))
    df = sheet.merge(key, on="id")
    df["you"] = df.your_label.astype(str).str.strip().str.lower().map(LABEL2ID)
    done = df[df.you.notna()]
    print(f"{len(done)}/{len(df)} labelled")
    out = {}
    for group, g in done.groupby("group"):
        out[group] = {"n": len(g), "you_agree_with_gold": float((g.you == g.label_id).mean()),
                      "you_agree_with_llm": float((g.you == g.pred_id).mean())}
        print(f"{group:>9}: n={len(g):2d}  you=gold {out[group]['you_agree_with_gold']:.0%}  "
              f"you=LLM {out[group]['you_agree_with_llm']:.0%}")
    resolve("reports/evaluation").mkdir(parents=True, exist_ok=True)
    resolve("reports/evaluation/relabel_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved reports/evaluation/relabel_summary.json (numbers only, no tweet text)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["make", "score"])
    {"make": make, "score": score}[parser.parse_args().cmd]()
