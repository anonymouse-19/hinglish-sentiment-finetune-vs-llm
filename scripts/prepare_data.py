"""Phase 1 pipeline: download -> parse -> clean -> filter -> dedupe -> split -> save + stats.

Usage:
    python scripts/prepare_data.py [--config configs/data.yaml]

Outputs (data/ is git-ignored; SentiMix has no explicit licence, so we don't redistribute it):
    data/processed/{train,val,test,test_youtube}.parquet
    data/processed/dropped.parquet        every removed row + the reason (audit trail)
    reports/data_stats.json               all numbers quoted in docs/DATA_CARD.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from ftfy.badness import is_bad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hinglish_sentiment import LABEL2ID, LABELS  # noqa: E402
from hinglish_sentiment.config import load_config, resolve  # noqa: E402
from hinglish_sentiment.data.clean import (  # noqa: E402
    MOJIBAKE_SIGNATURE,
    count_accented_latin,
    detokenize,
    has_content,
    has_devanagari,
    language_stats,
    normalize_tokens,
    repair_mojibake,
)
from hinglish_sentiment.data.dedupe import exact_key, near_duplicate_clusters, resolve_duplicates  # noqa: E402
from hinglish_sentiment.data.sentimix import load_sentimix  # noqa: E402
from hinglish_sentiment.data.split import MIX_BUCKETS, mix_bucket, stratified_split  # noqa: E402
from hinglish_sentiment.data.youtube import load_youtube  # noqa: E402

COLUMNS = [
    "id", "text", "label", "label_id", "mix_bucket", "cmi", "eng_ratio", "n_hin", "n_eng", "n_words",
    "n_chars", "has_devanagari", "is_retweet", "is_truncated", "is_foreign", "no_content",
    "test_internal_dup", "official_split", "uid", "text_orig",
]


def build_features(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = raw.copy()
    df["text_orig"] = df["tokens"].map(" ".join)
    df["had_mojibake"] = df["text_orig"].map(lambda s: is_bad(s) or bool(MOJIBAKE_SIGNATURE.search(s)))
    df["is_retweet"] = df["tokens"].map(lambda t: bool(t) and t[0].strip() == "RT")
    normalized = [normalize_tokens(t, g) for t, g in zip(df["tokens"], df["lang_tags"])]
    # second repair pass: detokenizing can make a mojibake sequence contiguous again
    df["text"] = [repair_mojibake(detokenize(toks)) for toks in normalized]
    df["n_words"] = [sum(t.kind == "word" for t in toks) for toks in normalized]
    df = pd.concat([df, pd.DataFrame([language_stats(t) for t in normalized], index=df.index)], axis=1)
    df["n_chars"] = df["text"].str.len()
    df["has_devanagari"] = df["text"].map(has_devanagari)
    df["is_truncated"] = df["text"].str.contains(r"…\s*(?:http)?$")
    df["no_content"] = ~df["text"].map(has_content)
    df["is_foreign"] = df["text"].map(count_accented_latin) >= cfg["cleaning"]["foreign_min_accented_letters"]
    share = cfg["robustness"]["dominant_share"]
    df["mix_bucket"] = [mix_bucket(h, e, share) for h, e in zip(df["n_hin"], df["n_eng"])]
    df["id"] = "sm_" + df["uid"]
    df["label_id"] = df["label"].map(LABEL2ID)
    if df["label_id"].isna().any():
        raise ValueError(f"Unknown labels: {df.loc[df['label_id'].isna(), 'label'].unique()}")
    return df


def dist(series: pd.Series, order=None) -> dict:
    counts = series.value_counts()
    if order is not None:
        counts = counts.reindex(order, fill_value=0)
    return {str(k): int(v) for k, v in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw_dir, out_dir, reports_dir = (resolve(cfg["paths"][k]) for k in ("raw_dir", "processed_dir", "reports_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {"config": cfg}

    # 1. Load + clean ---------------------------------------------------------------------------
    print("Loading SentiMix ...")
    raw = load_sentimix(cfg["sentimix"]["repo_id"], cfg["sentimix"]["revision"], raw_dir)
    stats["raw"] = {s: {"n": int((raw.official_split == s).sum()), "labels": dist(raw.label[raw.official_split == s], LABELS)}
                    for s in ("train", "dev", "test")}
    print("Cleaning tokens ...")
    df = build_features(raw, cfg)
    stats["cleaning"] = {
        "tweets_with_mojibake_repaired": int(df.had_mojibake.sum()),
        "retweets": int(df.is_retweet.sum()),
        "truncated_tweets": int(df.is_truncated.sum()),
        "tweets_with_devanagari": int(df.has_devanagari.sum()),
        "residual_split_mentions": int(df.text.str.contains(r"(?:^|\s)@\s").sum()),
        "residual_split_urls": int(df.text.str.contains(r"https?\s*//").sum()),
        "residual_mojibake": int(df.text.map(lambda s: is_bad(s) or bool(MOJIBAKE_SIGNATURE.search(s))).sum()),
    }

    # 2. Filters (train/dev only; the official test set is never modified) ---------------------
    is_test = df.official_split == "test"
    df["drop_reason"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[~is_test & df.no_content, "drop_reason"] = "no_content"
    df.loc[~is_test & df.drop_reason.isna() & df.is_foreign, "drop_reason"] = "non_hinglish_language"

    # 3. Dedupe --------------------------------------------------------------------------------
    keys = df.text.map(exact_key)
    exact_groups = df[df.text.map(has_content)].assign(key=keys).groupby("key")
    exact_multi = exact_groups.filter(lambda g: len(g) > 1).groupby("key")
    stats["exact_duplicates_after_masking"] = {
        "groups": int(exact_multi.ngroups),
        "rows": int(exact_multi.size().sum()),
        "groups_spanning_official_splits": int((exact_multi.official_split.nunique() > 1).sum()),
        "groups_with_conflicting_labels": int((exact_multi.label.nunique() > 1).sum()),
        "exact_duplicates_before_masking": int(df.text_orig.str.lower().duplicated(keep=False).sum()),
    }

    candidates = df[df.drop_reason.isna() & ~df.no_content]
    print(f"Near-duplicate search over {len(candidates)} tweets ...")
    clusters = near_duplicate_clusters(candidates.text.tolist(), cfg["dedupe"]["near_dup_threshold"])
    df["dup_cluster"] = -1
    df.loc[candidates.index, "dup_cluster"] = clusters
    dup_reasons = resolve_duplicates(df.loc[candidates.index])
    df.loc[dup_reasons.dropna().index, "drop_reason"] = dup_reasons.dropna()

    in_cluster = df[df.dup_cluster >= 0]
    cluster_sizes = in_cluster.groupby("dup_cluster").size()
    multi = in_cluster[in_cluster.dup_cluster.isin(cluster_sizes[cluster_sizes > 1].index)]
    test_counts = df[is_test & (df.dup_cluster >= 0)].groupby("dup_cluster").size()
    df["test_internal_dup"] = is_test & df.dup_cluster.isin(test_counts[test_counts > 1].index)
    stats["near_duplicates"] = {
        "threshold": cfg["dedupe"]["near_dup_threshold"],
        "clusters_with_2plus": int(multi.dup_cluster.nunique()),
        "rows_in_those_clusters": int(len(multi)),
        "largest_cluster": int(cluster_sizes.max()),
        "clusters_with_conflicting_labels": int((multi.groupby("dup_cluster").label.nunique() > 1).sum()),
        "test_rows_with_a_test_near_duplicate": int(df.test_internal_dup.sum()),
    }
    stats["dropped"] = {
        reason: dist(df.official_split[df.drop_reason == reason], ["train", "dev"])
        for reason in df.drop_reason.dropna().unique()
    }

    # 4. Split: pool cleaned train+dev -> stratified train/val; official test untouched --------
    kept = df[df.drop_reason.isna() & ~is_test]
    train, val = stratified_split(kept, cfg["split"]["val_size"], cfg["seed"], cfg["split"]["stratify_on"])
    test = df[is_test]
    splits = {"train": train, "val": val, "test": test}
    for name, part in splits.items():
        part[COLUMNS].reset_index(drop=True).to_parquet(out_dir / f"{name}.parquet", index=False)
    dropped = df[df.drop_reason.notna()]
    dropped[["id", "official_split", "label", "drop_reason", "dup_cluster", "text", "text_orig"]].to_parquet(
        out_dir / "dropped.parquet", index=False
    )

    stats["final"] = {}
    for name, part in splits.items():
        stats["final"][name] = {
            "n": int(len(part)),
            "labels": dist(part.label, LABELS),
            "mix_bucket": dist(part.mix_bucket, MIX_BUCKETS),
            "mix_bucket_x_label": {b: dist(part.label[part.mix_bucket == b], LABELS) for b in MIX_BUCKETS},
            "chars": {k: round(float(v), 1) for k, v in part.n_chars.describe(percentiles=[0.5, 0.95]).items()},
            "words": {k: round(float(v), 1) for k, v in part.n_words.describe(percentiles=[0.5, 0.95]).items()},
            "cmi_mean": round(float(part.cmi.mean()), 2),
            "has_devanagari": int(part.has_devanagari.sum()),
            "retweets": int(part.is_retweet.sum()),
            "truncated": int(part.is_truncated.sum()),
        }
    stats["test_flags"] = {
        "no_content": int(test.no_content.sum()),
        "non_hinglish_language": int(test.is_foreign.sum()),
        "has_near_duplicate_in_test": int(test.test_internal_dup.sum()),
    }

    # 5. Out-of-domain YouTube test set ----------------------------------------------------------
    print("Loading YouTube OOD set ...")
    yt_cfg = cfg["youtube_ood"]
    yt = load_youtube(yt_cfg["repo_id"], yt_cfg["revision"], yt_cfg["filename"], raw_dir / "youtube")
    yt_raw_n, yt_raw_labels = len(yt), dist(yt.label, LABELS)
    yt["drop_reason"] = pd.Series(pd.NA, index=yt.index, dtype="object")
    yt.loc[~yt.text.map(has_content), "drop_reason"] = "no_content"
    ok = yt.drop_reason.isna()
    yt["dup_cluster"] = -1
    yt.loc[ok, "dup_cluster"] = pd.factorize(yt.loc[ok, "text"].map(exact_key))[0]
    yt["official_split"] = "youtube"
    yt_reasons = resolve_duplicates(yt.loc[ok])
    yt.loc[yt_reasons.dropna().index, "drop_reason"] = yt_reasons.dropna()
    yt_final = yt[yt.drop_reason.isna()].copy()
    yt_final["label_id"] = yt_final.label.map(LABEL2ID)
    yt_final["n_chars"] = yt_final.text.str.len()
    yt_final[["id", "text", "label", "label_id", "n_chars", "has_devanagari", "video_id", "text_orig"]].reset_index(
        drop=True
    ).to_parquet(out_dir / "test_youtube.parquet", index=False)
    stats["youtube_ood"] = {
        "raw_n": yt_raw_n,
        "raw_labels": yt_raw_labels,
        "dropped": dist(yt.drop_reason.dropna()),
        "final_n": int(len(yt_final)),
        "final_labels": dist(yt_final.label, LABELS),
        "videos": int(yt_final.video_id.nunique()),
        "chars": {k: round(float(v), 1) for k, v in yt_final.n_chars.describe(percentiles=[0.5, 0.95]).items()},
    }

    with open(reports_dir / "data_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in stats.items() if k != "config"}, indent=2, ensure_ascii=False))
    for name, part in splits.items():
        print(f"{name:>5}: {len(part):6d}  {dist(part.label, LABELS)}")
    print(f"youtube: {len(yt_final):6d}  {dist(yt_final.label, LABELS)}")


if __name__ == "__main__":
    main()
