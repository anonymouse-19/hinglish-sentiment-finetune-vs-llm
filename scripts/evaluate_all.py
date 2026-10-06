"""Phase 4: compare every model on the same tweets; write tables, significance tests and figures.

Usage:
    python scripts/evaluate_all.py

Works with whatever results exist (missing models are skipped, partial LLM runs are flagged), so it can be
re-run as the LLM quota and the Colab runs complete.

Outputs:
    reports/evaluation/summary.json       every number below
    reports/evaluation/results.md         the tables, ready to paste into README / docs
    reports/figures/cost_vs_f1.png, robustness.png, confusion_matrices.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_data import BASELINE, GRID, INK, INK_2, MUTED, SURFACE, _title  # noqa: E402  (shared figure style)

from hinglish_sentiment import LABELS  # noqa: E402
from hinglish_sentiment.config import load_config, resolve  # noqa: E402
from hinglish_sentiment.eval.compare import (  # noqa: E402
    load_finetuned,
    load_llm_runs,
    load_tfidf,
    macro_f1,
    on_ids,
    paired_bootstrap,
)
from hinglish_sentiment.eval.metrics import bootstrap_ci, classification_metrics  # noqa: E402

# One colour per model *family* (identity), fixed order from the validated categorical palette.
FAMILY_COLORS = {"classical": "#2a78d6", "llm": "#eb6834", "fine-tuned": "#1baf7a"}
BUCKETS = [("code_mixed", "Code-mixed"), ("mostly_hindi", "Mostly Hindi"), ("mostly_english", "Mostly English")]


def subset_ids(models, split: str) -> tuple[list[str], bool]:
    """The tweets every available model answered on this split (the LLM runs cover a stratified subset)."""
    have = [m for m in models if split in m.predictions]
    if not have:
        return [], True
    ids = set(have[0].predictions[split].id)
    for m in have[1:]:
        ids &= set(m.predictions[split].id)
    order = have[0].predictions[split].id
    return [i for i in order if i in ids], all(m.complete for m in have)


def score_rows(models, split: str, ids: list[str]) -> list[dict]:
    rows = []
    for m in models:
        if split not in m.predictions:
            continue
        p = on_ids(m.predictions[split], ids)
        c = classification_metrics(p.label_id, p.pred_id)
        rows.append({"key": m.key, "name": m.name, "family": m.family, "n": len(p), "macro_f1": c["macro_f1"],
                     **bootstrap_ci(p.label_id, p.pred_id), "accuracy": c["accuracy"], "n_invalid": c["n_invalid"],
                     "f1_per_class": {k: v["f1"] for k, v in c["per_class"].items()},
                     "confusion": c["confusion_matrix"]["matrix"], "p50_ms": m.p50_ms,
                     "latency_basis": m.latency_basis, "usd_per_1k": m.usd_per_1k, "cost_basis": m.cost_basis,
                     "params": m.params, "size": m.size, "complete": m.complete})
    return sorted(rows, key=lambda r: -r["macro_f1"])


def by_bucket(models, split: str, ids: list[str]) -> dict:
    out = {}
    for m in models:
        if split not in m.predictions:
            continue
        p = on_ids(m.predictions[split], ids)
        if "mix_bucket" not in p:
            ref = next(x for x in models if split in x.predictions and "mix_bucket" in x.predictions[split])
            p = p.merge(ref.predictions[split][["id", "mix_bucket"]], on="id")
        out[m.key] = {b: {"n": int((p.mix_bucket == b).sum()),
                          "macro_f1": macro_f1(p.label_id[p.mix_bucket == b], p.pred_id[p.mix_bucket == b])}
                      for b, _ in BUCKETS}
    return out


def md_table(rows: list[dict]) -> str:
    lines = ["| Model | Macro-F1 [95% CI] | Accuracy | p50 latency | Cost / 1k predictions | Parameters | Size |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        cost = "–" if r["usd_per_1k"] is None else f"${r['usd_per_1k']:.4f}" if r["usd_per_1k"] >= 0.0001 else f"${r['usd_per_1k']:.6f}"
        lat = "–" if r["p50_ms"] is None else f"{r['p50_ms']:.1f} ms" if r["p50_ms"] < 100 else f"{r['p50_ms']:.0f} ms"
        flag = "" if r["complete"] else " *(partial)*"
        lines.append(f"| {r['name']}{flag} | **{r['macro_f1']:.3f}** [{r['macro_f1_ci_low']:.3f}, {r['macro_f1_ci_high']:.3f}] "
                     f"| {r['accuracy']:.3f} | {lat} ({r['latency_basis']}) | {cost} | {r['params']} | {r['size']} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------- figures
def fig_cost_vs_f1(rows: list[dict], path: Path, n: int, split: str) -> None:
    rows = [r for r in rows if r["usd_per_1k"]]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.14)
    for r in rows:
        ax.scatter(r["usd_per_1k"], r["macro_f1"], s=70, color=FAMILY_COLORS[r["family"]], edgecolor=SURFACE,
                   linewidth=2, zorder=3)
        ax.plot([r["usd_per_1k"]] * 2, [r["macro_f1_ci_low"], r["macro_f1_ci_high"]], color=FAMILY_COLORS[r["family"]],
                linewidth=2, alpha=0.45, zorder=2, solid_capstyle="round")
        ax.annotate(r["name"], (r["usd_per_1k"], r["macro_f1"]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("Cost per 1,000 predictions (USD, log scale)")
    ax.set_ylabel("Macro-F1")
    ax.grid(axis="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    xs = [r["usd_per_1k"] for r in rows]
    ax.set_xlim(min(xs) / 3, max(xs) * 12)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=8, label=f) for f, c in FAMILY_COLORS.items()]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.975), ncol=3, frameon=False, fontsize=9,
               labelcolor=INK_2)
    _title(fig, "Accuracy vs cost", f"Same {n} {split} tweets for every model · vertical line = 95% bootstrap CI · up-left is better")
    fig.savefig(path)
    plt.close(fig)


def fig_robustness(rows: list[dict], buckets: dict, yt_rows: list[dict], path: Path, split: str) -> None:
    """Dot plot: one row per data subset, one dot per model (identity = family colour + legend + labels)."""
    keys = [r["key"] for r in rows]
    names = {r["key"]: r["name"] for r in rows}
    fams = {r["key"]: r["family"] for r in rows}
    yt = {r["key"]: r["macro_f1"] for r in yt_rows}
    cats = [(lbl, {k: buckets[k][b]["macro_f1"] for k in keys if k in buckets}, buckets[keys[0]][b]["n"]) for b, lbl in BUCKETS]
    cats.append(("YouTube (out-of-domain)", {k: yt[k] for k in keys if k in yt}, yt_rows[0]["n"] if yt_rows else 0))
    markers = ["o", "s", "D", "^", "v", "P"]
    fig, ax = plt.subplots(figsize=(8, 0.75 * len(cats) + 1.6))
    fig.subplots_adjust(left=0.33, right=0.97, top=1 - 1.15 / (0.75 * len(cats) + 1.6), bottom=0.12)
    for y, (lbl, vals, n) in enumerate(cats):
        ax.axhline(y, color=GRID, linewidth=0.8, zorder=1)
        for j, k in enumerate(keys):
            if k in vals:
                ax.scatter(vals[k], y, s=60, marker=markers[j % len(markers)], color=FAMILY_COLORS[fams[k]],
                           edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_yticks(range(len(cats)), [f"{lbl}  (n={n})" for lbl, _, n in cats])
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1")
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    handles = [plt.Line2D([], [], marker=markers[j % len(markers)], linestyle="", color=FAMILY_COLORS[fams[k]],
                          markersize=7, label=names[k]) for j, k in enumerate(keys)]
    fig.legend(handles=handles, loc="lower center", ncol=min(3, len(handles)), frameon=False, fontsize=8.5,
               labelcolor=INK_2, bbox_to_anchor=(0.6, -0.01))
    fig.subplots_adjust(bottom=0.12 + 0.05 * ((len(handles) - 1) // 3 + 1))
    _title(fig, "Robustness by language mix and domain", f"{split.capitalize()} subsets by word-level language share; YouTube = different platform & topics")
    fig.savefig(path)
    plt.close(fig)


def fig_confusions(rows: list[dict], path: Path) -> None:
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n + 0.6, 3.6), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.66, bottom=0.2, wspace=0.35)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"])
    for ax, r in zip(axes[0], rows):
        cm = np.array(r["confusion"])[:, :3].astype(float)
        share = cm / cm.sum(axis=1, keepdims=True)  # row-normalised: recall per true class on the diagonal
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{share[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if share[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), ["neg", "neu", "pos"])
        ax.set_yticks(range(3), ["neg", "neu", "pos"])
        ax.set_title(r["name"], fontsize=9, color=INK, pad=6)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0][0].set_ylabel("True label")
    fig.text(0.5, 0.06, "Predicted label", ha="center", color=INK_2)
    _title(fig, "Where each model goes wrong", "Row-normalised confusion matrices on the same test tweets (diagonal = recall)")
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    cfg = load_config("configs/cost.yaml")
    hw = cfg["hardware"]
    base, ft = resolve(cfg["paths"]["baselines_dir"]), resolve(cfg["paths"]["finetune_dir"])
    out_dir, fig_dir = resolve(cfg["paths"]["out_dir"]), resolve(cfg["paths"]["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m for m in [load_tfidf(base, hw[cfg["model_hardware"]["tfidf"]]), *load_llm_runs(base),
                          load_finetuned(ft, "encoder", hw[cfg["model_hardware"]["encoder"]]),
                          load_finetuned(ft, "decoder", hw[cfg["model_hardware"]["decoder"]])] if m is not None]
    print("Models found:", ", ".join(f"{m.name} [{','.join(m.predictions)}]" for m in models))
    summary: dict = {"cost_assumptions": cfg, "tables": {}}
    md = ["# Phase 4 results (auto-generated by `scripts/evaluate_all.py`)\n"]

    # A. headline: every model on the same tweets (the LLM test subset); falls back to val while test is pending
    head_split = "test" if any(m.family == "llm" and "test" in m.predictions for m in models) else "val"
    ids, complete = subset_ids(models, head_split)
    head = score_rows(models, head_split, ids)
    summary["tables"]["headline"] = {"split": head_split, "n": len(ids), "complete": complete, "rows": head}
    md += [f"## A. All models on the same {len(ids)} {head_split} tweets"
           + ("" if complete else " *(LLM runs still partial)*") + "\n", md_table(head), ""]

    # B. full test set, self-hosted models only
    local = [m for m in models if m.family != "llm"]
    full_ids = list(local[0].predictions["test"].id) if local else []
    full = score_rows(local, "test", full_ids)
    summary["tables"]["full_test"] = {"n": len(full_ids), "rows": full}
    md += [f"## B. Self-hosted models on the full official test set ({len(full_ids)} tweets)\n", md_table(full), ""]

    # C. robustness: language-mix buckets on the headline tweets + YouTube
    buckets = by_bucket(models, head_split, ids)
    yt_ids, yt_complete = subset_ids(models, "test_youtube")
    yt = score_rows(models, "test_youtube", yt_ids) if yt_ids else []
    summary["tables"]["robustness"] = {"by_mix_bucket": buckets, "youtube": {"n": len(yt_ids), "complete": yt_complete, "rows": yt}}
    md += ["## C. Robustness (macro-F1)\n", "| Model | " + " | ".join(l for _, l in BUCKETS) + " | YouTube (OOD) |",
           "|---|" + "---|" * (len(BUCKETS) + 1)]
    yt_by = {r["key"]: r["macro_f1"] for r in yt}
    for r in head:
        cells = [f"{buckets[r['key']][b]['macro_f1']:.3f}" for b, _ in BUCKETS]
        md.append(f"| {r['name']} | " + " | ".join(cells) + f" | {yt_by.get(r['key'], float('nan')):.3f} |")
    md += [f"\nBucket sizes: " + ", ".join(f"{l} {buckets[head[0]['key']][b]['n']}" for b, l in BUCKETS)
           + f"; YouTube {len(yt_ids)}.", ""]

    # D. paired comparisons on the headline tweets
    preds = {m.key: on_ids(m.predictions[head_split], ids) for m in models if head_split in m.predictions}
    best_ft = max((r for r in head if r["family"] == "fine-tuned"), key=lambda r: r["macro_f1"], default=None)
    best_llm = max((r for r in head if r["family"] == "llm"), key=lambda r: r["macro_f1"], default=None)
    pairs = [(a["key"], b) for a in [best_ft] if a for b in ([best_llm["key"]] if best_llm else []) + ["tfidf"]]
    if best_llm:
        pairs.append((best_llm["key"], "tfidf"))
    tests = []
    for a, b in pairs:
        if a in preds and b in preds and a != b:
            t = paired_bootstrap(preds[a].label_id, preds[a].pred_id, preds[b].pred_id)
            tests.append({"a": a, "b": b, **t})
    summary["tables"]["paired"] = tests
    names = {m.key: m.name for m in models}
    md += ["## D. Paired bootstrap: is the difference real?\n", "| A vs B | Macro-F1(A) − Macro-F1(B) [95% CI] | P(A not better) |", "|---|---|---|"]
    md += [f"| {names[t['a']]} vs {names[t['b']]} | {t['diff']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] | {t['p_a_not_better']:.3f} |" for t in tests]

    # E. headline sentence
    if best_ft and best_llm:
        ratio_f1 = best_ft["macro_f1"] / best_llm["macro_f1"]
        ratio_cost = best_llm["usd_per_1k"] / best_ft["usd_per_1k"]
        summary["headline"] = {"fine_tuned": best_ft["name"], "llm": best_llm["name"], "f1_ratio": ratio_f1, "cost_ratio": ratio_cost}
        md += ["", f"**Headline:** {best_ft['name']} reaches **{ratio_f1:.0%}** of {best_llm['name']}'s macro-F1 "
                   f"at **{ratio_cost:,.0f}× lower cost** per prediction ({head_split}, n={len(ids)})."]

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    fig_cost_vs_f1(head, fig_dir / "cost_vs_f1.png", len(ids), head_split)
    fig_robustness(head, buckets, yt, fig_dir / "robustness.png", head_split)
    fig_confusions(head, fig_dir / "confusion_matrices.png")
    print("\n".join(md))


if __name__ == "__main__":
    main()
