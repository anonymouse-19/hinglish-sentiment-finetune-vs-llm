"""Data-card figures, drawn from reports/data_stats.json (run prepare_data.py first).

Usage:
    python scripts/plot_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hinglish_sentiment import LABELS  # noqa: E402
from hinglish_sentiment.config import load_config, resolve  # noqa: E402

# Sentiment is ordinal, so it gets a diverging encoding: red <-> gray midpoint <-> blue.
LABEL_COLORS = {"negative": "#e34948", "neutral": "#a3a19a", "positive": "#2a78d6"}
LABEL_TEXT = {"negative": "#ffffff", "neutral": "#0b0b0b", "positive": "#ffffff"}  # ink chosen per fill luminance
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BAR = 0.34  # bar thickness in row units: thin marks, the rest of the band is air

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": INK_2,
    "savefig.dpi": 200,
    "savefig.facecolor": SURFACE,
})


def _title(fig, title: str, subtitle: str) -> None:
    fig.text(0.015, 0.96, title, fontsize=13, fontweight="semibold", color=INK, va="top")
    fig.text(0.015, 0.875, subtitle, fontsize=10, color=INK_2, va="top")


def _legend(fig) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=LABEL_COLORS[l]) for l in LABELS]
    fig.legend(handles, LABELS, loc="upper right", bbox_to_anchor=(0.985, 0.975), ncol=3, frameon=False,
               handlelength=1.0, handleheight=1.0, fontsize=10, labelcolor=INK_2)


def stacked_share(ax, rows: list[tuple[str, dict]]) -> None:
    """100% stacked horizontal bars; one row per (name, {label: count})."""
    for y, (_, counts) in enumerate(rows):
        total = sum(counts.values())
        left = 0.0
        for label in LABELS:
            share = counts.get(label, 0) / total
            # a surface-coloured edge gives the 2px gap between touching segments
            ax.barh(y, share, left=left, height=BAR, color=LABEL_COLORS[label], edgecolor=SURFACE, linewidth=2)
            if share >= 0.07:  # label only where the text fits inside the segment
                ax.text(left + share / 2, y, f"{share:.0%}", ha="center", va="center", fontsize=9,
                        color=LABEL_TEXT[label])
            left += share
    ax.set_yticks(range(len(rows)), [name for name, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def plot_label_distribution(stats: dict, out: Path) -> None:
    rows = [(f"{s} (n={stats['final'][s]['n']:,})", stats["final"][s]["labels"]) for s in ("train", "val", "test")]
    rows.append((f"YouTube OOD (n={stats['youtube_ood']['final_n']:,})", stats["youtube_ood"]["final_labels"]))
    fig, ax = plt.subplots(figsize=(8, 3.1))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.72, bottom=0.12)
    stacked_share(ax, rows)
    _title(fig, "Label mix per split", "Share of examples in each sentiment class after cleaning")
    _legend(fig)
    fig.savefig(out)
    plt.close(fig)


def plot_mix_buckets(stats: dict, out: Path) -> None:
    test = stats["final"]["test"]
    names = {"mostly_hindi": "Mostly Hindi", "code_mixed": "Code-mixed", "mostly_english": "Mostly English"}
    rows = [(f"{names[b]} (n={test['mix_bucket'][b]:,})", test["mix_bucket_x_label"][b]) for b in names]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.7, bottom=0.2)
    stacked_share(ax, rows)
    _title(fig, "Sentiment shifts with language mix", "SentiMix test set; mostly = at least 80% of words in one language")
    _legend(fig)
    n_other = test["mix_bucket"]["no_lang_words"]
    fig.text(0.015, 0.03, f"Excludes {n_other} tweets with no tagged words.", fontsize=8.5, color=MUTED)
    fig.savefig(out)
    plt.close(fig)


def plot_drops(stats: dict, out: Path) -> None:
    names = {
        "duplicate_label_conflict": "Duplicate with conflicting label",
        "duplicate": "Duplicate (same label)",
        "near_duplicate_of_test": "Near-duplicate of a test tweet",
        "no_content": "No content (only @user / URL)",
        "non_hinglish_language": "Not Hinglish (e.g. Turkish, Polish)",
    }
    counts = {names[r]: sum(v.values()) for r, v in stats["dropped"].items()}
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    raw = stats["raw"]["train"]["n"] + stats["raw"]["dev"]["n"]
    kept = stats["final"]["train"]["n"] + stats["final"]["val"]["n"]
    fig, ax = plt.subplots(figsize=(8, 3.1))
    fig.subplots_adjust(left=0.33, right=0.93, top=0.74, bottom=0.08)
    ys = range(len(items))
    ax.barh(ys, [v for _, v in items], height=BAR * 1.3, color="#2a78d6")
    for y, (_, v) in zip(ys, items):
        ax.text(v + max(counts.values()) * 0.012, y, f"{v:,}", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(list(ys), [k for k, _ in items])
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.tick_params(length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    _title(fig, "Rows removed during cleaning",
           f"Official train+dev: {raw:,} tweets, {kept:,} kept ({raw - kept:,} removed). The test set is not modified.")
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    cfg = load_config("configs/data.yaml")
    stats = json.loads((resolve(cfg["paths"]["reports_dir"]) / "data_stats.json").read_text(encoding="utf-8"))
    fig_dir = resolve(cfg["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_label_distribution(stats, fig_dir / "label_distribution.png")
    plot_mix_buckets(stats, fig_dir / "mix_bucket_labels.png")
    plot_drops(stats, fig_dir / "cleaning_drops.png")
    print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
