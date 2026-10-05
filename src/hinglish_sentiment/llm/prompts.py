"""Prompts for zero-/few-shot sentiment classification, and strict parsing of the answer (D-018, D-020).

The model is asked for a single word. Few-shot examples are a fixed, seeded, class-balanced sample of the
TRAINING split, so every test tweet sees the identical prompt prefix (and the provider's prompt cache can
reuse it).
"""

from __future__ import annotations

import re

import pandas as pd

from hinglish_sentiment import LABEL2ID, LABELS
from hinglish_sentiment.eval.metrics import INVALID

SYSTEM_PROMPT = """You are a sentiment classifier for Hinglish social-media posts: Hindi and English mixed together, \
usually with Hindi typed in Roman letters (e.g. "yaar ye movie toh bakwas thi").

Classify the overall sentiment the author expresses in the post as exactly one of:
- positive: praise, happiness, support, gratitude, excitement, good wishes
- negative: criticism, anger, insults, sadness, complaints, sarcasm with a negative intent
- neutral: no clear sentiment, e.g. factual statements, questions, news, or a balanced mix

Notes: "@user" replaces a username and "http" replaces a link. Posts may be cut off with "…".

Answer with exactly one lowercase word: positive, negative, or neutral."""

FEW_SHOT_HEADER = "\n\nLabelled examples:"


def select_few_shot(train: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """A fixed class-balanced sample from train, skipping tweets too short or too long to be representative."""
    pool = train[(train.n_words >= 5) & (train.n_chars <= 140)]
    picked = [g.sample(per_class, random_state=seed) for _, g in pool.groupby("label", sort=True)]
    # interleave classes (neg, neu, pos, neg, ...) so label order carries no positional signal
    rows = [df.iloc[i] for i in range(per_class) for df in picked]
    return pd.DataFrame(rows).reset_index(drop=True)


def build_messages(text: str, examples: pd.DataFrame | None = None) -> list[dict]:
    system = SYSTEM_PROMPT
    if examples is not None and len(examples):
        lines = [f'Post: {ex.text}\nSentiment: {ex.label}' for ex in examples.itertuples()]
        system += FEW_SHOT_HEADER + "\n\n" + "\n\n".join(lines)
    return [{"role": "system", "content": system}, {"role": "user", "content": f"Post: {text}\nSentiment:"}]


_LABEL_RE = re.compile(r"\b(" + "|".join(LABELS) + r")\b")


def parse_label(answer: str | None) -> int:
    """Map the model's answer to a label id; INVALID if it names no label or more than one distinct label."""
    if not answer:
        return INVALID
    found = set(_LABEL_RE.findall(answer.strip().lower()))
    return LABEL2ID[found.pop()] if len(found) == 1 else INVALID
