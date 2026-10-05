"""TF-IDF + logistic regression baseline (DECISION_LOG D-017).

Two views of the text are combined:
- word 1-2-grams: "bahut accha", "not good"
- char 2-5-grams within word boundaries: robust to Hinglish's free spelling ("nahi" / "nhi" / "nahin")
  and they also capture emoji, which the default word tokenizer throws away.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

# Mentions/URLs were already masked to "@user" / "http" in Phase 1; drop them as features.
STOP_TOKENS = ["user", "http"]


def build_pipeline(features: str = "word+char", C: float = 1.0, class_weight: str | None = "balanced",
                   seed: int = 42) -> Pipeline:
    views = []
    if "word" in features:
        views.append(("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, lowercase=True,
                                              sublinear_tf=True, stop_words=STOP_TOKENS)))
    if "char" in features:
        views.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, lowercase=True,
                                              sublinear_tf=True, max_features=300_000)))
    if not views:
        raise ValueError(f"Unknown feature set: {features!r}")
    clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=3000, random_state=seed)
    return Pipeline([("features", FeatureUnion(views)), ("clf", clf)])


def out_of_fold_gold_probability(texts, labels, seed: int = 42, folds: int = 5, **pipeline_kwargs) -> np.ndarray:
    """P(given label) for each training example, predicted by a model that never saw that example.

    A low value flags a likely mislabelled (or very hard) example. Used to keep noisy labels out of the
    LLM's few-shot demonstrations (D-020).
    """
    labels = np.asarray(labels)
    proba = cross_val_predict(build_pipeline(seed=seed, **pipeline_kwargs), texts, labels, method="predict_proba",
                              cv=StratifiedKFold(folds, shuffle=True, random_state=seed))
    return proba[np.arange(len(labels)), labels]
