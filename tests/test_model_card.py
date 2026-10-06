from hinglish_sentiment.ship.model_card import render_model_card


def _split(f1, n=100):
    return {"n": n, "macro_f1": f1, "macro_f1_ci_low": f1 - 0.02, "macro_f1_ci_high": f1 + 0.02, "accuracy": f1,
            "per_class": {k: {"f1": f1} for k in ("negative", "neutral", "positive")},
            "by_mix_bucket": {"code_mixed": {"n": 50, "macro_f1": f1}, "no_lang_words": {"n": 1, "macro_f1": 0}}}


def _metrics(demojize):
    return {"best_run": {"name": "muril_demoji_lr3e-05", "hf_id": "google/muril-base-cased", "demojize": demojize,
                         "val_macro_f1": 0.7, "learning_rate": 3e-5, "best_epoch": 3, "train_minutes": 4.2},
            "splits": {"val": _split(0.70), "test": _split(0.74), "test_youtube": _split(0.45)},
            "serving": {"latency_batch1": {"p50_ms": 9.5}, "throughput_batch64_per_s": 900.0}}


def test_card_has_metadata_numbers_and_preprocessing_note():
    card = render_model_card("me/hinglish-sentiment-muril", _metrics(demojize=True))
    assert card.startswith("---\nlanguage: [hi, en]\nlicense: apache-2.0")
    assert "value: 0.7400" in card and "**0.740** [0.720, 0.760]" in card
    assert "emoji.demojize" in card and "clf(preprocess(text))" in card
    assert "no lang words" not in card  # the tiny emoji-only bucket is not reported
    assert "How it compares" not in card  # no evaluation summary given


def test_card_without_demojize_and_with_comparison():
    ev = {"tables": {"headline": {"n": 600, "split": "test", "rows": [
        {"name": "MuRIL", "macro_f1": 0.74, "p50_ms": 9.5, "usd_per_1k": 0.0002},
        {"name": "LLM", "macro_f1": 0.72, "p50_ms": 650.0, "usd_per_1k": 0.07}]}},
          "headline": {"fine_tuned": "MuRIL", "llm": "LLM", "f1_ratio": 1.03, "cost_ratio": 350.0}}
    card = render_model_card("me/x", _metrics(demojize=False), ev)
    assert "emoji.demojize" not in card and "clf(text)" in card
    assert "103% of LLM's macro-F1 at 350× lower cost" in card
