"""Generate the Hugging Face model card (README.md) from the project's own result files (no hand-copied numbers)."""

from __future__ import annotations

GITHUB = "https://github.com/anonymouse-19/hinglish-sentiment-finetune-vs-llm"
BASE_LICENCES = {"FacebookAI/xlm-roberta-base": "mit", "google/muril-base-cased": "apache-2.0"}
VARIANT_NAMES = {"xlmr": "XLM-R base", "muril": "MuRIL base", "muril_demoji": "MuRIL base (emoji → text)"}


def _f(x: float) -> str:
    return f"{x:.3f}"


def render_model_card(repo_id: str, ft_metrics: dict, evaluation: dict | None = None) -> str:
    best, splits = ft_metrics["best_run"], ft_metrics["splits"]
    base = best["hf_id"]
    test, yt = splits["test"], splits["test_youtube"]
    sv = ft_metrics["serving"]
    demojize = bool(best.get("demojize"))

    model_index = f"""model-index:
- name: {repo_id.split('/')[-1]}
  results:
  - task: {{type: text-classification, name: Sentiment analysis (3-class)}}
    dataset: {{type: sentimix-hinglish, name: SemEval-2020 Task 9 SentiMix (Hinglish), split: test}}
    metrics:
    - {{type: f1, name: Macro-F1, value: {test['macro_f1']:.4f}}}
    - {{type: accuracy, name: Accuracy, value: {test['accuracy']:.4f}}}"""

    rows = "\n".join(
        f"| {name} | {s['n']:,} | **{_f(s['macro_f1'])}** [{_f(s['macro_f1_ci_low'])}, {_f(s['macro_f1_ci_high'])}] | {_f(s['accuracy'])} |"
        for name, s in [("Validation", splits["val"]), ("SentiMix test (official)", test), ("YouTube comments (out-of-domain)", yt)])
    per_class = " · ".join(f"{k} {_f(v['f1'])}" for k, v in test["per_class"].items())
    buckets = test.get("by_mix_bucket", {})
    bucket_rows = "\n".join(f"| {b.replace('_', ' ')} | {v['n']} | {_f(v['macro_f1'])} |"
                            for b, v in buckets.items() if b != "no_lang_words")

    comparison = ""
    if evaluation and evaluation.get("tables", {}).get("headline"):
        head = evaluation["tables"]["headline"]
        lines = [f"| {r['name']} | {_f(r['macro_f1'])} | "
                 + ("–" if r["p50_ms"] is None else f"{r['p50_ms']:.0f} ms") + " | "
                 + ("–" if r["usd_per_1k"] is None else f"${r['usd_per_1k']:.5f}") + " |" for r in head["rows"]]
        hl = evaluation.get("headline")
        comparison = f"""
## How it compares with a large LLM

All models on the same {head['n']} {head['split']} tweets (latency: see the repo for hardware; LLM latency includes the network):

| Model | Macro-F1 | p50 latency | Cost / 1k predictions |
|---|---|---|---|
""" + "\n".join(lines) + ("" if not hl else f"""

**{hl['fine_tuned']} reaches {hl['f1_ratio']:.0%} of {hl['llm']}'s macro-F1 at {hl['cost_ratio']:,.0f}× lower cost per prediction.**""") + "\n"

    usage_pre = ("import emoji\n\ndef preprocess(text):  # this model was trained with emoji rewritten as text\n"
                 "    return \" \".join(emoji.demojize(text, delimiters=(\" :\", \": \")).split())\n\n") if demojize else ""
    usage_call = "clf(preprocess(text))" if demojize else "clf(text)"

    return f"""---
language: [hi, en]
license: {BASE_LICENCES.get(base, 'other')}
library_name: transformers
pipeline_tag: text-classification
base_model: {base}
tags: [sentiment-analysis, hinglish, code-mixed, hindi, sentimix, text-classification]
metrics: [f1, accuracy]
{model_index}
---

# {repo_id.split('/')[-1]}: Hinglish (Hindi–English code-mixed) sentiment

`{base}` fine-tuned for **3-class sentiment (negative / neutral / positive)** on romanised Hindi–English
code-mixed tweets from SemEval-2020 Task 9 (SentiMix). It was built for a study comparing small fine-tuned models
with a large LLM on accuracy, latency and cost. Code, data card, decision log and full results: **[{GITHUB.split('/')[-1]}]({GITHUB})**.

## Results

| Evaluation set | n | Macro-F1 [95% bootstrap CI] | Accuracy |
|---|---|---|---|
{rows}

Per-class F1 on test: {per_class}. For reference, the best SemEval-2020 system scored 0.750 F1 on the same test set.

By language mix (test, word-level language tags; mostly-X = ≥ 80% of words in X):

| Subset | n | Macro-F1 |
|---|---|---|
{bucket_rows}
{comparison}
## Usage

```python
from transformers import pipeline

{usage_pre}clf = pipeline("text-classification", model="{repo_id}", top_k=None)
text = "yaar ye movie toh ekdum bakwas thi, paisa barbaad"
print({usage_call})
```

Inputs should be cleaned like the training data: replace @mentions with `@user` and URLs with `http`.

## Training

- **Data:** SentiMix Hinglish train + dev, cleaned and de-duplicated, then re-split 85/15 (stratified by label × language mix):
  13,409 train / 2,367 validation. Mojibake was repaired, and 271 near-copies of *test* tweets were removed from training (leakage).
  The tweets are not redistributed (no licence); see the [data card]({GITHUB}/blob/main/docs/DATA_CARD.md).
- **Selection:** a 9-run sweep (XLM-R / MuRIL / MuRIL with emoji→text × learning rate 2e-5 / 3e-5 / 5e-5).
  This run (`{best['name']}`) had the best validation macro-F1 ({_f(best['val_macro_f1'])}). The test set was scored once, after selection.
- **Hyperparameters:** learning rate {best['learning_rate']}, batch 32, up to 4 epochs with early stopping on validation macro-F1
  (best epoch: {best['best_epoch']}), linear schedule with 10% warmup, weight decay 0.01, fp16, max length 128 tokens, seed 42.
- **Hardware:** a free Google Colab T4; training took {best['train_minutes']} min.
- **Serving (T4, fp16):** p50 latency {sv['latency_batch1']['p50_ms']:.1f} ms at batch size 1 (including tokenization);
  {sv.get('throughput_batch64_per_s', 0):,.0f} predictions/s at batch 64.

## Limitations and bias

- **Domain:** 2019 Indian Twitter, heavy on politics and cricket. Accuracy drops sharply on other domains:
  YouTube comments score {_f(yt['macro_f1'])} macro-F1, against {_f(test['macro_f1'])} on in-domain test.
- **Label noise:** SentiMix labels are noisy (35% of duplicate-tweet groups carry conflicting labels). "Neutral" is used
  broadly, including for many clearly emotional tweets, and the model has learned that convention.
- **Sarcasm,** which is common in political Hinglish, is often read literally.
- **Offensive content:** the training data contains profanity and abuse; predictions on such text reflect annotator conventions.
- **Script:** trained on Roman-script Hinglish; Devanagari makes up < 1% of the training data.
- Not for decisions about individuals (e.g. moderation without human review).

## Citation (data)

Patwa, P. et al. (2020). *SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets.* SemEval 2020.
"""
