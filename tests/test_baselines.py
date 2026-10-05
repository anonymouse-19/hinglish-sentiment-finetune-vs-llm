from types import SimpleNamespace

import numpy as np
import openai
import pandas as pd
import pytest

from hinglish_sentiment.data.split import stratified_subset
from hinglish_sentiment.eval.metrics import INVALID, bootstrap_ci, classification_metrics
from hinglish_sentiment.llm.client import LLMClient, ResponseCache, cost_usd
from hinglish_sentiment.llm.prompts import build_messages, parse_label, select_few_shot

# --- metrics ---------------------------------------------------------------------------------------


def test_invalid_predictions_count_as_errors_not_dropped():
    m = classification_metrics([0, 1, 2, 2], [0, 1, 2, INVALID])
    assert m["accuracy"] == 0.75
    assert m["n_invalid"] == 1
    assert m["per_class"]["positive"]["recall"] == 0.5
    assert m["confusion_matrix"]["matrix"][2] == [0, 0, 1, 1]  # last column = invalid


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)
    pred = np.where(rng.random(500) < 0.7, y, rng.integers(0, 3, 500))
    point = classification_metrics(y, pred)["macro_f1"]
    ci = bootstrap_ci(y, pred, n_resamples=200)
    assert ci["macro_f1_ci_low"] < point < ci["macro_f1_ci_high"]


# --- subsets / prompts -----------------------------------------------------------------------------


def _toy(n=300):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "id": [f"t{i}" for i in range(n)],
        "text": [f"tweet number {i} yaar" for i in range(n)],
        "label": rng.choice(["negative", "neutral", "positive"], n),
        "mix_bucket": rng.choice(["code_mixed", "mostly_hindi", "mostly_english"], n),
        "n_words": 6, "n_chars": 30,
    })


def test_stratified_subset_is_reproducible_and_keeps_proportions():
    df = _toy()
    a = stratified_subset(df, 90, seed=42, stratify_on=["label", "mix_bucket"])
    b = stratified_subset(df, 90, seed=42, stratify_on=["label", "mix_bucket"])
    assert len(a) == 90 and a.id.tolist() == b.id.tolist()
    assert abs(a.label.value_counts(normalize=True) - df.label.value_counts(normalize=True)).max() < 0.05
    assert stratified_subset(df, None, 42, ["label"]) is df


def test_few_shot_is_class_balanced_interleaved_and_deterministic():
    df = _toy()
    ex = select_few_shot(df, per_class=2, seed=42)
    assert ex.label.tolist() == ["negative", "neutral", "positive"] * 2
    assert ex.id.tolist() == select_few_shot(df, per_class=2, seed=42).id.tolist()


def test_few_shot_examples_go_in_the_system_prompt_only():
    ex = select_few_shot(_toy(), per_class=1, seed=0)
    msgs = build_messages("kya baat hai", ex)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert all(t in msgs[0]["content"] for t in ex.text)
    assert msgs[1]["content"] == "Post: kya baat hai\nSentiment:"
    assert "Labelled examples" not in build_messages("kya baat hai")[0]["content"]


@pytest.mark.parametrize("answer, expected", [
    ("negative", 0), ("Neutral", 1), (" positive.\n", 2), ("Sentiment: POSITIVE", 2),
    ("", INVALID), (None, INVALID), ("mixed", INVALID), ("positive or negative", INVALID),
    ("nonpositive", INVALID),  # whole words only
])
def test_parse_label(answer, expected):
    assert parse_label(answer) == expected


# --- LLM client (fake SDK, no network) -------------------------------------------------------------

CFG = {
    "model": "fake/model", "base_url": "http://unused",
    "prices": {"input_per_million": 0.15, "cached_input_per_million": 0.075, "output_per_million": 0.60},
    "generation": {"temperature": 0, "max_completion_tokens": 64, "reasoning_effort": "low", "extra_body": {}},
    "client": {"requests_per_minute": 600000, "max_retries": 2, "timeout_s": 5},
}


def _response(content="negative"):
    usage = SimpleNamespace(prompt_tokens=200, completion_tokens=50, total_tokens=250,
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=48),
                            prompt_tokens_details=SimpleNamespace(cached_tokens=100))
    return SimpleNamespace(model="fake/model", usage=usage,
                           choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))])


class FakeSDK:
    def __init__(self, failures=0):
        self.calls, self.failures = [], failures
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures:
            # minimal stand-in for the HTTP response object the SDK attaches to its errors
            response = SimpleNamespace(status_code=429, headers={"retry-after": "0"},
                                       request=SimpleNamespace(method="POST", url="http://unused"))
            raise openai.RateLimitError("slow down", response=response, body=None)
        return _response()


def test_cost_counts_cached_and_reasoning_tokens():
    # 100 uncached input, 100 cached input, 50 output (incl. reasoning)
    expected = (100 * 0.15 + 100 * 0.075 + 50 * 0.60) / 1e6
    assert cost_usd(200, 50, 100, CFG["prices"]) == pytest.approx(expected)
    assert cost_usd(200, 50, 100, {**CFG["prices"], "cached_input_per_million": None}) == pytest.approx(
        (200 * 0.15 + 50 * 0.60) / 1e6)


def test_client_logs_usage_and_caches(tmp_path):
    sdk = FakeSDK()
    client = LLMClient(CFG, "key", ResponseCache(tmp_path / "c.jsonl"), sdk_client=sdk)
    msgs = build_messages("bakwas movie")
    r1 = client.complete(msgs)
    assert (r1.content, r1.reasoning_tokens, r1.cached_tokens, r1.from_cache) == ("negative", 48, 100, False)
    assert sdk.calls[0]["reasoning_effort"] == "low" and sdk.calls[0]["temperature"] == 0

    # a fresh client with the same cache file answers from disk, with the original latency
    sdk2 = FakeSDK()
    r2 = LLMClient(CFG, "key", ResponseCache(tmp_path / "c.jsonl"), sdk_client=sdk2).complete(msgs)
    assert r2.from_cache and not sdk2.calls and r2.latency_ms == r1.latency_ms

    # a different generation setting is a different cache key
    other = {**CFG, "generation": {**CFG["generation"], "temperature": 1}}
    LLMClient(other, "key", ResponseCache(tmp_path / "c.jsonl"), sdk_client=sdk2).complete(msgs)
    assert len(sdk2.calls) == 1


def test_client_retries_rate_limits_then_gives_up():
    sdk = FakeSDK(failures=2)
    r = LLMClient(CFG, "key", sdk_client=sdk).complete(build_messages("x"))
    assert r.attempts == 3 and len(sdk.calls) == 3
    with pytest.raises(openai.RateLimitError):
        LLMClient(CFG, "key", sdk_client=FakeSDK(failures=5)).complete(build_messages("x"))
