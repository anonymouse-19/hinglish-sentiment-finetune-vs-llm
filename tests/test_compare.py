import numpy as np
import pandas as pd
import pytest

from hinglish_sentiment.eval.compare import cost_per_1k_self_hosted, on_ids, paired_bootstrap


def test_cost_per_1k_self_hosted():
    # $0.36/h at 100 predictions/s = 360,000 predictions per $0.36 -> $0.001 per 1k
    assert cost_per_1k_self_hosted(100, 0.36) == pytest.approx(0.001)


def test_paired_bootstrap_detects_a_clearly_better_model_and_not_an_identical_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600)
    good = np.where(rng.random(600) < 0.85, y, (y + 1) % 3)
    bad = np.where(rng.random(600) < 0.55, y, (y + 1) % 3)
    t = paired_bootstrap(y, good, bad, n_resamples=300)
    assert t["diff"] > 0.2 and t["ci_low"] > 0 and t["p_a_not_better"] == 0
    same = paired_bootstrap(y, good, good, n_resamples=100)
    assert same["diff"] == 0 and same["ci_low"] == 0 == same["ci_high"]


def test_on_ids_restricts_and_orders():
    df = pd.DataFrame({"id": ["a", "b", "c"], "pred_id": [0, 1, 2]})
    assert on_ids(df, ["c", "a"]).pred_id.tolist() == [2, 0]
