import pytest

pytest.importorskip("torch")

from hinglish_sentiment.config import load_config  # noqa: E402
from hinglish_sentiment.finetune.data import preprocess  # noqa: E402


def test_demojize_turns_emoji_into_words_and_leaves_text_alone():
    assert preprocess("kya baat 😂", demojize=True) == "kya baat :face_with_tears_of_joy:"
    assert preprocess("kya baat 😂", demojize=False) == "kya baat 😂"
    assert preprocess("no emoji here", demojize=True) == "no emoji here"


def test_sweep_grids_have_unique_names_and_expected_sizes():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("fs", Path(__file__).parents[1] / "scripts" / "finetune_sweep.py")
    fs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fs)
    cfg = load_config("configs/finetune.yaml")
    enc, dec = fs.grid_runs(cfg, "encoder"), fs.grid_runs(cfg, "decoder")
    assert len(enc) == 9 and len(dec) == 4
    assert len({r["name"] for r in enc + dec}) == 13
    assert {r["demojize"] for r in enc if r["variant"] == "muril_demoji"} == {True}
    assert all(r["model_key"] in cfg["models"] for r in enc + dec)


def test_prune_keeps_only_best_model(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("fs", Path(__file__).parents[1] / "scripts" / "finetune_sweep.py")
    fs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fs)
    results = [{"name": "a", "val_macro_f1": 0.6}, {"name": "b", "val_macro_f1": 0.7}]
    for r in results:
        (tmp_path / r["name"] / "model").mkdir(parents=True)
    fs.prune_models(tmp_path, results)
    assert not (tmp_path / "a" / "model").exists() and (tmp_path / "b" / "model").exists()
