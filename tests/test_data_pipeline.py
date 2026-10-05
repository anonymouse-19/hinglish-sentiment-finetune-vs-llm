import pandas as pd
import pytest

from hinglish_sentiment.data.dedupe import exact_key, near_duplicate_clusters, resolve_duplicates
from hinglish_sentiment.data.sentimix import parse_conll
from hinglish_sentiment.data.split import stratified_split

CONLL = (
    "meta\t4330\tneutral\n@\tO\nnehantics\tHin\nHaan\tHin\nyaar\tHin\n\n"
    "meta\t6648\tnegative\nmedia\tEng\nnhi\tHin\n.\t\n\n"
)


def test_parse_conll(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text(CONLL, encoding="utf-8")
    tweets = parse_conll(path)
    assert [t["uid"] for t in tweets] == ["4330", "6648"]
    assert tweets[0]["label"] == "neutral"
    assert tweets[0]["tokens"] == ["@", "nehantics", "Haan", "yaar"]
    assert tweets[1]["lang_tags"] == ["Eng", "Hin", ""]  # token with an empty tag is kept


def test_parse_conll_unlabelled_test_file(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("meta\t20803\nbahut\tHin\n\n", encoding="utf-8")
    assert parse_conll(path)[0]["label"] is None


def test_exact_key_ignores_case_punctuation_handles_and_urls():
    assert exact_key("@user This is also my city. http") == exact_key("this is also my city")


def test_near_duplicates_cluster_but_distinct_tweets_do_not():
    texts = [
        "@user Madam tab wo Kisi BHI party Mai Tod fod Kar sarkar Nahi banana chahte the lakin aaj BJP",
        "@user Madam tab wo Kisi BHI party Mai Tod fod Kar sarkar Nahi banana chahte the.... lakin aaj BJP",
        "Yeh pubg wale bohot chalak hain elite pass ke liye paise le lete hain",
    ]
    labels = near_duplicate_clusters(texts, threshold=0.9)
    assert labels[0] == labels[1] != labels[2]


def _frame(rows):
    return pd.DataFrame(rows, columns=["official_split", "label", "dup_cluster"])


def test_resolve_drops_train_copies_of_test_tweets_but_never_test():
    df = _frame([("test", "positive", 1), ("train", "positive", 1), ("dev", "neutral", 1)])
    reasons = resolve_duplicates(df)
    assert pd.isna(reasons[0])
    assert reasons[1] == reasons[2] == "near_duplicate_of_test"


def test_resolve_keeps_one_copy_when_labels_agree():
    reasons = resolve_duplicates(_frame([("train", "negative", 7), ("dev", "negative", 7)]))
    assert reasons.isna().sum() == 1 and (reasons == "duplicate").sum() == 1


def test_resolve_majority_vote_and_ties():
    majority = resolve_duplicates(_frame([("train", "positive", 3), ("train", "positive", 3), ("train", "neutral", 3)]))
    assert list(majority.fillna("kept")) == ["kept", "duplicate", "duplicate_label_conflict"]
    tie = resolve_duplicates(_frame([("train", "positive", 4), ("dev", "neutral", 4)]))
    assert (tie == "duplicate_label_conflict").all()


def test_stratified_split_preserves_joint_distribution():
    df = pd.DataFrame({"label": ["a", "b"] * 100, "mix_bucket": ["x"] * 100 + ["y"] * 100})
    train, val = stratified_split(df, val_size=0.2, seed=0, stratify_on=["label", "mix_bucket"])
    assert len(val) == 40
    assert val.groupby(["label", "mix_bucket"]).size().nunique() == 1  # every stratum equally represented


def test_stratified_split_rejects_singleton_strata():
    df = pd.DataFrame({"label": ["a"] * 10 + ["b"], "mix_bucket": ["x"] * 11})
    with pytest.raises(ValueError):
        stratified_split(df, val_size=0.2, seed=0, stratify_on=["label", "mix_bucket"])
