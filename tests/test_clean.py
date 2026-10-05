import math

import pytest

from hinglish_sentiment.data.clean import (
    count_accented_latin,
    detokenize,
    has_content,
    language_stats,
    normalize_tokens,
    repair_mojibake,
)
from hinglish_sentiment.data.split import mix_bucket


def clean(text: str, tags: str | None = None) -> str:
    toks = text.split(" ")
    tag_list = tags.split(" ") if tags else ["Hin"] * len(toks)
    return detokenize(normalize_tokens(toks, tag_list))


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("@ nehantics Haan yaar", "@user Haan yaar"),
        ("@ BTS _ army _ Fin kya baat", "@user kya baat"),
        ("@_ onlymanda hello", "@user hello"),
        ("dekho @… http", "dekho @user http"),
        ("RT @ user1 mast hai", "@user mast hai"),
    ],
)
def test_mentions_are_merged_and_masked(raw, expected):
    assert clean(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "accha hai https // t . co / hBg7zS0viy",
        "accha hai https // t co / XPCzJ7GFqC",
        "accha hai https // tco / 5RSlSbZNtt",
        "accha hai https // t …",
        "accha hai https …",
    ],
)
def test_url_variants_collapse_to_single_placeholder(raw):
    assert clean(raw) == "accha hai http"


def test_hashtags_are_rejoined_not_masked():
    assert clean("we miss you as a # videsh _ mantri") == "we miss you as a #videsh_mantri"


def test_mojibake_is_repaired():
    assert clean("itni importantce chaeay ni tou ðŸ˜…") == "itni importantce chaeay ni tou 😅"


def test_detokenize_fixes_punctuation_and_contractions():
    assert clean("They aren ’ t real issues . Pagal hai kya ?") == "They aren't real issues. Pagal hai kya?"


def test_language_stats_ignore_handles_and_urls():
    # The original tagger labels handle/URL pieces as Hin/Eng; those must not count.
    toks = "@ nehantics kab karega woh post https // t . co / abc".split(" ")
    tags = "O Hin Hin Hin Hin Eng Eng O Eng O Eng O Eng".split(" ")
    stats = language_stats(normalize_tokens(toks, tags))
    assert (stats["n_hin"], stats["n_eng"]) == (3, 1)
    assert stats["cmi"] == pytest.approx(25.0)  # 100 * (1 - 3/4)


def test_cmi_of_monolingual_tweet_is_zero():
    stats = language_stats(normalize_tokens(["bahut", "accha"], ["Hin", "Hin"]))
    assert stats["cmi"] == 0.0


def test_language_stats_without_words():
    stats = language_stats(normalize_tokens(["😂", "!"], ["EMT", "O"]))
    assert stats["n_hin"] == stats["n_eng"] == 0
    assert math.isnan(stats["eng_ratio"])


@pytest.mark.parametrize(
    "n_hin, n_eng, bucket",
    # 80/20 exactly is the boundary and counts as "mostly"; 70/30 is code-mixed.
    [(8, 2, "mostly_hindi"), (2, 8, "mostly_english"), (5, 5, "code_mixed"), (7, 3, "code_mixed"),
     (0, 0, "no_lang_words")],
)
def test_mix_bucket_boundaries(n_hin, n_eng, bucket):
    assert mix_bucket(n_hin, n_eng, dominant_share=0.8) == bucket


def test_has_content():
    assert not has_content("@user @user @user… http")
    assert has_content("@user 😂😂")
    assert has_content("@user nahi yaar http")


def test_accented_letter_count_flags_foreign_text():
    assert count_accented_latin("Tebliğ açısından dikkat çekici") >= 2
    assert count_accented_latin("Haan yaar kab karega") == 0


@pytest.mark.parametrize(
    "hinglish",
    [
        "shabaash Stokes buss issi tarhaan World Cup jeetna hai come on England 🏴󠁧󠁢󠁥󠁮󠁧󠁿",  # flag = TAG LATIN letters
        "Saturday class done ʟɪᴋᴇ ᴛᴀɢ ғʀɪᴇɴᴅ",  # decorative small caps
        "🌸ԼƖƔЄ ƖƝ 5! FORTNITE FRIDAY",  # look-alike letters without diacritics
    ],
)
def test_accented_letter_count_ignores_emoji_tags_and_fancy_fonts(hinglish):
    # Regression: a name-based check ("LATIN" in unicodedata.name) counted these and dropped real tweets.
    assert count_accented_latin(hinglish) == 0


@pytest.mark.parametrize(
    "broken, fixed",
    [("ÄŸ", "ğ"), ("ã…‹ã…‹", "ㅋㅋ"), ("Ú†", "چ"), ("Ã© Ã¨", "é è"), ("hotin Û”Û” stay", "hotin ۔۔ stay"),
     ("ðŸ˜…", "😅")],
)
def test_repair_mojibake_handles_short_sequences_ftfy_skips(broken, fixed):
    assert repair_mojibake(broken) == fixed


@pytest.mark.parametrize("legit", ["café… nice", "Señor", "naïve résumé", "Haan yaar…"])
def test_repair_mojibake_leaves_legitimate_text_alone(legit):
    assert repair_mojibake(legit) == legit
