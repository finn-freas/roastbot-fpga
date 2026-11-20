import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import tiktok_style as t


def test_style_marker_and_word_count_are_required():
    ok = "bro you really typed that with a straight face"
    plain = "this is a perfectly ordinary comment about shoes"
    short = "bro no way"
    got = t.candidates([ok, plain, short])
    assert got == {ok}


def test_word_count_bounds():
    four = "bro one two three"
    twenty = " ".join(["bro"] + ["w"] * 19)
    long = " ".join(["bro"] + ["w"] * 20)
    assert t.candidates([four, twenty, long]) == {four, twenty}


def test_non_strings_are_skipped_and_text_is_stripped():
    assert t.candidates([None, 3, ["bro"], "  bro this is the one  "]) == {"bro this is the one"}


def test_mentions_and_non_ascii_text_are_rejected():
    assert t.candidates(["@someone bro look at this one", "bro é this is the one here"]) == set()


def test_allowed_emoji_pass_and_other_symbols_fail():
    ok = "bro that is crazy \U0001f480"
    bad = "bro that is crazy ❤"
    assert t.candidates([ok, bad]) == {ok}


def test_filter_patterns_are_applied():
    assert t.candidates(["bro that was a kys moment for sure"]) == set()
    assert t.candidates(["bro the white shoes look great on you"]) == set()


def test_duplicates_collapse():
    assert t.candidates(["bro this is the one", "bro this is the one "]) == {"bro this is the one"}
