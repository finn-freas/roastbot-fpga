import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import tiktok_style as ts


def test_keeps_styled_comments_and_strips_whitespace():
    got = ts.candidates(["  bro really thought that was a good idea 💀 ", "nah this is wild to me"])
    assert got == {"bro really thought that was a good idea 💀", "nah this is wild to me"}


def test_rejects_wrong_length():
    assert ts.candidates(["bro no way", " ".join(["bro"] * 21)]) == set()


def test_rejects_without_style_marker():
    assert ts.candidates(["this blender works really well for me"]) == set()


def test_rejects_mentions_and_other_characters():
    assert ts.candidates(["@user bro that is crazy work 💀", "bro that is crazy work ñ", "bro that is crazy 日本語 work"]) == set()


def test_rejects_group_and_slur_terms():
    assert ts.candidates(["bro that is so gay right now 💀", "bro kys right now no way 💀"]) == set()


def test_ignores_non_strings():
    assert ts.candidates([None, 3, "bro that blender is crazy work 💀"]) == {"bro that blender is crazy work 💀"}
