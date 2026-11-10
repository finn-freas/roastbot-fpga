import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import emoji_fit as ef


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(ef, "ROOT", tmp_path)
    for name in ("rb", "rb_arg", "x"):
        d = tmp_path / "data" / "people" / name
        d.mkdir(parents=True)
        for split in ("train", "val"):
            (d / f"{split}.jsonl").write_text("")
    return tmp_path


def test_defaults(tree):
    assert ef.parse_args([]) == (0.4, 1.0, ["rb", "rb_arg"], tree / "data" / "people" / "rb3")


def test_numbers_and_options(tree):
    got = ef.parse_args(["0.5", "2", "src=x", "dst=out"])
    assert got == (0.5, 2.0, ["x"], tree / "data" / "people" / "out")


@pytest.mark.parametrize("argv, text", [
    (["abc"], "SHARE must be a number"),
    (["0.4", "abc"], "SPREAD must be a number"),
    (["1"], "below 1"),
    (["-0.1"], "below 1"),
    (["foo=1"], "unknown option"),
    (["0.1", "1", "2"], "usage"),
    (["src="], "at least one name"),
    (["src=missing"], "not found"),
    (["dst=a/b"], "plain directory name"),
    (["dst="], "plain directory name"),
])
def test_bad_arguments(tree, argv, text):
    with pytest.raises(SystemExit) as e:
        ef.parse_args(argv)
    assert text in str(e.value)


def test_unprompted_board_found_and_excused():
    c = [[0, "hi"], [1, "you are a tiny board"]]
    assert ef.unprompted_board(c) == 1
    c = [[0, "are you a robot"], [1, "i am a tiny board"]]
    assert ef.unprompted_board(c) is None


def test_load_filters_and_cuts(tmp_path, capsys):
    src = tmp_path / "rb"
    src.mkdir()
    keep = [[0, "yo"], [1, "hi"], [0, "sup"], [1, "ok"]]
    face = [[0, "yo"], [1, "your face"], [0, "sup"], [1, "ok"]]
    cut = [[0, "yo"], [1, "hi"], [0, "sup"], [1, "ok"], [0, "hm"], [1, "you are a toaster"]]
    short = [[0, "yo"], [1, "hi"]]
    for split in ("train", "val"):
        rows = [keep, face, cut, short] if split == "train" else [keep]
        (src / f"{split}.jsonl").write_text("".join(json.dumps({"turns": c}) + "\n" for c in rows))
    data = ef.load(src, random.Random(0))
    assert len(data["val"]) == 1
    assert len(data["train"]) == 2
    assert [len(c) for c in data["train"]] == [4, 4]
    assert "dropped" in capsys.readouterr().out
