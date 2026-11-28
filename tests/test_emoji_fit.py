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


def test_score_reuses_a_cache_with_matching_length(tmp_path):
    data = {"train": [[[0, "hi"], [1, "yo"], [0, "ok"], [1, "bye"]]], "val": [[[1, "x"]]]}
    cache = tmp_path / "scores.json"
    cache.write_text("[[0.1], [0.2], [0.3]]")

    def model():
        raise AssertionError("cache should be used")

    spots, scores = ef.score(data, cache, model)
    assert spots == [("train", 0, 1), ("train", 0, 3), ("val", 0, 0)]
    assert scores == [[0.1], [0.2], [0.3]]


def test_main_writes_dataset_from_cached_scores(tmp_path, monkeypatch, capsys):
    import types

    monkeypatch.setattr(ef, "ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "writer", types.SimpleNamespace(WRITER="w", q4=lambda w: 1 / 0))
    src = tmp_path / "data" / "people" / "rb"
    src.mkdir(parents=True)
    convs = [[[0, "yo"], [1, f"hi {n}"], [0, "sup"], [1, "ok then"]] for n in range(6)]
    for split in ("train", "val"):
        (src / f"{split}.jsonl").write_text("".join(json.dumps({"turns": c}) + "\n" for c in convs))
    dst = tmp_path / "data" / "people" / "out"
    n_spots = 2 * len(convs) * 2
    rows = [[-5.0 if q % 2 else -0.1] + [-3.0 - 0.1 * ((q + k) % 5) for k in range(20)] for q in range(n_spots)]
    dst.mkdir()
    (dst / "emoji_scores_rb.json").write_text(json.dumps(rows))
    ef.main(["0.5", "src=rb", "dst=out"])
    out = capsys.readouterr().out
    assert "B texts end in emoji" in out
    for split in ("train", "val"):
        got = [json.loads(line)["turns"] for line in (dst / f"{split}.jsonl").read_text().splitlines()]
        assert len(got) == len(convs)
        assert all(len(c) == 4 and c[0] == [0, "yo"] for c in got)
    texts = [t for line in (dst / "train.jsonl").read_text().splitlines() for k, t in json.loads(line)["turns"] if k == 1]
    assert any(t.split(" ")[-1] in ef.EMOJI or t[-1] in "".join(ef.EMOJI) for t in texts)
