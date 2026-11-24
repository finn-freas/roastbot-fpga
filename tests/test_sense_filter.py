import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import sense_filter as sf


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(sf, "ROOT", tmp_path)
    for n in ("rb", "rb_arg", "x"):
        d = tmp_path / "data" / "people" / n
        d.mkdir(parents=True)
        (d / "partial.jsonl").write_text("")
    return tmp_path


def test_defaults(data):
    assert sf.parse_args([]) == (0.15, ["rb", "rb_arg"])


def test_drop_and_sources(data):
    assert sf.parse_args(["0.2", "src=x"]) == (0.2, ["x"])
    assert sf.parse_args(["src=x,rb", "0"]) == (0.0, ["x", "rb"])


@pytest.mark.parametrize("argv, msg", [
    (["abc"], "DROP must be a number"),
    (["1"], "below 1"),
    (["-0.1"], "at least 0"),
    (["0.1", "0.2"], "usage"),
    (["dst=x"], "unknown option 'dst'"),
    (["src="], "at least one"),
    (["src=nope"], "partial.jsonl not found"),
])
def test_bad_arguments(data, argv, msg):
    with pytest.raises(SystemExit) as e:
        sf.parse_args(argv)
    assert msg in str(e.value)


def test_conv_score():
    assert sf.conv_score([]) == -99.0
    assert sf.conv_score([-1.0, -3.0]) == pytest.approx(-7 / 3)


def write_partial(path, convs):
    import json

    path.write_text("".join(json.dumps({"turns": c}) + "\n" for c in convs))


def read_split(path):
    import json

    return [json.loads(line)["turns"] for line in path.read_text().splitlines()]


def conv(text):
    return [[0, "hey"], [1, text]]


def test_main_dedupes_scores_and_splits_the_kept_conversations(data, monkeypatch, capsys):
    import json
    import types

    import writer

    d = data / "data" / "people" / "x"
    convs = [conv("aaa"), conv("bbbbb"), conv("ccccccc"), conv("ddddddddd")]
    write_partial(d / "partial.jsonl", convs + [convs[0], [[0, "only me"]]])
    monkeypatch.setattr(writer, "q4", lambda name: (types.SimpleNamespace(), None))
    monkeypatch.setattr(sf, "line_scores", lambda tok, net, cs: [[float(len(t)) for k, t in c if k] for c in cs])
    sf.main(["0.5", "src=x"])
    scores = json.loads((d / "sense_scores.json").read_text())
    assert scores == [3.0, 5.0, 7.0, 9.0]
    val, train = read_split(d / "val.jsonl"), read_split(d / "train.jsonl")
    assert len(val) == 1 and len(train) == 1
    assert sorted(t[1][1] for t in val + train) == ["ccccccc", "ddddddddd"]
    assert "kept 2 of 4" in capsys.readouterr().out
