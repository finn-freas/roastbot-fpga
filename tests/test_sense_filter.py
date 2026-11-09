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
