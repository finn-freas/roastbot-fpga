import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import mom_judge


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(mom_judge, "ROOT", tmp_path)
    d = tmp_path / "data" / "people" / "rb5"
    d.mkdir(parents=True)
    for split in ("train", "val"):
        (d / f"{split}.jsonl").write_text("")
    return d


def test_defaults(data):
    assert mom_judge.parse_args([]) == (0.25, data)


def test_drop_and_dst(data, tmp_path):
    other = tmp_path / "data" / "people" / "x"
    other.mkdir()
    for split in ("train", "val"):
        (other / f"{split}.jsonl").write_text("")
    assert mom_judge.parse_args(["0.5", "dst=x"]) == (0.5, other)


def test_bad_drop(data):
    with pytest.raises(SystemExit) as e:
        mom_judge.parse_args(["abc"])
    assert "number" in str(e.value)


@pytest.mark.parametrize("drop", ["1", "-0.1", "2"])
def test_drop_out_of_range(data, drop):
    with pytest.raises(SystemExit) as e:
        mom_judge.parse_args([drop])
    assert "below 1" in str(e.value)


def test_unknown_option(data):
    with pytest.raises(SystemExit) as e:
        mom_judge.parse_args(["foo=1"])
    assert "unknown option" in str(e.value)


def test_too_many_positionals(data):
    with pytest.raises(SystemExit):
        mom_judge.parse_args(["0.1", "0.2"])


def test_missing_data(tmp_path, monkeypatch):
    monkeypatch.setattr(mom_judge, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as e:
        mom_judge.parse_args([])
    assert "not found" in str(e.value)
