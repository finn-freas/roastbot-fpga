import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import mom_cap


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(mom_cap, "ROOT", tmp_path)
    d = tmp_path / "data" / "people" / "rb_mom"
    d.mkdir(parents=True)
    for split in ("val", "train"):
        (d / f"{split}.jsonl").write_text("")
    return d


def test_default_cap(data):
    assert mom_cap.parse_args([]) == (25, data)


def test_explicit_cap(data):
    assert mom_cap.parse_args(["3"])[0] == 3


def test_too_many_arguments(data):
    with pytest.raises(SystemExit) as e:
        mom_cap.parse_args(["1", "2"])
    assert "usage" in str(e.value)


@pytest.mark.parametrize("cap", ["x", "1.5", ""])
def test_non_integer_cap(data, cap):
    with pytest.raises(SystemExit) as e:
        mom_cap.parse_args([cap])
    assert "whole number" in str(e.value)


def test_cap_below_one(data):
    with pytest.raises(SystemExit) as e:
        mom_cap.parse_args(["0"])
    assert "at least 1" in str(e.value)


def test_missing_split_file_is_named(data):
    (data / "train.jsonl").unlink()
    with pytest.raises(SystemExit) as e:
        mom_cap.parse_args([])
    assert "train.jsonl not found" in str(e.value)
