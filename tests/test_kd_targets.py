import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import kd_targets


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(kd_targets, "ROOT", tmp_path)
    d = tmp_path / "data" / "people" / "pr"
    d.mkdir(parents=True)
    for split in ("val", "train"):
        (d / f"{split}.jsonl").write_text("")
    return d


def test_defaults(data):
    assert kd_targets.parse_args([]) == ("pr", 20)


def test_person_and_k(data):
    assert kd_targets.parse_args(["pr", "8"]) == ("pr", 8)


def test_too_many_arguments_print_usage(data):
    with pytest.raises(SystemExit) as e:
        kd_targets.parse_args(["pr", "8", "x"])
    assert "usage" in str(e.value)


@pytest.mark.parametrize("k", ["x", "2.5"])
def test_non_integer_k_is_rejected(data, k):
    with pytest.raises(SystemExit) as e:
        kd_targets.parse_args(["pr", k])
    assert "whole number" in str(e.value)


def test_k_must_be_positive(data):
    with pytest.raises(SystemExit) as e:
        kd_targets.parse_args(["pr", "0"])
    assert "at least 1" in str(e.value)


def test_missing_split_file_is_named(data):
    (data / "val.jsonl").unlink()
    with pytest.raises(SystemExit) as e:
        kd_targets.parse_args(["pr"])
    assert "val.jsonl not found" in str(e.value)


def test_unknown_person_is_rejected(data):
    with pytest.raises(SystemExit) as e:
        kd_targets.parse_args(["nobody"])
    assert "not found" in str(e.value)
