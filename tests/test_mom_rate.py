import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import mom_rate


@pytest.mark.parametrize("argv", [[], ["x"], ["x", "1", "2"]])
def test_wrong_argument_count_prints_usage(argv):
    with pytest.raises(SystemExit) as e:
        mom_rate.parse_args(argv)
    assert "usage" in str(e.value)


def test_missing_model_directory_is_named(tmp_path):
    with pytest.raises(SystemExit) as e:
        mom_rate.parse_args([str(tmp_path / "nope"), "3"])
    assert "nope is not a directory" in str(e.value)


@pytest.mark.parametrize("n", ["x", "1.5", ""])
def test_non_integer_chat_count_is_rejected(tmp_path, n):
    with pytest.raises(SystemExit) as e:
        mom_rate.parse_args([str(tmp_path), n])
    assert "whole number" in str(e.value)


def test_chat_count_must_be_positive(tmp_path):
    with pytest.raises(SystemExit) as e:
        mom_rate.parse_args([str(tmp_path), "0"])
    assert "at least 1" in str(e.value)


def test_valid_arguments(tmp_path):
    assert mom_rate.parse_args([str(tmp_path), "5"]) == (str(tmp_path), 5)
