import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chat
from scripts import convo_test as ct


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setattr(chat, "MODELS", tmp_path)
    (tmp_path / "pr_board.pkl").write_bytes(b"")
    return tmp_path


def test_defaults(models):
    assert ct.parse_args([]) == ("pr", False, None, None)


def test_options(models):
    assert ct.parse_args(["pr", "norepeat", "repeat=2.5", "emoji=1"]) == ("pr", True, 2.5, 1.0)


def test_compressed_model_is_accepted(models):
    (models / "x_board.pkl.xz").write_bytes(b"")
    assert ct.parse_args(["x"])[0] == "x"


def test_unknown_model_is_named(models):
    with pytest.raises(SystemExit) as e:
        ct.parse_args(["nobody"])
    assert "no board model for 'nobody'" in str(e.value)


def test_unknown_option(models):
    with pytest.raises(SystemExit) as e:
        ct.parse_args(["pr", "bogus"])
    assert "unknown option 'bogus'" in str(e.value)


def test_bad_number(models):
    with pytest.raises(SystemExit) as e:
        ct.parse_args(["pr", "repeat=abc"])
    assert "repeat= needs a number" in str(e.value)


@pytest.mark.parametrize("arg", ["repeat=nan", "emoji=inf", "emoji=-inf"])
def test_non_finite_number(models, arg):
    with pytest.raises(SystemExit) as e:
        ct.parse_args(["pr", arg])
    assert "needs a number" in str(e.value)
