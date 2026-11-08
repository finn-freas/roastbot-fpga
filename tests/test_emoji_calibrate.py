import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import emoji_calibrate as ec


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "ROOT", tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "pr.pt").write_bytes(b"")
    monkeypatch.setattr(ec, "MDIR", models)
    d = tmp_path / "data" / "people" / "pr"
    d.mkdir(parents=True)
    for split in ("train", "val"):
        (d / f"{split}.jsonl").write_text("")
    return tmp_path


def test_defaults_to_pr(tree):
    assert ec.parse_args([]) == (tree / "models", ["pr"])


def test_dir_option(tree):
    other = tree / "other"
    other.mkdir()
    (other / "pr.pt").write_bytes(b"")
    assert ec.parse_args(["--dir", str(other), "pr"]) == (other, ["pr"])


@pytest.mark.parametrize("argv", [["--dir"], ["--bogus"], ["pr", "-x"]])
def test_bad_options_print_usage(tree, argv):
    with pytest.raises(SystemExit) as e:
        ec.parse_args(argv)
    assert "usage" in str(e.value)


def test_missing_directory_is_named(tree):
    with pytest.raises(SystemExit) as e:
        ec.parse_args(["--dir", str(tree / "nope")])
    assert "nope is not a directory" in str(e.value)


def test_missing_model_is_named(tree):
    with pytest.raises(SystemExit) as e:
        ec.parse_args(["zz"])
    assert "zz.pt not found" in str(e.value)


def test_missing_data_is_named(tree):
    (tree / "models" / "zz.pt").write_bytes(b"")
    with pytest.raises(SystemExit) as e:
        ec.parse_args(["zz"])
    assert "train.jsonl not found" in str(e.value)
