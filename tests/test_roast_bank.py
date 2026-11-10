import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import roast_bank


def test_missing_file_names_the_generator(tmp_path):
    with pytest.raises(SystemExit) as e:
        roast_bank.load_mined(tmp_path / "none.json")
    assert "tiktok_style.py" in str(e.value)


def test_invalid_json(tmp_path):
    f = tmp_path / "t.json"
    f.write_text("[1,")
    with pytest.raises(SystemExit) as e:
        roast_bank.load_mined(f)
    assert "not valid JSON" in str(e.value)


@pytest.mark.parametrize("body", [{"a": 1}, [1, 2], ["ok", 3]])
def test_wrong_shape(tmp_path, body):
    f = tmp_path / "t.json"
    f.write_text(json.dumps(body))
    with pytest.raises(SystemExit) as e:
        roast_bank.load_mined(f)
    assert "list of strings" in str(e.value)


def test_long_comments_are_dropped(tmp_path):
    f = tmp_path / "t.json"
    f.write_text(json.dumps(["short one", " ".join(["w"] * (roast_bank.MAX_WORDS + 1))]))
    assert roast_bank.load_mined(f) == ["short one"]


def test_extra_argument_prints_usage():
    with pytest.raises(SystemExit) as e:
        roast_bank.main(["x"])
    assert "usage" in str(e.value)


def test_empty_after_filter_exits_before_model(tmp_path, monkeypatch):
    f = tmp_path / "t.json"
    f.write_text("[]")
    monkeypatch.setattr(roast_bank, "TIKTOK", f)
    with pytest.raises(SystemExit) as e:
        roast_bank.main([])
    assert "no comments" in str(e.value)
