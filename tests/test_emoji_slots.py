import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tok as T
from scripts import emoji_slots


def test_default_slot_list_is_valid():
    emoji_slots.check_slots(emoji_slots.EMOJI_SLOTS)


def test_too_many_slots_rejected():
    with pytest.raises(SystemExit) as e:
        emoji_slots.check_slots([f"e{i}" for i in range(T.N_RESERVED + 1)])
    assert "do not fit" in str(e.value)


def test_duplicates_named():
    with pytest.raises(SystemExit) as e:
        emoji_slots.check_slots(["a", "b", "a"])
    assert "duplicate" in str(e.value) and "a" in str(e.value)


def test_missing_tokenizer_is_named(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "TOK", tmp_path)
    with pytest.raises(SystemExit) as e:
        emoji_slots.main()
    assert "tokenizer.json not found" in str(e.value)
    assert not (tmp_path / "slots.json").exists()
