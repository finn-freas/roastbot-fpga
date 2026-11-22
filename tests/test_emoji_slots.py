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


def _tiny_tokenizer(tmp_path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    t.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=500, special_tokens=T.SPECIAL, min_frequency=1,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    t.train_from_iterator(["omg no way", "ok see u", "lol see u", "hearing that tho"] * 10, tr)
    t.save(str(tmp_path / "tokenizer.json"))
    return t


def test_main_fills_slots_and_writes_slots_file(tmp_path, monkeypatch, capsys):
    import json

    t = _tiny_tokenizer(tmp_path)
    monkeypatch.setattr(T, "TOK", tmp_path)
    monkeypatch.setattr(T, "CHAT_TOK", tmp_path / "tokenizer_chat.json")
    monkeypatch.setattr(T.load, "__defaults__", (tmp_path / "tokenizer.json",))
    monkeypatch.setattr(T, "VOCAB", t.get_vocab_size())
    emoji_slots.main()
    slots = json.loads((tmp_path / "slots.json").read_text())
    assert [s[1] for s in slots] == emoji_slots.EMOJI_SLOTS
    assert [s[0] for s in slots] == list(range(4, 4 + len(slots)))
    assert f"{len(slots)} emoji in slots" in capsys.readouterr().out
    chat_tok = T.load_chat()
    assert chat_tok.token_to_id("😂") == slots[emoji_slots.EMOJI_SLOTS.index("😂")][0]
