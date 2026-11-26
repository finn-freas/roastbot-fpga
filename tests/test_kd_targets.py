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


def test_byte_decoder_covers_every_byte_once():
    bd = kd_targets.byte_decoder()
    assert len(bd) == 256
    assert sorted(bd.values()) == list(range(256))


def test_byte_decoder_keeps_printable_ascii():
    bd = kd_targets.byte_decoder()
    assert all(bd[chr(b)] == b for b in range(ord("!"), ord("~") + 1))
    assert bd["Ġ"] == 32


def test_byte_encoder_inverts_the_decoder():
    assert all(kd_targets.BD[kd_targets.BE[b]] == b for b in range(256))


def test_piece_bytes_maps_byte_level_pieces_and_added_tokens():
    assert kd_targets.piece_bytes("Ġhi", False) == b" hi"
    assert kd_targets.piece_bytes("Ċ", False) == b"\n"
    assert kd_targets.piece_bytes("é", True) == "é".encode()


@pytest.fixture
def ours(tmp_path, monkeypatch):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    import tok as T
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=300, special_tokens=T.SPECIAL, min_frequency=1,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    t.train_from_iterator(["hello there my friend"] * 20, tr)
    t.add_tokens(["lol"])
    path = tmp_path / "tokenizer_chat.json"
    t.save(str(path))
    monkeypatch.setattr(T, "CHAT_TOK", path)
    return kd_targets.Ours(t), t


def test_ours_bytes_of_added_and_regular_tokens(ours):
    o, t = ours
    lol = t.token_to_id("lol")
    assert o.bytes(lol) == b"lol"
    assert o.bytes(t.token_to_id("h")) == b"h"


def test_ours_first_prefers_added_token_prefix(ours):
    o, t = ours
    assert o.first(b"lol!") == t.token_to_id("lol")


def test_ours_first_falls_back_to_the_tokenizer_model(ours):
    o, t = ours
    first = o.first(b"hello")
    assert first == t.model.tokenize("hello")[0].id
    assert o.memo[b"hello"] == first


def test_ours_first_of_empty_bytes_is_none(ours):
    o, _ = ours
    assert o.first(b"") is None
