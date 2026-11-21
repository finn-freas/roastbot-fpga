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
