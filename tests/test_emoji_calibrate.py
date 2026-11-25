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


@pytest.fixture
def small_vocab(monkeypatch):
    monkeypatch.setattr(ec.T, "VOCAB", 6)
    monkeypatch.setattr(ec.chat, "EMOJI", [False, True, False, True, True, False])


def test_mix_bias_lists_only_emoji_tokens(small_vocab):
    pred = ec.torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    real = ec.torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert [i for i, _ in ec.mix_bias(pred, real)] == [1, 3, 4]


def test_mix_bias_is_zero_when_prediction_matches(small_vocab):
    pred = ec.torch.tensor([0.0, 4.0, 0.0, 2.0, 1.0, 0.0])
    assert ec.mix_bias(pred, pred.clone()) == [[1, 0.0], [3, 0.0], [4, 0.0]]


def test_mix_bias_boosts_underpredicted_and_clamps(small_vocab):
    pred = ec.torch.tensor([0.0, 100.0, 0.0, 0.0, 100.0, 0.0])
    real = ec.torch.tensor([0.0, 100.0, 0.0, 100000.0, 0.0, 0.0])
    bias = dict((i, b) for i, b in ec.mix_bias(pred, real))
    assert bias[3] == ec.BIAS_RANGE[1]
    assert bias[4] == ec.BIAS_RANGE[0]
    assert ec.BIAS_RANGE[0] <= bias[1] <= ec.BIAS_RANGE[1]


def test_mix_bias_survives_zero_prediction(small_vocab):
    zero = ec.torch.zeros(6)
    assert ec.mix_bias(zero, zero) == [[1, 0.0], [3, 0.0], [4, 0.0]]


class FakeEnc:
    def __init__(self, ids):
        self.ids = ids


class FakeTok:
    def encode_batch(self, msgs, add_special_tokens):
        assert add_special_tokens is False
        return [FakeEnc([1] if "x" in m else [0, 2]) for m in msgs]


def test_real_rate_counts_their_messages_with_emoji(tree, small_vocab, monkeypatch):
    monkeypatch.setattr(ec.T, "load_chat", lambda: FakeTok())
    d = tree / "data" / "people" / "pr"
    (d / "train.jsonl").write_text(
        '{"turns": [[0, "hi"], [1, "x"], [0, "yo"], [1, "plain"]]}\n{"turns": [[1, "xx"]]}\n')
    (d / "val.jsonl").write_text('{"turns": [[1, "nothing"], [0, "x"]]}\n')
    assert ec.real_rate("pr") == pytest.approx(2 / 4)


def test_cold_mass_averages_bot_mass_over_every_text(monkeypatch):
    seen = []

    class FakePerson:
        def __init__(self, key, mdir):
            self.bot_mass = []

        def reset(self):
            self.bot_mass = []

    def fake_reply(p, tok, q, temp, cut, rng):
        seen.append((q, temp, cut))
        p.bot_mass += [0.5, 1.5] if len(seen) % 2 else [1.0]

    monkeypatch.setattr(ec.chat, "Person", FakePerson)
    monkeypatch.setattr(ec.chat, "reply", fake_reply)
    assert ec.cold_mass("pr", None) == pytest.approx(1.0)
    assert [s[0] for s in seen] == ec.TEXTS
    assert {s[1:] for s in seen} == {(0.8, 4.0)}


def test_main_writes_the_emoji_file_and_reports(tree, small_vocab, monkeypatch, capsys):
    import json

    torch = ec.torch
    pred = torch.tensor([0.0, 2.0, 0.0, 2.0, 0.0, 0.0])
    real = torch.tensor([0.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(ec, "measure", lambda key: (pred, real))
    monkeypatch.setattr(ec, "real_rate", lambda key: 0.25)
    monkeypatch.setattr(ec, "cold_mass", lambda key, tok: 0.123456)

    class Tok:
        def decode(self, ids):
            return f"<{ids[0]}>"

    monkeypatch.setattr(ec.T, "load_chat", lambda: Tok())
    monkeypatch.setattr(sys, "argv", ["emoji_calibrate.py"])
    ec.main()
    d = json.loads((tree / "models" / "pr_emoji.json").read_text())
    assert d["corr"] == 1.0
    assert d["rate"] == 0.25
    assert (d["pred"], d["actual"]) == (4.0, 4.0)
    assert d["cold_mass"] == 0.12346
    assert [i for i, _ in d["bias"]] == [1, 3, 4]
    out = capsys.readouterr().out
    assert out.startswith("pr: model expects 4 emoji starts, they made 4")
    assert "25.0% of their messages" in out
    assert "<1>" in out and "fresh-chat mass 0.1235" in out
