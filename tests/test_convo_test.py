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


class FakePerson:
    n_best = 3

    def __init__(self, key):
        self.key = key
        self.resets = 0

    def reset(self):
        self.resets += 1


@pytest.fixture
def stubbed(models, monkeypatch):
    calls = []

    def reply_best(p, tk, m, temp, k, rng, n_best):
        calls.append((p, m, rng, n_best, dict(chat.REPEAT), chat.REPEAT_EMOJI))
        return ["one�", "two"], 23, None

    monkeypatch.setattr(ct.T, "load_chat", lambda: None)
    monkeypatch.setattr(chat, "Person", FakePerson)
    monkeypatch.setattr(chat, "make_rng", lambda s: s)
    monkeypatch.setattr(chat, "reply_best", reply_best)
    monkeypatch.setattr(chat, "REPEAT", {"pr": 9.0})
    monkeypatch.setattr(chat, "REPEAT_EMOJI", 7.0)
    return calls


def test_main_replays_every_chat_and_prints_the_cost(stubbed, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["convo_test.py"])
    ct.main()
    out = capsys.readouterr().out
    assert len(stubbed) == sum(len(c) for c in ct.CHATS)
    assert stubbed[0][0].resets == len(ct.CHATS)
    assert [c[2] for c in stubbed[:4]] == [1, 1, 1, 1]
    assert stubbed[0][3] == 3
    assert out.count("--- chat ") == len(ct.CHATS)
    assert "  bot: one\n" in out and "  bot: two\n" in out
    assert "[23 tokens, ~1.0 s on the board]" in out


def test_main_applies_the_repeat_options(stubbed, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["convo_test.py", "pr", "repeat=2", "emoji=0.5"])
    ct.main()
    assert stubbed[0][4] == {"pr": 2.0}
    assert stubbed[0][5] == 0.5


def test_main_norepeat_clears_the_repeat_table(stubbed, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["convo_test.py", "pr", "norepeat"])
    ct.main()
    assert stubbed[0][4] == {}
    assert stubbed[0][5] == 7.0
