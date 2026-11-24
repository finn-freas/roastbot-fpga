import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import board as B
import chat as C
import tok as T


class Person:
    name = "pr"
    key = "pr"
    n_best = 3
    pos = 5
    p = {"cfg": SimpleNamespace(n_layers=2, dim=8, seq_len=64)}

    def __init__(self, *args):
        pass


@pytest.fixture
def session(monkeypatch, tmp_path):
    model = tmp_path / "pr_board.pkl"
    model.write_bytes(b"")
    monkeypatch.setattr(C, "board_file", lambda mdir, key: model)
    monkeypatch.setattr(T, "load_chat", lambda: None)
    monkeypatch.setattr(C, "Person", Person)
    monkeypatch.setattr(sys, "argv", ["chat.py"])

    def run(lines):
        feed = iter(lines)
        monkeypatch.setattr("builtins.input", lambda prompt="": next(feed))
        C.main()

    return run


def test_missing_model_exits_with_the_expected_path(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "board_file", lambda mdir, key: tmp_path / "absent.pkl")
    monkeypatch.setattr(T, "load_chat", lambda: None)
    monkeypatch.setattr(sys, "argv", ["chat.py"])
    with pytest.raises(SystemExit) as e:
        C.main()
    assert "models/pr_board.pkl" in str(e.value)


def test_eof_ends_the_session(monkeypatch, session, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError))
    C.main()
    assert "on this computer" in capsys.readouterr().out


def test_info_reports_the_model_and_settings(monkeypatch, session, capsys):
    monkeypatch.setattr(B, "memory_map", lambda p: {"total": 3 * 2 ** 20})
    session(["/temp 0.5", "/cut 2", "/emoji 1", "/best 2", "/info", "/quit"])
    out = capsys.readouterr().out
    assert "2 layers x 8" in out
    assert "5/64 tokens used" in out
    assert "temp 0.5, cut 2.0, emoji 1.0, best of 2, SDRAM 3.00 MB" in out


def test_blank_lines_are_ignored(monkeypatch, session, capsys):
    calls = []
    monkeypatch.setattr(C, "reply_best", lambda *a, **k: calls.append(a) or (["x"], 1, None))
    session(["", "   ", "/quit"])
    assert calls == []


def test_message_with_several_candidates_uses_reply_best(monkeypatch, session, capsys):
    seen = {}

    def fake(person, tok, text, temp, cut, rng, emoji, n_best):
        seen.update(text=text, temp=temp, cut=cut, emoji=emoji, n_best=n_best, rng=rng)
        return ["hey", "there�"], 7, None

    monkeypatch.setattr(C, "reply_best", fake)
    monkeypatch.setattr(C, "token_cycles", lambda p, pos: C.CLK_HZ // 1000)
    session(["hello", "/quit"])
    out = capsys.readouterr().out
    assert seen["text"] == "hello" and seen["n_best"] == 3 and seen["cut"] == 4.0 and seen["rng"] is not None
    assert "pr>\033[0m hey" in out and "pr>\033[0m there\n" in out
    assert "[7 tokens, board ~7 ms incl. reading your message]" in out


def test_single_reply_mode_uses_reply(monkeypatch, session, capsys):
    seen = {}

    def fake(person, tok, text, temp, cut, rng, emoji):
        seen.update(temp=temp, emoji=emoji)
        return ["ok"], 2, 0.0125

    monkeypatch.setattr(C, "reply", fake)
    session(["/best 1", "/temp 0.9", "/emoji 0.25", "hi", "/quit"])
    out = capsys.readouterr().out
    assert seen == {"temp": 0.9, "emoji": 0.25}
    assert "pr>\033[0m ok" in out
    assert "board ~12 ms" in out


def test_empty_reply_prints_an_ellipsis(monkeypatch, session, capsys):
    monkeypatch.setattr(C, "reply_best", lambda *a, **k: (["�", "  "], 1, None))
    monkeypatch.setattr(C, "token_cycles", lambda p, pos: 1)
    session(["hello", "/quit"])
    assert "pr>\033[0m ..." in capsys.readouterr().out
