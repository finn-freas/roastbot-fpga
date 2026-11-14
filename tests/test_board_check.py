import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import chat
from scripts import board_check


@pytest.fixture
def models(tmp_path, monkeypatch):
    (tmp_path / "pr_board.pkl").write_bytes(b"")
    monkeypatch.setattr(chat, "MODELS", tmp_path)
    return tmp_path


def test_defaults(models):
    assert board_check.parse_args([]) == ("pr", False, False)


def test_flags(models):
    assert board_check.parse_args(["pr", "--best", "--argue"]) == ("pr", True, True)


def test_unknown_option(models):
    with pytest.raises(SystemExit) as e:
        board_check.parse_args(["--bogus"])
    assert "unknown option: --bogus" in str(e.value)


def test_unknown_model(models):
    with pytest.raises(SystemExit) as e:
        board_check.parse_args(["zz"])
    assert "no model 'zz'" in str(e.value)


def test_too_many_models(models):
    with pytest.raises(SystemExit):
        board_check.parse_args(["pr", "pr"])


class FakeLink:
    def __init__(self):
        self.seeds = []

    def ping(self):
        return "pong"

    def seed(self, x):
        self.seeds.append(x)


class FakePerson:
    n_best = 2
    hist = ["h"]

    def reset(self):
        pass

    def sync(self):
        pass


def patch_main(monkeypatch, models, cpu_answer, board_answer):
    link = FakeLink()
    monkeypatch.setattr(board_check.T, "load_chat", lambda: object())
    monkeypatch.setattr(board_check, "connect", lambda: link)
    monkeypatch.setattr(chat, "BoardPerson", lambda key, lk: FakePerson())
    monkeypatch.setattr(chat, "Person", lambda key: FakePerson())
    monkeypatch.setattr(chat, "reply", lambda p, t, q, temp, pen, rng: (cpu_answer, 1, None))
    monkeypatch.setattr(chat, "board_reply", lambda p, t, q, temp, pen: (board_answer, 3, None))
    return link


def test_main_reports_identical_replies(models, monkeypatch, capsys):
    link = patch_main(monkeypatch, models, "ok", "ok")
    monkeypatch.setattr(sys, "argv", ["board_check.py"])
    board_check.main()
    out = capsys.readouterr().out
    n = 2 * len(board_check.PROMPTS)
    assert "ping pong" in out
    assert f"{n}/{n} replies identical to the CPU" in out
    assert "DIFFERENT" not in out
    assert len(link.seeds) == 2


def test_main_flags_a_reply_that_differs_from_the_cpu(models, monkeypatch, capsys):
    patch_main(monkeypatch, models, "cpu says", "board says")
    monkeypatch.setattr(sys, "argv", ["board_check.py", "--argue"])
    board_check.main()
    out = capsys.readouterr().out
    n = 2 * len(board_check.ARGUE)
    assert f"0/{n} replies identical to the CPU" in out
    assert out.count("DIFFERENT") == n
    assert "cpu: cpu says" in out
    assert "'ur a toaster'" in out
