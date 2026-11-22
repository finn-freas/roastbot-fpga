import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import mom_rate


class FakePerson:
    def __init__(self, *a):
        self.resets = 0

    def reset(self):
        self.resets += 1


def run(monkeypatch, tmp_path, reply, chats=2):
    seen = []

    def reply_best(p, tk, m, temp, k, rng, n_best, budget):
        seen.append(m)
        return reply(m), None, None

    monkeypatch.setattr(mom_rate.T, "load_chat", lambda: None)
    monkeypatch.setattr(mom_rate.chat, "Person", FakePerson)
    monkeypatch.setattr(mom_rate.chat, "make_rng", lambda s: s)
    monkeypatch.setattr(mom_rate.chat, "reply_best", reply_best)
    mom_rate.main([str(tmp_path), str(chats)])
    return seen


def test_counts_mom_lines_and_those_after_an_insult(monkeypatch, tmp_path, capsys):
    seen = run(monkeypatch, tmp_path, lambda m: ["your mom"] if m in mom_rate.INS else ["ok"])
    out = capsys.readouterr().out
    assert len(seen) == 60
    insults = sum(m in mom_rate.INS for m in seen)
    assert f"{insults} mom lines in 2 chats of 30 ({insults} after an insult)" in out
    assert out.count("chat 1 msg") + out.count("chat 2 msg") == insults


def test_no_mom_lines_reports_zero_with_empty_gaps(monkeypatch, tmp_path, capsys):
    run(monkeypatch, tmp_path, lambda m: ["fine"], chats=1)
    assert "0 mom lines in 1 chats of 30 (0 after an insult), gaps []" in capsys.readouterr().out


def test_gaps_between_consecutive_hits(monkeypatch, tmp_path, capsys):
    run(monkeypatch, tmp_path, lambda m: ["ur mama"], chats=1)
    out = capsys.readouterr().out
    assert "30 mom lines in 1 chats of 30" in out
    assert out.strip().endswith("gaps " + str([1] * 29))
