import json
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import gen_dialogs, mom_judge


def conv(trigger, joke="ur mom"):
    return [[0, "hi"], [1, "yo"], [0, "sup"], [1, "nm"], [0, trigger], [1, joke], [0, "x"], [1, "y"]]


GOOD = conv("fuck you")
LOW = conv("stfu")
CALM = conv("hello")
PLAIN = conv("hello", "nothing here")


class FakeTok:
    def __init__(self):
        self.prompts = []

    def convert_tokens_to_ids(self, c):
        return {"Y": 0, "N": 1}[c]

    def __call__(self, ps, return_tensors, padding):
        self.prompts = ps
        return types.SimpleNamespace(to=lambda dev: {})


class FakeNet:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, logits_to_keep):
        ys = [5.0 if "F: fuck you" in p else -5.0 for p in self.tok.prompts]
        logits = torch.tensor([[[y, 0.0]] for y in ys])
        return types.SimpleNamespace(logits=logits)


@pytest.fixture
def judge(tmp_path, monkeypatch):
    d = tmp_path / "data" / "people" / "rb5"
    d.mkdir(parents=True)
    for split in ("train", "val"):
        rows = [GOOD, LOW, CALM, PLAIN]
        (d / f"{split}.jsonl").write_text("".join(json.dumps({"turns": c}) + "\n" for c in rows))
    tok = FakeTok()
    writer = types.ModuleType("writer")
    writer.WRITER = "w"
    writer.chat = lambda t, system, ask: ask
    writer.q4 = lambda w: (tok, FakeNet(tok))
    monkeypatch.setitem(sys.modules, "writer", writer)
    fake_gen = types.ModuleType("gen_dialogs")
    fake_gen.MOM_RE = gen_dialogs.MOM_RE
    monkeypatch.setitem(sys.modules, "gen_dialogs", fake_gen)
    monkeypatch.setattr(mom_judge, "ROOT", tmp_path)
    return d


def test_cuts_conversations_with_unfitting_mom_lines(judge, capsys):
    mom_judge.main(["0.5"])
    rows = [json.loads(line)["turns"] for line in open(judge / "train.jsonl")]
    assert rows == [GOOD, LOW[:4], CALM[:4], PLAIN]
    out = capsys.readouterr().out
    assert "rb5 train: 3 your-mom lines, 1 not after an insult, 1 rated lowest (< 0.99); 2 conversations cut; 4 of 4 conversations kept" in out
    assert "rb5 val: 3 your-mom lines" in out
    assert out.count("doesn't fit") == 2 and out.count("fits") == 2


def test_zero_drop_cuts_only_lines_not_after_an_insult(judge, capsys):
    mom_judge.main(["0"])
    rows = [json.loads(line)["turns"] for line in open(judge / "val.jsonl")]
    assert rows == [GOOD, LOW, CALM[:4], PLAIN]
    assert "0 rated lowest" in capsys.readouterr().out


def test_conversation_cut_below_two_bot_turns_is_dropped(judge):
    short = [[0, "hello"], [1, "ur mom"], [0, "a"], [1, "b"], [0, "c"], [1, "d"]]
    (judge / "train.jsonl").write_text("".join(json.dumps({"turns": c}) + "\n" for c in [GOOD, short]))
    mom_judge.main(["0"])
    rows = [json.loads(line)["turns"] for line in open(judge / "train.jsonl")]
    assert rows == [GOOD]
