import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import roast_bank


def test_missing_file_names_the_generator(tmp_path):
    with pytest.raises(SystemExit) as e:
        roast_bank.load_mined(tmp_path / "none.json")
    assert "tiktok_style.py" in str(e.value)


def test_invalid_json(tmp_path):
    f = tmp_path / "t.json"
    f.write_text("[1,")
    with pytest.raises(SystemExit) as e:
        roast_bank.load_mined(f)
    assert "not valid JSON" in str(e.value)


@pytest.mark.parametrize("body", [{"a": 1}, [1, 2], ["ok", 3]])
def test_wrong_shape(tmp_path, body):
    f = tmp_path / "t.json"
    f.write_text(json.dumps(body))
    with pytest.raises(SystemExit) as e:
        roast_bank.load_mined(f)
    assert "list of strings" in str(e.value)


def test_long_comments_are_dropped(tmp_path):
    f = tmp_path / "t.json"
    f.write_text(json.dumps(["short one", " ".join(["w"] * (roast_bank.MAX_WORDS + 1))]))
    assert roast_bank.load_mined(f) == ["short one"]


def test_extra_argument_prints_usage():
    with pytest.raises(SystemExit) as e:
        roast_bank.main(["x"])
    assert "usage" in str(e.value)


def test_empty_after_filter_exits_before_model(tmp_path, monkeypatch):
    f = tmp_path / "t.json"
    f.write_text("[]")
    monkeypatch.setattr(roast_bank, "TIKTOK", f)
    with pytest.raises(SystemExit) as e:
        roast_bank.main([])
    assert "no comments" in str(e.value)


class Batch(dict):
    def to(self, device):
        return self


class ScoringTok:
    def __call__(self, prompts, return_tensors, padding):
        return Batch(prompts=prompts)

    def convert_tokens_to_ids(self, t):
        return int(t)


class ScoringNet:
    def __init__(self, scores):
        self.scores = scores

    def __call__(self, prompts, logits_to_keep):
        import torch

        out = torch.full((len(prompts), 1, 10), -1e4)
        for r, p in enumerate(prompts):
            comment = p.split("Comment: ")[1].split("\n")[0]
            out[r, 0, self.scores[comment]] = 1e4

        class R:
            logits = out

        return R()


def run_main(tmp_path, monkeypatch, comments, scores):
    f = tmp_path / "t.json"
    f.write_text(json.dumps(comments))
    monkeypatch.setattr(roast_bank, "TIKTOK", f)
    monkeypatch.setattr(roast_bank, "ROOT", tmp_path)
    monkeypatch.setattr(roast_bank, "chat", lambda tok, system, user: user)
    monkeypatch.setattr(roast_bank, "q4", lambda name: (ScoringTok(), ScoringNet(scores)))
    (tmp_path / "data" / "style").mkdir(parents=True)
    roast_bank.main([])
    return json.loads((tmp_path / "data" / "style" / "roast_bank.json").read_text())


def test_main_keeps_high_scoring_comments_after_hand_written_ones(tmp_path, monkeypatch, capsys):
    bank = run_main(tmp_path, monkeypatch, ["good one", "bad one"], {"good one": 9, "bad one": 2})
    assert bank[:len(roast_bank.HAND)] == roast_bank.HAND
    assert bank[len(roast_bank.HAND):] == ["good one"]
    assert "1 of 2 TikTok comments" in capsys.readouterr().out


def test_main_drops_blocked_and_duplicate_comments(tmp_path, monkeypatch):
    comments = ["skill issue", "kys lol", "the muslim thing", "fine one"]
    scores = {c: 9 for c in comments}
    bank = run_main(tmp_path, monkeypatch, comments, scores)
    assert bank.count("skill issue") == 1
    assert "kys lol" not in bank
    assert "the muslim thing" not in bank
    assert bank[-1] == "fine one"
