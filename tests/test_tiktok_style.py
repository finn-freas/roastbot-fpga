import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import tiktok_style as ts


def test_keeps_styled_comments_and_strips_whitespace():
    got = ts.candidates(["  bro really thought that was a good idea 💀 ", "nah this is wild to me"])
    assert got == {"bro really thought that was a good idea 💀", "nah this is wild to me"}


def test_rejects_wrong_length():
    assert ts.candidates(["bro no way", " ".join(["bro"] * 21)]) == set()


def test_rejects_without_style_marker():
    assert ts.candidates(["this blender works really well for me"]) == set()


def test_rejects_mentions_and_other_characters():
    assert ts.candidates(["@user bro that is crazy work 💀", "bro that is crazy work ñ", "bro that is crazy 日本語 work"]) == set()


def test_rejects_group_and_slur_terms():
    assert ts.candidates(["bro that is so gay right now 💀", "bro kys right now no way 💀"]) == set()


def test_ignores_non_strings():
    assert ts.candidates([None, 3, "bro that blender is crazy work 💀"]) == {"bro that blender is crazy work 💀"}


def run_main(monkeypatch, tmp_path, texts, top=500):
    import json
    import types

    import torch

    class FakeDs:
        def select_columns(self, cols):
            return self

        def iter(self, batch_size):
            yield {"text": texts}

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda name, split: FakeDs()
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    class Batch(dict):
        def to(self, device):
            return self

    class Tok:
        calls = []

        def convert_tokens_to_ids(self, t):
            return int(t)

        def __call__(self, prompts, return_tensors, padding):
            Tok.calls.append(prompts)
            return Batch(n=len(prompts))

    class Net:
        def __call__(self, n, logits_to_keep):
            logits = torch.full((n, 1, 10), -50.0)
            for i in range(n):
                logits[i, 0, i % 10] = 50.0
            return types.SimpleNamespace(logits=logits)

    monkeypatch.setattr(ts, "q4", lambda name: (Tok(), Net()))
    monkeypatch.setattr(ts, "chat", lambda tok, system, user: user)
    monkeypatch.setattr(ts, "ROOT", tmp_path)
    monkeypatch.setattr(ts, "TOP", top)
    ts.main()
    return json.loads((tmp_path / "data" / "style" / "tiktok_roasts.json").read_text()), Tok.calls


def test_main_writes_the_highest_scored_candidates(monkeypatch, tmp_path, capsys):
    texts = [f"bro comment number {i} is wild 💀" for i in range(4)] + ["not a styled comment at all here"]
    kept, calls = run_main(monkeypatch, tmp_path, texts, top=2)
    pool = sorted(t for t in texts if t.startswith("bro"))
    assert kept == [pool[3], pool[2]]
    assert len(calls) == 1 and len(calls[0]) == 4
    assert "4 candidates" in capsys.readouterr().out
