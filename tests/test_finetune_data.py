import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import finetune as F
import tok as T

TEXTS = ["Hello there my friend", "hello there my friend", "what are you doing tonight", "Nothing much"] * 20


@pytest.fixture(scope="module")
def tk():
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=400, special_tokens=T.SPECIAL, min_frequency=1,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    t.train_from_iterator(TEXTS, tr)
    return t


def test_encode_sessions_marks_only_him_tokens_and_markers(tk, tmp_path):
    path = tmp_path / "s.jsonl"
    turns = [[0, "hello there"], [1, "what are you doing"]]
    path.write_text(json.dumps({"turns": turns}) + "\n")
    (ids, lab), = F.encode_sessions(tk, path)
    a = tk.encode("hello there", add_special_tokens=False).ids
    b = tk.encode("what are you doing", add_special_tokens=False).ids
    assert ids.tolist() == [T.BOS, T.YOU] + a + [T.HIM] + b + [T.YOU]
    assert lab.tolist() == [False, True] + [False] * len(a) + [True] + [True] * len(b) + [True]
    assert ids.dtype == np.int64 and lab.dtype == bool


def test_encode_sessions_reads_every_line(tk, tmp_path):
    path = tmp_path / "s.jsonl"
    lines = [json.dumps({"turns": [[1, "hello"]]}), json.dumps({"turns": [[0, "nothing"], [1, "hello"]]})]
    path.write_text("\n".join(lines) + "\n")
    out = F.encode_sessions(tk, path)
    assert len(out) == 2
    assert out[0][0][0] == T.BOS and out[0][0][-1] == T.YOU


def test_load_kd_folds_case_and_newline_ids_and_sums_probabilities(tk, tmp_path):
    cap = tk.token_to_id("H")
    low = tk.token_to_id("h")
    assert cap is not None and low is not None and cap != low
    nl = tk.token_to_id("Ċ")
    assert nl is not None
    ids = np.array([T.BOS, T.YOU, 10, 11], np.int64)
    lab = np.array([False, True, True, True])
    pos = np.array([1, 2])
    idx = np.array([[cap, low, -1], [nl, 12, -1]])
    prob = np.array([[0.5, 0.25, 0.0], [0.125, 0.5, 0.0]], np.float32)
    path = tmp_path / "kd.pkl"
    path.write_bytes(pickle.dumps({"K": 3, "rows": [(pos, idx, prob)]}))
    (ki, kp), = F.load_kd(tk, path, [(ids, lab)])
    assert ki.shape == (4, 3) and kp.shape == (4, 3)
    assert ki[1].tolist() == [low, -1, -1]
    assert kp[1, 0] == pytest.approx(0.75)
    assert ki[2].tolist() == [12, T.YOU, -1]
    assert kp[2, :2].tolist() == pytest.approx([0.5, 0.125])
    assert ki[0].tolist() == [-1, -1, -1] and kp[3].sum() == 0
