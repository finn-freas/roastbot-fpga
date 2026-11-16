import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import board as B
import chat as C
import engine as E
import tok as T

W = [t for t in range(4, T.VOCAB) if not C.EMOJI[t]][:6]
EM = [t for t in range(4, T.VOCAB) if C.EMOJI[t]][0]


class ScriptModel:
    def __init__(self, skip, script):
        self.skip, self.script = skip, script
        self.calls = 0

    def reset(self):
        self.calls = 0

    def forward(self, t, pos, base=0):
        i = self.calls - (self.skip - 1)
        self.calls += 1
        want = self.script[min(max(i, 0), len(self.script) - 1)]
        lg = np.zeros(T.VOCAB, np.int64)
        lg[want] = 1 << 12
        return lg


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[W[5]] * len(text.split()))

    def decode(self, toks):
        return " ".join(f"w{t}" for t in toks)


def person(skip, script):
    p = object.__new__(C.Person)
    p.key = "pr"
    p.p = {
        "cfg": SimpleNamespace(dim=64, kv_dim=32, hidden=128, n_layers=2, n_heads=4, head_size=16, vocab_size=T.VOCAB),
        "fmt": {"logit": 8},
        "exp2": B.exp2_table(),
        "gumbel": B.gumbel_table(8),
    }
    p.m = ScriptModel(skip, script)
    p.emoji_corr, p.emoji_rate, p.emoji_cold = 1.0, None, None
    p.emoji_bias = np.zeros(T.VOCAB, np.int64)
    p.reset()
    return p


def run(p, text="hello there", emoji=0.0):
    return C.reply(p, FakeTok(), text, 1.0, 0.0, C.make_rng(1), emoji)


def test_reply_splits_messages_on_him_and_stops_on_you():
    p = person(5, [W[0], W[1], T.HIM, W[2], T.YOU])
    texts, n, secs = run(p)
    assert texts == [f"w{W[0]} w{W[1]}", f"w{W[2]}"]
    assert n == 5
    assert p.bot_msgs == 2
    assert p.hist[:2] == [T.BOS, T.YOU] and p.hist[-1] == W[2]


def test_reply_prompt_has_bos_only_on_the_first_turn():
    p = person(5, [W[0], T.YOU])
    run(p)
    assert p.hist[0] == T.BOS and p.hist.count(T.BOS) == 1
    p.m.skip = 4
    run(p, "again")
    assert p.hist.count(T.BOS) == 1


def test_reply_drops_empty_messages():
    p = person(5, [T.HIM, W[0], T.HIM, T.HIM, T.YOU])
    texts, n, _ = run(p)
    assert texts == [f"w{W[0]}"]
    assert p.bot_msgs == 1


def test_reply_is_capped_at_max_reply_tokens():
    p = person(5, [W[0]])
    texts, n, _ = run(p)
    assert n == C.MAX_REPLY
    assert len(texts) == 1 and len(texts[0].split()) == C.MAX_REPLY


def test_reply_time_matches_token_cycles():
    p = person(5, [W[0], T.YOU])
    _, n, secs = run(p)
    ids = 5
    want = sum(E.token_cycles(p.p, i) for i in range(ids)) + E.token_cycles(p.p, ids)
    assert secs == pytest.approx(want / E.CLK_HZ)


def test_reply_without_emoji_gain_never_emits_emoji():
    p = person(5, [W[0], T.HIM, EM, T.YOU])
    texts, _, _ = run(p, emoji=0.0)
    assert p.bot_emoji == 0
    assert not any(C.EMOJI[t] for t in p.hist[5:])


class ChainModel:
    def __init__(self, nxt):
        self.nxt = nxt

    def reset(self):
        pass

    def forward(self, t, pos, base=0):
        lg = np.zeros(T.VOCAB, np.int64)
        lg[self.nxt.get(t, T.YOU)] = 1 << 12
        return lg


class WordTok(FakeTok):
    def __init__(self, words):
        self.words = words

    def decode(self, toks):
        return " ".join(self.words.get(t, f"w{t}") for t in toks)


def chain_person(nxt):
    p = person(5, [T.YOU])
    p.m = ChainModel(nxt)
    p.mmi = False
    p.w_q = np.zeros(T.VOCAB, np.int64)
    p.w_base = p.w_q.copy()
    p.p["score_w"] = p.w_q
    p.common = set()
    p.reset()
    return p


def best(p, tok=None, text="hello there", n_best=2, budget=60):
    return C.reply_best(p, tok or FakeTok(), text, 1.0, 0.0, C.make_rng(1), emoji=0.0, n_best=n_best, budget=budget)


def test_reply_best_returns_the_chain_and_records_state():
    p = chain_person({T.HIM: W[0], W[0]: W[1], W[1]: T.HIM})
    p.m.nxt[W[1]] = T.YOU
    texts, used, win = best(p)
    assert texts == [f"w{W[0]} w{W[1]}"]
    assert 0 <= win < 2 + C.BEST_EXTRA
    assert used >= 3
    assert p.said == [[W[0], W[1]]]
    assert p.bot_msgs == 1 and p.bot_emoji == 0
    assert p.hist[:2] == [T.BOS, T.YOU]
    assert p.hist[-3:] == [T.HIM, W[0], W[1]]
    assert len(p.last_cands) >= 2 and all(ok for _, ok in p.last_cands)


def test_reply_best_stops_candidates_at_the_budget():
    p = chain_person({T.HIM: W[0], W[0]: W[0]})
    texts, used, win = best(p, n_best=3, budget=10)
    assert used <= 10 + C.BEST_EXTRA_BUDGET + 1
    assert not p.last_cands[0][1]


def test_reply_best_keeps_the_first_candidate_when_none_is_good():
    p = chain_person({T.HIM: W[0], W[0]: W[1], W[1]: T.YOU})
    texts, used, win = best(p, WordTok({W[1]: "kys"}))
    assert win == 0
    assert p.said == [[W[0], W[1]]]


def test_reply_best_without_a_cap_hits_max_reply():
    p = chain_person({T.HIM: W[0], W[0]: W[0]})
    texts, used, win = best(p, n_best=1, budget=1000)
    assert len(texts[0].split()) == C.MAX_REPLY
