import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import chat as C
import tok as T


class StubModel:
    def __init__(self):
        self.resets = 0
        self.fed = []

    def reset(self):
        self.resets += 1

    def forward(self, t, pos, base=0):
        self.fed.append((t, pos, base))
        return t


def person(**kw):
    p = object.__new__(C.Person)
    p.key = "pr"
    p.m = StubModel()
    p.emoji_corr, p.emoji_rate, p.emoji_cold = kw.get("corr", 1.0), kw.get("rate"), kw.get("cold")
    p.reset()
    return p


def test_reset_clears_conversation_state():
    p = person()
    p.feed([5, 6, 7])
    p.said.append([1])
    p.bot_msgs, p.bot_emoji = 3, 1
    p.reset()
    assert (p.pos, p.hist, p.said, p.bot_msgs, p.bot_emoji, p.bot_mass, p.base, p.last) == (0, [], [], 0, 0, [], 0, None)
    assert p.m.resets == 2


def test_feed_tracks_history_and_positions():
    p = person()
    p.window(9)
    p.feed([4, 5])
    assert p.hist == [4, 5] and p.pos == 2 and p.last == 5
    assert p.m.fed == [(4, 0, 9), (5, 1, 9)]


def test_rewind_drops_later_history():
    p = person()
    p.feed([4, 5, 6, 7])
    p.rewind(2)
    assert p.hist == [4, 5] and p.pos == 2


def test_skip_extends_history_without_running_the_model():
    p = person()
    p.skip(np.array([8, 9]))
    assert p.hist == [8, 9] and p.pos == 2 and p.m.fed == []


def test_make_room_keeps_history_when_it_fits():
    p = person()
    p.feed([T.BOS, T.YOU, 10])
    p.make_room(10)
    assert p.hist == [T.BOS, T.YOU, 10] and p.m.resets == 1


def test_make_room_restarts_from_recent_turn_marker():
    p = person()
    p.hist = [T.BOS, T.YOU, 10, T.HIM, 11, T.YOU, 12]
    p.pos = C.NULL_BASE
    p.make_room(1)
    assert p.hist == [T.BOS, T.YOU, 10, T.HIM, 11, T.YOU, 12]
    p.hist = [T.BOS] + [50] * (C.KEEP + 20) + [T.YOU, 12]
    p.pos = C.NULL_BASE
    p.make_room(1)
    assert p.hist == [T.BOS, T.YOU, 12] and p.pos == 3


def test_make_room_without_marker_keeps_only_bos():
    p = person()
    p.hist = [50] * 10
    p.pos = C.NULL_BASE
    p.make_room(1)
    assert p.hist == [T.BOS] and p.pos == 1


def test_candidate_selection_picks_best_average_score():
    p = person()
    p.best_reset()
    assert p.cand_end() == 0
    for s in (4, 4):
        p.score(s)
    assert p.cand_end() == 1
    for s in (9, 1):
        p.score(s)
    assert p.cand_end() == 2
    for s in (7, 7, 7):
        p.score(s)
    assert p.cand_end() == 3
    assert p.best == (21, 3, 3)


def test_candidate_end_can_discard_a_candidate():
    p = person()
    p.best_reset()
    p.score(100)
    assert p.cand_end(keep=False) == 0
    assert p.best is None and p.cand == 1 and p.acc == [0, 0]


def test_emoji_gain_without_a_rate_is_the_corrected_scale():
    p = person(corr=2.0)
    assert p.emoji_gain(0.5) == 2.0 * C.emoji_scale(0.5)
    assert p.emoji_gain(0) == 0


def test_emoji_gain_is_minimal_when_quota_is_met():
    p = person(rate=0.2)
    p.bot_msgs, p.bot_emoji = 100, 100
    assert p.emoji_gain(0.5) == C.EMOJI_GAIN_MIN


def test_emoji_gain_grows_when_behind_and_stays_in_bounds():
    p = person(rate=0.2, cold=1.0)
    p.bot_msgs, p.bot_emoji = 10, 0
    low = p.emoji_gain(0.5)
    p.bot_mass = [0.01] * 8
    high = p.emoji_gain(0.5)
    assert C.EMOJI_GAIN_MIN <= low <= C.EMOJI_GAIN_MAX
    assert C.EMOJI_GAIN_MIN <= high <= C.EMOJI_GAIN_MAX
    assert high > low
    assert math.isfinite(high)
