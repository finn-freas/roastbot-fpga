import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import chat as C
import tok as T


class Link:
    def __init__(self):
        self.calls = []
        self.rng = None
        self.best_reply = (0, 0, 0)

    def __getattr__(self, name):
        def call(*a, **kw):
            self.calls.append((name, a, kw))
            return None
        return call

    def feed(self, t):
        self.calls.append(("feed", (t,), {}))
        return len([c for c in self.calls if c[0] == "feed"])

    def step(self, t):
        self.calls.append(("step", (t,), {}))
        return (7, 0, 1, 2)

    def best(self, clear=False, end=False, keep=True):
        self.calls.append(("best", (), {"clear": clear, "end": end, "keep": keep}))
        return self.best_reply


def board(pipe=False):
    p = object.__new__(C.BoardPerson)
    p.link = Link()
    p.pipe = pipe
    p.key = "pr"

    class M:
        def reset(self):
            pass

    p.m = M()
    p.w_q = np.arange(200, dtype=np.int64)
    p.w_addr = 1000
    p.emoji_bias = np.zeros(T.VOCAB, np.int64)
    p.run, p.last_tok, p.j, p.cap, p.overrun, p.undone = None, None, 0, C.MAX_REPLY, False, 0
    C.Person.reset(p)
    return p


def names(p):
    return [c[0] for c in p.link.calls]


def test_feed_and_step_track_position_without_pipeline():
    p = board()
    p.feed([5, 6])
    assert p.pos == 2 and p.hist == [5, 6]
    assert p.step(9) == (7, 0, 1, 2)
    assert p.pos == 3 and p.hist == [5, 6, 9]


def test_rewind_and_skip_jump_the_board():
    p = board()
    p.feed([1, 2, 3])
    p.rewind(1)
    assert p.pos == 1 and p.hist == [1]
    p.skip([8, 9])
    assert p.pos == 3 and p.hist == [1, 8, 9]
    assert p.link.calls[-1] == ("jump", (3,), {})


def test_window_sets_base():
    p = board()
    p.window(40)
    assert p.base == 40 and p.link.calls[-1] == ("window", (40,), {})


def test_push_table_sends_runs_and_reinitialises():
    p = board()
    p.push_table(np.array([], dtype=np.int64))
    assert p.link.calls == []
    p.push_table(np.array([3, 4, 5, 50]))
    writes = [c for c in p.link.calls if c[0] == "write_bytes"]
    assert [w[1][0] for w in writes] == [1003, 1050]
    assert writes[0][1][1] == b"".join(int(v).to_bytes(8, "little") for v in (3, 4, 5))
    assert names(p)[-2:] == ["init_engine", "emoji_table"]


def test_push_table_encodes_negative_scores_as_64_bit():
    p = board()
    p.w_q[7] = -1
    p.push_table(np.array([7]))
    assert p.link.calls[0][1][1] == b"\xff" * 8


def test_cand_end_checks_the_board_best():
    p = board()
    p.best_reset()
    p.score(10)
    p.link.best_reply = (0, 10, 1)
    assert p.cand_end() == 0
    p.score(5)
    p.link.best_reply = (0, 99, 9)
    with pytest.raises(IOError, match="board best"):
        p.cand_end()


def test_make_room_resets_and_replays_the_recent_turns():
    p = board()
    p.feed([T.BOS, T.YOU, 5, T.HIM, 6])
    p.pos = C.NULL_BASE
    p.link.calls.clear()
    p.make_room(10)
    assert p.hist[0] == T.BOS and p.pos == len(p.hist)
    assert names(p)[0] == "new_conversation"
    p.link.calls.clear()
    p.make_room(0)
    assert p.link.calls == []


def test_reset_starts_a_new_conversation():
    p = board()
    p.reset()
    assert names(p) == ["new_conversation"]
    p.pipe = True
    p.link.calls.clear()
    p.reset()
    assert names(p) == ["post_new_conversation"]


def test_pipelined_feed_posts_without_waiting():
    p = board(pipe=True)
    p.feed([4, 5])
    assert p.pos == 2 and p.hist == [4, 5]
    assert [(c[0], c[1], c[2]) for c in p.link.calls] == [
        ("post_feed", (4,), {"expect": 1}), ("post_feed", (5,), {"expect": 2})]


class Run:
    def __init__(self, vals, ran=0):
        self.vals, self.ran, self.done = list(vals), ran, False

    def next(self):
        return self.vals.pop(0) if self.vals else None

    def stop(self):
        return [0] * self.ran


def test_pipelined_step_reuses_the_run_while_tokens_follow():
    p = board(pipe=True)
    runs = []

    def start(t, n):
        runs.append((t, n))
        return Run([(11, 0, 1, 2), (12, 0, 1, 2)])

    p.link.start_run = start
    assert p.step(3)[0] == 11
    assert p.step(11)[0] == 12
    assert runs == [(3, 80)]
    assert p.pos == 2 and p.hist == [3, 11]


def test_pipelined_step_fails_when_the_run_ends_early():
    p = board(pipe=True)
    p.link.start_run = lambda t, n: Run([])
    with pytest.raises(IOError, match="ended a run"):
        p.step(3)


def test_cancel_reseeds_when_steps_overran():
    p = board(pipe=True)
    p.pos = 5
    p.link.rng = 99
    p.run = Run([], ran=2)
    p._cancel()
    assert p.run is None and p.overrun and p.undone == 2
    assert [c[:2] for c in p.link.calls] == [("post_seed", (99,)), ("post_jump", (5,))]


def test_cancel_without_a_seed_is_an_error():
    p = board(pipe=True)
    p.run = Run([], ran=1)
    with pytest.raises(IOError, match="no seed"):
        p._cancel()


def test_cancel_is_a_noop_when_nothing_ran():
    p = board(pipe=True)
    p.run = Run([], ran=0)
    p._cancel()
    assert p.run is None and p.link.calls == [] and not p.overrun


def test_kept_candidate_with_undone_steps_is_rejected():
    p = board(pipe=True)
    p.best_reset()
    p.overrun = True
    with pytest.raises(IOError, match="undone"):
        p.cand_end(keep=True)


class Posted:
    def __init__(self, value):
        self.value = value


def test_settings_post_or_send_by_pipeline_mode():
    p = board()
    p.settings(1, 2, 3, None)
    assert p.link.calls == [("settings", (1, 2, 3), {})]
    p = board(pipe=True)
    p.settings(1, 2, 3, None)
    assert p.link.calls == [("post_settings", (1, 2, 3), {})]


def test_pipelined_rewind_posts_a_jump_and_restarts_the_run_count():
    p = board(pipe=True)
    p.feed([1, 2, 3])
    p.j = 4
    p.link.calls.clear()
    p.rewind(1)
    assert p.pos == 1 and p.hist == [1] and p.j == 0
    assert p.link.calls == [("post_jump", (1,), {})]


def test_pipelined_skip_posts_a_jump_to_the_new_position():
    p = board(pipe=True)
    p.feed([1])
    p.link.calls.clear()
    p.skip([8, 9])
    assert p.pos == 3 and p.hist == [1, 8, 9]
    assert p.link.calls == [("post_jump", (3,), {})]


def test_pipelined_window_posts_and_sets_base():
    p = board(pipe=True)
    p.window(7)
    assert p.base == 7
    assert p.link.calls == [("post_window", (7,), {})]


def test_pipelined_score_after_waits_for_the_value_and_advances():
    p = board(pipe=True)
    p.link.post_f = lambda op, t, nxt: Posted((nxt, op, t))
    assert p.score_after(5, 6) == (6, 2, 5)
    assert p.pos == 1 and p.hist == [5] and p.j == 1


def test_pipelined_score_seq_posts_every_pair_before_reading_any():
    p = board(pipe=True)
    posted = []
    p.link.post_f = lambda op, t, nxt: posted.append((t, nxt)) or Posted(nxt * 10)
    out = p.score_seq([(1, 2), (2, 3)])
    assert posted == [(1, 2), (2, 3)]
    assert p.pos == 2 and p.hist == [1, 2]
    assert list(out) == [20, 30]


def test_pipelined_best_reset_clears_the_board_best_and_the_overrun_flag():
    p = board(pipe=True)
    p.overrun = True
    p.best_reset()
    assert not p.overrun
    assert p.link.calls[-1] == ("post_best", (), {"clear": True})
