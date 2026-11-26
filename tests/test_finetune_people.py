import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import finetune as F
import tok as T


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)


def session(n, first=10):
    ids = np.arange(first, first + n, dtype=np.int64)
    lab = np.arange(n) % 2 == 1
    return ids, lab


def test_weights_follow_session_length():
    p = F.People([session(10), session(30)], 8, 0)
    assert p.p.tolist() == [0.25, 0.75]


def test_batch_shapes_and_shift():
    p = F.People([session(40)], 8, 0)
    x, y = p.batch(5)
    assert x.shape == y.shape == (5, 8)
    for r in range(5):
        for t in range(8):
            if y[r, t] >= 0:
                assert y[r, t] == x[r, t] + 1


def test_unlabelled_targets_are_masked():
    p = F.People([session(40)], 8, 1)
    x, y = p.batch(20)
    assert ((y == -1) | (y % 2 == 1)).all()
    assert (y == -1).any() and (y >= 0).any()


def test_short_session_is_padded():
    p = F.People([session(4)], 8, 0)
    x, y = p.batch(3)
    assert (x[:, :3] == torch.tensor([10, 11, 12])).all()
    assert (x[:, 3:] == T.PAD).all()
    assert (y[:, 3:] == -1).all()


def test_same_seed_gives_same_batches():
    a = F.People([session(40), session(25)], 8, 5).batch(6)
    b = F.People([session(40), session(25)], 8, 5).batch(6)
    assert all(torch.equal(u, v) for u, v in zip(a, b))


def test_distillation_targets_follow_the_labelled_tokens():
    s = session(40)
    K = 3
    ki = np.tile(np.arange(40, dtype=np.int64)[:, None], (1, K))
    kp = np.tile(np.linspace(0, 1, 40, dtype=np.float32)[:, None], (1, K))
    p = F.People([s], 8, 2, kd=[(ki, kp)])
    x, y, ti, tp = p.batch(4)
    assert ti.shape == tp.shape == (4, 8, K)
    for r in range(4):
        for t in range(8):
            if x[r, t] != T.PAD:
                assert ti[r, t, 0] == x[r, t] - 10 + 1
                assert tp[r, t, 0] == pytest.approx((x[r, t] - 10 + 1) / 39)


class Uniform(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return (self.w + torch.zeros(x.shape[0], x.shape[1], self.vocab))


def test_eval_people_counts_labelled_targets_and_restores_train_mode():
    net = Uniform(64)
    net.train()
    nll, n = F.eval_people(net, [session(10)], 4)
    assert n == 5
    assert nll == pytest.approx(float(np.log(64)), rel=1e-5)
    assert net.training


def test_eval_people_covers_every_segment_of_a_long_session():
    net = Uniform(64)
    _, n = F.eval_people(net, [session(21), session(9, first=40)], 4)
    assert n == 10 + 4


def test_eval_people_without_labelled_tokens_returns_zero():
    ids = np.arange(10, 20, dtype=np.int64)
    nll, n = F.eval_people(Uniform(64), [(ids, np.zeros(10, dtype=bool))], 4)
    assert (nll, n) == (0.0, 0)
