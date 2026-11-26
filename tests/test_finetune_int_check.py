import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import board as B
import finetune as F
import train as T


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = {"dim": 256, "hidden": 256, "layers": 2, "heads": 8, "kv_heads": 2, "vocab": 3072, "seq": 16, "dropout": 0.0}
    net = T.Net(cfg)
    with torch.no_grad():
        for prm in net.parameters():
            if prm.dim() == 2:
                prm.mul_(8.0)
    net.eval()
    x = torch.randint(0, cfg["vocab"], (4, cfg["seq"]))
    fmt, _ = B.calibrate(net, x)
    p = B.quantize(net, fmt)
    B.kv8_calibrate(p, [[int(t) for t in x[0]]])
    return p, net, x[0].numpy().astype(np.int64)


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)


def test_int_check_counts_labelled_targets_only(tiny):
    p, net, ids = tiny
    lab = np.zeros(len(ids), dtype=bool)
    lab[[3, 4, 9]] = True
    r = F.int_check(p, net, [(ids, lab)])
    assert r["tokens"] == 3


def test_int_check_reports_close_integer_and_float_models(tiny):
    p, net, ids = tiny
    lab = np.ones(len(ids), dtype=bool)
    r = F.int_check(p, net, [(ids, lab)])
    assert r["tokens"] == len(ids) - 1
    assert 0.0 <= r["top1_agree"] <= 1.0
    assert r["nll_int"] > 0 and r["nll_float"] > 0
    assert abs(r["nll_int"] - r["nll_float"]) < 1.0


def test_int_check_stops_after_max_tokens(tiny):
    p, net, ids = tiny
    lab = np.ones(len(ids), dtype=bool)
    r = F.int_check(p, net, [(ids, lab)] * 5, max_tokens=20)
    assert r["tokens"] == 2 * (len(ids) - 1)


def test_int_check_truncates_to_the_model_context(tiny):
    p, net, ids = tiny
    long_ids = np.concatenate([ids, ids])
    r = F.int_check(p, net, [(long_ids, np.ones(len(long_ids), dtype=bool))])
    assert r["tokens"] == p["cfg"].seq_len - 1
