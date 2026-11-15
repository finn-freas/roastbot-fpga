import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import board as B


def test_cfg_head_and_kv_sizes():
    c = B.Cfg(dim=64, hidden=128, n_layers=2, n_heads=8, n_kv_heads=2, vocab_size=100, seq_len=16)
    assert c.head_size == 8
    assert c.kv_dim == 16


def test_requant_reconstructs_scale():
    s = np.array([0.5, 0.013, 1.7, 0.0004])
    m, sh = B._requant(s, 8, 10)
    assert np.all(m < (1 << B.M_BITS))
    assert np.all(m >= (1 << (B.M_BITS - 2)))
    assert np.allclose(m / 2.0 ** sh, s * 4.0, rtol=2e-4)


def test_requant_rejects_shift_underflow():
    with pytest.raises(AssertionError):
        B._requant(np.array([1e9]), 0, 0)


def test_quant8_matrix_range_and_row_scale():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(5, 12))
    out = B.quant8_matrix(w, 4, 6)
    assert out["w"].shape == w.shape
    assert np.abs(out["w"]).max(axis=1).tolist() == [127] * 5
    assert out["m"].shape == out["s"].shape == (5,)
    s = np.abs(w).max(axis=1) / 127.0
    assert np.allclose(out["m"] / 2.0 ** out["s"], s * 2.0 ** (6 - 4), rtol=2e-4)


def test_quant4_matrix_layout_and_bounds():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(4, 3 * B.G))
    out = B.quant4_matrix(w, 15, 8)
    assert out["q4"].shape == w.shape
    assert out["g8"].shape == (4, 3)
    assert np.abs(out["q4"]).max() <= B.Q4
    assert out["g8"].min() >= 1 and out["g8"].max() <= 255
    assert np.array_equal(out["w"], (out["q4"].reshape(4, 3, B.G) * out["g8"][..., None]).reshape(4, -1))
    assert (out["g8"].max(axis=1) == 255).all()


def test_quant4_matrix_extra_scales_weights():
    rng = np.random.default_rng(2)
    w = rng.normal(size=(2, B.G))
    a = B.quant4_matrix(w, 15, 8, 2.0)
    b = B.quant4_matrix(w * 2.0, 15, 8)
    assert np.array_equal(a["q4"], b["q4"])
    assert np.array_equal(a["g8"], b["g8"])


def test_quant4_dequantises_close_to_input():
    rng = np.random.default_rng(3)
    w = rng.normal(size=(3, 2 * B.G))
    out = B.quant4_matrix(w, 15, 15)
    scale = out["m"] / 2.0 ** out["s"]
    approx = out["w"] * scale[:, None]
    assert np.abs(approx - w).max() < np.abs(w).max() / B.Q4


def test_fq4_matches_quant4_matrix():
    rng = np.random.default_rng(4)
    w = rng.normal(size=(3, 2 * B.G))
    out = B.quant4_matrix(w, 15, 15)
    amax_r = np.abs(w.reshape(3, 2, B.G)).max(-1).max(-1, keepdims=True)
    fq = B.fq4(torch.tensor(w)).numpy().reshape(3, 2, B.G)
    step = (amax_r * out["g8"] / (255.0 * B.Q4))[..., None]
    assert np.allclose(fq, out["q4"].reshape(3, 2, B.G) * step)


def test_fq8_matches_quant8_matrix():
    rng = np.random.default_rng(5)
    w = rng.normal(size=(4, 10))
    out = B.quant8_matrix(w, 0, 0)
    s = np.abs(w).max(axis=1, keepdims=True) / 127.0
    assert np.allclose(B.fq8(torch.tensor(w)).numpy(), out["w"] * s)


def test_fq_straight_through_gradient():
    w = torch.randn(2, B.G, dtype=torch.float64, requires_grad=True)
    B.fq4(w).sum().backward()
    assert torch.equal(w.grad, torch.ones_like(w))
    v = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    B.fq8(v).sum().backward()
    assert torch.equal(v.grad, torch.ones_like(v))


def test_memory_map_totals():
    rng = np.random.default_rng(6)
    cfg = B.Cfg(dim=B.G, hidden=B.G, n_layers=2, n_heads=2, n_kv_heads=1, vocab_size=10, seq_len=4)
    layer = {k: B.quant4_matrix(rng.normal(size=(3, B.G)), 15, 8) for k in ("a", "b")}
    p = {"cfg": cfg, "layers": [layer, layer], "table": np.zeros((10, B.G), np.int64)}
    m = B.memory_map(p)
    assert m["weights_4bit"] == 4 * 3 * B.G // 2
    assert m["group_scales"] == 4 * 3
    assert m["row_requant"] == 4 * 3 * 3
    assert m["table_8bit"] == 10 * B.G + 10 * 6
    assert m["kv_cache"] == 2 * 2 * 4 * cfg.kv_dim * 2
    assert m["streamed_per_token"] == m["weights_4bit"] + m["group_scales"] + m["row_requant"] + m["table_8bit"]
    assert m["total"] == m["streamed_per_token"] + m["kv_cache"]
    p["kv8"] = True
    assert B.memory_map(p)["kv_cache"] == m["kv_cache"] // 2
