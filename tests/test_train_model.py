import struct
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import train as T

CFG = {"dim": 16, "hidden": 24, "layers": 2, "heads": 4, "kv_heads": 2, "vocab": 32, "seq": 8, "dropout": 0.0}


def tiny_net():
    torch.manual_seed(0)
    return T.Net(CFG)


def test_sizes_are_consistent():
    for dim, hidden, layers, heads, kv in T.SIZES.values():
        assert dim % heads == 0
        assert heads % kv == 0
        assert layers > 0 and hidden > 0


def test_rope_tables_shape_and_first_position():
    cos, sin = T.rope_tables(8, 5)
    assert cos.shape == (5, 4) and sin.shape == (5, 4)
    assert torch.all(cos[0] == 1) and torch.all(sin[0] == 0)


def test_apply_rope_position_zero_is_identity_and_preserves_norm():
    cos, sin = T.rope_tables(8, 5)
    x = torch.randn(2, 5, 3, 8)
    y = T.apply_rope(x, cos, sin)
    assert torch.allclose(y[:, 0], x[:, 0])
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_unit_rms_normalises():
    x = torch.randn(3, 16) * 5
    r = T.unit_rms(x).pow(2).mean(-1)
    assert torch.allclose(r, torch.ones(3), atol=1e-3)


def test_forward_shape_and_causality():
    net = tiny_net().eval()
    idx = torch.randint(0, CFG["vocab"], (2, CFG["seq"]))
    out = net(idx)
    assert out.shape == (2, CFG["seq"], CFG["vocab"])
    other = idx.clone()
    other[:, -1] = (other[:, -1] + 1) % CFG["vocab"]
    assert torch.allclose(net(other)[:, :-1], out[:, :-1], atol=1e-5)


def test_export_layout(tmp_path):
    net = tiny_net()
    path = tmp_path / "m.bin"
    T.export(net, path)
    raw = path.read_bytes()
    head = struct.unpack("7i", raw[:28])
    assert head == (16, 24, 2, 4, 2, 32, 8)
    hs = CFG["dim"] // CFG["heads"]
    d, h, kvd = CFG["dim"], CFG["hidden"], CFG["kv_heads"] * hs
    per_layer = d + d * d + 2 * kvd * d + d * d + d + 3 * h * d
    count = CFG["vocab"] * d + per_layer * CFG["layers"] + d + CFG["seq"] * hs
    assert len(raw) == 28 + 4 * count
    emb = np.frombuffer(raw[28:28 + 4 * CFG["vocab"] * d], np.float32).reshape(CFG["vocab"], d)
    assert np.array_equal(emb, net.emb.weight.detach().numpy())
    tail = np.frombuffer(raw[-4 * CFG["seq"] * hs:], np.float32)
    assert not tail.any()


def test_set_qat_propagates():
    net = tiny_net()
    net.set_qat(True)
    assert net.qat and all(b.qat for b in net.blocks)
    net.set_qat(False)
    assert not net.qat and not any(b.qat for b in net.blocks)


def test_stream_opens_token_file(tmp_path):
    p = tmp_path / "t.bin"
    np.arange(100, dtype=np.uint16).tofile(p)
    s = T.Stream(p, 8, 0)
    assert s.d.shape == (100,) and s.seq == 8
