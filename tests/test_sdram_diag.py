import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import sdram_diag


def test_defaults():
    assert sdram_diag.parse_args([]) == (20, 65536)


def test_explicit_values():
    assert sdram_diag.parse_args(["0"]) == (0, 65536)
    assert sdram_diag.parse_args(["5", "4096"]) == (5, 4096)


@pytest.mark.parametrize("argv", [["x"], ["1", "y"], ["-1"], ["1", "100"], ["1", "4096", "3"]])
def test_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit) as e:
        sdram_diag.parse_args(argv)
    assert "usage" in str(e.value)


class FakeLink:
    def __init__(self, flip=False, bad_checks=0):
        self.mem = {}
        self.flip = flip
        self.bad_checks = bad_checks
        self.sent = []

    def ping(self):
        return "pong"

    def send(self, b):
        self.sent.append(b)

    def recv(self, n):
        return bytes([0, 0x35, 0x00])

    def write_bytes(self, addr, data):
        self.addr, self.data = addr, data

    def checksum(self, addr, n):
        if self.bad_checks:
            self.bad_checks -= 1
            return -1
        return sdram_diag.checksum(self.data)

    def read(self, addr, n):
        import struct
        off = (addr - self.addr) * 8
        words = list(struct.unpack(f"<{n}Q", self.data[off:off + n * 8]))
        if self.flip:
            words[0] ^= 1
        return words


def run_main(monkeypatch, capsys, argv, link):
    monkeypatch.setattr(sdram_diag, "connect", lambda: link)
    monkeypatch.setattr(sys, "argv", ["sdram_diag.py"] + argv)
    sdram_diag.main()
    return capsys.readouterr().out


def test_main_reports_calibration_and_a_clean_run(monkeypatch, capsys):
    out = run_main(monkeypatch, capsys, ["3", "4096"], FakeLink())
    assert "ping pong" in out
    assert "calibration: passing read latencies (first = CAS - 1) XX..  pick 5" in out
    assert "checksum fails 0/3, word errors 0/4096" in out


def test_main_counts_failed_checksums_and_wrong_words(monkeypatch, capsys):
    out = run_main(monkeypatch, capsys, ["4", "4096"], FakeLink(flip=True, bad_checks=2))
    assert "checksum fails 2/4, word errors 16/4096" in out


def test_main_with_zero_rounds_only_calibrates(monkeypatch, capsys):
    link = FakeLink()
    out = run_main(monkeypatch, capsys, ["0"], link)
    assert "calibration:" in out and "checksum fails" not in out
    assert link.sent == [b"D"] and not hasattr(link, "data")
