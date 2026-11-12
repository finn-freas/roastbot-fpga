import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import link


class FakeLink(link.Link):
    def __init__(self, replies=()):
        self.sent = bytearray()
        self.replies = bytearray(b"".join(replies))
        self.pending = []
        self.rng = None
        self.run = None

    def drain(self, quiet=0.2):
        pass

    def _send(self, data):
        self.sent += data

    def recv(self, n, timeout=5.0):
        if len(self.replies) < n:
            raise TimeoutError("short")
        out = bytes(self.replies[:n])
        del self.replies[:n]
        return out


def test_header_layout():
    assert link.Link._hdr(b"W", 0x010203, 0x0A0B0C) == b"W\x03\x02\x01\x0c\x0b\x0a"


def test_checksum_pads_to_eight_bytes():
    assert link.checksum(b"\x01\x00\x00\x00") == 1
    assert link.checksum(bytes(range(16))) == link.checksum(bytes(range(16)) + b"\x00" * 8)


def test_checksum_wraps_to_32_bits():
    data = struct.pack("<2I", 0xFFFFFFFF, 2)
    assert link.checksum(data) == 1


def test_ping_decodes_flags():
    lk = FakeLink([b"K\x07\x01"])
    assert lk.ping() == {"version": 7, "sdram_ready": True, "cal_err": False}
    assert lk.sent == b"P"


def test_ping_bad_reply_raises():
    lk = FakeLink([b"xyz", b"xyz"])
    with pytest.raises(IOError):
        lk.ping()


def test_write_frames_words():
    lk = FakeLink([b"w"])
    lk.write(5, [1, 2])
    assert lk.sent == b"W\x05\x00\x00\x02\x00\x00" + struct.pack("<2Q", 1, 2)


def test_write_requires_ack():
    with pytest.raises(IOError):
        FakeLink([b"?"]).write(0, [1])


def test_write_bytes_chunks_and_pads():
    n_words = link.CHUNK + 3
    data = bytes(8 * n_words - 5)
    lk = FakeLink([b"w", b"w"])
    seen = []
    lk.write_bytes(0, data, lambda done, n: seen.append((done, n)))
    assert seen == [(link.CHUNK, n_words), (n_words, n_words)]
    assert len(lk.sent) == 2 * 7 + 8 * n_words


def test_read_unpacks_words():
    lk = FakeLink([struct.pack("<2Q", 9, 2 ** 63)])
    assert lk.read(1, 2) == [9, 2 ** 63]
    assert lk.sent == b"R\x01\x00\x00\x02\x00\x00"


def test_seed_masks_to_32_bits():
    lk = FakeLink([b"s"])
    lk.seed(2 ** 32 + 5)
    assert lk.sent == b"S" + struct.pack("<I", 5)


def test_settings_clamps_gain():
    lk = FakeLink([b"g"])
    lk.settings(1, 2, 10 ** 6)
    assert lk.sent == b"G" + struct.pack("<HHH", 1, 2, 0xFFFF)


def test_feed_and_step():
    lk = FakeLink([b"a" + struct.pack("<H", 42)])
    assert lk.feed(300) == 42
    assert lk.sent == struct.pack("<cBHH", b"F", 0, 300, 0)
    reply = b"f" + struct.pack("<H", 7) + (1).to_bytes(6, "little") + (2).to_bytes(6, "little") + (-3).to_bytes(3, "little", signed=True)
    lk = FakeLink([reply])
    assert lk.step(1) == (7, 1, 2, -3)
    assert lk.sent == struct.pack("<cBHH", b"F", 1, 1, 0)


def test_score_after_forces_the_token():
    reply = b"f" + struct.pack("<H", 9) + (1).to_bytes(6, "little") + (2).to_bytes(6, "little") + (-5).to_bytes(3, "little", signed=True)
    lk = FakeLink([reply])
    assert lk.score_after(4, 9) == (9, 1, 2, -5)
    assert lk.sent == struct.pack("<cBHH", b"F", 2, 4, 9)


def test_window_frame():
    lk = FakeLink([b"x"])
    lk.window(424)
    assert lk.sent == b"X" + struct.pack("<H", 424)


def test_best_flags_and_reply():
    lk = FakeLink([b"y" + bytes([3]) + struct.pack("<i", -9) + bytes([4])])
    assert lk.best(clear=True, end=True) == (3, -9, 4)
    assert lk.sent == b"Y\x03"


def test_feed_reports_a_position_mismatch():
    lk = FakeLink([b"a" + struct.pack("<H", 5)])
    with pytest.raises(IOError, match="position 5"):
        lk.post_feed(300, expect=6).value


def test_feed_bad_reply_raises():
    with pytest.raises(IOError, match="feed"):
        FakeLink([b"zzz"]).feed(1)


def test_best_bad_reply_raises():
    with pytest.raises(IOError, match="best"):
        FakeLink([b"z" + bytes(6)]).best()


def test_jump_and_new_conversation_frames():
    lk = FakeLink([b"j", b"z"])
    lk.jump(300)
    lk.new_conversation()
    assert lk.sent == b"J" + struct.pack("<H", 300) + b"Z"


def test_ack_mismatch_raises():
    with pytest.raises(IOError):
        FakeLink([b"x"]).init_engine()


def test_emoji_table_size():
    lk = FakeLink([b"e"])
    lk.emoji_table([1] * 128, [-1] * 128)
    assert len(lk.sent) == 1 + 128 * 3


def run_main(monkeypatch, *argv):
    opened = []
    monkeypatch.setattr(link, "Link", lambda: opened.append(1))
    monkeypatch.setattr(sys, "argv", ["link.py", *argv])
    return opened


def test_main_unknown_command_exits_before_opening_the_port(monkeypatch):
    opened = run_main(monkeypatch, "bogus")
    with pytest.raises(SystemExit) as e:
        link.main()
    assert "usage" in str(e.value)
    assert not opened


def test_main_load_without_a_path_exits_with_usage(monkeypatch):
    opened = run_main(monkeypatch, "load")
    with pytest.raises(SystemExit) as e:
        link.main()
    assert "usage" in str(e.value)
    assert not opened


@pytest.mark.parametrize("argv", [("ping", "extra"), ("load", "m_board", "extra")])
def test_main_extra_arguments_exit_with_usage(monkeypatch, argv):
    opened = run_main(monkeypatch, *argv)
    with pytest.raises(SystemExit) as e:
        link.main()
    assert "usage" in str(e.value)
    assert not opened


def test_main_load_missing_image_names_the_file(monkeypatch, tmp_path):
    opened = run_main(monkeypatch, "load", str(tmp_path / "m_board"))
    with pytest.raises(SystemExit) as e:
        link.main()
    assert "m_board.sdram.bin" in str(e.value)
    assert not opened


def test_missing_port_gives_clear_error(tmp_path):
    with pytest.raises(ConnectionError, match="cannot open the serial port"):
        link.Link(port=str(tmp_path / "ttyMissing"))


class LoadStub:
    def __init__(self, remote):
        self.remote = remote

    def write_bytes(self, addr, data, progress=None):
        if progress:
            progress(len(data) // 8, len(data) // 8)

    def checksum(self, addr, n):
        return self.remote


def run_load(monkeypatch, tmp_path, remote):
    img = bytes(range(16))
    (tmp_path / "m_board.sdram.bin").write_bytes(img)
    stub = LoadStub(link.checksum(img) if remote is None else remote)
    monkeypatch.setattr(link, "Link", lambda: stub)
    monkeypatch.setattr(sys, "argv", ["link.py", "load", str(tmp_path / "m_board")])
    link.main()


def test_main_load_succeeds_when_the_checksums_match(monkeypatch, tmp_path, capsys):
    run_load(monkeypatch, tmp_path, None)
    assert "checksum OK" in capsys.readouterr().out


def test_main_load_exits_nonzero_on_checksum_mismatch(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        run_load(monkeypatch, tmp_path, 1)
    assert e.value.code == 1
    assert "MISMATCH" in capsys.readouterr().out


def step_reply(tok, score=0):
    return b"f" + struct.pack("<H", tok) + (1).to_bytes(6, "little") + (2).to_bytes(6, "little") + score.to_bytes(3, "little", signed=True)


def test_quick_commands_are_not_waited_for():
    lk = FakeLink([b"j", b"x", b"a" + struct.pack("<H", 6)])
    lk.post_jump(5)
    lk.post_window(0)
    assert lk.sent == b"J\x05\x00X\x00\x00"
    assert lk.feed(9) == 6
    assert lk.pending == []


def test_nothing_follows_an_unanswered_step():
    lk = FakeLink([step_reply(7), b"j"])
    r = lk.post_f(1, 3)
    lk.post_jump(1)
    assert r.done and r.value[0] == 7
    assert lk.sent == struct.pack("<cBHH", b"F", 1, 3, 0) + b"J\x01\x00"


def test_run_reads_steps_until_e():
    lk = FakeLink([step_reply(10), step_reply(2), b"e"])
    run = lk.start_run(3, 40)
    assert lk.sent == struct.pack("<cBHH", b"F", 5, 3, 40)
    assert run.next()[0] == 10
    assert run.next()[0] == 2
    assert run.next() is None and run.done and lk.run is None
    assert run.stop() == []


def test_run_stop_sends_zero_and_returns_extra_steps():
    lk = FakeLink([step_reply(10), step_reply(11), b"e"])
    run = lk.start_run(3, 40)
    assert [r[0] for r in run.stop()] == [10, 11]
    assert lk.sent.endswith(b"\x00") and lk.run is None


def test_rng_copy_follows_sampled_steps():
    x = 0x12345678
    lk = FakeLink([b"s", step_reply(5), step_reply(6)])
    lk.seed(x)
    lk.step(1)
    lk.step(5)
    for _ in range(2 * link.DRAWS_PER_STEP):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
    assert lk.rng == x


def test_connect_exits_with_message_when_port_missing(tmp_path):
    with pytest.raises(SystemExit) as e:
        link.connect(str(tmp_path / "missing"))
    assert "cannot open the serial port" in str(e.value)


@pytest.fixture
def pty_link():
    import os
    master, slave = os.openpty()
    lk = link.Link(port=os.ttyname(slave), baud=115200)
    yield lk, master
    os.close(lk.fd)
    os.close(master)
    os.close(slave)


def test_link_over_a_serial_device_sends_bytes_unchanged(pty_link):
    import os
    lk, master = pty_link
    payload = bytes(range(256))
    lk.send(payload)
    got = b""
    while len(got) < len(payload):
        got += os.read(master, 512)
    assert got == payload


def test_link_recv_returns_exactly_the_requested_bytes(pty_link):
    import os
    lk, master = pty_link
    os.write(master, b"abcdef")
    assert lk.recv(4) == b"abcd"
    assert lk.recv(2) == b"ef"


def test_link_recv_times_out_and_reports_the_bytes_seen(pty_link):
    import os
    lk, master = pty_link
    os.write(master, b"xy")
    with pytest.raises(TimeoutError, match="board sent 2 of 4 bytes: b'xy'"):
        lk.recv(4, timeout=0.3)


def test_link_drain_discards_pending_input(pty_link):
    import os
    lk, master = pty_link
    os.write(master, b"stale")
    lk.drain(quiet=0.1)
    os.write(master, b"ok")
    assert lk.recv(2) == b"ok"
