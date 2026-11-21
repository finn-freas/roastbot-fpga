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


def test_ping_retries_after_a_bad_reply():
    lk = FakeLink([b"xyz", b"K\x03\x02"])
    assert lk.ping() == {"version": 3, "sdram_ready": False, "cal_err": True}
    assert lk.sent == b"PP"


def test_ping_with_one_try_does_not_retry():
    lk = FakeLink([b"xyz", b"K\x03\x02"])
    with pytest.raises(IOError, match="bad ping reply"):
        lk.ping(tries=1)
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


def test_a_file_that_is_not_a_serial_port_gives_clear_error(tmp_path):
    path = tmp_path / "plain"
    path.write_bytes(b"")
    with pytest.raises(ConnectionError, match="is not a serial port"):
        link.Link(str(path))
    with pytest.raises(SystemExit) as e:
        link.connect(str(path))
    assert "is not a serial port" in str(e.value)


def test_checksum_command_reads_a_word():
    lk = FakeLink([struct.pack("<I", 0xDEADBEEF)])
    assert lk.checksum(3, 4) == 0xDEADBEEF
    assert lk.sent == link.Link._hdr(b"C", 3, 4)


def test_write_bytes_reports_the_word_that_was_not_acknowledged():
    lk = FakeLink([b"w", b"?"])
    data = bytes(8 * (link.CHUNK + 1))
    with pytest.raises(IOError, match="not acknowledged"):
        lk.write_bytes(0, data)


def test_write_bytes_reports_progress():
    lk = FakeLink([b"w"])
    seen = []
    lk.write_bytes(0, bytes(16), progress=lambda done, total: seen.append((done, total)))
    assert seen == [(2, 2)]


def test_settings_requires_ack():
    with pytest.raises(IOError):
        FakeLink([b"?"]).settings(1, 2, 3)


def test_post_settings_bad_ack_raises_when_collected():
    lk = FakeLink([b"?"])
    r = lk.post_settings(1, 2, 3)
    with pytest.raises(IOError):
        r.value


def test_post_settings_clamps_gain():
    lk = FakeLink([b"g"])
    lk.post_settings(1, 2, 0x12345).value
    assert lk.sent == b"G" + struct.pack("<HHH", 1, 2, 0xFFFF)


def test_post_is_refused_while_a_run_is_going():
    lk = FakeLink([b"e"])
    lk.start_run(3, 4)
    with pytest.raises(IOError, match="still going"):
        lk.post_jump(1)


def test_start_run_is_refused_while_another_run_is_going():
    lk = FakeLink([b"e"])
    lk.start_run(3, 4)
    with pytest.raises(IOError, match="still going"):
        lk.start_run(3, 4)


def test_start_run_collects_pending_long_replies_first():
    lk = FakeLink([step_reply(7), b"e"])
    r = lk.post_f(1, 3)
    lk.start_run(2, 4)
    assert r.done and r.value[0] == 7
    assert lk.sent == struct.pack("<cBHH", b"F", 1, 3, 0) + struct.pack("<cBHH", b"F", 5, 2, 4)


def test_send_collects_pending_replies_before_writing():
    lk = FakeLink([b"j"])
    r = lk.post_jump(1)
    lk.send(b"P")
    assert r.done
    assert lk.sent == b"J\x01\x00P"


def test_sync_reports_a_run_that_went_on_past_the_host():
    lk = FakeLink([step_reply(1), step_reply(2), b"e"])
    lk.start_run(3, 40)
    lk.run.stop = lambda: [1]
    with pytest.raises(IOError, match="went on past"):
        lk.sync()


def test_read_image_adds_the_sdram_suffix(tmp_path):
    (tmp_path / "pr_board.sdram.bin").write_bytes(b"\x01\x02\x03")
    assert link.read_image(["link.py", "load", str(tmp_path / "pr_board")]) == b"\x01\x02\x03"


def test_read_image_replaces_an_existing_suffix(tmp_path):
    (tmp_path / "pr_board.sdram.bin").write_bytes(b"\x09")
    assert link.read_image(["link.py", "load", str(tmp_path / "pr_board.pt")]) == b"\x09"


def test_read_image_reports_a_missing_image(tmp_path):
    with pytest.raises(SystemExit) as e:
        link.read_image(["link.py", "load", str(tmp_path / "none")])
    assert "pack.py" in str(e.value)


def test_read_image_needs_a_model_argument():
    with pytest.raises(SystemExit) as e:
        link.read_image(["link.py", "load"])
    assert str(e.value) == link.USAGE


def xorshift32(x):
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    return x


@pytest.mark.parametrize("n", [0, 1, 2, 7, 100])
def test_xs_matrix_jumps_the_generator_n_steps(n):
    seed = 0x2545F491
    expect = seed
    for _ in range(n):
        expect = xorshift32(expect)
    assert link._xs_apply(link._xs_matrix(n), seed) == expect


def test_xs_step_matches_draws_per_step_iterations():
    seed = 0xDEADBEEF
    expect = seed
    for _ in range(link.DRAWS_PER_STEP):
        expect = xorshift32(expect)
    assert link._xs_apply(link.XS_STEP, seed) == expect


def test_xs_apply_of_zero_is_zero():
    assert link._xs_apply(link.XS_STEP, 0) == 0


def test_post_seed_records_the_masked_state_only_once_acknowledged():
    lk = FakeLink([b"s"])
    r = lk.post_seed(2 ** 32 + 7)
    assert lk.rng is None
    r.value
    assert lk.rng == 7
    assert lk.sent == b"S" + struct.pack("<I", 7)


def test_post_seed_leaves_the_state_alone_on_a_bad_ack():
    lk = FakeLink([b"?"])
    r = lk.post_seed(7)
    with pytest.raises(IOError):
        r.value
    assert lk.rng is None


def test_post_best_runs_the_check_on_the_decoded_reply():
    seen = []
    lk = FakeLink([b"y" + bytes([9]) + struct.pack("<i", -3) + bytes([4])])
    v = lk.post_best(clear=True, keep=True, check=seen.append).value
    assert v == (9, -3, 4)
    assert seen == [(9, -3, 4)]
    assert lk.sent == b"Y" + bytes([5])


def test_post_best_check_may_reject_a_reply():
    def check(v):
        raise ValueError(v)

    lk = FakeLink([b"y" + bytes([1]) + struct.pack("<i", 2) + bytes([3])])
    with pytest.raises(ValueError):
        lk.post_best(check=check).value


def test_post_window_and_post_jump_send_their_frames():
    lk = FakeLink([b"x", b"j"])
    lk.post_window(0x0102)
    lk.post_jump(0x0304)
    lk.sync()
    assert lk.sent == b"X\x02\x01" + b"J\x04\x03"


def test_reply_value_collects_earlier_replies_in_order():
    lk = FakeLink([b"x", b"z"])
    first = lk.post_window(1)
    second = lk.post_new_conversation()
    assert second.value is None
    assert first.done and second.done
