import glob
import os
import struct
import sys
import termios
import time
from pathlib import Path

PORT = next(iter(sorted(glob.glob("/dev/serial/by-id/*SIPEED*if01*"))), "/dev/ttyUSB1")
BAUD = 1_000_000
CHUNK = 4096
DRAWS_PER_STEP = 1 + 3072


def _xs_matrix(n):
    def step(x):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        return x
    cols = [step(1 << i) for i in range(32)]
    out = [1 << i for i in range(32)]
    while n:
        if n & 1:
            out = [_xs_apply(cols, c) for c in out]
        cols = [_xs_apply(cols, c) for c in cols]
        n >>= 1
    return out


def _xs_apply(cols, x):
    y = 0
    i = 0
    while x:
        if x & 1:
            y ^= cols[i]
        x >>= 1
        i += 1
    return y


XS_STEP = _xs_matrix(DRAWS_PER_STEP)


class Reply:
    __slots__ = ("link", "long", "parse", "done", "val")

    def __init__(self, link, long, parse):
        self.link, self.long, self.parse, self.done, self.val = link, long, parse, False, None

    @property
    def value(self):
        while not self.done:
            self.link._take()
        return self.val


class Link:
    def __init__(self, port=PORT, baud=BAUD):
        try:
            self.fd = os.open(port, os.O_RDWR | os.O_NOCTTY)
        except OSError as e:
            raise ConnectionError(f"cannot open the serial port {port}: {e.strerror}; "
                                  "check that the board is plugged in and no other program is using the port") from e
        a = termios.tcgetattr(self.fd)
        a[0] = a[1] = a[3] = 0
        a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        a[4] = a[5] = getattr(termios, f"B{baud}")
        a[6][termios.VMIN], a[6][termios.VTIME] = 0, 1
        termios.tcsetattr(self.fd, termios.TCSANOW, a)
        termios.tcflush(self.fd, termios.TCIOFLUSH)
        self.drain()
        self.pending = []
        self.rng = None
        self.run = None

    def drain(self, quiet=0.2):
        end = time.time() + quiet
        while time.time() < end:
            if os.read(self.fd, 256):
                end = time.time() + quiet

    def send(self, data):
        self.sync()
        self._send(data)

    def sync(self):
        if self.run and self.run.stop():
            raise IOError("a run of steps went on past what the host read, and nothing undid it")
        while self.pending:
            self._take()

    def _take(self):
        r = self.pending.pop(0)
        r.val = r.parse(self)
        r.done = True

    def post(self, cmd, parse):
        if self.run:
            raise IOError("a run of steps is still going on")
        while any(r.long for r in self.pending):
            self._take()
        r = Reply(self, cmd[:1] in (b"F", b"I", b"Y"), parse)
        self.pending.append(r)
        self._send(cmd)
        return r

    def start_run(self, tok, n):
        if self.run:
            raise IOError("a run of steps is still going on")
        while any(r.long for r in self.pending):
            self._take()
        self._send(struct.pack("<cBHH", b"F", 5, tok, n))
        self.run = Run(self)
        return self.run

    def _send(self, data):
        view = memoryview(data)
        while view:
            n = os.write(self.fd, view)
            view = view[n:]

    def recv(self, n, timeout=5.0):
        out = bytearray()
        end = time.time() + timeout
        while len(out) < n:
            chunk = os.read(self.fd, n - len(out))
            if chunk:
                out += chunk
                end = time.time() + timeout
            elif time.time() > end:
                raise TimeoutError(f"board sent {len(out)} of {n} bytes: {bytes(out[:16])!r}")
        return bytes(out)

    @staticmethod
    def _hdr(cmd, addr, n):
        return cmd + addr.to_bytes(3, "little") + n.to_bytes(3, "little")

    def ping(self, tries=2):
        for i in range(tries):
            self.send(b"P")
            r = self.recv(3)
            if r[0:1] == b"K":
                break
            self.drain()
        else:
            raise IOError(f"bad ping reply {r!r}")
        return {"version": r[1], "sdram_ready": bool(r[2] & 1), "cal_err": bool(r[2] & 2)}

    def write(self, addr, words):
        self.send(self._hdr(b"W", addr, len(words)) + struct.pack(f"<{len(words)}Q", *words))
        if self.recv(1, timeout=10) != b"w":
            raise IOError("write not acknowledged")

    def write_bytes(self, addr, data, progress=None):
        data = data + bytes(-len(data) % 8)
        n = len(data) // 8
        for i in range(0, n, CHUNK):
            k = min(CHUNK, n - i)
            self.send(self._hdr(b"W", addr + i, k) + data[8 * i:8 * (i + k)])
            if self.recv(1, timeout=10) != b"w":
                raise IOError(f"write at word {addr + i:#x} not acknowledged")
            if progress:
                progress(i + k, n)

    def read(self, addr, n):
        self.send(self._hdr(b"R", addr, n))
        return list(struct.unpack(f"<{n}Q", self.recv(8 * n)))

    def checksum(self, addr, n):
        self.send(self._hdr(b"C", addr, n))
        return struct.unpack("<I", self.recv(4, timeout=10))[0]

    def _ack(self, cmd, want, timeout=5.0):
        self.send(cmd)
        r = self.recv(1, timeout)
        if r != want:
            raise IOError(f"{cmd[:1]!r}: expected {want!r}, got {r!r}")

    def post_ack(self, cmd, want, then=None):
        def parse(lk):
            r = lk.recv(1)
            if r != want:
                raise IOError(f"{cmd[:1]!r}: expected {want!r}, got {r!r}")
            if then:
                then(lk)
        return self.post(cmd, parse)

    def init_engine(self):
        self._ack(b"I", b"i")

    def new_conversation(self):
        self._ack(b"Z", b"z")

    def seed(self, state):
        self.post_seed(state).value

    def post_seed(self, state):
        state &= 0xFFFFFFFF

        def then(lk):
            lk.rng = state
        return self.post_ack(b"S" + struct.pack("<I", state), b"s", then)

    def settings(self, inv_t, cut_q, gain_q):
        self._ack(b"G" + struct.pack("<HHH", inv_t, cut_q, min(gain_q, 0xFFFF)), b"g")

    def post_settings(self, inv_t, cut_q, gain_q):
        return self.post_ack(b"G" + struct.pack("<HHH", inv_t, cut_q, min(gain_q, 0xFFFF)), b"g")

    def emoji_table(self, mask, bias):
        body = b"".join(struct.pack("<Bh", int(bool(mask[t])), int(bias[t])) for t in range(128))
        self._ack(b"E" + body, b"e")

    def feed(self, tok):
        return self.post_feed(tok).value

    def post_feed(self, tok, expect=None):
        def parse(lk):
            r = lk.recv(3)
            if r[0:1] != b"a":
                raise IOError(f"feed: bad reply {r!r}")
            pos = struct.unpack("<H", r[1:3])[0]
            if expect is not None and pos != expect:
                raise IOError(f"feed: the board is at position {pos}, the host at {expect}")
            return pos
        return self.post(struct.pack("<cBHH", b"F", 0, tok, 0), parse)

    def step(self, tok):
        return self._f(1, tok, 0)

    def score_after(self, tok, nxt):
        return self._f(2, tok, nxt)

    def _f(self, flags, tok, nxt):
        return self.post_f(flags, tok, nxt).value

    def post_f(self, flags, tok, nxt=0):
        def parse(lk):
            r = lk.recv(1)
            if r != b"f":
                raise IOError(f"step: bad reply {r!r}")
            r += lk.recv(17)
            if flags & 1 and not flags & 2 and lk.rng is not None:
                lk.rng = _xs_apply(XS_STEP, lk.rng)
            return (struct.unpack("<H", r[1:3])[0], int.from_bytes(r[3:9], "little"),
                    int.from_bytes(r[9:15], "little"), int.from_bytes(r[15:18], "little", signed=True))
        return self.post(struct.pack("<cBHH", b"F", flags, tok, nxt), parse)

    def window(self, base):
        self._ack(b"X" + struct.pack("<H", base), b"x")

    def post_window(self, base):
        return self.post_ack(b"X" + struct.pack("<H", base), b"x")

    def post_jump(self, pos):
        return self.post_ack(b"J" + struct.pack("<H", pos), b"j")

    def post_new_conversation(self):
        return self.post_ack(b"Z", b"z")

    def jump(self, pos):
        self._ack(b"J" + struct.pack("<H", pos), b"j")

    def best(self, clear=False, end=False, keep=False):
        return self.post_best(clear, end, keep).value

    def post_best(self, clear=False, end=False, keep=False, check=None):
        def parse(lk):
            r = lk.recv(7)
            if r[0:1] != b"y":
                raise IOError(f"best: bad reply {r!r}")
            v = r[1], struct.unpack("<i", r[2:6])[0], r[6]
            if check:
                check(v)
            return v
        return self.post(b"Y" + bytes([int(clear) | int(end) << 1 | int(keep) << 2]), parse)


class Run:
    def __init__(self, link):
        self.link, self.done = link, False

    def next(self):
        if self.done:
            return None
        lk = self.link
        while lk.pending:
            lk._take()
        r = lk.recv(1)
        if r == b"e":
            self.done = True
            lk.run = None
            return None
        if r != b"f":
            raise IOError(f"run: bad reply {r!r}")
        r += lk.recv(17)
        if lk.rng is not None:
            lk.rng = _xs_apply(XS_STEP, lk.rng)
        return (struct.unpack("<H", r[1:3])[0], int.from_bytes(r[3:9], "little"),
                int.from_bytes(r[9:15], "little"), int.from_bytes(r[15:18], "little", signed=True))

    def stop(self):
        if self.done:
            return []
        self.link._send(b"\x00")
        extra = []
        while (r := self.next()) is not None:
            extra.append(r)
        return extra


def checksum(data):
    data = data + bytes(-len(data) % 8)
    return sum(struct.unpack(f"<{len(data) // 4}I", data)) & 0xFFFFFFFF


USAGE = "usage: python src/link.py ping | load models/pr_board"


def read_image(argv):
    if len(argv) < 3:
        sys.exit(USAGE)
    path = Path(argv[2]).with_suffix(".sdram.bin")
    if not path.is_file():
        sys.exit(f"{path} not found; build it with pack.py first")
    return path.read_bytes()


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ping"
    if cmd not in ("ping", "load") or len(sys.argv) > (3 if cmd == "load" else 2):
        sys.exit(USAGE)
    img = read_image(sys.argv) if cmd == "load" else None
    link = Link()
    if cmd == "ping":
        print(link.ping())
    elif cmd == "load":
        t = time.time()

        def progress(done, n):
            print(f"\r  {done * 8 / 2 ** 20:.2f} / {n * 8 / 2 ** 20:.2f} MB", end="", flush=True)

        link.write_bytes(0, img, progress)
        dt = time.time() - t
        want, got = checksum(img), link.checksum(0, -(-len(img) // 8))
        print(f"\nuploaded {len(img) / 2 ** 20:.2f} MB in {dt:.1f}s ({len(img) / dt / 1e3:.0f} KB/s), "
              f"checksum {'OK' if want == got else f'MISMATCH {got:#x} != {want:#x}'}")
        if want != got:
            sys.exit(1)


if __name__ == "__main__":
    main()
