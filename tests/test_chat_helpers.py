import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import chat as C
import tok as T


def plain(n):
    return [t for t in range(4, T.VOCAB) if not C.EMOJI[t]][:n]


class WordTok:
    def __init__(self, words):
        self.words = words

    def decode(self, toks):
        return " ".join(self.words[t] for t in toks)


def test_make_rng_is_deterministic_and_seed_dependent():
    a = [C.make_rng(7).next() for _ in range(3)]
    assert a == [C.make_rng(7).next() for _ in range(3)]
    assert C.make_rng(7).next() != C.make_rng(8).next()


def test_make_rng_masks_seed_to_32_bits():
    assert C.make_rng(5).x == C.make_rng(5 + (1 << 32)).x


def test_pick_seed_numeric(capsys):
    assert C.pick_seed("42") == 42
    assert capsys.readouterr().out.strip() == "seed 42"


def test_pick_seed_random_in_range(capsys):
    seed = C.pick_seed("random")
    assert 0 <= seed < 1_000_000
    assert capsys.readouterr().out.strip() == f"seed {seed}"


def test_pick_seed_rejects_garbage():
    with pytest.raises(ValueError):
        C.pick_seed("abc")


def test_emoji_scale():
    assert C.emoji_scale(0.5) == 1.0
    assert C.emoji_scale(0) == 0.0
    assert C.emoji_scale(-3) == 0.0
    assert C.emoji_scale(1.0) == 4.0


def test_emoji_mask_shape():
    assert C.EMOJI.shape == (T.VOCAB,)
    assert C.EMOJI.dtype == bool


def test_loops_detects_repeated_trigram():
    base = plain(6)
    assert not C.loops(base)
    assert C.loops(base[:3] + base[:3])


def test_loops_detects_triple_repeat():
    t = plain(1)[0]
    assert C.loops([t, t, t])
    assert not C.loops([t, t])


def test_loops_ignores_him_marker():
    a, b = plain(2)
    assert not C.loops([a, T.HIM, b, T.HIM, a])
    assert C.loops([a, T.HIM, a, T.HIM, a])


def test_blocked_matches_slurs_and_groups():
    assert C.blocked(WordTok(["kys now"]), [0])
    assert C.blocked(WordTok(["you are gay"]), [0])
    assert not C.blocked(WordTok(["nice shoes"]), [0])


def test_has_mom_variants():
    tk = WordTok(["ur mom", "Your Mama", "you’re momma", "mom", "your moment"])
    assert C.has_mom(tk, [0])
    assert C.has_mom(tk, [1])
    assert C.has_mom(tk, [2])
    assert not C.has_mom(tk, [3])
    assert not C.has_mom(tk, [4])


def test_mom_bad_requires_aggression_and_no_recent_repeat():
    tk = WordTok(["ur mom", "hello"])

    class P:
        said = []

    assert C.mom_bad(P, tk, [0], "hello there")
    assert not C.mom_bad(P, tk, [0], "you stupid")
    assert not C.mom_bad(P, tk, [1], "hello there")

    class Q:
        said = [[0]]

    assert C.mom_bad(Q, tk, [0], "you stupid")


def test_mom_bad_normalises_curly_apostrophe():
    tk = WordTok(["ur mom"])

    class P:
        said = []

    assert not C.mom_bad(P, tk, [0], "shut up")
    assert C.mom_bad(P, tk, [0], "nothing")


class FakePerson:
    name = "pr"
    key = "pr"

    def __init__(self, *args):
        pass


def run_main(monkeypatch, tmp_path, lines):
    model = tmp_path / "pr_board.pkl"
    model.write_bytes(b"")
    monkeypatch.setattr(C, "board_file", lambda mdir, key: model)
    monkeypatch.setattr(T, "load_chat", lambda: None)
    monkeypatch.setattr(C, "Person", FakePerson)
    monkeypatch.setattr(sys, "argv", ["chat.py"])
    feed = iter(lines)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(feed))
    C.main()


@pytest.mark.parametrize("command", ["/temp abc", "/temp 0", "/temp -1", "/cut x", "/emoji y", "/seed z", "/best 1.5",
                                     "/temp inf", "/temp nan", "/cut inf", "/cut nan", "/cut -1", "/emoji inf", "/emoji -inf", "/emoji -1"])
def test_bad_command_values_do_not_end_the_session(monkeypatch, tmp_path, capsys, command):
    run_main(monkeypatch, tmp_path, [command, "/quit"])
    out = capsys.readouterr().out
    assert "not a valid value" in out
    assert command.split()[0] in out


def test_startup_seed_default(capsys):
    assert C.startup_seed(["chat.py"]) == 1
    assert capsys.readouterr().out == ""


def test_startup_seed_value(capsys):
    assert C.startup_seed(["chat.py", "--seed", "7"]) == 7


@pytest.mark.parametrize("argv", [["chat.py", "--seed"], ["chat.py", "--seed", "abc"]])
def test_startup_seed_rejects_bad_value(argv):
    with pytest.raises(SystemExit) as e:
        C.startup_seed(argv)
    assert "--seed" in str(e.value)


@pytest.mark.parametrize("argv", [["chat.py"], ["chat.py", "--board"], ["chat.py", "--seed", "7"],
                                  ["chat.py", "--board", "--seed", "random"], ["chat.py", "--seed", "--board"]])
def test_check_options_accepts_known_options(argv):
    C.check_options(argv)


@pytest.mark.parametrize("argv", [["chat.py", "--bord"], ["chat.py", "extra"], ["chat.py", "--seed", "1", "-x"]])
def test_check_options_rejects_unknown_arguments(argv):
    with pytest.raises(SystemExit) as e:
        C.check_options(argv)
    assert "unknown argument" in str(e.value)


def test_finite_accepts_numbers_and_rejects_nan_and_inf():
    assert C.finite("2.5") == 2.5
    assert C.finite("-1") == -1.0
    for bad in ["nan", "inf", "-inf", "x"]:
        with pytest.raises(ValueError):
            C.finite(bad)


def test_cut_zero_is_accepted(monkeypatch, tmp_path, capsys):
    run_main(monkeypatch, tmp_path, ["/cut 0", "/quit"])
    out = capsys.readouterr().out
    assert "cutoff: only words within 0.0" in out
    assert "not a valid value" not in out


def test_board_file_returns_existing_pickle_untouched(tmp_path):
    f = tmp_path / "pr_board.pkl"
    f.write_bytes(b"keep")
    (tmp_path / "pr_board.pkl.xz").write_bytes(b"not xz")
    assert C.board_file(tmp_path, "pr") == f
    assert f.read_bytes() == b"keep"


def test_board_file_unpacks_the_compressed_copy(tmp_path):
    import lzma
    (tmp_path / "pr_board.pkl.xz").write_bytes(lzma.compress(b"payload"))
    f = C.board_file(tmp_path, "pr")
    assert f == tmp_path / "pr_board.pkl"
    assert f.read_bytes() == b"payload"


def test_board_file_missing_model_creates_nothing(tmp_path):
    f = C.board_file(tmp_path, "pr")
    assert not f.exists()
    assert list(tmp_path.iterdir()) == []


def test_emoji_mask_marks_slot_tokens_and_skips_modifiers(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(T, "TOK", tmp_path)
    assert not C.emoji_mask().any()
    slots = [[10, "\U0001F600", 5], [11, "️", 5], [12, "\U0001F3FD", 5], [13, "❤", 5]]
    (tmp_path / "slots.json").write_text(json.dumps(slots))
    mask = C.emoji_mask()
    assert mask.shape == (T.VOCAB,)
    assert sorted(mask.nonzero()[0].tolist()) == [10, 13]


class BoardStub:
    def __init__(self, script, pos=0):
        self.script = list(script)
        self.pos = pos
        self.p = {"fmt": {"logit": 8}}
        self.fed = []
        self.stepped = []
        self.bot_mass = []
        self.bot_msgs = 0
        self.bot_emoji = 0

    def emoji_gain(self, emoji):
        return 1.0

    def make_room(self, need):
        pass

    def feed(self, ids):
        self.fed.append(list(ids))
        self.pos += len(ids)

    def step(self, t):
        self.stepped.append(t)
        self.pos += 1
        return self.script.pop(0)


class EncodedText:
    def __init__(self, ids):
        self.ids = ids


class IdTok:
    def encode(self, text, add_special_tokens=False):
        return EncodedText([plain(10)[int(c)] for c in text])

    def decode(self, toks):
        return "".join(str(plain(10).index(t)) for t in toks)


def board_stub(script, pos=0):
    stub = BoardStub(script, pos)
    stub.link = type("L", (), {"settings": lambda s, *a: setattr(stub, "settings", a)})()
    return stub


def test_board_reply_splits_messages_at_the_marker_and_stops_at_you():
    a, b, c = plain(3)
    script = [(a, 0, 1, 0), (b, 0, 1, 0), (T.HIM, 0, 1, 0), (c, 0, 1, 0), (T.YOU, 0, 1, 0)]
    stub = board_stub(script)
    texts, n, secs = C.board_reply(stub, IdTok(), "12", 1.0, 4.0)
    assert texts == ["01", "2"]
    assert n == 5 and secs == 0.0
    assert stub.fed == [[T.BOS, T.YOU, plain(10)[1], plain(10)[2]]]
    assert stub.stepped == [T.HIM, a, b, T.HIM, c]
    assert stub.bot_msgs == 2
    assert stub.settings[0] == 256 and stub.settings[1] == 4 * 2 ** 8


def test_board_reply_omits_bos_after_the_first_message():
    stub = board_stub([(T.YOU, 0, 1, 0)], pos=7)
    texts, n, _ = C.board_reply(stub, IdTok(), "1", 1.0, 4.0)
    assert texts == [] and n == 1
    assert stub.fed == [[T.YOU, plain(10)[1]]]
    assert stub.bot_msgs == 0


def test_board_reply_is_capped_at_max_reply():
    a = plain(1)[0]
    stub = board_stub([(a, 0, 1, 0)] * C.MAX_REPLY)
    texts, n, _ = C.board_reply(stub, IdTok(), "1", 1.0, 4.0)
    assert n == C.MAX_REPLY
    assert len(stub.stepped) == C.MAX_REPLY
    assert texts == ["0" * C.MAX_REPLY]
    assert stub.fed[-1] == [a]


@pytest.mark.parametrize("command, shown", [
    ("/temp 0.5", "temperature 0.5 (board inv_t 512)"),
    ("/temp auto", "each person's own temperature"),
    ("/emoji 0", "emoji 0.0"),
    ("/best 1", "one reply per message"),
    ("/best 99", f"writing up to {C.BEST_N} replies"),
    ("/best auto", "each person's own number of candidates"),
    ("/seed 7", "seed 7"),
])
def test_valid_command_values_are_acknowledged(monkeypatch, tmp_path, capsys, command, shown):
    run_main(monkeypatch, tmp_path, [command, "/quit"])
    out = capsys.readouterr().out
    assert shown in out
    assert "not a valid value" not in out


@pytest.mark.parametrize("command", ["/help", "/nonsense", "/temp", "/cut", "/best"])
def test_unknown_or_incomplete_commands_print_the_help(monkeypatch, tmp_path, capsys, command):
    run_main(monkeypatch, tmp_path, [command, "/quit"])
    assert C.HELP in capsys.readouterr().out


@pytest.mark.parametrize("quit_word", ["/q", "/quit", "/exit"])
def test_quit_aliases_end_the_session(monkeypatch, tmp_path, capsys, quit_word):
    run_main(monkeypatch, tmp_path, [quit_word])
    assert "not a valid value" not in capsys.readouterr().out


def test_reset_clears_the_conversation(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(FakePerson, "reset", lambda self: calls.append(1), raising=False)
    run_main(monkeypatch, tmp_path, ["/reset", "/quit"])
    assert calls == [1]
    assert "conversation cleared" in capsys.readouterr().out
