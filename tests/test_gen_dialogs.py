import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import gen_dialogs as g


def convo(*lines):
    return "\n".join(lines)


GOOD = convo("F: yo what is up", "B: nothing much and you", "F: just bored", "B: of course you are", "F: ok", "B: ok then")


def test_parse_reads_alternating_turns():
    turns = g.parse(GOOD)
    assert turns[0] == [0, "yo what is up"]
    assert [s for s, _ in turns] == [0, 1, 0, 1, 0, 1]


def test_parse_strips_bold_markers_and_quotes():
    turns = g.parse(convo("**F**: \"hello there\"", "**B**: hi", "F: x", "B: y", "F: z", "B: w"))
    assert turns[0] == [0, "hello there"]
    assert turns[1] == [1, "hi"]


def test_parse_ignores_non_dialogue_lines():
    turns = g.parse("Sure, here it is:\n" + GOOD + "\nthe end")
    assert len(turns) == 6


def test_parse_requires_three_bot_lines():
    assert g.parse(convo("F: a", "B: b", "F: c", "B: d")) is None


def test_parse_requires_f_to_start():
    assert g.parse(convo("B: a", "F: b", "B: c", "F: d", "B: e", "F: f", "B: g")) is None


def test_parse_rejects_long_lines():
    long = " ".join(["word"] * 26)
    assert g.parse(GOOD + "\nF: " + long) is None


def test_parse_rejects_blocked_content():
    assert g.parse(GOOD.replace("of course you are", "you should kys")) is None
    assert g.parse(GOOD.replace("ok then", "that is suicidal")) is None


def test_stock_lines_counts_bot_turns_only():
    turns = [[0, "who asked"], [1, "who asked lol"], [1, "skill issue"]]
    assert g.stock_lines(turns) == 2
    assert g.stock_lines(turns, ("skill issue",)) == 1


def test_parse_rejects_repeated_stock_lines():
    text = GOOD.replace("of course you are", "who asked").replace("ok then", "ur cooked")
    assert g.parse(text) is None


def test_mom_lines_and_mom_mode():
    base = GOOD.replace("ok then", "your mom said otherwise")
    assert g.parse(base) is not None
    assert g.parse(base, mom=True) is not None
    assert g.parse(GOOD, mom=True) is None
    two = base.replace("of course you are", "your mom too")
    assert g.parse(two, mom=True) is None


def test_mom_mode_rejects_yo_mama_jokes():
    text = GOOD.replace("ok then", "your mom is so slow")
    assert g.parse(text, mom=True) is None


def test_i_had_pattern_rejected_in_mode_three():
    text = convo("F: fuck you", "B: i had your mom last night", "F: a", "B: b", "F: c", "B: d")
    assert g.parse(text, mom=True, strict=3) is None


def test_parse_args_defaults_and_modes():
    assert g.parse_args([]) == (1500, ["topic"], 24)
    assert g.parse_args(["10", "both", "batch=4"]) == (10, ["topic", "argue"], 4)
    assert g.parse_args(["5", "mom3"]) == (5, ["mom3"], 24)


def test_parse_args_rejects_bad_input():
    import pytest

    for argv in (["x"], ["0"], ["5", "mon"], ["5", "batch=x"], ["5", "batch=0"], ["1", "mom", "batch=2", "extra"]):
        with pytest.raises(SystemExit):
            g.parse_args(argv)


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTok:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, prompts, return_tensors, padding):
        g.torch.manual_seed(0)
        return FakeBatch(input_ids=g.torch.zeros(len(prompts), 3, dtype=g.torch.long))

    def batch_decode(self, y, skip_special_tokens):
        assert skip_special_tokens
        return [GOOD] * len(y)


class FakeNet:
    def generate(self, input_ids, **kw):
        assert kw["do_sample"] and kw["max_new_tokens"] in (260, 320)
        return g.torch.zeros(len(input_ids), 5, dtype=g.torch.long)


def run_main(tmp_path, monkeypatch, argv):
    import json

    (tmp_path / "data" / "style").mkdir(parents=True)
    (tmp_path / "data" / "people").mkdir(exist_ok=True)
    (tmp_path / "data" / "style" / "roast_bank.json").write_text(json.dumps(["a", "b", "c", "d"]))
    (tmp_path / "data" / "style" / "tiktok_roasts.json").write_text(json.dumps(["one two three"] * 6))
    monkeypatch.setattr(g, "ROOT", tmp_path)
    monkeypatch.setattr(g, "q4", lambda name: (FakeTok(), FakeNet()))
    monkeypatch.setattr(g, "chat", lambda tok, system, user: system)
    monkeypatch.setattr(sys, "argv", ["gen_dialogs.py"] + argv)
    g.main()
    return tmp_path / "data" / "people"


def read_turns(path):
    import json

    return [json.loads(line)["turns"] for line in path.read_text().splitlines()]


def test_main_writes_partial_and_splits_train_and_val(tmp_path, monkeypatch, capsys):
    people = run_main(tmp_path, monkeypatch, ["13", "batch=4"])
    d = people / "rb"
    assert len(read_turns(d / "partial.jsonl")) == 16
    assert len(read_turns(d / "val.jsonl")) == 1
    assert len(read_turns(d / "train.jsonl")) == 15
    out = capsys.readouterr().out
    assert "topic: starting" in out
    assert "16 conversations ->" in out


def test_main_resumes_from_the_partial_file(tmp_path, monkeypatch, capsys):
    import json

    d = tmp_path / "data" / "people" / "rb_arg"
    d.mkdir(parents=True)
    old = g.parse(GOOD)
    (d / "partial.jsonl").write_text(json.dumps({"turns": old}) + "\n")
    people = run_main(tmp_path, monkeypatch, ["3", "argue", "batch=2"])
    assert "argue: resuming with 1 conversations" in capsys.readouterr().out
    assert len(read_turns(people / "rb_arg" / "partial.jsonl")) == 3
    assert len(read_turns(people / "rb_arg" / "val.jsonl")) + len(read_turns(people / "rb_arg" / "train.jsonl")) == 3


def test_mom_lines_counts_only_b_turns_that_mention_a_mom():
    turns = [[0, "your mom is nice"], [1, "ur mom says hi"], [1, "nothing here"], [1, "yo mama said so"]]
    assert g.mom_lines(turns) == 2


def test_mom_lines_handles_curly_apostrophe():
    assert g.mom_lines([[1, "you’re mom was busy"]]) == 1
    assert g.mom_lines([[1, "my mom is fine"]]) == 0


def test_mom_fits_requires_an_insult_before_the_mom_line():
    quiet = [[0, "what time is it"], [1, "your mom knows"]]
    assert g.mom_fits(quiet) is False
    insulted = [[0, "fuck you"], [1, "your mom likes me more"]]
    assert g.mom_fits(insulted) is True


def test_mom_fits_rejects_copied_stock_lines():
    turns = [[0, "fuck you"], [1, g.MOMS[3]]]
    assert g.mom_fits(turns) is False


def test_mom_fits_ignores_turns_without_a_mom_line():
    assert g.mom_fits([[0, "hello"], [1, "hi"]]) is True
    assert g.mom_fits([[1, "your mom opens"]]) is True


def test_style_lines_reads_the_style_file(monkeypatch, tmp_path):
    (tmp_path / "data" / "style").mkdir(parents=True)
    (tmp_path / "data" / "style" / "tiktok_roasts.json").write_text('["a b", "c"]')
    monkeypatch.setattr(g, "ROOT", tmp_path)
    assert g.style_lines() == ["a b", "c"]
