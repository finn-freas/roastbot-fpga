import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import writer as w


class FakeTok:
    def __init__(self, name, tail=""):
        self.name_or_path = name
        self.tail = tail
        self.seen = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        self.seen = (messages, tokenize, add_generation_prompt, enable_thinking)
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>" + self.tail


def test_chat_without_system():
    tok = FakeTok("other")
    out = w.chat(tok, None, "hi")
    assert out == "<user>hi<assistant>"
    assert tok.seen == ([{"role": "user", "content": "hi"}], False, True, False)


def test_chat_with_system():
    tok = FakeTok("other")
    w.chat(tok, "be brief", "hi")
    assert [m["role"] for m in tok.seen[0]] == ["system", "user"]


def test_chat_closes_think_block_for_writer_model():
    out = w.chat(FakeTok(w.WRITER), None, "hi")
    assert out.endswith("<assistant><think>\n\n</think>\n\n")


def test_chat_keeps_existing_think_block():
    out = w.chat(FakeTok(w.WRITER, "<think>\n\n</think>\n"), None, "hi")
    assert out.count("</think>") == 1


def test_chat_leaves_other_models_alone():
    assert "<think>" not in w.chat(FakeTok("other"), None, "hi")


def test_group_pattern_matches_word_stems_only_at_word_start():
    assert w.GROUPS.search("the Christian choir")
    assert w.GROUPS.search("Chinese food")
    assert not w.GROUPS.search("a nontransferable ticket")
    assert not w.GROUPS.search("plain text")


def test_tiktok_path_under_data_style():
    assert w.TIKTOK == w.ROOT / "data" / "style" / "tiktok_roasts.json"
    assert w.ROOT.name == Path(__file__).resolve().parent.parent.name
