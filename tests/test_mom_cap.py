import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import mom_cap


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(mom_cap, "ROOT", tmp_path)
    d = tmp_path / "data" / "people" / "rb_mom"
    d.mkdir(parents=True)
    for split in ("val", "train"):
        (d / f"{split}.jsonl").write_text("")
    return d


def test_default_cap(data):
    assert mom_cap.parse_args([]) == (25, data)


def test_explicit_cap(data):
    assert mom_cap.parse_args(["3"])[0] == 3


def test_too_many_arguments(data):
    with pytest.raises(SystemExit) as e:
        mom_cap.parse_args(["1", "2"])
    assert "usage" in str(e.value)


@pytest.mark.parametrize("cap", ["x", "1.5", ""])
def test_non_integer_cap(data, cap):
    with pytest.raises(SystemExit) as e:
        mom_cap.parse_args([cap])
    assert "whole number" in str(e.value)


def test_cap_below_one(data):
    with pytest.raises(SystemExit) as e:
        mom_cap.parse_args(["0"])
    assert "at least 1" in str(e.value)


def test_missing_split_file_is_named(data):
    (data / "train.jsonl").unlink()
    with pytest.raises(SystemExit) as e:
        mom_cap.parse_args([])
    assert "train.jsonl not found" in str(e.value)


def write_split(d, split, convs):
    (d / f"{split}.jsonl").write_text("".join(json.dumps({"turns": c}) + "\n" for c in convs))


def read_split(d, split):
    return [json.loads(line)["turns"] for line in (d / f"{split}.jsonl").read_text().splitlines()]


def test_main_caps_each_example_line_and_keeps_other_conversations(data, capsys):
    from gen_dialogs import MOMS

    mom = [[0, "hey"], [1, MOMS[0]]]
    plain = [[0, "hey"], [1, "nothing about family here"]]
    write_split(data, "val", [mom, plain, mom])
    write_split(data, "train", [mom] * 5 + [plain] * 2)
    mom_cap.main(["2"])
    val, train = read_split(data, "val"), read_split(data, "train")
    assert val.count(mom) == 2 and val.count(plain) == 1
    assert train.count(mom) == 0 and train.count(plain) == 2
    out = capsys.readouterr().out
    assert "rb_mom val: kept 3 of 3" in out
    assert "rb_mom train: kept 2 of 7" in out


def test_main_ignores_mom_lines_spoken_by_the_user_side(data):
    from gen_dialogs import MOMS

    conv = [[0, MOMS[0]], [1, "ok"]]
    write_split(data, "val", [conv] * 4)
    write_split(data, "train", [])
    mom_cap.main(["1"])
    assert len(read_split(data, "val")) == 4
