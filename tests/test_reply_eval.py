import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import reply_eval as re_


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(re_, "OUT", tmp_path)
    for n in ("a", "b"):
        (tmp_path / f"{n}.json").write_text("[]")
    return tmp_path


@pytest.mark.parametrize("argv", [[], ["bogus"], ["reference", "x"], ["gen"], ["gen", "a", "2"],
                                  ["gen", "a", "2", "3", "1", "2", "3", "4", "5"], ["sense"], ["judge"],
                                  ["compare"], ["compare", "a"]])
def test_wrong_arguments_print_usage(out, argv):
    with pytest.raises(SystemExit) as e:
        re_.parse_args(argv)
    assert "usage" in str(e.value)


def test_gen_defaults_and_options(out):
    assert re_.parse_args(["gen", "n", "4", "100"]) == ("gen", ("n", 4, 100))
    assert re_.parse_args(["gen", "n", "4", "100", "-1", "0.5"]) == ("gen", ("n", 4, 100, None, 0.5))
    assert re_.parse_args(["gen", "n", "4", "100", "0.3", "0.8", "4", "3"]) == ("gen", ("n", 4, 100, 0.3, 0.8, 4.0, 3))


@pytest.mark.parametrize("argv,msg", [(["gen", "n", "x", "1"], "must be integers"),
                                      (["gen", "n", "0", "1"], "at least 1"),
                                      (["gen", "n", "1", "1", "a"], "must be numbers"),
                                      (["gen", "n", "1", "1", "0", "1", "1", "1.5"], "ROUNDS")])
def test_gen_rejects_bad_values(out, argv, msg):
    with pytest.raises(SystemExit) as e:
        re_.parse_args(argv)
    assert msg in str(e.value)


def test_sense_judge_compare_take_existing_runs(out):
    assert re_.parse_args(["sense", "a", "b"]) == ("sense", (["a", "b"],))
    assert re_.parse_args(["judge", "a"]) == ("judge", (["a"],))
    assert re_.parse_args(["compare", "a", "b"]) == ("compare", ("a", ["b"]))
    assert re_.parse_args(["reference"]) == ("reference", ())


def test_missing_run_is_named(out):
    with pytest.raises(SystemExit) as e:
        re_.parse_args(["compare", "a", "nope"])
    assert "nope.json not found" in str(e.value)


def test_reference_creates_output_directory_and_pairs_replies(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(re_, "ROOT", tmp_path)
    monkeypatch.setattr(re_, "OUT", tmp_path / "models" / "eval")
    d = tmp_path / "data" / "people" / "rb3"
    d.mkdir(parents=True)
    convs = [[[0, "hi"], [1, "r1"], [0, "yo"], [1, "r2"]], [[0, "a"], [1, "r3"]]]
    (d / "val.jsonl").write_text("".join(json.dumps({"turns": c}) + "\n" for c in convs))
    re_.reference()
    real = json.loads((re_.OUT / "ref_writer.json").read_text())
    fake = json.loads((re_.OUT / "ref_shuffled.json").read_text())
    assert real == [[["hi", "r1", 0], ["yo", "r2", 0]], [["a", "r3", 0]]]
    assert [[t[0] for t in c] for c in fake] == [["hi", "yo"], ["a"]]
    assert sorted(t[1] for c in fake for t in c) == ["r1", "r2", "r3"]


def test_reference_names_missing_validation_file(tmp_path, monkeypatch):
    monkeypatch.setattr(re_, "ROOT", tmp_path)
    monkeypatch.setattr(re_, "OUT", tmp_path / "eval")
    with pytest.raises(SystemExit, match="val.jsonl not found"):
        re_.reference()
