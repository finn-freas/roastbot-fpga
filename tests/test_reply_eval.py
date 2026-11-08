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
