import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import kv8_add


@pytest.mark.parametrize("argv", [[], ["a"], ["a", "b", "c", "d"]])
def test_wrong_argument_count_prints_usage(argv):
    with pytest.raises(SystemExit) as e:
        kv8_add.parse_args(argv)
    assert "usage" in str(e.value)


def test_missing_source_is_named(tmp_path):
    with pytest.raises(SystemExit) as e:
        kv8_add.parse_args([str(tmp_path / "pr_board.pkl"), str(tmp_path / "out")])
    assert "pr_board.pkl not found" in str(e.value)


def test_output_next_to_the_source_is_rejected(tmp_path):
    src = tmp_path / "pr_board.pkl"
    src.write_bytes(b"")
    with pytest.raises(SystemExit) as e:
        kv8_add.parse_args([str(src), str(tmp_path)])
    assert "somewhere else" in str(e.value)


def test_missing_training_data_is_named(tmp_path):
    src = tmp_path / "zz_nokey_board.pkl"
    src.write_bytes(b"")
    with pytest.raises(SystemExit) as e:
        kv8_add.parse_args([str(src), str(tmp_path / "out")])
    assert "train.jsonl not found" in str(e.value)


def test_key_defaults_to_the_file_name_prefix(tmp_path, monkeypatch):
    src = tmp_path / "zz_board.pkl"
    src.write_bytes(b"")
    data = tmp_path / "data" / "people" / "zz"
    data.mkdir(parents=True)
    (data / "train.jsonl").write_text("")
    monkeypatch.setattr(kv8_add, "ROOT", tmp_path)
    assert kv8_add.parse_args([str(src), str(tmp_path / "out")]) == (src, tmp_path / "out", "zz")
    assert kv8_add.parse_args([str(src), str(tmp_path / "out"), "zz"])[2] == "zz"


def test_main_calibrates_writes_model_and_links_companions(tmp_path, monkeypatch, capsys):
    import pickle
    from types import SimpleNamespace

    src_dir = tmp_path / "models"
    src_dir.mkdir()
    src = src_dir / "zz_board.pkl"
    src.write_bytes(pickle.dumps({"cfg": SimpleNamespace(n_layers=3)}))
    (src_dir / "zz_extra.npy").write_bytes(b"x")
    (src_dir / "generic_unigram.npy").write_bytes(b"u")
    (src_dir / "other_board.pkl").write_bytes(b"o")
    data = tmp_path / "data" / "people" / "zz"
    data.mkdir(parents=True)
    (data / "train.jsonl").write_text("")
    out = tmp_path / "out"
    seen = {}

    def calibrate(p, ids):
        seen["ids"] = ids
        p["kv8"] = [1, 2, 3]

    monkeypatch.setattr(kv8_add, "ROOT", tmp_path)
    monkeypatch.setattr(kv8_add.T, "load_chat", lambda: "tok")
    monkeypatch.setattr(kv8_add, "encode_sessions", lambda t, path: [([i], [i]) for i in range(60)])
    monkeypatch.setattr(kv8_add.B, "kv8_calibrate", calibrate)
    monkeypatch.setattr(kv8_add.P, "build", lambda p: (b"img", None))
    monkeypatch.setattr(kv8_add.P, "memory_report", lambda img: "report")
    monkeypatch.setattr(sys, "argv", ["kv8_add.py", str(src), str(out)])
    kv8_add.main()
    assert len(seen["ids"]) == 48
    assert pickle.loads((out / "zz_board.pkl").read_bytes())["kv8"] == [1, 2, 3]
    assert (out / "zz_extra.npy").is_symlink()
    assert (out / "generic_unigram.npy").is_symlink()
    assert not (out / "other_board.pkl").exists()
    text = capsys.readouterr().out
    assert "3 layers, kv8 shifts [1, 2, 3]" in text and "report" in text
