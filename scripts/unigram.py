import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import tok as T


def main():
    if len(sys.argv) > 2:
        sys.exit("usage: unigram.py [output_dir]")
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models"
    val = T.TOK / "val.bin"
    if not val.is_file():
        sys.exit(f"{val} not found; tokenize the corpus first")
    ids = np.fromfile(val, np.uint16)
    if not ids.size:
        sys.exit(f"{val} is empty; tokenize the corpus first")
    out.mkdir(parents=True, exist_ok=True)
    c = np.bincount(ids, minlength=T.VOCAB)[:T.VOCAB].astype(float)
    np.save(out / "generic_unigram.npy", (c / c.sum() + 1e-7) / (1 + 1e-7 * T.VOCAB))


if __name__ == "__main__":
    main()
