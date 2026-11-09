import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

norm = lambda s: " ".join(re.sub(r"[^a-z ]", " ", s.lower().replace("’", "")).split())


def parse_args(argv):
    if len(argv) > 1:
        sys.exit("usage: mom_cap.py [CAP]")
    cap = 25
    if argv:
        try:
            cap = int(argv[0])
        except ValueError:
            sys.exit(f"CAP must be a whole number, got {argv[0]!r}")
        if cap < 1:
            sys.exit("CAP must be at least 1")
    d = ROOT / "data" / "people" / "rb_mom"
    for split in ("val", "train"):
        if not (d / f"{split}.jsonl").is_file():
            sys.exit(f"{d / (split + '.jsonl')} not found")
    return cap, d


def main(argv):
    cap, d = parse_args(argv)
    from gen_dialogs import MOM_RE, MOMS

    ex = [norm(m) for m in MOMS]
    r = random.Random(4)
    seen = Counter()
    for split in ("val", "train"):
        convs = [json.loads(line)["turns"] for line in open(d / f"{split}.jsonl")]
        r.shuffle(convs)
        out = []
        for c in convs:
            hit = [e for k, t in c if k and MOM_RE.search(t.lower().replace("’", "'")) for e in ex if e in norm(t)]
            if hit and seen[hit[0]] >= cap:
                continue
            seen.update(hit[:1])
            out.append(c)
        (d / f"{split}.jsonl").write_text("".join(json.dumps({"turns": c}, ensure_ascii=False) + "\n" for c in out))
        print(f"rb_mom {split}: kept {len(out)} of {len(convs)}")
    print("example lines kept:", dict(seen))


if __name__ == "__main__":
    main(sys.argv[1:])
