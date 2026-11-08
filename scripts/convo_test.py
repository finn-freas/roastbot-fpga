import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import chat
import tok as T

CHATS = [
    ["yo i just lost 5 games in a row in fortnite", "it wasnt my fault my teammates were trash", "ok fine maybe i suck a little", "wanna play later tho"],
    ["i got a new haircut today", "my mom said it looks good", "why are you being so mean", "ok what should i get next time"],
    ["what should we eat tonight", "i was thinking pizza", "pineapple on it obviously", "youre just jealous"],
    ["goku would beat superman easy", "superman doesnt even have ultra instinct", "name one thing superman beat", "thats not even canon"],
    ["i have a math test tomorrow", "i havent studied at all", "can you help me", "its on fractions"],
    ["yo", "woah woah", "ur a bot bro", "ok shut up", "how are you", "calm down", "whatever"],
]


def parse_args(argv):
    key = argv[0] if argv else "pr"
    norepeat = False
    repeat = emoji = None
    for a in argv[1:]:
        k, _, v = a.partition("=")
        if a == "norepeat":
            norepeat = True
        elif k in ("repeat", "emoji"):
            try:
                x = float(v)
            except ValueError:
                sys.exit(f"{k}= needs a number, got {v!r}")
            if k == "repeat":
                repeat = x
            else:
                emoji = x
        else:
            sys.exit(f"unknown option {a!r}; usage: convo_test.py [model] [norepeat] [repeat=X] [emoji=X]")
    if not (chat.MODELS / f"{key}_board.pkl").exists() and not (chat.MODELS / f"{key}_board.pkl.xz").exists():
        sys.exit(f"no board model for {key!r} in {chat.MODELS}")
    return key, norepeat, repeat, emoji


def main():
    key, norepeat, repeat, emoji = parse_args(sys.argv[1:])
    if norepeat:
        chat.REPEAT = {}
    if repeat is not None:
        chat.REPEAT = {key: repeat}
    if emoji is not None:
        chat.REPEAT_EMOJI = emoji
    tok = T.load_chat()
    p = chat.Person(key)
    for seed, msgs in enumerate(CHATS, 1):
        p.reset()
        rng = chat.make_rng(seed)
        print(f"--- chat {seed}")
        for m in msgs:
            texts, used, _ = chat.reply_best(p, tok, m, 0.8, 4.0, rng, n_best=p.n_best)
            print(f"  you: {m}")
            for t in texts:
                print(f"  bot: {t.replace(chr(0xfffd), '')}")
            print(f"       [{used} tokens, ~{used / 23:.1f} s on the board]")


if __name__ == "__main__":
    main()
