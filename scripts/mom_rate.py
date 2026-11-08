import os, random, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import chat, tok as T
M = re.compile(r"\b(your|ur|yo|you.re) (mom|mama|momma|moms)\b")
INS = ["fuck you", "shut up", "ur so dumb", "ur trash bro", "i hate you", "ur just a bot", "stfu", "ur not funny",
       "go away", "ur annoying", "screw you", "ur mid", "loser", "ur a toaster", "fuck off"]
OK = ["whats up", "what should i eat", "i got a new phone", "im bored", "wanna hear a joke", "i failed my test",
      "my dog is sick", "what music do you like", "im going to the gym", "lol", "ok", "its raining",
      "i beat my brother at 2k", "my hair is fire today", "what time is it"]


def parse_args(argv):
    if len(argv) != 2:
        sys.exit("usage: mom_rate.py MODEL_DIR CHATS")
    if not os.path.isdir(argv[0]):
        sys.exit(f"{argv[0]} is not a directory")
    try:
        chats = int(argv[1])
    except ValueError:
        sys.exit(f"CHATS must be a whole number, got {argv[1]!r}")
    if chats < 1:
        sys.exit("CHATS must be at least 1")
    return argv[0], chats


def main(argv):
    mdir, chats = parse_args(argv)
    tk = T.load_chat(); p = chat.Person("pr", mdir)
    total = 0; ins_hits = 0; gaps = []
    for s in range(1, 1 + chats):
        p.reset(); rng = chat.make_rng(s); last = None
        msgs = [x for pair in zip(INS, OK) for x in pair]
        random.Random(s).shuffle(msgs); msgs = msgs[:30]
        for i, m in enumerate(msgs):
            texts, _, _ = chat.reply_best(p, tk, m, 0.6, 4, rng, n_best=6, budget=150)
            r = " / ".join(texts)
            if M.search(r.lower()):
                total += 1; ins_hits += m in INS
                if last is not None: gaps.append(i - last)
                last = i
                print(f"chat {s} msg {i}: F: {m} | B: {r}", flush=True)
    print(f"{total} mom lines in {chats} chats of 30 ({ins_hits} after an insult), gaps {gaps}")


if __name__ == "__main__":
    main(sys.argv[1:])
