#!/usr/bin/env python3
"""What sentence shapes does Bucket actually learn a fact from?

The original grammar only matched is/are/was/were/means/equals, so "Ana does
have wet puh" was silently dropped and /literal ana came back empty. This checks
the wider grammar catches real speech without turning every sentence into a fact.

    python factoid_test.py
"""

import os
import sys
import tempfile

os.environ["BUCKET_LLM_BACKEND"] = "none"
os.environ["BUCKET_EMBED_BACKEND"] = "none"
os.environ["BUCKET_SEED"] = "0"

from bucket import Bucket  # noqa: E402

# (sentence, expected subject or None if it should be ignored)
CASES = [
    # the shape that started this
    ("Ana does have wet puh remember that", "ana"),
    # linking verbs, as before
    ("God is good", "god"),
    ("cheese is the best food in the world", "cheese"),
    ("rocks are heavy", "rocks"),
    # possession
    ("Elle has a hammer", "elle"),
    ("Cleo owns three cats", "cleo"),
    ("the server room has no windows", "the server room"),
    # negation, canonicalised
    ("Ana isn't awake", "ana"),
    ("the printer doesn't have paper", "the printer"),
    # preference and state
    ("Ben likes cold pizza", "ben"),
    ("bucket hates fridays", "bucket"),                  # facts about itself
    ("bucket, cheese is delicious", "cheese"),           # still reads as addressing
    ("Elle wants a new keyboard", "elle"),
    ("ana lives in a shed", "ana"),
    ("the alarm sounds like a dying goat", "the alarm"),
    # definitional
    ("BRB stands for be right back", "brb"),
    # possessive subject
    ("Ana's mom is a teacher", "ana's mom"),
    # chat spelling — no apostrophes, which is how people actually type
    ("ana wont be here tomorrow", "ana"),
    ("claude cant be trusted", "claude"),
    ("the server doesnt have ram", "the server"),
    # auxiliaries where the predicate trails into the object
    ("claude can be trusted with nuclear codes", "claude"),
    ("the server needs to be restarted", "the server"),
    ("the alarm keeps going off", "the alarm"),

    # --- things that must NOT become facts ---------------------------------
    ("i have no idea what you mean", None),          # junk subject "i"
    ("you are being weird", None),                   # junk subject "you"
    ("what is that", None),                          # junk subject "what"
    ("it is fine", None),                            # junk subject "it"
    ("https://example.com is a website", None),      # url
    ("hello", None),                                 # no predicate
    ("stop licking urself", None),                   # no predicate

    # --- questions are not assertions, with or without a "?" --------------
    ("bucket did you know you are gay", None),
    ("toadlya_bot can you draw a self portrait", None),
    ("do you remember our previous convo", None),
    ("what do you want me to do with it", None),
    ("is ana awake", None),
    ("Elle has a hammer?", None),                     # explicit question mark
    # subject that is really a parse artefact
    ("bro you have to eat that", None),
    ("i don't know what you just said", None),
    ("so you want to stab dario", None),
]

# Messages containing several sentences: the junk half must not hide the real
# fact in the other half. (subject that must be learned, subject that must not)
MULTI = [
    ("bro you have to. Wet puh is SO GOOD bro", "wet puh"),
    ("i dunno man. the basement has no windows", "the basement"),
    ("who cares? dario is a menace", "dario"),
    ("hello! cheese is good", "cheese"),
]


def main() -> int:
    path = os.path.join(tempfile.mkdtemp(), "factoid.sqlite3")
    bot = Bucket(db_path=path, quiet=True)
    failures = []

    print(f"{'sentence':<48} {'learned as':<22} {'verdict'}")
    print("-" * 84)

    for sentence, expected in CASES:
        before = {(r["subject"], r["verb"], r["object"])
                  for r in bot.db.conn.execute("SELECT * FROM factoids")}
        bot.learner.ingest(sentence)
        after = {(r["subject"], r["verb"], r["object"])
                 for r in bot.db.conn.execute("SELECT * FROM factoids")}
        new = after - before

        got = sorted(new)[0] if new else None
        subject = got[0] if got else None
        ok = subject == expected

        shown = f"{got[0]} | {got[1]} | {got[2][:18]}" if got else "(nothing)"
        print(f"{sentence[:46]:<48} {shown[:20]:<22} {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{sentence!r}: expected {expected!r}, got {subject!r}")

    print("\n=== multi-sentence messages ===")
    for sentence, expected in MULTI:
        before = {(r["subject"], r["verb"], r["object"])
                  for r in bot.db.conn.execute("SELECT * FROM factoids")}
        bot.learner.ingest(sentence)
        after = {(r["subject"], r["verb"], r["object"])
                 for r in bot.db.conn.execute("SELECT * FROM factoids")}
        subjects = {s for s, _, _ in after - before}
        ok = expected in subjects
        print(f"  [{'ok ' if ok else 'FAIL'}] {sentence[:44]:<46} -> {sorted(subjects) or 'nothing'}")
        if not ok:
            failures.append(f"{sentence!r}: did not learn about {expected!r}")

    print("\n=== /literal on the sentence that started this ===")
    print(bot.cmd_literal("ana"))
    if "wet puh" not in bot.cmd_literal("ana"):
        failures.append("/literal ana still doesn't know about the 'does have' fact")

    print("\n=== canonicalised negations ===")
    print(bot.cmd_literal("the printer"))

    total = bot.db.conn.execute("SELECT COUNT(*) n FROM factoids").fetchone()["n"]
    expected_total = sum(1 for _, e in CASES if e)
    print(f"\nlearned {total} facts from {len(CASES)} sentences "
          f"({expected_total} were meant to stick)")

    bot.close()

    print()
    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("grammar is wider, and still ignores the things it should. ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
