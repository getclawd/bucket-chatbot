#!/usr/bin/env python3
"""Does it avoid saying something that was just said?

From a real transcript. Elle said "so true queen"; two turns later Bucket said
"so true queen" back at him. The parrot guard didn't catch it because it only
ever compared the reply against the *current* message ("i didnt"), which shares
no words with it. Bucket's own replies weren't tracked at all, so it could also
repeat itself indefinitely.

The window is deliberately short. Reusing the group's catchphrases is what the
phrase and obsession strategies are *for* — this only stops it happening twice
inside one exchange, so the third check below asserts the suppression expires.

    python repeat_test.py
"""

import os
import sys
import tempfile

os.environ["BUCKET_LLM_BACKEND"] = "none"
os.environ["BUCKET_EMBED_BACKEND"] = "none"  # this is about selection, not recall
os.environ.setdefault("BUCKET_SEED", "0")

from bucket import Bucket  # noqa: E402
from bucket.brain import Brain  # noqa: E402
from bucket.config import config  # noqa: E402
from bucket.text import DANGLING_ENDINGS  # noqa: E402


def main() -> int:
    failures = []

    # --- 1. the guard itself, in isolation ------------------------------
    print("=== the guard sees more than the current message ===")
    cases = [
        # (reply, current message, earlier lines, should be suppressed)
        ("so true queen", "i didnt", ("so true queen",), True),
        ("so true queen", "i didnt", (), False),
        ("SO TRUE QUEEN", "i didnt", ("so true queen",), True),
        ("chickens were once dinosaurs", "i didnt", ("so true queen",), False),
    ]
    for reply, current, earlier, expected in cases:
        recent = (current, *earlier)
        got = Brain._repeats(reply, recent)
        mark = "ok " if got == expected else "BAD"
        print(f"  {mark} {reply!r} after {earlier or '(nothing)'} -> suppressed={got}")
        if got != expected:
            failures.append(f"_repeats({reply!r}, {recent!r}) was {got}, wanted {expected}")

    # --- 2. end to end, through a real conversation ---------------------
    print("\n=== it doesn't repeat itself across turns ===")
    path = os.path.join(tempfile.mkdtemp(), "repeat.sqlite3")
    bot = Bucket(db_path=path, quiet=True)

    # A corpus small enough that repeating is the path of least resistance.
    for line in [
        "so true queen",
        "thats so wise",
        "to die is fake and gay",
        "chickens were once dinosaurs",
        "why are you pinging dima why",
        "your mom is dead",
        "i didnt",
    ]:
        bot.learn_only(line, author="elle", chat="dc:1")

    chat = "dc:1"
    said = []
    for turn in ["die", "thats so wise", "so true queen", "i didnt", "hi", "what", "ok"]:
        reply = bot.handle(turn, author="elle", chat=chat, is_private=True)
        if reply:
            said.append(reply)
            print(f"  elle: {turn!r}")
            print(f"       -> {reply!r}")

    # Nothing Bucket says should equal the line it was replying to, or one of
    # its own recent lines. Duplicates *far* apart are allowed by design.
    window = config.NO_REPEAT_WINDOW
    for i, line in enumerate(said):
        clash = [p for p in said[max(0, i - window):i] if p.strip().lower() == line.strip().lower()]
        if clash:
            failures.append(f"repeated its own line within {window} turns: {line!r}")

    print(f"\n  {len(said)} replies, {len(set(s.lower() for s in said))} distinct")

    # --- 3. suppression must expire ------------------------------------
    # If it were permanent, the phrase and obsession strategies would be dead:
    # the group's catchphrases are meant to come back around.
    print("\n=== suppression is a window, not a ban ===")
    recent = tuple(f"filler line number {i}" for i in range(config.NO_REPEAT_WINDOW))
    still_blocked = Brain._repeats("so true queen", recent)
    print(f"  'so true queen' blocked after {len(recent)} unrelated lines: {still_blocked}")
    if still_blocked:
        failures.append("suppression did not expire — phrase/obsession strategies would die")

    # --- 4. replies don't stop mid-thought -----------------------------
    # Composition and markov both like to stop on a word that needs something
    # after it, which reads as the message being cut off rather than as insane.
    print("\n=== no reply ends on a word that can't end a sentence ===")
    dangling = []
    total = 0
    for prompt in ["hi", "who is ana", "ok", "tell me something", "why"]:
        for _ in range(40):
            reply = bot.brain.respond(prompt)
            total += 1
            words = reply.text.split()
            if words and words[-1].strip(".,!?;:\"()").lower() in DANGLING_ENDINGS:
                dangling.append(reply.text)
    print(f"  {len(dangling)}/{total} ended on a dangling word")
    for text in dangling[:3]:
        print(f"      ...{text[-60:]}")
    if dangling:
        failures.append(f"{len(dangling)}/{total} replies ended mid-thought")

    bot.close()
    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("it won't say back what was just said, and the window expires. ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
