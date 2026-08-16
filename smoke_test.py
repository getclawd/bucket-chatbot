#!/usr/bin/env python3
"""Exercise the engine end to end without Telegram or an LLM.

    python smoke_test.py
"""

import os
import tempfile

os.environ["BUCKET_LLM_BACKEND"] = "none"
os.environ["BUCKET_CHATTINESS"] = "1.0"

from bucket import Bucket  # noqa: E402
from bucket.brain import Brain  # noqa: E402

CONVERSATION = [
    "hey bucket, what's up",
    "cheese is the best food in the world",
    "no, pizza is the best food in the world",
    "gives bucket a rusty hammer",
    "what do you think about cheese",
    "all work and no play makes jack a dull boy",
    "all work and no play makes jack a dull boy",
    "all work and no play makes jack a dull boy",
    "a hammer is for hitting things you disagree with",
    "bucket, tell me about pizza",
    "i think the internet was a mistake",
    "hands bucket a small glass owl",
]


def main() -> int:
    path = os.path.join(tempfile.mkdtemp(), "smoke.sqlite3")
    bot = Bucket(db_path=path)
    failures = []

    print("=== feeding it a conversation ===")
    for line in CONVERSATION:
        reply = bot.handle(line, chat="test", is_private=True)
        print(f"  input: {line}")
        print(f"  bucket> {reply}")
        if not reply:
            failures.append(f"no reply to {line!r}")

    print("\n=== what it learned ===")
    stats = bot.db.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    for key in ("utterances", "pairs", "factoids", "transitions"):
        if stats[key] == 0:
            failures.append(f"{key} table is empty")

    print("\n=== factoid recall ===")
    print(bot.cmd_literal("cheese"))
    if not bot.db.factoids_for("cheese"):
        failures.append("did not learn 'cheese is ...'")

    print("\n=== inventory ===")
    print(bot.cmd_inventory())
    if not bot.db.items():
        failures.append("did not pick up any items")

    print("\n=== every strategy fires at least once ===")
    brain = Brain(bot.db)
    seen: dict[str, str] = {}
    for _ in range(400):
        reply = brain.respond("what do you think about cheese and pizza")
        seen.setdefault(reply.strategy, reply.text)
    for strategy, example in sorted(seen.items()):
        print(f"  {strategy:<12} {example[:70]}")
    for expected in ("pair", "factoid", "markov", "echo", "mashup"):
        if expected not in seen:
            failures.append(f"strategy {expected!r} never fired")

    print("\n=== obsession ===")
    obsessed = bot.db.most_repeated(3)
    print(f"  {obsessed['text'] if obsessed else '(none)'}")
    if not obsessed:
        failures.append("repeated line did not become an obsession")

    print("\n=== polish layer degrades safely ===")
    print(f"  status: {bot.polisher.status()}")
    if bot.polisher.polish("raw line", "hello", "markov") != "raw line":
        failures.append("disabled polisher altered the raw line")

    bot.close()

    print()
    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
