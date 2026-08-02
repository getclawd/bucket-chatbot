#!/usr/bin/env python3
"""Can Bucket tell that two names are one person, and answer about them?

    python identity_test.py
"""

import os
import sys
import tempfile

os.environ["BUCKET_LLM_BACKEND"] = "none"
os.environ["BUCKET_EMBED_BACKEND"] = "none"
os.environ["BUCKET_SEED"] = "0"

from bucket import Bucket  # noqa: E402


def main() -> int:
    path = os.path.join(tempfile.mkdtemp(), "identity.sqlite3")
    bot = Bucket(db_path=path, quiet=True)
    failures = []

    # Facts arrive under both names, as they would in a real chat.
    for line, who in [
        ("ana has good puh", "Ben"),
        ("ana is never awake", "Cleo"),
        ("ben owns a shed", "Cleo"),
        ("cleo is the loudest one here", "Ben"),
        ("elle hates fridays", "Ben"),
    ]:
        bot.learner.ingest(line, author=who, chat="t")

    print("=== before linking ===")
    print(f"  facts under 'ana':        {len(bot.db.factoids_for('ana'))}")
    print(f"  facts under 'ben': {len(bot.db.factoids_for('ben'))}")

    print("\n=== linking ===")
    print("  " + bot.cmd_alias("ben ana"))
    print("  " + bot.cmd_alias("cleo elle"))

    merged = bot.db.factoids_for("ana")
    print(f"\n  facts under 'ana' after:  {len(merged)}")
    if len(merged) < 3:
        failures.append(f"aliasing did not merge the facts (got {len(merged)})")

    # Either name must reach the same place.
    if bot.db.resolve("ben") != bot.db.resolve("ana"):
        failures.append("ben and ana did not resolve to one subject")
    if bot.db.resolve("cleo") != bot.db.resolve("elle"):
        failures.append("cleo and elle did not resolve to one subject")

    print("\n=== /about ana (asked by the alias) ===")
    about = bot.cmd_about("ben")
    print("  " + about.replace("\n", "\n  "))
    if "good puh" not in about:
        failures.append("/about did not surface the fact taught under the other name")
    if "has said" not in about:
        failures.append("/about did not include what they actually say")

    print("\n=== /who is dead style reverse lookup ===")
    bot.learner.ingest("anas mom is dead", author="Cleo", chat="t")
    who = bot.cmd_who("is dead")
    print("  " + who.replace("\n", "\n  "))
    if "mom" not in who:
        failures.append("reverse lookup did not find the matching fact")

    print("\n=== 'aka' phrasing works without the command ===")
    bot.learner.ingest("dima is also known as the toad", author="Cleo", chat="t")
    if bot.db.resolve("the toad") != bot.db.resolve("dima"):
        failures.append("'also known as' did not create an alias")
    else:
        print("  dima = the toad")

    print("\n=== an ordinary fact must NOT become an alias ===")
    bot.learner.ingest("cleo is annoying", author="Ben", chat="t")
    if bot.db.resolve("annoying") == bot.db.resolve("cleo"):
        failures.append("'X is annoying' was misread as an identity link")
    else:
        print("  'cleo is annoying' stayed a fact, not a rename")

    bot.close()

    print()
    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("two names, one person. ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
