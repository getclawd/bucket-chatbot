#!/usr/bin/env python3
"""Bucket talking to itself must not create an obsession.

`count` drives the obsession strategy. If Bucket's own replies bumped it, saying
a line would make it likelier to say that line again — a loop with nothing
damping it, which is how a piece of seed filler ended up as its catchphrase.
Only humans repeating something should count.

    python feedback_test.py
"""

import os
import tempfile

os.environ["BUCKET_LLM_BACKEND"] = "none"
os.environ["BUCKET_EMBED_BACKEND"] = "none"

from bucket import Bucket  # noqa: E402

LINE = "keep talking i'm writing it down"


def count_of(bot, text):
    row = bot.db.conn.execute(
        "SELECT count FROM utterances WHERE norm = ?",
        (__import__("bucket.text", fromlist=["normalize"]).normalize(text),),
    ).fetchone()
    return row["count"] if row else 0


def main() -> int:
    path = os.path.join(tempfile.mkdtemp(), "feedback.sqlite3")
    bot = Bucket(db_path=path, quiet=True)
    failures = []

    # --- Bucket saying something 20 times must not inflate it -------------
    # The first utterance creates the row at count 1 whoever says it; what
    # matters is that saying it again never moves the number.
    bot.learner.ingest(LINE, reinforce=False)
    baseline = count_of(bot, LINE)
    for _ in range(20):
        bot.learner.ingest(LINE, reinforce=False)
    after_self = count_of(bot, LINE)

    print("=== bucket repeating itself ===")
    print(f"  count after bucket first said it: {baseline}")
    print(f"  after saying it 20 more times: {after_self}")
    if after_self != baseline:
        failures.append(f"self-repetition inflated count from {baseline} to {after_self}")

    # --- humans repeating it must still count ----------------------------
    for _ in range(4):
        bot.learner.ingest(LINE)
    after_humans = count_of(bot, LINE)
    print(f"  after 4 humans said it: {after_humans}")
    if after_humans != baseline + 4:
        failures.append(f"human repetition did not count ({baseline} -> {after_humans})")

    # --- and the line is still learned, just not reinforced ---------------
    print("\n=== bucket's own words are still learned ===")
    novel = "the hexagonal cupboard resents me"
    uid = bot.learner.ingest(novel, reinforce=False)
    stored = bot.db.get_utterance(uid) if uid else None
    print(f"  stored: {bool(stored)}  count: {stored['count'] if stored else '-'}")
    if not stored:
        failures.append("bucket's own line was not stored at all")

    hits = bot.db.search("hexagonal cupboard", 3)
    print(f"  retrievable by search: {bool(hits)}")
    if not hits:
        failures.append("bucket's own line is not retrievable")

    # --- its replies must not become generative material -----------------
    print("\n=== bucket's replies stay out of the chain and the pair graph ===")
    probe = Bucket(db_path=os.path.join(tempfile.mkdtemp(), "sources.sqlite3"), quiet=True)

    chain_before = probe.db.conn.execute("SELECT COUNT(*) n FROM chain").fetchone()["n"]
    pairs_before = probe.db.conn.execute("SELECT COUNT(*) n FROM pairs").fetchone()["n"]

    first = probe.learner.ingest("humans said this first")
    probe.learner.ingest(
        "an entirely novel sentence bucket invented itself",
        previous_id=first, reinforce=False,
    )

    chain_after = probe.db.conn.execute("SELECT COUNT(*) n FROM chain").fetchone()["n"]
    pairs_after = probe.db.conn.execute("SELECT COUNT(*) n FROM pairs").fetchone()["n"]
    human_edges = chain_after - chain_before

    # The human line adds edges; the bot line must add none beyond those.
    probe2 = Bucket(db_path=os.path.join(tempfile.mkdtemp(), "sources2.sqlite3"), quiet=True)
    probe2.learner.ingest("humans said this first")
    human_only = probe2.db.conn.execute("SELECT COUNT(*) n FROM chain").fetchone()["n"]

    print(f"  chain edges after human + bot line: {chain_after}")
    print(f"  chain edges after human line alone: {human_only}")
    print(f"  pair edges created by the bot reply: {pairs_after - pairs_before}")
    if chain_after != human_only:
        failures.append("bucket's reply added markov transitions")
    if pairs_after != pairs_before:
        failures.append("bucket's reply created a pair edge")

    stored = probe.db.search("novel sentence invented", 3)
    print(f"  still searchable afterwards: {bool(stored)}")
    if not stored:
        failures.append("bucket's reply was not stored at all")
    probe.close()
    probe2.close()

    # --- end to end: a live conversation must not run away ---------------
    print("\n=== 60 turns of live conversation ===")
    fresh = os.path.join(tempfile.mkdtemp(), "live.sqlite3")
    bot2 = Bucket(db_path=fresh, quiet=True)
    for index in range(60):
        bot2.handle(f"message {index} about assorted things",
                    chat="tg:2", is_private=True)

    top = bot2.db.conn.execute(
        "SELECT count, text FROM utterances ORDER BY count DESC LIMIT 5"
    ).fetchall()
    print("  most repeated after the run:")
    for row in top:
        print(f"    x{row['count']:<3} {row['text'][:48]}")
    if top and top[0]["count"] > 3:
        failures.append(f"a generated line reached count {top[0]['count']} unexpectedly")

    bot.close()
    bot2.close()

    print()
    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("bucket can no longer talk itself into an obsession. ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
