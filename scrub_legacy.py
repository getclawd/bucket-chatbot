#!/usr/bin/env python3
"""Purge blocked names from everything that feeds generation.

The 2008 transcripts are worth keeping — the flavour is the reason they were
imported. What isn't worth keeping is the *handles*: those people never agreed to
be quoted by a bot in a different chat eighteen years later, and their names came
back out recombined ("and someone finds cypress_z and tells him we won't let
you"). Plus the Warhammer/Final Fantasy roleplay names, which nobody here cares
about.

So this is deliberately narrow. It does NOT stop legacy lines feeding the chain —
"BLOOD FOR THE BLOOD GOD" and the rest of the 2008 character stay exactly as they
are. It removes only the chain edges, phrases and factoids that contain a blocked
name, whoever said them.

"whoever said them" is the important part. A blocklist that filtered by author or
by chat would miss the case that actually happened: Bucket said `cypress_z`, a
present-day user asked "who is cypress_z", and that question — from a live,
reinforcing speaker — put the name into the chain. The name launders itself
through the current conversation.

Going forward, learn.py refuses to learn from any line containing a blocked name
and brain.py refuses to say one, so this only has to clean up the backlog.

    python scrub_legacy.py            # show what would change
    python scrub_legacy.py --apply    # do it (backs the database up first)
"""

import shutil
import sys
from pathlib import Path

from bucket.blocklist import DEFAULT_BLOCKED, Blocklist
from bucket.config import config
from bucket.db import SEP, BucketDB
from bucket.learn import NON_REINFORCING_AUTHORS, Learner

BLOCKED = Blocklist(DEFAULT_BLOCKED | config.BLOCKED_NAMES)


def main() -> int:
    apply = "--apply" in sys.argv
    path = Path(config.DB_PATH)
    if not path.exists():
        print(f"no database at {path}", file=sys.stderr)
        return 1

    db = BucketDB(str(path))

    print(f"blocked names ({len(BLOCKED.names)}):")
    print(f"  {', '.join(sorted(BLOCKED.names))}\n")

    # Chain contexts are SEP-joined, so they have to be unpacked before matching
    # or the first word of a two-word context runs into the second.
    blocked_chain = [
        (r["context"], r["word"])
        for r in db.conn.execute("SELECT context, word FROM chain")
        if BLOCKED.blocks(r["context"].replace(SEP, " ")) or BLOCKED.blocks(r["word"])
    ]
    blocked_phrases = [
        r["text"] for r in db.conn.execute("SELECT text FROM phrases")
        if BLOCKED.blocks(r["text"])
    ]
    blocked_facts = [
        (r["id"], f"{r['subject']} {r['verb']} {r['object']}")
        for r in db.conn.execute("SELECT id, subject, verb, object FROM factoids")
        if BLOCKED.blocks(f"{r['subject']} {r['verb']} {r['object']}")
    ]
    # Utterances aren't touched — they stay stored, and the introspection commands
    # redact blocked names on the way out (Bucket.redact).
    mentions = db.conn.execute(
        "SELECT COUNT(*) n FROM utterances"
    ).fetchone()["n"]

    print("would delete, whoever said them:")
    print(f"  {len(blocked_chain)} chain edges")
    print(f"  {len(blocked_phrases)} phrases")
    print(f"  {len(blocked_facts)} factoids")
    if blocked_facts:
        for _, label in blocked_facts[:8]:
            print(f"      {label[:64]}")
    print(f"\nleaving all {mentions} stored utterances alone. legacy lines keep")
    print("feeding the chain — only the names come out.")

    if not apply:
        db.close()
        print("\nthis was a dry run. re-run with --apply to do it.")
        return 0

    db.close()
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    print(f"\nbacked up to {backup.name}")
    db = BucketDB(str(path))
    learner = Learner(db)
    before = db.stats()

    for context, word in blocked_chain:
        db.conn.execute(
            "DELETE FROM chain WHERE context = ? AND word = ?", (context, word)
        )
    for text in blocked_phrases:
        db.conn.execute("DELETE FROM phrases WHERE text = ?", (text,))
    for fid, _ in blocked_facts:
        db.conn.execute("DELETE FROM factoids WHERE id = ?", (fid,))
    db.conn.commit()

    # Re-mine phrases so any n-gram that only existed because of a blocked line
    # is rebuilt from what's left. Legacy authors are deliberately included here:
    # their content stays, only their names go.
    excluded = tuple(NON_REINFORCING_AUTHORS | {config.NAME})
    marks = ",".join("?" * len(excluded))
    keepers = [
        r for r in db.conn.execute(
            f"SELECT text FROM utterances WHERE author IS NOT NULL "
            f"AND author NOT IN ({marks}) ORDER BY id",
            excluded,
        )
        if not BLOCKED.blocks(r["text"])
    ]
    db.conn.execute("DELETE FROM phrases")
    for row in keepers:
        learner._learn_phrases(row["text"])
    db.conn.commit()

    db.prune_fact_vectors()
    after = db.stats()

    print(f"\nre-mined phrases from {len(keepers)} lines")
    for key in ("utterances", "factoids", "phrases", "transitions"):
        if key in before:
            print(f"  {key:<14} {before[key]:>6} -> {after[key]}")

    # Verify rather than assume.
    leaks = {
        "chain": sum(
            1 for r in db.conn.execute("SELECT context, word FROM chain")
            if BLOCKED.blocks(r["context"].replace(SEP, " ")) or BLOCKED.blocks(r["word"])
        ),
        "phrases": sum(
            1 for r in db.conn.execute("SELECT text FROM phrases")
            if BLOCKED.blocks(r["text"])
        ),
        "factoids": sum(
            1 for r in db.conn.execute("SELECT subject, verb, object FROM factoids")
            if BLOCKED.blocks(f"{r['subject']} {r['verb']} {r['object']}")
        ),
    }
    print("\nblocked names remaining in generative tables:")
    for table, n in leaks.items():
        print(f"  {table:<10} {n}")

    db.close()
    return 0 if not any(leaks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
