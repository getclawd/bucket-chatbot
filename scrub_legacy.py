#!/usr/bin/env python3
"""Purge blocked names from everything that feeds generation.

An imported corpus is worth keeping — the flavour is the reason it was imported.
What isn't worth keeping is the *handles*: those people never agreed to be quoted
by a bot in a different chat, and their names come back out recombined ("and
someone finds oldname_7 and tells him we won't let you").

So this is deliberately narrow. It does NOT stop imported lines feeding the chain
— the character of the corpus stays exactly as it is. It removes only the chain
edges, phrases and factoids that contain a blocked name. The database has no
author or chat metadata to filter on; matching is content-based.

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
from bucket.learn import Learner

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

    print("would delete from generative tables:")
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
    # is rebuilt from what's left. The remaining content stays; only blocked
    # names are excluded.
    keepers = [
        r for r in db.conn.execute(
            "SELECT text FROM utterances ORDER BY id",
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
