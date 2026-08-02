#!/usr/bin/env python3
"""Mine phrases out of everything already said, and re-clean the facts.

Phrases are recurring 2-4 word runs — the group's verbal tics. They're learned
going forward automatically; this recovers them from the existing corpus, and
re-runs fact extraction so trailing filler ("...remember that") is stripped from
objects and emphatic verbs ("does have") collapse to their plain form.

    python rebuild_phrases.py            # show what it would find
    python rebuild_phrases.py --apply    # write it (backs the database up first)
"""

import shutil
import sys
from pathlib import Path

from bucket.config import config
from bucket.db import BucketDB
from bucket.learn import NON_REINFORCING_AUTHORS, Learner


def main() -> int:
    apply = "--apply" in sys.argv
    path = Path(config.DB_PATH)
    if not path.exists():
        print(f"no database at {path}", file=sys.stderr)
        return 1

    if apply:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        print(f"backed up to {backup.name}\n")

    db = BucketDB(str(path))
    learner = Learner(db)

    # Skip anything Bucket said, the seed corpus, and imported 2008 Bucket
    # output — all of it is bot text, and mining phrases/facts out of it feeds
    # generation with already-degraded material.
    excluded = tuple(NON_REINFORCING_AUTHORS | {config.NAME})
    marks = ",".join("?" * len(excluded))
    rows = db.conn.execute(
        f"SELECT text, author FROM utterances WHERE author IS NOT NULL "
        f"AND author NOT IN ({marks}) ORDER BY id",
        excluded,
    ).fetchall()
    print(f"reading {len(rows)} lines said by people\n")

    if not apply:
        scratch = BucketDB(":memory:")
        trial = Learner(scratch)
        for row in rows:
            trial._learn_phrases(row["text"])
        found = scratch.top_phrases(limit=30)
        total = scratch.conn.execute(
            "SELECT COUNT(*) n FROM phrases WHERE count >= 2"
        ).fetchone()["n"]
        print(f"would learn {total} recurring phrases. the most repeated:\n")
        for r in found:
            print(f"  x{r['count']:<3} {r['text']}")
        scratch.close()
        db.close()
        print("\nthis was a dry run. re-run with --apply to write them.")
        return 0

    # Reset first. Counts are cumulative, so re-running would stack a second
    # count onto every phrase and promote one-offs to "recurring".
    db.conn.execute("DELETE FROM phrases")
    for row in rows:
        learner._learn_phrases(row["text"])
    db.conn.commit()

    # Re-extract facts so filler stripping and verb collapsing apply to old ones.
    before = db.conn.execute("SELECT COUNT(*) n FROM factoids").fetchone()["n"]
    db.conn.execute("DELETE FROM factoids")
    for row in rows:
        learner._learn_factoid(row["text"], row["author"])
    after = db.conn.execute("SELECT COUNT(*) n FROM factoids").fetchone()["n"]

    phrases = db.conn.execute(
        "SELECT COUNT(*) n FROM phrases WHERE count >= 2"
    ).fetchone()["n"]
    print(f"phrases (said 2+ times): {phrases}")
    print(f"facts re-extracted: {before} -> {after}\n")

    print("its favourite phrases now:")
    for r in db.top_phrases(limit=15):
        print(f"  x{r['count']:<3} {r['text']}")

    print("\nfacts about people, cleaned:")
    for r in db.conn.execute(
        "SELECT subject, verb, object FROM factoids ORDER BY RANDOM() LIMIT 10"
    ):
        print(f"  {r['subject']} | {r['verb']} | {r['object'][:46]}")

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
