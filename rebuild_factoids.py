#!/usr/bin/env python3
"""Re-mine facts from everything Bucket has already heard.

The factoid grammar started out matching only is/are/was/were/means/equals, so
anything phrased "Ana does have...", "claude can be trusted with...", "ana wont
be..." was stored as an utterance but never became a fact. Widening the grammar
only helps new messages; this recovers the ones already in the database.

Bucket's own lines are skipped — it recombines what it heard, so mining its
output for facts turns markov noise into knowledge it recites as true.

    python rebuild_factoids.py            # show what would be learned
    python rebuild_factoids.py --apply    # write it (backs the database up first)
"""

import shutil
import sqlite3
import sys
from pathlib import Path

from bucket.config import config
from bucket.db import BucketDB
from bucket.learn import Learner


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

    before = db.conn.execute("SELECT COUNT(*) n FROM factoids").fetchone()["n"]

    rows = db.conn.execute(
        "SELECT text, author FROM utterances WHERE author IS NOT NULL "
        "AND author NOT IN (?, 'seed') ORDER BY id",
        (config.NAME,),
    ).fetchall()
    print(f"re-reading {len(rows)} lines said by people "
          f"(skipping {config.NAME}'s own and any seed)\n")

    if not apply:
        # Mine into a throwaway copy so a dry run cannot touch the real data.
        scratch = BucketDB(":memory:")
        trial = Learner(scratch)
        for row in rows:
            trial._learn_factoid(row["text"], row["author"])
        found = scratch.conn.execute(
            "SELECT subject, verb, object FROM factoids ORDER BY subject"
        ).fetchall()

        known = {
            (r["subject"], r["verb"], r["object"])
            for r in db.conn.execute("SELECT subject, verb, object FROM factoids")
        }
        fresh = [f for f in found
                 if (f["subject"], f["verb"], f["object"]) not in known]

        print(f"would learn {len(fresh)} new facts (database currently has {before}):\n")
        for f in fresh[:40]:
            print(f"  {f['subject']} | {f['verb']} | {f['object'][:44]}")
        if len(fresh) > 40:
            print(f"  ... and {len(fresh) - 40} more")
        scratch.close()
        db.close()
        print("\nthis was a dry run. re-run with --apply to write them.")
        return 0

    for row in rows:
        learner._learn_factoid(row["text"], row["author"])

    after = db.conn.execute("SELECT COUNT(*) n FROM factoids").fetchone()["n"]
    print(f"facts: {before} -> {after}  (+{after - before})\n")

    print("subjects it now knows most about:")
    for r in db.conn.execute(
        "SELECT subject, COUNT(*) n FROM factoids GROUP BY subject "
        "ORDER BY n DESC, subject LIMIT 15"
    ):
        print(f"  {r['n']:>2}  {r['subject']}")

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
