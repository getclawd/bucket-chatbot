#!/usr/bin/env python3
"""Remove pastes that were never conversation.

Generation-parameter dumps, stack traces and copypasta get learned like any
other line, and then recited verbatim — Bucket once answered with 200 words of
Stable Diffusion metadata. New ones are rejected on the way in; this clears out
anything already stored.

    python clean_corpus.py            # show what would go
    python clean_corpus.py --apply    # remove it (backs the database up first)
"""

import shutil
import sys
from pathlib import Path

from bucket.config import config
from bucket.db import BucketDB
from bucket.learn import is_machine_output
from bucket.text import tokenize


def main() -> int:
    apply = "--apply" in sys.argv
    path = Path(config.DB_PATH)
    if not path.exists():
        print(f"no database at {path}", file=sys.stderr)
        return 1

    db = BucketDB(str(path))
    rows = db.conn.execute("SELECT id, text, author FROM utterances").fetchall()
    doomed = [r for r in rows if is_machine_output(r["text"])]

    print(f"{len(rows)} lines stored, {len(doomed)} of them not conversation\n")
    if not doomed:
        print("nothing to clean.")
        db.close()
        return 0

    for row in doomed[:12]:
        words = len(tokenize(row["text"]))
        print(f"  [{row['author'] or '?'}] {words} words: {row['text'][:70]}...")
    if len(doomed) > 12:
        print(f"  ... and {len(doomed) - 12} more")

    if not apply:
        db.close()
        print("\nthis was a dry run. re-run with --apply to remove them.")
        return 0

    db.close()
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    print(f"\nbacked up to {backup.name}")
    db = BucketDB(str(path))

    ids = [r["id"] for r in doomed]
    placeholders = ",".join("?" * len(ids))
    db.conn.execute(f"DELETE FROM word_index WHERE utterance_id IN ({placeholders})", ids)
    db.conn.execute(f"DELETE FROM vectors WHERE utterance_id IN ({placeholders})", ids)
    db.conn.execute(
        f"DELETE FROM pairs WHERE prompt_id IN ({placeholders}) "
        f"OR reply_id IN ({placeholders})",
        ids + ids,
    )
    db.conn.execute(f"DELETE FROM utterances WHERE id IN ({placeholders})", ids)
    db.conn.commit()
    # The full-text index mirrors utterances but isn't touched by that DELETE.
    db.rebuild_fts()

    left = db.conn.execute("SELECT COUNT(*) n FROM utterances").fetchone()["n"]
    print(f"removed {len(ids)} lines; {left} remain")
    print("\nnote: their n-gram transitions stay in the chain. run")
    print("  python rebuild_phrases.py --apply")
    print("if you want the markov table rebuilt from the cleaned corpus too.")
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
