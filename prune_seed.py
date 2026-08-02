#!/usr/bin/env python3
"""Remove the starter corpus from a Bucket that has real conversation in it.

The seed exists so a fresh bot isn't mute. Once people are actually talking to
it the seed stops being scaffolding and becomes noise — worse, generic filler
like "keep talking, i'm writing it down" sits close to everything in embedding
space and matches any input, so it gets picked constantly.

Unpicking it touches five tables. Deleting the utterance rows alone would leave
the seed's n-gram transitions behind, and Bucket would carry on generating seed
phrasing out of the markov chain with no row to point at.

    python prune_seed.py                   # show what would change
    python prune_seed.py --apply           # do it (backs the database up first)
    python prune_seed.py --apply --bot-facts   # also drop facts Bucket invented
"""

import shutil
import sqlite3
import sys
from pathlib import Path

from bucket.config import config
from bucket.db import SEP
from bucket.seed import SEED_LINES
from bucket.text import content_words, normalize, tokenize

END = "\x00"


def chain_transitions(text: str):
    """Exactly what Learner._learn_chain would have added for this line."""
    words = tokenize(text)
    if len(words) < 2:
        return
    yield ("",), words[0]
    for a, b in zip(words, words[1:]):
        yield (a,), b
    yield (words[-1],), END
    for a, b, c in zip(words, words[1:], words[2:]):
        yield (a, b), c
    yield (words[-2], words[-1]), END


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

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")

    drop_bot_facts = "--bot-facts" in sys.argv

    # Factoids carry their own author column, so removing the seed utterances
    # leaves the facts they taught behind. They have to go too.
    seed_facts = conn.execute(
        "SELECT COUNT(*) n FROM factoids WHERE author = 'seed'"
    ).fetchone()["n"]
    bot_facts = conn.execute(
        "SELECT COUNT(*) n FROM factoids WHERE author = ?", (config.NAME,)
    ).fetchone()["n"]

    rows = conn.execute(
        "SELECT id, text, count FROM utterances WHERE author = 'seed'"
    ).fetchall()
    if not rows and not seed_facts and not (drop_bot_facts and bot_facts):
        print("nothing left to prune.")
        return 0

    if not rows:
        # Only facts left to clean.
        print(f"utterances: already clean\n")
        print(f"seed-taught facts to remove: {seed_facts}")
        if drop_bot_facts:
            print(f"bucket-invented facts to remove: {bot_facts}")
        if not apply:
            print("\nthis was a dry run. re-run with --apply to do it.")
            return 0
        conn.execute("DELETE FROM factoids WHERE author = 'seed'")
        if drop_bot_facts:
            conn.execute("DELETE FROM factoids WHERE author = ?", (config.NAME,))
        conn.commit()
        left = conn.execute("SELECT COUNT(*) n FROM factoids").fetchone()["n"]
        print(f"\nfacts remaining: {left}, all taught by people.")
        conn.close()
        return 0

    ids = [r["id"] for r in rows]
    placeholders = ",".join("?" * len(ids))
    pair_count = conn.execute(
        f"SELECT COUNT(*) n FROM pairs WHERE prompt_id IN ({placeholders}) "
        f"OR reply_id IN ({placeholders})",
        ids + ids,
    ).fetchone()["n"]
    vector_count = conn.execute(
        f"SELECT COUNT(*) n FROM vectors WHERE utterance_id IN ({placeholders})", ids
    ).fetchone()["n"]

    total = conn.execute("SELECT COUNT(*) n FROM utterances").fetchone()["n"]
    print(f"corpus: {total} utterances, {len(rows)} of them seed\n")
    print("seed lines to remove (most repeated first):")
    for row in sorted(rows, key=lambda r: -r["count"])[:10]:
        print(f"  x{row['count']:<3} {row['text'][:62]}")
    if len(rows) > 10:
        print(f"  ... and {len(rows) - 10} more")
    print(f"\nalso removes {pair_count} conversation pairs and {vector_count} vectors,")
    print("and unwinds their n-gram transitions from the markov chain.")

    if not apply:
        print("\nthis was a dry run. re-run with --apply to do it.")
        return 0

    # --- markov chain: reverse each seeded transition ---------------------
    # SEED_LINES is iterated with its duplicates intact, because a line listed
    # three times contributed three times.
    touched = removed_edges = 0
    for line in SEED_LINES:
        for context, word in chain_transitions(line):
            key = SEP.join(context)
            cur = conn.execute(
                "UPDATE chain SET count = count - 1 WHERE context = ? AND word = ?",
                (key, word),
            )
            touched += cur.rowcount
    cur = conn.execute("DELETE FROM chain WHERE count <= 0")
    removed_edges = cur.rowcount

    # --- word document frequencies ---------------------------------------
    for row in rows:
        for word in set(content_words(row["text"])):
            conn.execute("UPDATE words SET df = df - 1 WHERE word = ?", (word,))
    conn.execute("DELETE FROM words WHERE df <= 0")

    # --- the rows themselves ---------------------------------------------
    conn.execute(f"DELETE FROM word_index WHERE utterance_id IN ({placeholders})", ids)
    conn.execute(f"DELETE FROM vectors WHERE utterance_id IN ({placeholders})", ids)
    conn.execute(
        f"DELETE FROM pairs WHERE prompt_id IN ({placeholders}) "
        f"OR reply_id IN ({placeholders})",
        ids + ids,
    )
    conn.execute(f"DELETE FROM utterances WHERE id IN ({placeholders})", ids)
    conn.execute("DELETE FROM factoids WHERE author = 'seed'")
    if drop_bot_facts:
        conn.execute("DELETE FROM factoids WHERE author = ?", (config.NAME,))

    # Stop /wipe from putting it all back.
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('seeded', '1') "
        "ON CONFLICT(key) DO UPDATE SET value = '1'"
    )
    # The full-text index mirrors utterances but survives that DELETE, so it
    # would keep matching ids that no longer resolve. Absent if this build has
    # no FTS5, or if BUCKET_FTS was off when the database was last opened.
    try:
        conn.execute("INSERT INTO utterances_fts (utterances_fts) VALUES ('rebuild')")
    except sqlite3.Error:
        pass
    conn.commit()

    print(f"\nremoved {len(rows)} seed lines, {pair_count} pairs, {vector_count} vectors")
    print(f"decremented {touched} chain edges, deleted {removed_edges} that hit zero")

    left = conn.execute("SELECT COUNT(*) n FROM utterances").fetchone()["n"]
    print(f"\ncorpus is now {left} utterances, all of it from real conversation.")
    print("\nmost repeated now:")
    for row in conn.execute(
        "SELECT count, author, text FROM utterances ORDER BY count DESC LIMIT 8"
    ):
        print(f"  x{row['count']:<3} [{row['author'] or '?':<10}] {row['text'][:52]}")

    conn.execute("VACUUM")
    conn.close()
    print("\nrestart bucket so it reloads its vector index.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
