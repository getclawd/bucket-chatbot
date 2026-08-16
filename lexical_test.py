#!/usr/bin/env python3
"""Does FTS5/BM25 recall at least as well as the hand-rolled IDF scan?

Bucket has two lexical retrieval paths (see BucketDB.lexical_search):

  search()      a sqrt-IDF sum over the word_index/words tables. Every shared
                word adds its own weight independently, so one rare word in
                common outranks four common ones — and length is ignored, so a
                long line that happens to contain a rare word beats a short
                line that's actually about it.
  fts_search()  SQLite's own FTS5 index scored with BM25, which saturates
                repeated terms and normalizes for document length.

This measures both against fixed queries with known-relevant answers, so the
swap is decided on precision numbers rather than on BM25's reputation. Runs
without a network or an embedding model: this is the *lexical* half only.

    python lexical_test.py
"""

import os
import sys
import tempfile

os.environ["BUCKET_LLM_BACKEND"] = "none"
os.environ["BUCKET_EMBED_BACKEND"] = "none"
os.environ.setdefault("BUCKET_SEED", "0")

from bucket.db import BucketDB  # noqa: E402

# A corpus with the shapes that break sqrt-IDF: near-duplicates, one very long
# line that name-drops a rare word it isn't about, and repeated common words.
CORPUS = [
    "ana has wet puh",
    "wet puh is so good bro",
    "bro you have to, wet puh is SO GOOD bro",
    "what about wet puh?",
    "the server room floods every time it rains",
    "the server room is flooded again",
    "it rains here constantly",
    "blood for the blood god",
    "no i want it for the blood god",
    "we kill for the blood god",
    "dario beats claude like a puppy",
    "claude can be trusted with the nuclear codes",
    "dima is a strange little bot",
    "i gave bucket a broken keyboard yesterday",
    # The trap: mentions "puh" and "blood" once each, in a line about neither.
    "anyway i was telling ana about the time someone spilled blood on the "
    "keyboard in the server room and everyone said puh, which honestly says "
    "more about this group than anything else i could possibly write down here",
]

# (query, substrings that mark a hit as relevant)
QUERIES = [
    ("what do you think about wet puh", ["wet puh"]),
    ("is the server room flooded", ["server room"]),
    ("blood god", ["blood god"]),
    ("can claude be trusted", ["claude"]),
    ("tell me about the broken keyboard", ["broken keyboard"]),
    ("does it rain a lot", ["rains", "floods", "flooded"]),
]

TOP_N = 3


def precision(hits, db, markers) -> float:
    """Fraction of the top hits that are actually about the query."""
    if not hits:
        return 0.0
    good = 0
    for uid, _score in hits:
        row = db.get_utterance(uid)
        text = (row["text"] if row else "").lower()
        if any(m in text for m in markers):
            good += 1
    return good / len(hits)


def main() -> int:
    path = os.path.join(tempfile.mkdtemp(), "lexical.sqlite3")
    db = BucketDB(path)

    if not db.fts:
        print("this SQLite build has no FTS5 — nothing to compare.")
        return 0

    for line in CORPUS:
        db.add_utterance(line)
    print(f"corpus: {db.conn.execute('SELECT COUNT(*) n FROM utterances').fetchone()['n']} lines\n")

    # The index must actually contain everything, or high precision below would
    # just mean "returned nothing wrong because it returned nothing".
    indexed = db.conn.execute("SELECT COUNT(*) n FROM utterances_fts_docsize").fetchone()["n"]
    total = db.conn.execute("SELECT COUNT(*) n FROM utterances").fetchone()["n"]
    failures = []
    if indexed != total:
        failures.append(f"fts index has {indexed} rows, corpus has {total}")

    print(f"=== precision@{TOP_N}, per query ===")
    idf_total = bm25_total = 0.0
    for query, markers in QUERIES:
        idf = db.search(query, TOP_N)
        bm25 = db.fts_search(query, TOP_N)
        p_idf = precision(idf, db, markers)
        p_bm25 = precision(bm25, db, markers)
        idf_total += p_idf
        bm25_total += p_bm25

        flag = "  " if p_bm25 >= p_idf else "<-"
        print(f"\n  {query!r}")
        print(f"    idf  {p_idf:.2f}   bm25 {p_bm25:.2f} {flag}")
        for label, hits in (("idf", idf), ("bm25", bm25)):
            for uid, _ in hits:
                row = db.get_utterance(uid)
                print(f"      {label:5} {(row['text'] if row else '?')[:58]}")

        # Only a *relative* regression is a failure. A query can legitimately
        # have no lexical hits at all (that's what the semantic half is for);
        # what must not happen is bm25 going blind where idf could see.
        if idf and not bm25:
            failures.append(f"idf found {len(idf)} hits for {query!r}, bm25 found none")
        if p_bm25 < p_idf:
            failures.append(
                f"bm25 precision {p_bm25:.2f} < idf {p_idf:.2f} for {query!r}"
            )

    n = len(QUERIES)
    print(f"\n=== mean precision@{TOP_N} ===")
    print(f"  idf   {idf_total / n:.3f}")
    print(f"  bm25  {bm25_total / n:.3f}")

    # The point of the exercise: BM25 has to be at least as good overall, or
    # there is no reason to carry the new code path at all.
    if bm25_total < idf_total:
        failures.append(
            f"bm25 mean precision {bm25_total / n:.3f} is worse than idf {idf_total / n:.3f}"
        )

    # Query sanitizing: FTS5 ANDs bare terms and treats punctuation as syntax.
    print("\n=== query sanitizing ===")
    for hostile in ['what "about" wet puh?', "puh*", "-blood god", "NEAR(a b)", "^ana", "???"]:
        try:
            hits = db.fts_search(hostile, 3)
            print(f"  ok  {hostile!r} -> {len(hits)} hits")
        except Exception as exc:  # noqa: BLE001 - that's the thing being tested
            print(f"  RAISED {hostile!r}: {exc}")
            failures.append(f"{hostile!r} raised {type(exc).__name__}")

    # A multi-word query must not be AND-joined: "wet puh rains blood" shares
    # no single line, but should still recall the lines about each part.
    spread = db.fts_search("wet puh rains blood keyboard", 5)
    print(f"\n  or-joined multi-topic query -> {len(spread)} hits")
    if len(spread) < 3:
        failures.append("multi-topic query recalled almost nothing (AND-joined?)")

    # Deleting content must not leave the index pointing at dead ids.
    print("\n=== index stays in sync after deletion ===")
    doomed = db.conn.execute(
        "SELECT id FROM utterances WHERE text LIKE 'blood for%'"
    ).fetchone()["id"]
    db.conn.execute("DELETE FROM utterances WHERE id = ?", (doomed,))
    db.conn.commit()
    db.rebuild_fts()
    stale = [uid for uid, _ in db.fts_search("blood god", 5) if db.get_utterance(uid) is None]
    print(f"  hits resolving to deleted rows: {len(stale)}")
    if stale:
        failures.append(f"{len(stale)} fts hits point at deleted utterances")

    db.close()
    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("bm25 recalls at least as well as the idf scan, and stays in sync. ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
