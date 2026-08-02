#!/usr/bin/env python3
"""Can it reach facts by meaning, and resolve a pronoun to an earlier turn?

Two measured gaps this covers:
  * facts were findable only by substring-matching the subject, so "who is
    violent" could never reach "dario is an abuser"
  * replies saw only the current message, so "what about him" retrieved the
    word "what"

    python recall_test.py
"""

import os
import sys
import tempfile

os.environ["BUCKET_LLM_BACKEND"] = "none"
os.environ.setdefault("BUCKET_SEED", "0")

from bucket import Bucket  # noqa: E402

FACTS = [
    ("dario is an abuser", "Cleo"),
    ("dario beats claude like a puppy", "Ben"),
    ("claude can be trusted with nuclear codes", "Cleo"),
    ("ana has wet puh", "Elle"),
    ("the server room is flooded every time it rains", "Elle"),
    ("dima is a strange little bot", "Ana"),
]

# (query, a word that should appear in the fact it finds)
SEMANTIC = [
    ("who is violent", "dario"),
    ("who hurts animals", "dario"),
    ("who can be relied on", "claude"),
    ("what gets wet", "puh"),
    ("what happens in bad weather", "flooded"),
]


def main() -> int:
    path = os.path.join(tempfile.mkdtemp(), "recall.sqlite3")
    bot = Bucket(db_path=path, quiet=True)
    failures = []

    if bot.fact_index is None:
        print("no embedding model — semantic fact lookup can't be tested")
        return 0

    # learn_only embeds as well as stores; learner.ingest alone would leave the
    # utterance index empty and semantic recall with nothing to search.
    for line, who in FACTS:
        bot.learn_only(line, author=who, chat="t")
    bot._remember_facts()
    print(f"learned {bot.db.stats()['factoids']} facts\n")

    print("=== finding facts by meaning, not by name ===")
    for query, expected in SEMANTIC:
        hits = bot._semantic_facts(query, 3)
        rows = [bot.db.get_factoid(fid) for fid, _ in hits]
        texts = [f"{r['subject']} {r['verb']} {r['object']}" for r in rows if r]
        found = any(expected in t.lower() for t in texts)

        # The old path: does the subject appear literally in the query?
        lexical = bool(bot.db.known_subjects_in(query))

        print(f"  [{'ok ' if found else 'FAIL'}] {query:<28} "
              f"subject-match:{'hit' if lexical else 'miss':<5} "
              f"semantic:{texts[0][:34] if texts else '-'}")
        if not found:
            failures.append(f"{query!r} did not reach a fact about {expected!r}")

    print("\n=== a pronoun resolving to an earlier turn ===")
    bot._note_turn("t", "dario beats claude like a puppy")
    bare = bot.context_for("t", "why does he do that")
    print(f"  context supplied for 'why does he do that': {bare!r}")
    if not bare:
        failures.append("no context supplied for a pronoun-only message")

    with_ctx = bot._semantic(f"{bare} why does he do that", 3)
    without = bot._semantic("why does he do that", 3)

    def top(hits):
        row = bot.db.get_utterance(hits[0][0]) if hits else None
        return row["text"] if row else "-"

    print(f"  without context -> {top(without)}")
    print(f"  with context    -> {top(with_ctx)}")
    if "dario" not in top(with_ctx).lower():
        failures.append("context did not steer recall toward the antecedent")

    print("\n=== a self-contained message must NOT drag in context ===")
    full = bot.context_for("t", "tell me everything about the nuclear codes please")
    print(f"  context supplied: {full!r}")
    if full:
        failures.append("context was folded into a message that stood on its own")

    bot.close()

    print()
    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("facts reachable by meaning, pronouns resolved. ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
