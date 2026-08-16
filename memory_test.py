#!/usr/bin/env python3
"""Does semantic recall actually beat word matching?

Teaches Bucket a set of lines, then queries with paraphrases that share no
content words with what it learned. Lexical search should miss; semantic
should hit.

    python memory_test.py
"""

import os
import tempfile

os.environ["BUCKET_LLM_BACKEND"] = "none"  # test memory, not voice

from bucket import Bucket  # noqa: E402

CORPUS = [
    "my favourite food is cheese",
    "i drive a rusty old pickup truck",
    "the cat knocked the lamp off the table again",
    "i work night shifts at the hospital",
    "learning to play guitar is harder than i expected",
    "we went hiking up the mountain last weekend",
]

# (query, which corpus line it should retrieve) — deliberately no shared words.
PARAPHRASES = [
    ("what do you like to eat", "my favourite food is cheese"),
    ("what kind of vehicle do you own", "i drive a rusty old pickup truck"),
    ("tell me about your job", "i work night shifts at the hospital"),
    ("did you climb anything recently", "we went hiking up the mountain last weekend"),
    ("are you picking up a musical instrument", "learning to play guitar is harder than i expected"),
]


def main() -> int:
    path = os.path.join(tempfile.mkdtemp(), "memory.sqlite3")
    bot = Bucket(db_path=path, quiet=True)

    print(f"memory backend: {bot.embedder.status()}")
    if not bot.index:
        print("\nSemantic memory is off, so this test can only show the old behavior.")

    for line in CORPUS:
        uid = bot.learner.ingest(line)
        if uid:
            bot._remember(uid, line)

    def texts(hits):
        out = []
        for uid, _score in hits:
            row = bot.db.get_utterance(uid)
            if row:
                out.append(row["text"])
        return out

    # The brain considers several candidates, not just the best one, so top-3
    # recall is the number that actually matters. Top-1 is reported too.
    scores = {"lexical": [0, 0], "semantic": [0, 0]}
    print(f"\n{'query':<45} {'lexical':<18} {'semantic'}")
    print("-" * 82)

    for query, expected in PARAPHRASES:
        results = {
            "lexical": texts(bot.db.search(query, 3)),
            "semantic": texts(bot._semantic(query, 3)) if bot.index else [],
        }
        cells = []
        for key in ("lexical", "semantic"):
            found = results[key]
            first = bool(found) and found[0] == expected
            within = expected in found
            scores[key][0] += first
            scores[key][1] += within
            cells.append("top-1" if first else ("top-3" if within else "miss"))
        print(f"{query:<45} {cells[0]:<18} {cells[1]}")

        if bot.index and expected not in results["semantic"]:
            print(f"    wanted: {expected}")
            print(f"    got:    {results['semantic'][0] if results['semantic'] else None}")

    total = len(PARAPHRASES)
    print("-" * 82)
    print(f"{'top-1':<45} {scores['lexical'][0]}/{total:<16} {scores['semantic'][0]}/{total}")
    print(f"{'top-3':<45} {scores['lexical'][1]}/{total:<16} {scores['semantic'][1]}/{total}")
    lexical_hits, semantic_hits = scores["lexical"][1], scores["semantic"][1]

    print("\n=== blended ranking used by the brain ===")
    for query, _ in PARAPHRASES[:3]:
        merged = bot.brain._candidates(query, limit=2)
        print(f"\n{query}")
        for uid, score in merged:
            row = bot.db.get_utterance(uid)
            print(f"  {score:.3f}  {row['text'] if row else '?'}")

    bot.close()

    if bot.index and semantic_hits <= lexical_hits:
        print("\nFAILED: semantic recall did not beat lexical recall")
        return 1
    print("\nok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
