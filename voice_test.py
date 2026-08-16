#!/usr/bin/env python3
"""Does it sound like Bucket, or does it sound broken?

The 2008 original was *semi*-coherent: on-topic material in the wrong shape,
stated with total confidence. Two failure modes sit either side of that — noise
(nothing to do with what you said) and parroting (your own sentence handed back).
This measures both, and prints samples so you can judge the middle.

    python voice_test.py            # against your real corpus, read only
    python voice_test.py --samples  # just show me conversations
"""

import os
import random
import sys

os.environ.setdefault("BUCKET_LLM_BACKEND", "none")  # judge the engine, not the voice
# Keep this probe offline and reproducible. Semantic recall is tested separately;
# this test should not depend on an Ollama service or a live model.
os.environ.setdefault("BUCKET_EMBED_BACKEND", "none")

from bucket import Bucket, config  # noqa: E402
from bucket.text import content_words, normalize  # noqa: E402

PROMPTS = [
    "what do you think about god",
    "ana has wet puh",
    "tell me about claude",
    "who is dario",
    "i met dima, hes strange",
    "what are you carrying",
    "do you remember the nuclear codes",
    "say something new",
    "why are you like this",
    "cheese is good",
    # Ends with the bot's own name, so strip_addressing removes it before the
    # parrot check sees the sentence. That mismatch let exact echoes through.
    "put candy in the bucket",
    "bucket are you better than the 2008 bucket?",
]

RANDOM_SEED = 0
SAMPLES_PER_PROMPT = 6


def main() -> int:
    # The engine uses both Python's RNG and SQLite's RANDOM() function. Seeding
    # only Python left ORDER BY RANDOM() nondeterministic, so identical CI runs
    # could land on different strategy mixes.
    random.seed(RANDOM_SEED)
    bot = Bucket(quiet=True)
    sqlite_rng = random.Random(RANDOM_SEED)
    bot.db.conn.create_function(
        "random",
        0,
        lambda: sqlite_rng.randint(-(2**63), 2**63 - 1),
    )
    if bot.db.stats()["utterances"] < 20:
        print("corpus is too small to judge; talk to it first.")
        return 0

    print(f"corpus: {bot.db.stats()['utterances']} lines, "
          f"{bot.db.stats()['factoids']} facts\n")

    strategies: dict[str, int] = {}
    parroted = unanchored = no_recall = recallable = 0
    total = 0

    print("=" * 78)
    for prompt in PROMPTS:
        print(f"\nyou:    {prompt}")
        # A fresh seed corpus has never heard words such as "claude" or
        # "nuclear". A reply to those prompts cannot be judged by exact lexical
        # overlap, so use the material actually retrieved for the prompt as the
        # anchor set and report no-recall prompts separately.
        candidates = bot.brain._candidates(prompt, limit=6)
        anchor_words = set(content_words(prompt))
        for uid, _score in candidates:
            row = bot.db.get_utterance(uid)
            if row:
                anchor_words.update(content_words(row["text"]))
        # Six samples per prompt smooth the intended strategy distribution. At
        # three, one reply moves the composition score by almost three points
        # and routinely flips the 35% release gate despite unchanged behavior.
        for _ in range(SAMPLES_PER_PROMPT):
            reply = bot.brain.respond(prompt)
            total += 1
            strategies[reply.strategy] = strategies.get(reply.strategy, 0) + 1

            said = set(content_words(prompt))
            mine = set(content_words(reply.text))
            shares_prompt = bool(said & mine)
            has_anchor = bool(anchor_words & mine)
            if candidates:
                recallable += 1
                if not has_anchor:
                    unanchored += 1
            else:
                no_recall += 1
            if mine and normalize(reply.text) == normalize(prompt):
                parroted += 1

            if not candidates:
                flag = "   (no corpus recall)"
            elif not has_anchor:
                flag = "   (no corpus anchor)"
            elif not shares_prompt:
                flag = "   (anchored via recall)"
            else:
                flag = ""
            print(f"bucket: {reply.text}")
            print(f"        [{reply.strategy}]{flag}")

    print("\n" + "=" * 78)
    print("strategy mix:")
    composing = {"mashup", "tangent", "collision", "phrase"}
    composed = sum(n for s, n in strategies.items() if s in composing)
    for name, count in sorted(strategies.items(), key=lambda kv: -kv[1]):
        mark = " *composed" if name in composing else ""
        print(f"  {name:<12} {count:>3}  ({100 * count / total:.0f}%){mark}")

    print(f"\n  composed replies : {composed}/{total} ({100 * composed / total:.0f}%)")
    print(f"  parroted input   : {parroted}/{total}")
    if recallable:
        anchored = recallable - unanchored
        print(f"  corpus anchors   : {anchored}/{recallable}"
              f" ({100 * anchored / recallable:.0f}% of recallable replies)")
    else:
        print("  corpus anchors   : no recallable prompts")
    print(f"  no corpus recall : {no_recall}/{total} ({100 * no_recall / total:.0f}%)")

    bot.close()

    failures = []
    if parroted:
        failures.append(f"{parroted} replies were the input handed straight back")
    if composed < total * 0.35:
        failures.append(f"only {100 * composed / total:.0f}% of replies mixed anything")
    if recallable and unanchored > recallable * 0.6:
        failures.append(
            f"{100 * unanchored / recallable:.0f}% of recallable replies ignored their corpus anchor"
        )

    print()
    if failures:
        print("NEEDS WORK:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("on topic, badly assembled, said with confidence. ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
