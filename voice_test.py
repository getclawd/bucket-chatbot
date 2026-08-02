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
import sys

os.environ.setdefault("BUCKET_LLM_BACKEND", "none")  # judge the engine, not the voice

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


def main() -> int:
    bot = Bucket(quiet=True)
    if bot.db.stats()["utterances"] < 20:
        print("corpus is too small to judge; talk to it first.")
        return 0

    print(f"corpus: {bot.db.stats()['utterances']} lines, "
          f"{bot.db.stats()['factoids']} facts\n")

    strategies: dict[str, int] = {}
    parroted = offtopic = 0
    total = 0

    print("=" * 78)
    for prompt in PROMPTS:
        print(f"\nyou:    {prompt}")
        for _ in range(3):
            reply = bot.brain.respond(prompt)
            total += 1
            strategies[reply.strategy] = strategies.get(reply.strategy, 0) + 1

            said = set(content_words(prompt))
            mine = set(content_words(reply.text))
            shares_topic = bool(said & mine)
            if not shares_topic:
                offtopic += 1
            if mine and normalize(reply.text) == normalize(prompt):
                parroted += 1

            flag = "" if shares_topic else "   (no shared topic)"
            print(f"bucket: {reply.text}")
            print(f"        [{reply.strategy}]{flag}")

    print("\n" + "=" * 78)
    print("strategy mix:")
    composing = {"mashup", "tangent", "collision", "attribution", "phrase"}
    composed = sum(n for s, n in strategies.items() if s in composing)
    for name, count in sorted(strategies.items(), key=lambda kv: -kv[1]):
        mark = " *composed" if name in composing else ""
        print(f"  {name:<12} {count:>3}  ({100 * count / total:.0f}%){mark}")

    print(f"\n  composed replies : {composed}/{total} ({100 * composed / total:.0f}%)")
    print(f"  parroted input   : {parroted}/{total}")
    print(f"  no shared topic  : {offtopic}/{total} ({100 * offtopic / total:.0f}%)")

    bot.close()

    failures = []
    if parroted:
        failures.append(f"{parroted} replies were the input handed straight back")
    if composed < total * 0.35:
        failures.append(f"only {100 * composed / total:.0f}% of replies mixed anything")
    if offtopic > total * 0.6:
        failures.append(f"{100 * offtopic / total:.0f}% shared no topic — that's noise, not insanity")

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
