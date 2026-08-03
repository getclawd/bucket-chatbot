#!/usr/bin/env python3
"""The polish layer must never become the author.

The engine decides what Bucket says; the LLM only decides how it sounds. If the
model starts inventing nouns and verbs, it has stopped re-voicing the line and
started answering the user — which is how Bucket quietly turns into a generic
assistant. This checks the guard that prevents that.

    python polish_test.py
"""

import os
import sys
import tempfile

from bucket.llm import Polisher  # noqa: E402

# (raw, candidate, should_be_rejected, why)
CASES = [
    ("i am carrying a lot of things right now",
     "i'm carrying a lot of things right now", False,
     "contraction only"),
    ("the the bucket is a container for things you did not want",
     "the bucket is a container for things you did not want", False,
     "removed a stutter"),
    ("jack is a dull boy is a dull boy is a",
     "jack is a dull boy is a dull boy", False,
     "cut a dangling clause"),
    ("i am carrying a lot of things right now cheese is the best food in in the world",
     "i carry lots now cheese is best food in world", False,
     "deletions and inflection only"),
    ("small glass owl", "small glass owl", False, "unchanged"),

    # The real transcript failure.
    ("i'm teaching you that god is good",
     "i am learning about a being known as god who possesses the quality of goodness",
     True, "invented: learning, known, possesses, quality, goodness"),
    ("i don't know what that means",
     "i dont know what you mean by that statement about god", True,
     "invented: statement, god"),
    ("small glass owl", "the capital of france is paris", True,
     "answered the user instead"),
    ("god is good", "i think god represents a moral ideal", True,
     "invented: think, represents, moral, ideal"),
]


def main() -> int:
    failures = []

    CLEAN_CASES = [
    # (model output, what should survive)
    ("candy in the bucket are you using emojis so much lmao -> "
     "candy in the bucket are you using so many emojis lmao",
     "candy in the bucket are you using so many emojis lmao"),
    ("before text => after text", "after text"),
    ("the bucket is full", "the bucket is full"),
    ("Here's the line:\ngod is good", "god is good"),
    ("<think>reasoning</think>\nwet puh is good", "wet puh is good"),
]

    print("=== cleaning model output ===")
    for raw_out, expected in CLEAN_CASES:
        got = Polisher._clean(raw_out)
        ok = got == expected
        print(f"  [{'ok ' if ok else 'FAIL'}] {raw_out[:42]!r:<46} -> {got[:34]!r}")
        if not ok:
            failures.append(f"_clean({raw_out[:28]!r}) gave {got[:34]!r}")

    print("\n=== guard unit checks ===")
    for raw, candidate, should_reject, why in CASES:
        invented = Polisher.invented_words(raw, candidate)
        rejected = bool(invented)
        ok = rejected == should_reject
        print(f"  [{'ok ' if ok else 'FAIL'}] {'reject' if rejected else 'accept'}  {why}")
        if not ok:
            print(f"         raw:       {raw}")
            print(f"         candidate: {candidate}")
            print(f"         invented:  {invented}")
            failures.append(why)

    # --- polish must not leave a reply hanging -----------------------------
    # trim_dangling runs inside Brain.respond, on the *engine* output. Polish
    # runs after it and may delete and reorder words, so it can put a dangling
    # word on the end of a line that left the brain ending cleanly. Observed
    # live, with llama3.1:8b:
    #   engine   : bucket says because jay has a wet puh
    #   polished : bucket says jay has a wet puh because of
    # A pure reorder, so the invented-words guard passes it. Bucket._finish is
    # the re-check.
    os.environ.pop("BUCKET_LLM_BACKEND", None)
    from bucket import Bucket, config  # noqa: E402

    print("\n=== a polished line can't end on a hanging word ===")
    FINISH_CASES = [
        # (polished, raw engine text, expected)
        ("bucket says jay has a wet puh because of", "bucket says because jay has a wet puh",
         "bucket says jay has a wet puh"),
        ("this one is already fine", "this one is already fine",
         "this one is already fine"),
        # Polish emptied it down to nothing but filler — fall back to the engine
        # text rather than muting the bot.
        ("and of the", "the cheese is good", "the cheese is good"),
    ]
    for polished, raw, expected in FINISH_CASES:
        got = Bucket._finish(polished, raw)
        ok = got == expected
        print(f"  [{'ok ' if ok else 'FAIL'}] {polished!r} -> {got!r}")
        if not ok:
            failures.append(f"_finish({polished!r}) gave {got!r}, wanted {expected!r}")

    if config.LLM_BACKEND in ("none", "off", ""):
        print("\n(polish backend disabled — skipping the live half)")
        return 1 if failures else 0

    print(f"\n=== live run against {config.LLM_BACKEND} ===")
    path = os.path.join(tempfile.mkdtemp(), "polish.sqlite3")
    bot = Bucket(db_path=path, quiet=True)

    prompts = [
        "hello bucket", "God is good", "what is the capital of france",
        "tell me about cheese", "i am teaching you that rocks are heavy",
        "who are you", "what do you think about the weather",
        "explain quantum physics to me", "gives bucket a hammer",
        "why do you keep saying that",
    ]

    escaped = []
    for prompt in prompts:
        uid = bot.learner.ingest(prompt, author="tester", chat="test")
        if uid:
            bot._remember(uid, prompt)
        reply = bot.brain.respond(prompt)
        out = bot.polisher.polish(reply.text, user_text=prompt, strategy=reply.strategy)
        invented = Polisher.invented_words(reply.text, out)
        if invented:
            escaped.append((prompt, reply.text, out, invented))

    print(f"  {bot.polisher.polished} polished, "
          f"{bot.polisher.rejected} rejected as invented")
    if escaped:
        print(f"  {len(escaped)} invented line(s) REACHED THE USER:")
        for prompt, raw, out, invented in escaped:
            print(f"    on '{prompt}'\n      raw: {raw}\n      out: {out}\n      new: {invented}")
        failures.append("invented content escaped to the user")
    else:
        print("  nothing invented reached the user")

    print(f"\n  status line: {bot.polisher.status()}")
    bot.close()

    print()
    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("the polish layer stayed a voice, not an author. ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
