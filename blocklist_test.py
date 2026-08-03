#!/usr/bin/env python3
"""Can Bucket be made to say a name it must never say?

A corpus drawn from a real chat community is worth keeping for its flavour. The
handles in it are not: those people never agreed to be quoted by a bot in a
different chat.

Three enforcement points, and the test covers all three because any one alone
leaks:

  learning   a line naming someone blocked never enters the chain or phrases
  reply      a candidate reply naming someone blocked is rejected
  display    the introspection commands redact rather than print

The interesting case is the third-party one. A name can reach the live chain
because Bucket said it and a *real, present-day user* asked "who is oldname_7" —
so the name arrived via a reinforcing speaker in a live chat, where no author or
chat filter can see it. Filtering by text is the only thing that catches that.

The names below are synthetic. `DEFAULT_BLOCKED` ships empty, so this also
exercises the path every real deployment uses: names supplied through
BUCKET_BLOCKED_NAMES. It has to be set before `bucket` is imported, since
config reads the environment once at import time.

    python blocklist_test.py
"""

import json
import os
import sys
import tempfile

os.environ["BUCKET_LLM_BACKEND"] = "none"
os.environ["BUCKET_EMBED_BACKEND"] = "none"
os.environ.setdefault("BUCKET_SEED", "0")
os.environ["BUCKET_BLOCKED_NAMES"] = "pengu, oldname_7, corvid"

from bucket import Bucket  # noqa: E402
from bucket.blocklist import Blocklist  # noqa: E402
from bucket.db import SEP  # noqa: E402


def main() -> int:
    failures = []

    # --- 1. whole-token matching ----------------------------------------
    print("=== matches whole names, not substrings ===")
    bl = Blocklist({"pengu", "oldname_7", "corvid"})
    cases = [
        ("ALL HAIL PENGU", True),
        ("penguin is a snack", False),        # substring must not match
        ("who is oldname_7", True),
        ("oldname on its own", False),        # underscore matters
        ("i said corvid!", True),
        ("corvids", False),
        ("nothing to see", False),
    ]
    for text, expected in cases:
        got = bl.blocks(text)
        mark = "ok " if got == expected else "BAD"
        print(f"  {mark} {text!r} -> blocked={got}")
        if got != expected:
            failures.append(f"{text!r} blocked={got}, wanted {expected}")

    # --- 2. it never enters generation ----------------------------------
    print("\n=== a blocked name can't get into the chain or phrases ===")
    path = os.path.join(tempfile.mkdtemp(), "blocked.sqlite3")
    bot = Bucket(db_path=path, quiet=True)

    # A live, reinforcing, present-day speaker asking about a blocked name. This
    # is the exact shape that leaked before: the filter has to look at the text,
    # because the author and chat are both entirely legitimate.
    for line in ["who is oldname_7", "oldname_7 is my friend", "pengu demands blood"]:
        bot.handle(line, author="Elle", chat="dc:1", is_private=True)

    # Something unblocked, so the corpus isn't empty and the comparison is fair.
    for line in ["blood for the blood god", "wet puh is good", "ana has wet puh"]:
        bot.handle(line, author="Elle", chat="dc:1", is_private=True)

    # A blocked handle as an *author*. This is the shape /about turns into a
    # targeted dump: the name is the thing you look someone up by, and their
    # lines come back verbatim. The line text itself is innocuous on purpose —
    # the leak is the attribution, not the content.
    for line in ["the server room floods when it rains", "i hate mondays"]:
        bot.handle(line, author="corvid", chat="dc:1", is_private=True)

    from bucket.learn import BLOCKED as LEARN_BLOCKED

    chain_leaks = [
        (r["context"], r["word"]) for r in bot.db.conn.execute("SELECT context, word FROM chain")
        if LEARN_BLOCKED.blocks(r["context"].replace(SEP, " "))
        or LEARN_BLOCKED.blocks(r["word"])
    ]
    phrase_leaks = [
        r["text"] for r in bot.db.conn.execute("SELECT text FROM phrases")
        if LEARN_BLOCKED.blocks(r["text"])
    ]
    fact_leaks = [
        dict(r) for r in bot.db.conn.execute("SELECT subject, verb, object FROM factoids")
        if LEARN_BLOCKED.blocks(f"{r['subject']} {r['verb']} {r['object']}")
    ]
    print(f"  chain edges naming one : {len(chain_leaks)}")
    print(f"  phrases naming one     : {len(phrase_leaks)}")
    print(f"  factoids naming one    : {len(fact_leaks)}")
    for label, leaks in (("chain", chain_leaks), ("phrase", phrase_leaks),
                         ("factoid", fact_leaks)):
        if leaks:
            failures.append(f"{len(leaks)} blocked names reached the {label} table: {leaks[:3]}")

    # The unblocked content must still be there — the point is to remove names,
    # not to gut the corpus.
    kept = bot.db.conn.execute(
        "SELECT COUNT(*) n FROM phrases WHERE text LIKE '%blood%'"
    ).fetchone()["n"]
    print(f"  'blood' phrases kept   : {kept}")
    if kept == 0:
        failures.append("scrubbing names also removed unrelated content")

    # --- 3. it never comes out of a reply -------------------------------
    # Even with the names stored (they are — only learning and replying filter),
    # no amount of prompting should surface one.
    print("\n=== 200 replies, none may name a blocked person ===")
    said = []
    for i in range(200):
        out = bot.speak("who is oldname_7 and pengu", chat="dc:1")
        if out:
            said.append(out)
    offenders = [s for s in said if LEARN_BLOCKED.blocks(s)]
    print(f"  replies generated : {len(said)}")
    print(f"  naming someone    : {len(offenders)}")
    for s in offenders[:3]:
        print(f"      {s[:70]}")
    if offenders:
        failures.append(f"{len(offenders)}/{len(said)} replies named a blocked person")

    # --- 4. introspection redacts --------------------------------------
    # Every command that prints corpus text or factoid columns, not just the two
    # that originally did it. /about and /who both shipped unredacted: /about
    # dumps an author's lines on request, and /who prints factoid subjects
    # straight out of the table.
    print("\n=== commands redact instead of printing ===")
    checks = [
        ("literal", "oldname_7"),
        ("recall", "oldname_7"),
        ("about", "corvid"),
        ("about", "oldname_7"),
        ("who", "my friend"),
        ("who", "demands blood"),
    ]
    for cmd, arg in checks:
        out = getattr(bot, f"cmd_{cmd}")(arg)
        leaked = LEARN_BLOCKED.hits(out)
        print(f"  /{cmd} {arg!r:<16} -> leaks: {sorted(leaked) or 'nothing'}")
        if leaked:
            failures.append(f"/{cmd} {arg!r} printed {sorted(leaked)}")
    print(f"\n  /about corvid output:\n      " + "\n      ".join(
        bot.cmd_about("corvid").splitlines()[:4]))

    # --- 4b. the mini app is the same corpus over http ------------------
    # Same material, different surface. These return raw rows, so they need the
    # same redaction the chat commands have.
    print("\n=== mini app endpoints redact ===")
    from webapp.server import Handler  # noqa: E402

    api = Handler.__new__(Handler)
    Handler.bot = bot
    payloads = {
        "/api/people": api._people(),
        "/api/about?name=corvid": api._about("corvid"),
        "/api/about?name=oldname_7": api._about("oldname_7"),
        "/api/facts": api._facts("", 200),
        "/api/recall?q=friend": api._recall("friend"),
        "explain()": bot.explain("who is oldname_7", learn=False),
    }
    for name, payload in payloads.items():
        leaked = LEARN_BLOCKED.hits(json.dumps(payload))
        print(f"  {name:<26} leaks: {sorted(leaked) or 'nothing'}")
        if leaked:
            failures.append(f"{name} returned {sorted(leaked)}")

    # --- 5. non-people are never credited ------------------------------
    # "webapp says ana has wet puh" — webapp is the Mini App's author tag.
    print("\n=== 'webapp' is not a person ===")
    bot.handle("dario is an abuser", author="webapp", chat="dc:1", is_private=True)
    attributions = []
    for _ in range(120):
        reply = bot.brain._from_attribution("dario")
        if reply:
            attributions.append(reply.text)
    bad = [a for a in attributions
           if a.lower().startswith(("webapp", "seed", "bucket2008"))
           or " webapp " in f" {a.lower()} "]
    print(f"  attribution replies : {len(attributions)}")
    print(f"  crediting a non-person : {len(bad)}")
    for a in bad[:3]:
        print(f"      {a[:70]}")
    if bad:
        failures.append(f"{len(bad)} attributions credited a non-person author")

    bot.close()
    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("blocked names don't enter generation, don't come out, and aren't printed. ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
