"""Names Bucket must never say.

The 2008 transcripts are full of people who never consented to being quoted by a
bot in a different chat eighteen years later. Their handles are the one part of
that material that shouldn't come back out.

Two enforcement points, because either alone leaks:

  learning    a blocked name never enters the chain, phrases or factoids, so
              composition can't emit it in the first place
  reply       any candidate reply containing one is rejected, which catches names
              that got into the corpus before they were blocked, or that arrive
              via live conversation

The second is not redundant. `cypress_z` reached the live chain because a real
user asked "who is cypress_z" after Bucket said it — a legacy name laundered
through a present-day speaker, where no author- or chat-based filter can see it.

Matching is whole-token and case-insensitive, on the same tokenization the rest
of the engine uses, so "khorne" is blocked but "khornetto" is not. Substring
matching would silently eat innocent words.
"""

import re

# Speakers in the imported 2008 logs. Exact, from import_legacy.py's parse of the
# `:Speaker:` labels — not guessed from capitalization, which is meaningless in
# chat logs where people SHOUT and Capitalize At Random.
LEGACY_SPEAKERS = frozenset({
    "orkkaptin", "terranarachnid", "wibble", "richy", "bucket2008",
    "khorne", "moogle",
})

# Names that appear *inside* the legacy transcripts rather than as speakers, so
# no author or chat filter can reach them. Add to this as they surface; there is
# no reliable way to extract them automatically (token-diffing the two corpora
# fails once Bucket has said one out loud, and capitalization heuristics flag
# "blood", "death" and "hey" as names).
LEGACY_MENTIONED = frozenset({
    "cypress_z", "ccrraaiikkyy3333", "orgmemberswinxx001",
})

# Deliberately excluded, despite being legacy speaker labels: "You", "The Spy"
# and "The Scout" are a pronoun and TF2 class names, not personal handles.
# Blocking "you" would make most of the corpus unsayable.

# Sources that are not people, and so must never be named as one. Bucket used to
# say "webapp says ana has wet puh" — `webapp` is the author tag the Mini App
# writes, not a person, and neither is `seed` (scaffolding) or `bucket2008` (a
# dead bot). Distinct from the blocklist above: lines from these authors are still
# learned from normally, they just can't be *credited* to anyone.
# config.NAME is added at the use site, since it's configurable.
NON_PERSON_AUTHORS = frozenset({"webapp", "seed", "bucket2008", "console"})

DEFAULT_BLOCKED = LEGACY_SPEAKERS | LEGACY_MENTIONED

# Same shape as text.tokenize's word notion, but keeps underscores and digits so
# handles like `cypress_z` survive as one token.
_TOKEN_RE = re.compile(r"[a-z0-9_']+")


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class Blocklist:
    """A set of names, checked against text as whole tokens."""

    def __init__(self, names=DEFAULT_BLOCKED):
        self.names = frozenset(n.strip().lower() for n in names if n and n.strip())

    def __bool__(self) -> bool:
        # Explicit, so an empty blocklist isn't confused with "not configured".
        return True

    def hits(self, text: str) -> set[str]:
        """Which blocked names appear in this text."""
        if not self.names:
            return set()
        return self.names & set(tokens(text))

    def blocks(self, text: str) -> bool:
        return bool(self.hits(text))
