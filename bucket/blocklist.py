"""Names Bucket must never say.

Any corpus drawn from a real chat community contains handles belonging to people
who never agreed to be quoted by a bot — someone who left and asked to be
scrubbed, a handle that is also a legal name, an old name someone no longer uses.
Their words can stay. Their names shouldn't come back out.

Set `BUCKET_BLOCKED_NAMES` to the handles your corpus needs suppressed. Three
enforcement points, because any one alone leaks:

  learning    a blocked name never enters the chain, phrases or factoids, so
              composition can't emit it in the first place
  reply       any candidate reply containing one is rejected, which catches names
              that got into the corpus before they were blocked, or that arrive
              via live conversation
  display     the introspection commands print corpus text verbatim, so they
               redact — including the subject the user asked about

The reply guard is not redundant with the learning guard. A name already in the
corpus can be repeated by a present-day message, so matching is on the message
*text* itself.

Matching is whole-token and case-insensitive, on the same tokenization the rest
of the engine uses, so "pengu" is blocked but "penguin" is not. Substring
matching would silently eat innocent words.
"""

import re

# Ships empty on purpose. A blocklist is data about one specific corpus: the
# handles that matter are the ones in *your* logs, and nobody else's deployment
# is helped by carrying them. Hardcoding them here would also publish the exact
# list of names the feature exists to keep unpublished.
#
# Populate it per-deployment via BUCKET_BLOCKED_NAMES in .env, which is unioned
# in at every use site. Run scrub_legacy.py --apply after adding names to purge
# them from the chain, phrases and factoids retroactively.
#
# Pick whole handles, not common words: matching is per-token, so blocking a
# pronoun or an ordinary noun makes much of the corpus unsayable.
DEFAULT_BLOCKED = frozenset()

# Same shape as text.tokenize's word notion, but keeps underscores and digits so
# handles like `oldname_7` survive as one token.
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
