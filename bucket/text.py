"""Tokenizing and normalizing. Deliberately dumb, like the original."""

import re

WORD_RE = re.compile(r"[\w'’-]+", re.UNICODE)

# Stuff that carries no signal when matching one line against another.
STOPWORDS = frozenset("""
a an the and or but if then so of to in on at by for with from as is are was
were be been being do does did done have has had i you he she it we they me
him her us them my your his its our their this that these those what who whom
whose which when where why how not no yes ok okay just really very too also
""".split())


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def normalize(text: str) -> str:
    """Canonical form used for dedupe and lookup keys.

    Curly apostrophes are folded to straight ones — phones insert them and
    laptops don't, which otherwise files "i'm" and "i’m" as different subjects.
    """
    return " ".join(tokenize(text.replace("’", "'")))


def content_words(text: str) -> list[str]:
    return [w for w in tokenize(text) if w not in STOPWORDS and len(w) > 1]


# Words that cannot be the last word of a sentence: they exist to introduce
# something that has to follow. A reply ending on one reads as cut off mid-
# thought — "which means dario beats his", "i notice you bought up the".
#
# Deliberately strict. "that", "it", "you", "are" and "those" all *look* like
# function words but end perfectly good sentences ("claude never told me that"),
# so they're excluded. Better to leave a few odd endings than to amputate valid
# ones — this trims text, and anything it removes is gone.
DANGLING_ENDINGS = frozenset("""
a an the and or but nor of to for with in on at as from by like than into onto
upon about my your his her their our its every each another
because which whose whom
""".split())


def trim_dangling(text: str, max_trim: int = 3) -> str:
    """Drop trailing words that can't end a sentence.

    Returns "" if the whole thing was dangling words, which the caller should
    treat as "this strategy produced nothing" rather than sending an empty reply.
    `max_trim` bounds how much is chopped, so a reply that is *mostly* dangling
    words is rejected rather than whittled down to one token.
    """
    words = text.split()
    trimmed = 0
    while words and trimmed < max_trim:
        bare = words[-1].strip(".,!?;:\"'()").lower()
        if bare not in DANGLING_ENDINGS:
            break
        words.pop()
        trimmed += 1
    if not words:
        return ""
    # Ran out of budget with a dangling word still on the end: the line is mostly
    # filler, so reject it rather than hand back a trimmed version that's just as
    # cut off as the original.
    if trimmed >= max_trim and words[-1].strip(".,!?;:\"'()").lower() in DANGLING_ENDINGS:
        return ""
    out = " ".join(words)
    # A trailing comma or "and," left behind by the trim reads worse than the
    # dangling word did.
    return out.rstrip(" ,;:-") if trimmed else out


def looks_like_url(text: str) -> bool:
    return "http://" in text or "https://" in text or "www." in text


def strip_addressing(text: str, name: str, require_punctuation: bool = False) -> str:
    """Remove a leading/trailing 'bucket,' so the rest can be matched cleanly.

    `require_punctuation` makes the leading form need a separator, so "bucket,
    cheese is good" is still treated as addressing but "bucket hates fridays"
    keeps its subject. Fact extraction needs that distinction; reply matching
    doesn't care and strips either way.
    """
    separator = r"[,:;-]" if require_punctuation else r"[,:;-]?"
    pattern = re.compile(
        rf"^\s*{re.escape(name)}\s*{separator}\s*"
        rf"|\s*[,:;-]?\s*{re.escape(name)}\s*[.!?]?\s*$",
        re.IGNORECASE,
    )
    stripped = pattern.sub("", text).strip()
    return stripped or text
