"""Ingestion. Everything anyone says gets shredded into the database.

There is no filter here by design — this is the part of the original that made
it famous. See README for how to turn the chaos down if you change your mind.
"""

import re
from dataclasses import dataclass

from .blocklist import DEFAULT_BLOCKED, Blocklist
from .config import config
from .db import BucketDB
from .text import STOPWORDS, looks_like_url, normalize, strip_addressing, tokenize

# Names that must never be learned from, so composition can't emit them. Built
# once from the legacy-transcript list plus BUCKET_BLOCKED_NAMES; see blocklist.py.
BLOCKED = Blocklist(DEFAULT_BLOCKED | config.BLOCKED_NAMES)

# Authors whose lines are stored and recallable but never learned *from*.
#
# "seed" is scaffolding. "bucket2008" is imported transcript of the original
# Bucket — which was itself markov output, so feeding it back into a markov
# chain compounds the degradation it already has. Same reasoning as the
# `reinforce=False` path for this bot's own replies; see Learner.ingest.
# The *humans* in those transcripts are ordinary authors and do reinforce.
NON_REINFORCING_AUTHORS = frozenset({"seed", "bucket2008"})

# Predicates Bucket recognises as "someone is telling me a fact".
#
# Order matters: the alternation is matched left to right, so multi-word and
# negated forms must come before the bare verb they contain, or "does have"
# matches as "does" and the object becomes "have wet puh".
FACTOID_VERBS = [
    # Negated and auxiliary forms first. Apostrophe-less spellings are listed
    # explicitly because nobody punctuates in chat — without "wont" here,
    # "ana wont be here" parses as subject "ana wont", verb "be".
    "does not have", "doesn't have", "doesnt have",
    "do not have", "don't have", "dont have",
    "did not have", "didn't have", "didnt have",
    "does have", "do have", "did have",
    "is not", "isn't", "isnt", "are not", "aren't", "arent", "ain't", "aint",
    "was not", "wasn't", "wasnt", "were not", "weren't", "werent",
    "has not", "hasn't", "hasnt", "have not", "haven't", "havent",
    "can not", "cannot", "can't", "cant",
    "will not", "won't", "wont",
    "should not", "shouldn't", "shouldnt",
    "would not", "wouldn't", "wouldnt",
    "could not", "couldn't", "couldnt",
    "used to be", "used to",
    # Linking. Bare "be" is deliberately absent: as a standalone predicate it is
    # rare, and it swallows the tail of any auxiliary spelling we failed to list.
    "is", "are", "was", "were",
    # possession
    "has", "have", "had", "owns", "own", "carries", "carry",
    # preference and state
    "likes", "like", "loves", "love", "hates", "hate",
    "wants", "want", "needs", "need", "fears", "fear",
    "knows", "know", "remembers", "remember",
    "keeps", "keep", "becomes", "become", "stays", "stay",
    # perception — multi-word forms win over the bare verb by length
    "smells like", "sounds like", "looks like", "feels like", "seems like",
    "smells", "sounds", "looks", "feels", "seems", "seem",
    "lives in", "comes from", "works at", "belongs to",
    # definitional
    "means", "equals", "stands for", "refers to",
    # modal
    "can", "will", "should", "must", "might", "always", "never",
]

# Longest first so "is not" wins over "is"; then a stable order for equal lengths.
_VERB_ALTERNATION = "|".join(
    re.escape(v) for v in sorted(FACTOID_VERBS, key=lambda v: (-len(v), v))
)

FACTOID_RE = re.compile(
    rf"^(?P<subject>.{{2,60}}?)\s+(?P<verb>{_VERB_ALTERNATION})\s+(?P<object>.{{2,300}})$",
    re.IGNORECASE,
)

# Written back in a canonical form so /literal reads consistently.
# Discourse filler that rides along on the end of a claim and isn't part of it.
# "ana does have wet puh remember that" is a fact about wet puh, not about
# remembering. Stripped repeatedly, so "..., silly lol" comes off in one pass.
TRAILING_FILLER = re.compile(
    r"[\s,.!-]*\b("
    r"remember that|remember|you know|ya know|i think|i guess|i mean|or something|"
    r"apparently|honestly|actually|basically|seriously|literally|"
    r"lol|lmao|lmfao|rofl|haha+|hehe+|xd|smh|fr|ngl|imo|imho|btw|tbh|"
    r"though|tho|right|yeah|yea|ok|okay|k|"
    r"silly|bro|bruh|man|dude|mate|kid|fam|sis|"
    r"no cap|for real|i swear|trust me|periodt|deadass"
    r")\b[\s,.!?-]*$",
    re.IGNORECASE,
)

LEADING_FILLER = re.compile(
    r"^[\s,]*\b("
    r"remember that|remember|listen|look|see|i mean|i think|you know|"
    r"like|so|well|and|but|honestly|actually|basically|"
    r"ok|okay|yeah|yea|yes|yep|nah|no|oh|hey|um|uh|erm|lol|wait"
    r")\b[\s,]+",
    re.IGNORECASE,
)


# Fields that show up in image-generation dumps and log pastes. Several of them
# together, with numbers, means a machine wrote it.
MACHINE_MARKERS = frozenset(
    "seed steps cfg sampler scheduler checkpoint model prompt negative "
    "denoising clipskip lora vae resolution size iterations tokens "
    "traceback exception stacktrace http https www href".split()
)


def is_machine_output(text: str) -> bool:
    """Is this a paste rather than something a person said?

    A 200-word Stable Diffusion parameter dump is not conversation, and learning
    it verbatim means reciting it verbatim — which is exactly what happened.
    """
    words = tokenize(text)
    if len(words) > config.MAX_LEARN_WORDS:
        return True
    # Shorter, but structured like metadata.
    markers = sum(1 for w in set(words) if w in MACHINE_MARKERS)
    digits = sum(1 for w in words if w.isdigit())
    return markers >= 3 and digits >= 2


# Markup and quote artifacts that ride along from chat clients: a stray ">" from
# a quoted reply, a "\" from an escape, leftover markdown. Stored inside a fact
# they resurface every time it's recited ("dario is an abuser>").
EDGE_JUNK = re.compile(r"^[\s>|\\*_~`^\-–—]+|[\s>|\\*_~`^\-–—]+$")


def tidy(text: str) -> str:
    """Trim markup litter from the edges of a claim, leaving the words alone."""
    text = EDGE_JUNK.sub("", text.strip())
    # An opening bracket with no partner is litter too.
    for opener, closer in (("(", ")"), ("[", "]"), ("{", "}")):
        if text.count(opener) != text.count(closer):
            text = text.replace(opener, "").replace(closer, "")
    return text.strip()


def strip_filler(text: str) -> str:
    """Peel discourse filler and markup litter off both ends of a claim."""
    previous = None
    while previous != text:
        previous = text
        text = TRAILING_FILLER.sub("", text).strip()
        text = LEADING_FILLER.sub("", text).strip()
        text = tidy(text)
    return text


VERB_CANONICAL = {
    # Emphatic forms mean the same thing as the plain verb, and collapsing them
    # keeps "ana does have X" and "ana has X" as one fact instead of two.
    "does have": "has", "do have": "have", "did have": "had",
    "isn't": "is not", "isnt": "is not",
    "aren't": "are not", "arent": "are not",
    "ain't": "is not", "aint": "is not",
    "wasn't": "was not", "wasnt": "was not",
    "weren't": "were not", "werent": "were not",
    "hasn't": "has not", "hasnt": "has not",
    "haven't": "have not", "havent": "have not",
    "doesn't have": "does not have", "doesnt have": "does not have",
    "don't have": "do not have", "dont have": "do not have",
    "didn't have": "did not have", "didnt have": "did not have",
    "can't": "can not", "cant": "can not", "cannot": "can not",
    "won't": "will not", "wont": "will not",
    "shouldn't": "should not", "shouldnt": "should not",
    "wouldn't": "would not", "wouldnt": "would not",
    "couldn't": "could not", "couldnt": "could not",
}

# "bucket, X is Y" — an explicit teach, which we weight the same as anything else
TEACH_RE = re.compile(r"^\s*\S+\s*[,:]\s*(?P<rest>.+)$")

# Inventory is deliberately parsed as a small event grammar rather than a
# generic "verb + noun" regex. The direction matters: "alice gives bucket a
# rock" receives an item, while "bucket gives alice the rock" removes one.
INVENTORY_RECEIVE = "receive"
INVENTORY_DROP = "drop"
INVENTORY_ACTIVATE = "activate"

INVENTORY_RECEIVE_VERBS = r"gives?|hands?|offers?|passes?"
INVENTORY_DROP_VERBS = r"drops?|throws?|puts\s+down|gives?|hands?|offers?|passes?"
INVENTORY_ACTIVE_VERBS = r"uses?|wields?|holds?|(?:is\s+)?(?:using|wielding|holding)"
INVENTORY_ARTICLES = r"(?:a|an|the|some|his|her|their|my|your)"
INVENTORY_DISCARD_REPLY_RE = re.compile(
    rf"\b(?:puts?\s+down|offers?\s+you|takes?)\s+"
    rf"(?P<item>(?:{INVENTORY_ARTICLES}\s+)?[^.,!?*\n]+?)(?=$|[.,!?*\n])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class InventoryEvent:
    action: str
    item: str


def _clean_inventory_item(fragment: str) -> str | None:
    """Normalize the noun phrase while leaving the user's words intact."""
    item = re.sub(rf"^\s*{INVENTORY_ARTICLES}\s+", "", fragment, flags=re.IGNORECASE)
    item = re.split(r"\s+(?:because|while|and then)\b", item, maxsplit=1,
                    flags=re.IGNORECASE)[0]
    item = re.sub(r"\s+", " ", item).strip(" .,!?*_-")
    if len(item) < 2 or len(item) > 60:
        return None
    return item


def _inventory_name_pattern() -> str:
    return rf"\b{re.escape(config.NAME)}\b"


def inventory_discard_items(text: str) -> list[str]:
    """Extract items from Bucket's existing discard-style reply phrases.

    These phrases can come from any reply strategy, not just the inventory
    strategy, so this is intentionally separate from the input event parser.
    In generated speech, ``takes the knife`` is an established discard phrase
    even though ``takes`` means receiving an item when a human teaches it.
    """
    found: list[str] = []
    for match in INVENTORY_DISCARD_REPLY_RE.finditer(text or ""):
        item = _clean_inventory_item(match.group("item"))
        if item and item not in found:
            found.append(item)
    return found


def parse_inventory_event(text: str) -> InventoryEvent | None:
    """Recognize one directional inventory event involving Bucket.

    Incoming events are the role-play forms already documented by the bot,
    such as ``alice gives bucket a rock``. Outgoing events require Bucket to be
    the grammatical subject, so ordinary messages that merely mention the bot
    cannot accidentally remove an item.
    """
    text = (text or "").strip()
    if not text or not config.NAME:
        return None

    name = _inventory_name_pattern()
    if not re.search(name, text, re.IGNORECASE):
        return None

    # Receive: "alice gives bucket a rock" and "alice gives a rock to bucket".
    for match in re.finditer(
        rf"\b(?:{INVENTORY_RECEIVE_VERBS})\b(?P<rest>[^.,!?*]*)",
        text,
        re.IGNORECASE,
    ):
        rest = match.group("rest")
        recipient = re.search(name, rest, re.IGNORECASE)
        if recipient:
            item = _clean_inventory_item(rest[recipient.end():])
            if item:
                return InventoryEvent(INVENTORY_RECEIVE, item)

            before = re.sub(r"\bto\s*$", "", rest[:recipient.start()],
                            flags=re.IGNORECASE)
            item = _clean_inventory_item(before)
            if item:
                return InventoryEvent(INVENTORY_RECEIVE, item)

    # Pick up / take: "bucket picks up the hammer".
    match = re.search(
        rf"{name}\s+(?:picks?\s+up|picked\s+up|takes?|took)\s+"
        rf"(?P<item>[^.,!?*]+)",
        text,
        re.IGNORECASE,
    )
    if match:
        item = _clean_inventory_item(match.group("item"))
        if item:
            return InventoryEvent(INVENTORY_RECEIVE, item)

    # Outgoing events: Bucket must precede the action. This prevents a human
    # saying "Alice gives Bucket a rock" from taking the remove path.
    match = re.search(
        rf"{name}\s+(?P<verb>{INVENTORY_DROP_VERBS})\s+"
        rf"(?P<rest>[^.,!?*]+)",
        text,
        re.IGNORECASE,
    )
    if match:
        verb = match.group("verb").lower()
        rest = match.group("rest").strip()
        rest = re.sub(r"^(?:away|back)\s+", "", rest, flags=re.IGNORECASE)

        if verb.startswith(("drop", "throw", "put")):
            item = _clean_inventory_item(rest)
        else:
            # Support both "Bucket gives the knife to Alice" and
            # "Bucket gives Alice the knife". The latter intentionally only
            # treats a recipient as such when an article introduces the item.
            explicit_target = re.match(
                r"(?P<item>.+?)\s+to\s+\S+(?:\s+\S+)*$", rest, re.IGNORECASE
            )
            recipient_first = re.match(
                rf"(?:me|you|us|him|her|them|[\w'@.-]+)\s+"
                rf"(?P<item>{INVENTORY_ARTICLES}\s+.+)$",
                rest,
                re.IGNORECASE,
            )
            item = _clean_inventory_item(
                explicit_target.group("item") if explicit_target else
                recipient_first.group("item") if recipient_first else rest
            )
        if item:
            return InventoryEvent(INVENTORY_DROP, item)

    # Use / wield / hold: these do not create an item, but make an existing
    # one active so the inventory strategy can prefer it.
    match = re.search(
        rf"{name}\s+(?:{INVENTORY_ACTIVE_VERBS})\s+"
        rf"(?P<item>[^.,!?*]+)",
        text,
        re.IGNORECASE,
    )
    if match:
        item = _clean_inventory_item(match.group("item"))
        if item:
            return InventoryEvent(INVENTORY_ACTIVATE, item)

    return None

# A sentence opening with one of these is a question, not an assertion.
# "bucket did you know you are gay" would otherwise be stored as a fact about
# "bucket did you", and chat questions rarely carry a question mark.
QUESTION_STARTERS = frozenset(
    "what who whom whose where when why how which "
    "do does did doesnt dont didnt "
    "is are was were am "
    "can could will would should shall may might must "
    "have has had".split()
)

# A subject whose final word is a pronoun or auxiliary is a parse artefact —
# "bro you", "so you", "i don't", "bucket did you" — not something to remember.
BAD_SUBJECT_TAILS = frozenset(
    "i you he she it we they me him her us them my your his its our their "
    "do does did dont don't doesnt doesn't didnt didn't "
    "am is are was were be been being "
    "have has had havent haven't hasnt hasn't "
    "can cant can't could couldnt couldn't will wont won't would wouldnt "
    "should shouldnt must might may shall that this these those and or but "
    "so if then than as of to in on at by for with from "
    # contracted pronouns — "bucket i'd like you to..." is not a fact about
    # "bucket i'd", and tokenize() keeps the apostrophe
    "i'd id i'll ill i'm im i've ive you'd youd you're youre you'll youll "
    "we'd wed we're were we'll they'd theyd they're theyre he'd she'd "
    "he's hes she's shes that's thats there's theres it's its what's whats".split()
)

# An auxiliary followed by a subject pronoun is inverted question order:
# "toadlya_bot can you draw...", "bucket have you ever...". The name in front
# means the sentence doesn't start with a question word, so this catches what
# QUESTION_STARTERS misses.
AUXILIARY_VERBS = frozenset(
    "is are was were do does did have has had "
    "can could will would should must might may".split()
)
SUBJECT_PRONOUNS = frozenset("i you he she it we they".split())

# One chunk per sentence, keeping the terminator so questions stay detectable.
SENTENCE_RE = re.compile(r"[^.!?;]+[.!?;]?")

# "ana is also known as anabird", "elle aka the hammer guy"
ALIAS_RE = re.compile(
    r"^(?P<name>.{2,40}?)\s+(?:is\s+)?(?:also\s+known\s+as|also\s+called|aka|a\.k\.a\.?"
    r"|goes\s+by|short\s+for)\s+(?P<alias>.{2,40})$",
    re.IGNORECASE,
)

# Subjects too generic to be worth remembering as factoids.
JUNK_SUBJECTS = frozenset(
    "it that this there here he she they we you i who what which everything "
    "nothing something anything someone everyone".split()
)


class Learner:
    def __init__(self, db: BucketDB):
        self.db = db

    def ingest(self, text: str, author: str = "", chat: str = "",
               previous_id: int | None = None, reinforce: bool = True) -> int | None:
        """Absorb one line. Returns the utterance id so it can be chained to the next.

        `reinforce=False` (used for Bucket's own replies) learns the line without
        counting it as another repetition — see BucketDB.add_utterance.
        """
        text = (text or "").strip()
        if not text or is_machine_output(text):
            return None

        uid = self.db.add_utterance(text, author=author, chat=chat, reinforce=reinforce)
        if uid is None:
            return None

        # Everything below this line is Bucket learning. Its own replies are
        # stored and remain searchable, but they are not learned *from*:
        #
        #   chain    - generating from a chain trained on its own recombinations
        #              degrades toward noise as its share of the corpus grows
        #   pairs    - an edge whose reply is its own output means answering a
        #              prompt makes it likelier to answer the same way again
        #   factoids - markov noise ("dario is my god") hardening into knowledge
        #   items    - it shouldn't hand things to itself
        #
        # Only people teach Bucket things. It just talks.
        if not reinforce:
            return uid

        # A line naming someone from the 2008 transcripts is stored (so /recall
        # and /literal can still show it) but never learned from. Letting it into
        # the chain or the phrase table is what puts the name in Bucket's mouth
        # later, recombined and stripped of any context. Checked on the text
        # rather than the author because these names arrive through present-day
        # speakers too — a real user asking "who is cypress_z" is how that one
        # got into the live chain in the first place.
        if BLOCKED.blocks(text):
            return uid

        if previous_id is not None:
            self.db.add_pair(previous_id, uid)
        self._learn_chain(text)
        self._learn_phrases(text)
        self._learn_factoid(text, author)
        self._learn_item(text, author)
        return uid

    # ------------------------------------------------------------------
    def _learn_phrases(self, text: str) -> None:
        """Collect every 2-4 word run, so recurring ones can surface later.

        A phrase only counts as one the group actually uses once it has been
        said more than once — that threshold is applied when reading, not here,
        so a chunk can become a phrase later without re-reading the corpus.
        """
        words = tokenize(text)
        if len(words) < 2:
            return

        # Count each distinct chunk once per line. Otherwise one message that
        # repeats a word eighty times contributes eighty counts and buries every
        # phrase the group actually uses.
        found: dict[str, int] = {}
        for size in (2, 3, 4):
            for start in range(len(words) - size + 1):
                chunk = words[start:start + size]
                # A single word repeated isn't a phrase, it's a stuck key.
                if len(set(chunk)) < 2:
                    continue
                # All-stopword runs ("of the", "you are") carry nothing.
                if all(word in STOPWORDS for word in chunk):
                    continue
                # Nor do runs that are mostly single letters or digits.
                if sum(len(word) for word in chunk) < size + 3:
                    continue
                if any(word.isdigit() for word in chunk):
                    continue
                found[" ".join(chunk)] = size

        for phrase, size in found.items():
            self.db.add_phrase(phrase, size)
        self.db.conn.commit()

    # ------------------------------------------------------------------
    def _learn_chain(self, text: str) -> None:
        """Feed the order-2 markov table, plus an order-1 backoff table."""
        words = tokenize(text)
        if len(words) < 2:
            return

        # Order 1, including the empty context which acts as the sentence-start bucket.
        self.db.add_transition(("",), words[0])
        for a, b in zip(words, words[1:]):
            self.db.add_transition((a,), b)
        self.db.add_transition((words[-1],), "\x00")  # end marker

        # Order 2.
        for a, b, c in zip(words, words[1:], words[2:]):
            self.db.add_transition((a, b), c)
        self.db.add_transition((words[-2], words[-1]), "\x00")
        self.db.conn.commit()

    # ------------------------------------------------------------------
    def _learn_factoid(self, text: str, author: str) -> None:
        """Mine a message for facts, one sentence at a time.

        Matching the whole message as a single unit meant the first predicate
        swallowed everything after it, sentence boundaries included: "bro you
        have to. Wet puh is SO GOOD bro" parsed as subject "bro you", object
        "to. Wet puh is SO GOOD bro" — junk, and it hid the real fact in the
        second sentence. One message can legitimately assert several things.
        """
        if looks_like_url(text):
            return

        candidate = text
        # "bucket, cheese is delicious" -> "cheese is delicious"
        teach = TEACH_RE.match(text)
        if teach and config.NAME in normalize(text.split(",")[0] + " " + text.split(":")[0]):
            candidate = teach.group("rest")
        # Only strip the name when punctuation marks it as a salutation, so
        # "bucket hates fridays" keeps "bucket" as the subject.
        candidate = strip_addressing(candidate, config.NAME, require_punctuation=True)

        for sentence in SENTENCE_RE.findall(candidate):
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue
            # A question is not an assertion, marked or unmarked.
            if sentence.endswith("?"):
                continue
            cleaned = sentence.strip(" .!?;,")
            if self._extract_alias(cleaned):
                continue
            self._extract_factoid(cleaned, author)

    def _extract_alias(self, candidate: str) -> bool:
        """'ana is also known as anabird' — file both names under one subject."""
        match = ALIAS_RE.match(candidate)
        if not match:
            return False
        name = normalize(match.group("name")).lstrip("@")
        alias = normalize(match.group("alias")).lstrip("@")
        if not name or not alias or len(alias.split()) > 4:
            return False
        if name in JUNK_SUBJECTS or alias in JUNK_SUBJECTS:
            return False
        # The longer-established name wins as canonical, so aliases collapse
        # toward whatever Bucket already knows things about.
        if self.db.factoids_for(alias) and not self.db.factoids_for(name):
            name, alias = alias, name
        return self.db.add_alias(alias, name)

    def _extract_factoid(self, candidate: str, author: str) -> None:
        """Pull one subject/verb/object out of a single sentence, or nothing."""
        # Clean the whole sentence, not just the object — otherwise the filler
        # ends up welded to the subject instead ("yes dario", "remember that
        # chickens").
        candidate = strip_filler(candidate)
        if len(candidate) < 5:
            return

        # A question phrased without a question mark is still a question.
        opening = tokenize(candidate)
        if opening and opening[0] in QUESTION_STARTERS:
            return

        match = FACTOID_RE.match(candidate)
        if not match:
            return

        # "@ana is asleep" and "ana is asleep" are about the same person.
        subject = normalize(match.group("subject")).lstrip("@")
        subject_words = subject.split()
        if subject_words and subject_words[-1] in BAD_SUBJECT_TAILS:
            return
        obj = strip_filler(match.group("object").strip())
        if not subject or len(obj) < 2:
            return
        if subject in JUNK_SUBJECTS or len(subject.split()) > 5:
            return
        # A subject that is only a possessive fragment ("ana's") carries nothing.
        if subject.endswith("'s") and len(subject.split()) == 1:
            return

        verb = match.group("verb").lower()

        object_words = tokenize(obj)
        if (verb in AUXILIARY_VERBS and object_words
                and object_words[0] in SUBJECT_PRONOUNS):
            return

        self.db.add_factoid(subject, VERB_CANONICAL.get(verb, verb), obj, author=author)

    # ------------------------------------------------------------------
    def _learn_item(self, text: str, author: str) -> str | None:
        """Apply a directional inventory event, if the line contains one."""
        event = parse_inventory_event(text)
        if event is None:
            return None

        if event.action == INVENTORY_RECEIVE:
            return self.db.add_item(event.item, author, config.INVENTORY_SIZE)
        if event.action == INVENTORY_DROP:
            return event.item if self.db.remove_item(event.item) else None
        if event.action == INVENTORY_ACTIVATE:
            return event.item if self.db.activate_item(event.item) else None
        return None
