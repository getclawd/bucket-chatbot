"""The bot itself: storage + learning + brain + polish, wired together.

Surfaces (console, telegram) drive this and stay dumb.
"""

import random
import re
import threading
import time

from .blocklist import DEFAULT_BLOCKED, Blocklist
from .brain import Brain, Reply
from .config import config
from .db import BucketDB
from .embed import Embedder, VectorIndex, pack
from .learn import Learner, inventory_discard_items
from .llm import Polisher
from .seed import load_seed
from .text import content_words, normalize, tokenize

# Names redacted from anything shown to a user. Same list the learning and reply
# guards use — see blocklist.py.
BLOCKED = Blocklist(DEFAULT_BLOCKED | config.BLOCKED_NAMES)


class Bucket:
    def __init__(self, db_path: str | None = None, quiet: bool = False):
        # One Bucket can be shared by several surfaces at once (Telegram in a
        # thread, Discord on the event loop). Everything that touches the
        # database or the vector index goes through this lock.
        self._lock = threading.RLock()
        self.db = BucketDB(db_path or config.DB_PATH)
        self.learner = Learner(self.db)
        self.polisher = Polisher()
        # Last thing said in each chat, so replies can be chained into pairs.
        self._last_utterance: dict[str, int] = {}
        # When it last spoke uninvited in each chat, for pacing.
        self._last_spoke: dict[str, float] = {}
        load_seed(self.db, self.learner)

        # A chattiness set at runtime should outlive a restart.
        stored = self.db.get_meta("chattiness", "")
        if stored:
            try:
                config.CHATTINESS = float(stored)
            except ValueError:
                pass

        # Recent conversation content per in-memory chat key, so a reply to
        # "what about him" has something to attach "him" to.
        self._recent: dict[str, list[str]] = {}
        # The last few lines said in each chat by *anyone*, Bucket included,
        # used only to stop it repeating one of them. See _note_said.
        self._said: dict[str, list[str]] = {}

        self.embedder = Embedder()
        self.index: VectorIndex | None = None
        self.fact_index: VectorIndex | None = None
        if self.embedder.available:
            self.index = VectorIndex(self.embedder.dim)
            self.index.load(
                self.db.all_vectors(self.embedder.dim, self.embedder.fingerprint)
            )
            self.fact_index = VectorIndex(self.embedder.dim)
            self.db.prune_fact_vectors()
            self.fact_index.load(
                self.db.all_fact_vectors(self.embedder.dim, self.embedder.fingerprint)
            )
            self.backfill(quiet=quiet)
            self.backfill_facts(quiet=quiet)

        self.brain = Brain(
            self.db,
            semantic=self._semantic if self.index is not None else None,
            semantic_facts=self._semantic_facts if self.fact_index is not None else None,
        )

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------------
    # semantic memory
    # ------------------------------------------------------------------
    def _semantic(self, text: str, limit: int = 12) -> list[tuple[int, float]]:
        if self.index is None:
            return []
        vector = self.embedder.embed(text)
        if not vector:
            return []
        return self.index.search(vector, limit)

    # Words that point at something said earlier and mean nothing on their own.
    _DANGLING = frozenset(
        "him her it he she they them that this those these his hers their there".split()
    )

    def _note_turn(self, chat: str, text: str) -> None:
        turns = self._recent.setdefault(chat, [])
        turns.append(text.strip())
        del turns[:-3]

    def _note_said(self, chat: str, text: str) -> None:
        """Record a line for repeat suppression only.

        Separate from `_recent` on purpose. `_recent` feeds `context_for()`,
        which widens *recall* — putting Bucket's own replies in there would let
        its output steer what it retrieves next, which is the feedback loop this
        whole codebase is built to avoid. This list is never used for recall; it
        only tells the brain what not to say again (see Brain.respond's `avoid`).
        """
        # 0 disables suppression. Guarded because `del said[:-0]` is
        # `del said[:0]`, which trims nothing — the list would grow forever and
        # suppression would become permanent, the exact opposite of "off".
        if config.NO_REPEAT_WINDOW <= 0:
            return
        said = self._said.setdefault(chat, [])
        said.append(text.strip())
        del said[: -config.NO_REPEAT_WINDOW]

    def recently_said(self, chat: str) -> tuple[str, ...]:
        return tuple(self._said.get(chat, ()))

    def context_for(self, chat: str, text: str) -> str:
        """Earlier turns worth folding into recall, or '' if the line stands alone.

        Only used when the message can't be understood by itself — a short line
        or one leaning on a pronoun. Blending context into everything would
        drown out what was actually just said.
        """
        turns = [t for t in self._recent.get(chat, []) if t.strip() != text.strip()]
        if not turns:
            return ""

        words = content_words(text)
        leans_on_earlier = any(w in self._DANGLING for w in tokenize(text))
        if len(words) > 3 and not leans_on_earlier:
            return ""
        return " ".join(turns[-2:])

    def _remember_facts(self) -> None:
        """Embed any fact learned in the last message or two."""
        if self.fact_index is None:
            return
        rows = self.db.unvectorized_facts(
            self.embedder.dim, limit=8, model_key=self.embedder.fingerprint
        )
        if not rows:
            return
        texts = [f"{r['subject']} {r['verb']} {r['object']}".strip() for r in rows]
        vectors = self.embedder.embed_batch(texts)
        if not vectors or len(vectors) != len(rows):
            return
        for row, vector in zip(rows, vectors):
            self.db.add_fact_vector(
                row["id"], self.embedder.dim, pack(vector), self.embedder.fingerprint
            )
            self.fact_index.add(row["id"], vector)

    def _remember(self, uid: int, text: str) -> None:
        """Give a newly stored utterance a vector, if it doesn't have one."""
        if (
            self.index is None
            or self.db.has_vector(uid, self.embedder.dim, self.embedder.fingerprint)
        ):
            return
        vector = self.embedder.embed(text)
        if not vector:
            return
        self.db.add_vector(uid, self.embedder.dim, pack(vector), self.embedder.fingerprint)
        self.index.add(uid, vector)

    def _semantic_facts(self, text: str, limit: int = 6) -> list[tuple[int, float]]:
        if self.fact_index is None:
            return []
        vector = self.embedder.embed(text)
        if not vector:
            return []
        return self.fact_index.search(vector, limit)

    def backfill_facts(self, quiet: bool = False) -> int:
        """Embed the fact table. Safe to re-run; only touches what's missing."""
        if self.fact_index is None:
            return 0

        done = 0
        while True:
            rows = self.db.unvectorized_facts(
                self.embedder.dim, limit=64, model_key=self.embedder.fingerprint
            )
            if not rows:
                break
            texts = [
                f"{r['subject']} {r['verb']} {r['object']}".strip() for r in rows
            ]
            vectors = self.embedder.embed_batch(texts)
            if not vectors or len(vectors) != len(rows):
                break
            for row, vector in zip(rows, vectors):
                self.db.add_fact_vector(
                    row["id"],
                    self.embedder.dim,
                    pack(vector),
                    self.embedder.fingerprint,
                )
                self.fact_index.add(row["id"], vector)
            done += len(rows)

        if done and not quiet:
            print(f"  embedded {done} fact(s).")
        return done

    def backfill(self, quiet: bool = False) -> int:
        """Embed anything in the corpus that has no vector yet. Safe to re-run."""
        if self.index is None:
            return 0

        outstanding = self.db.count_unvectorized(
            self.embedder.dim, self.embedder.fingerprint
        )
        if not outstanding:
            return 0
        if not quiet:
            print(f"embedding {outstanding} remembered line(s) with {self.embedder.model}...")

        done = 0
        while True:
            rows = self.db.unvectorized(
                self.embedder.dim, limit=64, model_key=self.embedder.fingerprint
            )
            if not rows:
                break
            vectors = self.embedder.embed_batch([row["text"] for row in rows])
            if not vectors or len(vectors) != len(rows):
                break
            for row, vector in zip(rows, vectors):
                self.db.add_vector(
                    row["id"],
                    self.embedder.dim,
                    pack(vector),
                    self.embedder.fingerprint,
                )
                self.index.add(row["id"], vector)
            done += len(rows)
            if not quiet and outstanding > 200:
                print(f"  {done}/{outstanding}", end="\r", flush=True)

        if not quiet:
            print(f"  embedded {done} line(s).      ")
        return done

    # ------------------------------------------------------------------
    def is_addressed(self, text: str, is_private: bool = False,
                     is_reply_to_bot: bool = False) -> bool:
        if is_private or is_reply_to_bot:
            return True
        return bool(re.search(rf"\b{re.escape(config.NAME)}\b", text or "", re.IGNORECASE))

    def should_speak(self, text: str, is_private: bool = False,
                     is_reply_to_bot: bool = False, chat: str = "") -> bool:
        """Addressed directly it always answers. Otherwise it has a rhythm.

        A flat coin-flip per message makes it reply five times in a row in a busy
        group and then vanish. Two adjustments fix that: a hard quiet period
        after speaking uninvited, and a raised chance during the couple of
        minutes after it spoke, when a conversation is actually happening.
        """
        if self.is_addressed(text, is_private, is_reply_to_bot):
            return True

        now = time.time()
        since = now - self._last_spoke.get(chat, 0.0)
        if since < config.MIN_GAP:
            return False

        chance = config.CHATTINESS
        if since < config.FOLLOWUP_WINDOW:
            chance *= config.FOLLOWUP_BOOST
        return random.random() < min(chance, 1.0)

    # ------------------------------------------------------------------
    def handle(self, text: str, chat: str = "",
               is_private: bool = False, is_reply_to_bot: bool = False) -> str | None:
        """Learn from a message, then reply if it feels like it.

        Blocking — it may call out to an embedding model and an LLM. Surfaces
        running an event loop should call this off the loop.
        """
        text = (text or "").strip()
        if not text:
            return None

        with self._lock:
            uid = None
            if config.LEARN:
                uid = self.learner.ingest(
                    text, previous_id=self._last_utterance.get(chat)
                )
                if uid is not None:
                    self._last_utterance[chat] = uid
                    self._remember(uid, text)
                    self._remember_facts()

            self._note_turn(chat, text)
            self._note_said(chat, text)

            addressed = self.is_addressed(text, is_private, is_reply_to_bot)
            if not self.should_speak(text, is_private, is_reply_to_bot, chat):
                return None

            reply = self.speak(text, chat=chat, context=self.context_for(chat, text))
            # Only uninvited replies start the quiet period — being spoken to
            # should never be rate limited.
            if not addressed:
                self._last_spoke[chat] = time.time()
            return reply

    @staticmethod
    def _cap(text: str) -> str:
        """Bucket is terse. Nothing it says should be a wall of text."""
        words = text.strip().split()
        if len(words) <= config.MAX_REPLY_WORDS:
            return text.strip()
        return " ".join(words[: config.MAX_REPLY_WORDS])

    def learn_only(self, text: str) -> None:
        """Absorb a line without considering a reply — edits, backfill, imports."""
        text = (text or "").strip()
        if not text or not config.LEARN:
            return
        with self._lock:
            uid = self.learner.ingest(text)
            if uid is not None:
                self._remember(uid, text)

    def speak(self, text: str, chat: str = "", context: str = "") -> str:
        """Generate a reply and remember that it was said."""
        with self._lock:
            reply: Reply = self.brain.respond(
                text, context=context, avoid=self.recently_said(chat)
            )
            out = (
                reply.text
                if reply.state_changed
                else self.polisher.polish(
                    reply.text, user_text=text, strategy=reply.strategy
                )
            )
            out = self._cap(out or reply.text)

            # Its own line joins the no-repeat window, so the next reply in this
            # chat won't be the same one again. Polished text, not reply.text —
            # what matters is what actually got said.
            self._note_said(chat, out)
            self._apply_spoken_inventory_actions(out)

            # Its own line joins the corpus, chained to what prompted it. This is
            # the feedback loop that lets a phrase snowball into an obsession.
            if config.LEARN:
                own_id = self.learner.ingest(
                    out,
                    # Its own voice doesn't count as another vote for the line.
                    # Only repeated input can make it an obsession.
                    reinforce=False,
                )
                if own_id is not None:
                    self._remember(own_id, out)
                # Deliberately not updating _last_utterance: the pair graph
                # should record what conversation input followed what, not what
                # Bucket interjected. Leaving it means the next message chains
                # to the previous message.

            return out

    def _apply_spoken_inventory_actions(self, text: str) -> list[str]:
        """Discard items named by Bucket's established action phrases."""
        removed = []
        for item in inventory_discard_items(text):
            if self.db.remove_item(item):
                removed.append(item)
        return removed

    # ------------------------------------------------------------------
    # commands shared by every surface
    # ------------------------------------------------------------------
    def cmd_stats(self) -> str:
        stats = self.db.stats()
        lines = [f"{key}: {value}" for key, value in stats.items()]
        lines.append(f"memory: {self.embedder.status()}")
        lines.append(f"polish: {self.polisher.status()}")
        lines.append(f"chattiness: {config.CHATTINESS}")
        lines.append(f"learning: {'on' if config.LEARN else 'off'}")
        return "\n".join(lines)

    @staticmethod
    def redact(text: str) -> str:
        """Blank out blocked names in anything shown to a user.

        The introspection commands print corpus text verbatim, which is their
        whole value — but a blocked name is exactly as unwanted in `/recall`
        output as in a reply. Redacting rather than hiding the line keeps them
        useful for debugging: you can still see the line exists.
        """
        hits = BLOCKED.hits(text)
        if not hits:
            return text
        for name in hits:
            text = re.sub(rf"(?<![\w']){re.escape(name)}(?![\w'])", "[redacted]",
                          text, flags=re.IGNORECASE)
        return text

    def cmd_literal(self, subject: str) -> str:
        """Show exactly what it believes about a subject, unmangled.

        Reports facts *and* raw mentions. Facts are only the subset the extractor
        managed to parse into a triple; the chain generates from the utterance
        text, so it can freely emit a name it has no fact about. Reporting facts
        alone made it deny things it had demonstrably just said — it produced a
        line about `oldname_7`, and answered `/literal oldname_7` with "i don't
        know anything about oldname_7" while holding three lines containing it.
        """
        subject = normalize(subject)
        if not subject:
            return "literal what"
        # Echoed back in every branch below, so redact once here. Bucket printing
        # a blocked name is unwanted even when the user typed it themselves.
        shown = self.redact(subject)

        rows = self.db.factoids_for(subject)
        seen = self.db.mentions(subject, limit=6)

        if not rows and not seen:
            return f"i don't know anything about {shown}"

        lines = []
        if rows:
            lines.append(f"{shown}:")
            for row in rows[:20]:
                lines.append(
                    f"  {row['verb']} {row['object']}  (x{row['count']})"
                )
            if len(rows) > 20:
                lines.append(f"  ... and {len(rows) - 20} more")
        else:
            # The honest version of "i don't know": no parsed facts, but the
            # material the chain draws on is right there.
            lines.append(f"no facts about {shown}, but i've heard it said:")

        if seen:
            if rows:
                lines.append("mentioned in:")
            for row in seen:
                lines.append(f"  {self.redact(row['text'])[:70]}")
        return "\n".join(lines)
    def cmd_who(self, predicate: str) -> str:
        """Reverse lookup: '/who is dead' searches the object side of the facts."""
        predicate = predicate.strip()
        if not predicate:
            return "who what"

        rows = self.db.factoids_by_predicate(predicate)
        if not rows:
            return f"nobody i know of {predicate}"

        seen: dict[str, str] = {}
        for row in rows:
            seen.setdefault(row["subject"], f"{row['verb']} {row['object']}")
        lines = [f"things that match '{predicate}':"]
        for subject, predicate_text in list(seen.items())[:12]:
            lines.append(f"  {subject} {predicate_text[:60]}")
        return "\n".join(lines)

    def cmd_forget(self, subject: str) -> str:
        subject = normalize(subject)
        if not subject:
            return "forget what"
        with self._lock:
            removed = self.db.forget(subject)
        if not removed:
            return f"i never knew anything about {subject}"
        return f"forgot {removed} thing(s) about {subject}"

    def cmd_chattiness(self, argument: str) -> str:
        """Set how often it butts in, and remember it across restarts."""
        argument = argument.strip()
        if not argument:
            return (
                f"chattiness is {config.CHATTINESS} "
                f"(min gap {config.MIN_GAP:.0f}s, x{config.FOLLOWUP_BOOST} for "
                f"{config.FOLLOWUP_WINDOW:.0f}s after i speak). "
                "give me a number from 0 to 1."
            )
        try:
            value = float(argument)
        except ValueError:
            return f"'{argument}' is not a number between 0 and 1"

        value = max(0.0, min(1.0, value))
        config.CHATTINESS = value
        with self._lock:
            self.db.set_meta("chattiness", str(value))

        if value == 0:
            mood = "i'll only speak when spoken to"
        elif value < 0.1:
            mood = "i'll keep to myself mostly"
        elif value < 0.3:
            mood = "that's about right"
        elif value < 0.6:
            mood = "i'm going to be in the way a lot"
        else:
            mood = "you will regret this"
        return f"chattiness set to {value}. {mood}."

    def cmd_inventory(self) -> str:
        items = self.db.items()
        if not items:
            return "i'm not carrying anything"
        return "i am carrying: " + ", ".join(items)

    def cmd_recall(self, text: str) -> str:
        """Show what the memory actually retrieves for a query. Debugging aid."""
        text = text.strip()
        if not text:
            return "recall what"

        lines = [f"recall for: {self.redact(text)}"]
        with self._lock:
            semantic = self._semantic(text, 5) if self.index is not None else []
            # lexical_search, not search: this command exists to show what recall
            # actually does, so it has to use the same path the replies use.
            lexical = self.db.lexical_search(text, 5)

        for label, hits in (("semantic", semantic), ("lexical", lexical)):
            lines.append(f"  {label}:")
            if not hits:
                lines.append("    (nothing)")
                continue
            for uid, score in hits:
                row = self.db.get_utterance(uid)
                if row:
                    lines.append(f"    {score:.3f}  {self.redact(row['text'])[:70]}")
        return "\n".join(lines)

    def cmd_wipe(self) -> str:
        with self._lock:
            self.db.wipe()
            self.db.set_meta("seeded", "0")
            load_seed(self.db, self.learner)
            self._last_utterance.clear()
            if self.index is not None:
                self.index.load([])
                self.backfill(quiet=True)
        return "everything is gone. starting over."

