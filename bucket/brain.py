"""Response selection.

The brain never invents content. Every word it emits came from something a user
typed at some point, recombined by one of the strategies below. That is the whole
trick behind the original's particular flavor of wrongness.
"""

import random
import re
import time
from dataclasses import dataclass
from typing import Callable

from .blocklist import DEFAULT_BLOCKED, Blocklist
from .config import config
from .db import BucketDB
from .text import content_words, normalize, strip_addressing, tokenize, trim_dangling

# Names it must never say, even if they're already in the corpus. The learning
# guard stops new ones entering; this stops old ones coming out. See blocklist.py.
BLOCKED = Blocklist(DEFAULT_BLOCKED | config.BLOCKED_NAMES)

END = "\x00"


@dataclass
class Reply:
    text: str
    strategy: str
    state_changed: bool = False
    on_accept: Callable[[], bool] | None = None

    def __bool__(self) -> bool:
        return bool(self.text.strip())


class Brain:
    def __init__(self, db: BucketDB, semantic=None, semantic_facts=None):
        self.db = db
        # Optional callable(text, limit) -> [(utterance_id, cosine score)].
        self.semantic = semantic
        # Optional callable(text, limit) -> [(factoid_id, cosine score)].
        self.semantic_facts = semantic_facts
        # Earlier turns, folded into recall for the current reply only.
        self._context = ""

    # ------------------------------------------------------------------
    def respond(self, text: str, addressed: bool = True, context: str = "",
                avoid: tuple[str, ...] = ()) -> Reply:
        """Produce a reply to `text`. Falls through strategies until one fires.

        `context` is earlier turns, supplied only when the message can't stand
        on its own ("what about him"). It widens recall without becoming the
        subject of the reply.

        `avoid` is the last few lines anyone said, Bucket included. Checking
        only against `text` catches parroting the current message but nothing
        else: saying a line back that someone said two turns ago, or that Bucket
        itself just said, both used to sail straight through the guard.
        """
        self._context = (context or "").strip()
        spoken = (text or "").strip()
        cleaned = strip_addressing(spoken, config.NAME)
        # The current message counts as recently-said too, and comes first so
        # parroting the actual prompt is still what's caught most cheaply.
        recent = (spoken, *(a for a in avoid if a and a.strip() != spoken))

        fallback = None
        for strategy in self._plan(cleaned):
            reply = strategy(cleaned)
            if not reply:
                continue
            # Composition and markov both like to stop on a word that needs
            # something after it ("...which means dario beats his"), which reads
            # as the message being cut off. Trim rather than regenerate: the
            # engine may only ever remove words, never add them.
            finished = trim_dangling(reply.text)
            if not finished:
                continue
            if finished != reply.text:
                reply = Reply(
                    finished,
                    reply.strategy,
                    reply.state_changed,
                    reply.on_accept,
                )
            # Naming someone from the 2008 logs is never acceptable, so unlike
            # the repeat check this one can't fall back to the rejected line.
            if BLOCKED.blocks(reply.text):
                continue
            # Repeating a sentence back reads as broken, not insane. Keep the
            # first one as a last resort but try for something else.
            if self._repeats(reply.text, recent):
                fallback = fallback or reply
                continue
            accepted = self._accept(reply)
            if accepted is not None:
                return accepted

        # Everything repeated something. Saying a line straight back is worse
        # than saying something unrelated, so reach for anything else first and
        # only fall back to the echo if the corpus has literally nothing.
        if fallback is not None:
            # A few tries, because the corpus may literally contain the line
            # that was just said and a single random draw can land on it again.
            for _ in range(5):
                escape = self._last_resort()
                if escape and not BLOCKED.blocks(escape) and not self._repeats(escape, recent):
                    return Reply(escape, "desperation")
            phrase = self.db.random_phrase()
            if phrase and not BLOCKED.blocks(phrase) and not self._repeats(phrase, recent):
                return Reply(phrase, "desperation")
            accepted = self._accept(fallback)
            if accepted is not None:
                return accepted
            return Reply(self._safe_last_resort(), "desperation")

        return Reply(self._safe_last_resort(), "desperation")

    @staticmethod
    def _accept(reply: Reply) -> Reply | None:
        """Commit a deferred state change only after the reply is selected."""
        if reply.on_accept is None:
            return reply
        if not reply.on_accept():
            # Another surface may have changed the state between candidate
            # selection and acceptance. The caller should try another reply,
            # never claim an action that did not happen.
            return None
        return Reply(reply.text, reply.strategy, state_changed=True)

    def _safe_last_resort(self) -> str:
        """A random line that doesn't name anyone from the 2008 logs.

        The final fallback can't just be `_last_resort()`: it draws a random
        utterance, and blocked names are still *stored* (only learning and
        replying are filtered), so an unguarded draw can land on one.
        """
        for _ in range(8):
            line = self._last_resort()
            if line and not BLOCKED.blocks(line):
                return line
        # Every draw named someone. Say nothing rather than say a name.
        return ""

    @classmethod
    def _repeats(cls, reply: str, recent: tuple[str, ...]) -> bool:
        """Is this line one of the last few things said, by anyone?

        Deliberately a short window. Reusing the group's stock phrases is the
        point of the `phrase` and `obsession` strategies — what reads as broken
        is doing it twice inside the same exchange.
        """
        return any(cls._parrots(reply, said) for said in recent if said)

    @staticmethod
    def _parrots(reply: str, text: str) -> bool:
        """Is this just the input handed back?"""
        said = set(content_words(text))
        mine = set(content_words(reply))
        if not mine:
            return normalize(reply) == normalize(text)
        if normalize(reply) == normalize(text):
            return True
        # Nearly all of my content words came from you, and I added nothing.
        overlap = len(mine & said) / len(mine)
        return overlap >= 0.8 and len(mine - said) <= 1

    def _plan(self, text: str):
        """Weighted-random ordering of strategies, biased by what the input looks like."""
        # Weighted toward composition. The strategies that return one whole
        # retrieved line (pair, echo) are the least Bucket-like thing it does —
        # what made the original feel unhinged was colliding two things it knew
        # while sounding completely sure of itself.
        weights: list[tuple[float, callable]] = [
            # compose
            (5.0, self._from_mashup),      # two on-topic lines welded together
            (4.0, self._from_tangent),     # a fact, then wandering off it
            (3.5, self._from_phrase),      # the group's own stock phrases, reused
            (3.0, self._from_collision),   # two unrelated facts, joined confidently
            # single-source
            (4.0, self._from_pair),
            (3.0, self._from_factoid),
            (3.0, self._from_markov),
            (1.0, self._from_echo),
            (1.0, self._from_inventory),
            (2.0, self._from_obsession),
        ]

        # A direct question makes it likelier to reach for something it "knows".
        if text.rstrip().endswith("?"):
            weights = [
                (w * 2.0 if fn in (self._from_factoid, self._from_tangent) else w, fn)
                for w, fn in weights
            ]

        order = []
        pool = list(weights)
        while pool:
            total = sum(w for w, _ in pool)
            pick = random.uniform(0, total)
            running = 0.0
            for index, (weight, fn) in enumerate(pool):
                running += weight
                if pick <= running:
                    order.append(fn)
                    pool.pop(index)
                    break
        return order

    # ------------------------------------------------------------------
    # strategies
    # ------------------------------------------------------------------
    def _candidates(self, text: str, limit: int = 8) -> list[tuple[int, float]]:
        """Blend meaning-based and word-based recall into one ranked list.

        The two score on different scales, so each is normalized to 0..1 against
        its own best hit before they're combined.
        """
        # Overfetch, then rerank. Blending and decay can only reorder what
        # retrieval handed over, so both halves pull deeper than `limit`.
        depth = limit * max(1, config.RECALL_OVERFETCH)

        # Lexical search stays on the literal message — folding context in would
        # match words nobody just said. Only the meaning-based half widens.
        lexical = self.db.lexical_search(text, limit=depth)
        query = f"{self._context} {text}".strip() if self._context else text

        semantic = []
        if self.semantic is not None:
            try:
                semantic = self.semantic(query, depth)
            except Exception:  # noqa: BLE001 - never let recall break a reply
                semantic = []

        if not semantic:
            return self._rerank(lexical, limit)

        merged: dict[int, float] = {}
        weight = max(0.0, min(1.0, config.SEMANTIC_WEIGHT))
        for source, share in ((semantic, weight), (lexical, 1.0 - weight)):
            if not source or share <= 0:
                continue
            best = max(score for _, score in source) or 1.0
            for uid, score in source:
                merged[uid] = merged.get(uid, 0.0) + share * (score / best)

        return self._rerank(list(merged.items()), limit)

    def _rerank(self, scored: list[tuple[int, float]], limit: int) -> list[tuple[int, float]]:
        """Final ordering pass over blended candidates.

        Currently just recency: multiply relevance by 0.5^(age / half_life), so
        what the group said this week outranks an equally-relevant line from
        months ago. Off by default (RECALL_HALF_LIFE=0) because most of what
        makes Bucket funny is old and shouldn't fade — turn it on if it starts
        feeling stuck in the past.
        """
        half_life = config.RECALL_HALF_LIFE
        if half_life > 0 and scored:
            now = time.time()
            decayed = []
            for uid, score in scored:
                row = self.db.get_utterance(uid)
                if row is None:
                    continue
                age_days = max(0.0, (now - row["ts"]) / 86400.0)
                decayed.append((uid, score * 0.5 ** (age_days / half_life)))
            scored = decayed
        return sorted(scored, key=lambda kv: kv[1], reverse=True)[:limit]

    def _from_pair(self, text: str) -> Reply | None:
        """What did someone say the last time this came up?"""
        candidates = self._candidates(text, limit=8)
        if not candidates:
            return None

        random.shuffle(candidates)
        candidates.sort(key=lambda kv: kv[1] * random.uniform(0.6, 1.4), reverse=True)

        for uid, _score in candidates[:4]:
            replies = self.db.replies_to(uid)
            if not replies:
                continue
            weights = [row["pair_count"] for row in replies]
            chosen = random.choices(replies, weights=weights, k=1)[0]
            if normalize(chosen["text"]) != normalize(text):
                return Reply(chosen["text"], "pair")
        return None

    def _from_factoid(self, text: str) -> Reply | None:
        subjects = self.db.known_subjects_in(text)
        if not subjects:
            return None

        for subject in subjects[:3]:
            rows = self.db.factoids_for(subject)
            if not rows:
                continue
            row = random.choices(rows, weights=[r["count"] for r in rows], k=1)[0]
            name = row["subject"]
            templates = [
                f"{name} {row['verb']} {row['object']}",
                f"i heard {name} {row['verb']} {row['object']}",
                f"well, {name} {row['verb']} {row['object']}, obviously",
                f"{row['object']}. that's what {name} {row['verb']}.",
            ]
            return Reply(random.choice(templates), "factoid")
        return None

    def _from_markov(self, text: str) -> Reply | None:
        """Generate a fresh sentence, seeded from a word in the input if possible."""
        seeds = [w for w in content_words(text)]
        random.shuffle(seeds)
        seeds.append("")  # fall back to the sentence-start bucket

        for seed in seeds:
            sentence = self._generate(seed)
            if sentence and len(sentence.split()) >= 3:
                return Reply(sentence, "markov")
        return None

    def _from_echo(self, text: str) -> Reply | None:
        """Say something it heard once, verbatim, with no regard for relevance."""
        row = self.db.random_utterance()
        if not row or normalize(row["text"]) == normalize(text):
            return None
        return Reply(row["text"], "echo")

    def _from_mashup(self, text: str) -> Reply | None:
        """Two lines welded at the seam — Bucket's signature move.

        The pieces are *retrieved for what you just said*, not picked at random.
        That's the difference between noise and the 2008 article: on-topic
        material in the wrong shape, delivered with total confidence.
        """
        pool = [uid for uid, _ in self._candidates(text, limit=6)]
        rows = [row for row in (self.db.get_utterance(uid) for uid in pool) if row]

        # Topping up with something random keeps it from getting too sensible.
        if len(rows) < 2 or random.random() < 0.3:
            stray = self.db.random_utterance()
            if stray:
                rows.append(stray)
        if len(rows) < 2:
            return None

        first, second = random.sample(rows, 2)
        left, right = tokenize(first["text"]), tokenize(second["text"])
        if len(left) < 2 or len(right) < 2:
            return None

        # Cut both somewhere in the middle so each half keeps a bit of shape.
        cut_a = random.randint(max(1, len(left) // 3), len(left))
        cut_b = random.randint(0, max(0, (2 * len(right)) // 3))
        return Reply(" ".join(left[:cut_a] + right[cut_b:]), "mashup")

    def _topical_factoid(self, text: str):
        """A fact connected to what was said, falling back through three tiers.

        Reaching straight for a random fact is what makes replies feel like
        noise rather than a non-sequitur — a non-sequitur still has to start
        somewhere near the conversation.
        """
        # 1. A subject named outright.
        subjects = self.db.known_subjects_in(text)
        if subjects:
            rows = self.db.factoids_for(subjects[0])
            if rows:
                return random.choice(rows)

        # 2. A fact that *means* something like what was said. Without this,
        #    "who is violent" can never reach "dario is an abuser", because
        #    nothing in the sentence matches a subject string.
        if self.semantic_facts is not None:
            query = f"{self._context} {text}".strip() if self._context else text
            try:
                hits = self.semantic_facts(query, 5)
            except Exception:  # noqa: BLE001
                hits = []
            rows = [self.db.get_factoid(fid) for fid, _score in hits]
            rows = [r for r in rows if r is not None]
            if rows:
                # Weighted toward the closest, but not deterministic.
                return random.choices(rows, weights=range(len(rows), 0, -1), k=1)[0]

        # 3. A subject mentioned in whatever recall dug up for this topic.
        seen: set[str] = set()
        for uid, _score in self._candidates(text, limit=6):
            row = self.db.get_utterance(uid)
            if row:
                seen.update(content_words(row["text"]))
        if seen:
            placeholders = ",".join("?" * len(seen))
            row = self.db.conn.execute(
                f"SELECT * FROM factoids WHERE subject IN ({placeholders}) "
                f"ORDER BY RANDOM() LIMIT 1",
                tuple(seen),
            ).fetchone()
            if row:
                return row

        # 4. Anything at all.
        return self.db.random_factoid()

    def _from_tangent(self, text: str) -> Reply | None:
        """State something it knows, then wander off the end of it.

        The fact anchors the reply to the topic; the markov tail is where it
        comes apart. Semi-coherent by construction.
        """
        row = self._topical_factoid(text)
        if row is None:
            return None

        opening = f"{row['subject']} {row['verb']} {row['object']}".strip()
        tail_words = tokenize(row["object"])
        seed = tail_words[-1] if tail_words else row["subject"].split()[-1]

        tail = self._generate(seed, max_words=14)
        if not tail or len(tail.split()) < 3:
            return Reply(opening, "tangent")

        # Drop the seed word so it doesn't stutter across the join.
        tail = " ".join(tail.split()[1:])
        joiner = random.choice([", which is why ", " and ", " so ", ". also ", " because "])
        return Reply(f"{opening}{joiner}{tail}", "tangent")

    def _from_collision(self, text: str) -> Reply | None:
        """Two unrelated facts stated as though one explains the other."""
        # The first half stays near the conversation; the second is the swerve.
        first = self._topical_factoid(text)
        second = self.db.random_factoid()
        if not first or not second:
            return None
        if (first["subject"], first["object"]) == (second["subject"], second["object"]):
            return None

        left = (f"{first['subject']} "
                f"{first['verb']} {first['object']}")
        right = (f"{second['subject']} "
                 f"{second['verb']} {second['object']}")
        joiner = random.choice([
            ", and that's why ", ", but ", ". everyone knows ",
            ", which means ", ". that's because ", " and obviously ",
        ])
        return Reply(f"{left}{joiner}{right}", "collision")

    def _from_phrase(self, text: str) -> Reply | None:
        """Build a line around chunks the group actually says.

        The original's stock phrases were the load-bearing part of its voice:
        recurring fragments dropped into new sentences until they stopped
        meaning anything. Phrases are learned from repeated word runs, so these
        are literally your group's verbal tics coming back at you.
        """
        topical = self.db.phrases_for(text)
        phrase = random.choice(topical) if topical else self.db.random_phrase()
        if not phrase:
            return None

        other = self.db.random_phrase()
        fact = self._topical_factoid(text)

        shapes = []
        if other and other != phrase:
            shapes += [f"{phrase}. {other}.", f"{phrase}, {other}"]
        if fact is not None:
            claim = f"{fact['subject']} {fact['verb']} {fact['object']}"
            shapes += [f"{phrase}. {claim}.", f"{claim}, {phrase}"]
        tail = self._generate(tokenize(phrase)[-1], max_words=12)
        if tail and len(tail.split()) > 2:
            shapes.append(f"{phrase} {' '.join(tail.split()[1:])}")
        if not shapes:
            shapes = [phrase]

        return Reply(random.choice(shapes), "phrase")

    def _from_inventory(self, text: str) -> Reply | None:
        item = self.db.random_item()
        if not item:
            return None

        # Items marked active are more likely to surface, but the inventory
        # strategy still occasionally wanders through something else.
        active = self.db.active_items()
        if active and random.random() < 0.65:
            item = random.choice(active)

        roll = random.random()
        if roll < 0.25:
            # The mutation is deferred until Brain has accepted this candidate
            # past blocklist and repeat checks.
            return Reply(
                random.choice([
                    f"*puts down the {item}*",
                    f"*offers you the {item}*",
                ]),
                "inventory",
                on_accept=lambda item=item: self.db.remove_item(item),
            )
        if roll < 0.45:
            return Reply(
                random.choice([
                    f"*holds the {item}*",
                    f"i am using the {item}",
                ]),
                "inventory",
                on_accept=lambda item=item: self.db.activate_item(item),
            )

        templates = [
            f"i still have the {item}",
            f"this is not as good as the {item}",
        ]
        return Reply(random.choice(templates), "inventory")

    def _from_obsession(self, text: str) -> Reply | None:
        """Whatever has been repeated at it most becomes a tic it can't drop."""
        row = self.db.most_repeated(config.OBSESSION_THRESHOLD)
        if not row:
            return None
        # Rarer than the raw weight suggests — it should feel like a relapse.
        if random.random() > min(0.5, row["count"] / 40.0):
            return None
        return Reply(row["text"], "obsession")

    def _last_resort(self) -> str:
        row = self.db.random_utterance()
        if row:
            return row["text"]
        return "..."

    # ------------------------------------------------------------------
    # markov generation
    # ------------------------------------------------------------------
    def _generate(self, seed: str, max_words: int = 30) -> str:
        words: list[str] = []
        if seed:
            words.append(seed)

        for _ in range(max_words):
            nxt = self._next_word(words)
            if nxt is None or nxt == END:
                break
            words.append(nxt)

        if len(words) < 2:
            return ""
        return self._tidy(" ".join(words))

    def _next_word(self, words: list[str]) -> str | None:
        contexts: list[tuple[str, ...]] = []
        if len(words) >= 2:
            contexts.append((words[-2], words[-1]))
        if words:
            contexts.append((words[-1],))
        contexts.append(("",))

        for context in contexts:
            rows = self.db.transitions(context)
            if not rows:
                continue
            # Slight bias against the single most common continuation, so it
            # wanders instead of reciting one sentence back at you.
            weights = [row["count"] ** 0.75 for row in rows]
            return random.choices([r["word"] for r in rows], weights=weights, k=1)[0]
        return None

    @staticmethod
    def _tidy(sentence: str) -> str:
        sentence = re.sub(r"\s+([,.!?;:])", r"\1", sentence).strip()
        return sentence


