"""SQLite storage for everything Bucket has ever been told.

Five things live in here:
  utterances   every line it has heard, with a repeat counter
  pairs        prompt/reply adjacency (Cleverbot-style)
  chain        n-gram transition counts (order 2 with order 1 backoff)
  factoids     "X is Y" triples
  inventory    the things it is carrying

No account, author, username, or chat identifier is stored. Surface adapters may
use an opaque in-memory conversation key for pacing and short-term context, but
that key never reaches SQLite.
"""

import sqlite3
import time
from pathlib import Path

from .config import config
from .text import content_words, normalize

SCHEMA = """
CREATE TABLE IF NOT EXISTS utterances (
    id      INTEGER PRIMARY KEY,
    text    TEXT NOT NULL,
    norm    TEXT NOT NULL UNIQUE,
    count   INTEGER NOT NULL DEFAULT 1,
    ts      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_utterances_norm ON utterances(norm);

CREATE TABLE IF NOT EXISTS pairs (
    prompt_id INTEGER NOT NULL,
    reply_id  INTEGER NOT NULL,
    count     INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (prompt_id, reply_id)
);
CREATE INDEX IF NOT EXISTS idx_pairs_prompt ON pairs(prompt_id);

CREATE TABLE IF NOT EXISTS word_index (
    word         TEXT NOT NULL,
    utterance_id INTEGER NOT NULL,
    PRIMARY KEY (word, utterance_id)
);
CREATE INDEX IF NOT EXISTS idx_word_index_word ON word_index(word);

CREATE TABLE IF NOT EXISTS words (
    word TEXT PRIMARY KEY,
    df   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chain (
    context TEXT NOT NULL,
    word    TEXT NOT NULL,
    count   INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (context, word)
);
CREATE INDEX IF NOT EXISTS idx_chain_context ON chain(context);

CREATE TABLE IF NOT EXISTS factoids (
    id      INTEGER PRIMARY KEY,
    subject TEXT NOT NULL,
    verb    TEXT NOT NULL,
    object  TEXT NOT NULL,
    count   INTEGER NOT NULL DEFAULT 1,
    ts      REAL NOT NULL,
    UNIQUE (subject, verb, object)
);
CREATE INDEX IF NOT EXISTS idx_factoids_subject ON factoids(subject);

CREATE TABLE IF NOT EXISTS inventory (
    id       INTEGER PRIMARY KEY,
    item     TEXT NOT NULL UNIQUE,
    ts       REAL NOT NULL,
    active   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS phrases (
    text  TEXT PRIMARY KEY,
    words INTEGER NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    ts    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_phrases_count ON phrases(count DESC);

CREATE TABLE IF NOT EXISTS vectors (
    utterance_id INTEGER PRIMARY KEY,
    dim          INTEGER NOT NULL,
    model_key    TEXT NOT NULL DEFAULT '',
    vec          BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_vectors (
    factoid_id INTEGER PRIMARY KEY,
    dim        INTEGER NOT NULL,
    model_key  TEXT NOT NULL DEFAULT '',
    vec        BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# FTS5 gives real BM25 ranking in C, replacing the hand-rolled sqrt-IDF loop in
# search(). Kept in a separate script because FTS5 is a compile-time option —
# if this SQLite build lacks it, the executescript raises and we fall back.
FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS utterances_fts USING fts5(
    text,
    -- porter stems on both sides of the query, so "does it rain a lot" reaches
    -- "it rains here constantly". The old word_index path matched raw tokens
    -- and could not: that query scored zero against a corpus that discussed
    -- nothing else. Stemming is the main recall win here, not BM25 itself.
    content='utterances',
    content_rowid='id',
    tokenize='porter unicode61'
);
"""

# Bump whenever FTS_SCHEMA changes in a way that invalidates existing index
# content — a different tokenizer, most likely. `CREATE TABLE IF NOT EXISTS`
# won't rebuild an index that already exists, and the row count still matches,
# so nothing else would notice that the stored terms are stemmed differently
# from the terms the query now produces. Stored in `meta`; a mismatch drops and
# recreates the index (cheap: it's derived data).
FTS_VERSION = "2-porter"

# FTS5 treats a bare multi-word MATCH as AND-joined, which annihilates recall on
# conversational input ("what do you think about the nuclear codes"). Every
# query gets OR-joined and phrase-quoted instead — see _fts_query().
FTS_SPECIAL = '"()*^:-+'

SEP = "\x1f"  # context separator inside the chain table


class BucketDB:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # WAL lets readers run alongside a writer, and busy_timeout stops a
        # second surface from erroring out instead of waiting its turn.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript(SCHEMA)
        self._migrate_privacy()
        self._migrate_inventory()
        self._migrate_vectors()
        self.conn.commit()
        self.fts = self._init_fts() if config.FTS else False

    def _migrate_privacy(self) -> None:
        """Remove identity columns from databases created by older releases.

        The old schema kept message authors, chat identifiers, inventory givers,
        and an alias table. Rebuild only those tables so their data is physically
        removed while the learned text and derived indexes remain usable. The
        FTS table is derived data and is recreated after this migration.
        """
        def columns(table: str) -> set[str]:
            return {
                row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")
            }

        utterance_columns = columns("utterances")
        factoid_columns = columns("factoids")
        inventory_columns = columns("inventory")
        has_aliases = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'aliases'"
        ).fetchone() is not None

        needs_utterance = bool({"author", "chat"} & utterance_columns)
        needs_factoid = "author" in factoid_columns
        needs_inventory = "giver" in inventory_columns
        if not (needs_utterance or needs_factoid or needs_inventory or has_aliases):
            return

        # FTS5 is an external-content index over utterances. Dropping it before
        # rebuilding the content table avoids leaving an index tied to the old
        # schema; _init_fts() rebuilds it from the clean table below.
        self.conn.execute("DROP TABLE IF EXISTS utterances_fts")

        if needs_utterance:
            self.conn.execute("DROP TABLE IF EXISTS utterances_public")
            self.conn.execute(
                """CREATE TABLE utterances_public (
                    id      INTEGER PRIMARY KEY,
                    text    TEXT NOT NULL,
                    norm    TEXT NOT NULL UNIQUE,
                    count   INTEGER NOT NULL DEFAULT 1,
                    ts      REAL NOT NULL
                )"""
            )
            self.conn.execute(
                """INSERT INTO utterances_public (id, text, norm, count, ts)
                   SELECT id, text, norm, count, ts FROM utterances"""
            )
            self.conn.execute("DROP TABLE utterances")
            self.conn.execute("ALTER TABLE utterances_public RENAME TO utterances")
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_utterances_norm ON utterances(norm)"
            )

        if needs_factoid:
            self.conn.execute("DROP TABLE IF EXISTS factoids_public")
            self.conn.execute(
                """CREATE TABLE factoids_public (
                    id      INTEGER PRIMARY KEY,
                    subject TEXT NOT NULL,
                    verb    TEXT NOT NULL,
                    object  TEXT NOT NULL,
                    count   INTEGER NOT NULL DEFAULT 1,
                    ts      REAL NOT NULL,
                    UNIQUE (subject, verb, object)
                )"""
            )
            self.conn.execute(
                """INSERT INTO factoids_public
                           (id, subject, verb, object, count, ts)
                   SELECT id, subject, verb, object, count, ts FROM factoids"""
            )
            self.conn.execute("DROP TABLE factoids")
            self.conn.execute("ALTER TABLE factoids_public RENAME TO factoids")
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_factoids_subject ON factoids(subject)"
            )

        if needs_inventory:
            self.conn.execute("DROP TABLE IF EXISTS inventory_public")
            self.conn.execute(
                """CREATE TABLE inventory_public (
                    id       INTEGER PRIMARY KEY,
                    item     TEXT NOT NULL UNIQUE,
                    ts       REAL NOT NULL,
                    active   INTEGER NOT NULL DEFAULT 0
                )"""
            )
            active = "active" if "active" in inventory_columns else "0"
            self.conn.execute(
                f"""INSERT INTO inventory_public (id, item, ts, active)
                    SELECT id, item, ts, {active} FROM inventory"""
            )
            self.conn.execute("DROP TABLE inventory")
            self.conn.execute("ALTER TABLE inventory_public RENAME TO inventory")

        if has_aliases:
            # Alias relationships were an identity-resolution feature. They are
            # not needed for content recall and can expose a map of real names.
            self.conn.execute("DROP TABLE aliases")

        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('privacy_schema', '1') "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        )

    def _migrate_inventory(self) -> None:
        """Add inventory state to databases created before action tracking."""
        columns = {
            row["name"] for row in self.conn.execute("PRAGMA table_info(inventory)")
        }
        if "active" not in columns:
            self.conn.execute(
                "ALTER TABLE inventory ADD COLUMN active INTEGER NOT NULL DEFAULT 0"
            )

    def _migrate_vectors(self) -> None:
        """Track the embedding configuration that produced each vector.

        Vector rows are derived data, so rows from before this column existed
        deliberately keep the empty default and are backfilled on the next
        startup with an available embedder.
        """
        for table in ("vectors", "fact_vectors"):
            columns = {
                row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")
            }
            if "model_key" not in columns:
                self.conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN model_key TEXT NOT NULL DEFAULT ''"
                )

    def _init_fts(self) -> bool:
        """Create the FTS5 index and backfill it. False if FTS5 isn't available.

        FTS5 is a compile-time SQLite option, so this can legitimately fail on
        a stock Python build. The caller keeps working off the word_index path
        when it does.
        """
        # An existing index with no recorded version predates versioning, so its
        # tokenizer is unknown — treat that as a mismatch too rather than
        # trusting it.
        exists = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'utterances_fts'"
        ).fetchone()
        if exists and self.get_meta("fts_version") != FTS_VERSION:
            self.conn.execute("DROP TABLE IF EXISTS utterances_fts")
            self.conn.commit()
        try:
            self.conn.executescript(FTS_SCHEMA)
        except sqlite3.Error:
            return False
        self.set_meta("fts_version", FTS_VERSION)
        # An external-content table isn't populated by CREATE, and it silently
        # drifts if rows were added while FTS was switched off. Its own docsize
        # table is the only honest count — querying `rowid` on the FTS table
        # reads through to `utterances`, so it always looks complete.
        # Mismatch means rebuild; on a corpus this size that costs milliseconds.
        indexed = self.conn.execute(
            "SELECT COUNT(*) AS n FROM utterances_fts_docsize"
        ).fetchone()["n"]
        total = self.conn.execute("SELECT COUNT(*) AS n FROM utterances").fetchone()["n"]
        if indexed != total:
            self.conn.execute(
                "INSERT INTO utterances_fts (utterances_fts) VALUES ('rebuild')"
            )
        self.conn.commit()
        return True

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # utterances
    # ------------------------------------------------------------------
    def add_utterance(self, text: str, reinforce: bool = True) -> int | None:
        """Store a line. Returns its id, or None if there was nothing to store.

        `reinforce=False` stores the line without bumping its repeat counter.
        Bucket's own replies use this: it still learns from what it said, but
        saying something no longer makes it likelier to say it again. Without
        that, `count` feeds the obsession strategy, which makes it repeat the
        line, which raises `count` — a loop with nothing damping it.
        """
        text = text.strip()
        norm = normalize(text)
        if not norm:
            return None

        row = self.conn.execute(
            "SELECT id FROM utterances WHERE norm = ?", (norm,)
        ).fetchone()
        if row:
            if reinforce:
                self.conn.execute(
                    "UPDATE utterances SET count = count + 1, ts = ? WHERE id = ?",
                    (time.time(), row["id"]),
                )
            else:
                self.conn.execute(
                    "UPDATE utterances SET ts = ? WHERE id = ?", (time.time(), row["id"])
                )
            self.conn.commit()
            return row["id"]

        cur = self.conn.execute(
            "INSERT INTO utterances (text, norm, ts) VALUES (?, ?, ?)",
            (text, norm, time.time()),
        )
        uid = int(cur.lastrowid)
        if self.fts:
            self.conn.execute(
                "INSERT INTO utterances_fts (rowid, text) VALUES (?, ?)", (uid, text)
            )
        for word in set(content_words(text)):
            self.conn.execute(
                "INSERT OR IGNORE INTO word_index (word, utterance_id) VALUES (?, ?)",
                (word, uid),
            )
            self.conn.execute(
                "INSERT INTO words (word, df) VALUES (?, 1) "
                "ON CONFLICT(word) DO UPDATE SET df = df + 1",
                (word,),
            )
        self.conn.commit()
        return uid

    def get_utterance(self, uid: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM utterances WHERE id = ?", (uid,)).fetchone()

    def random_utterance(self) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM utterances ORDER BY RANDOM() LIMIT 1"
        ).fetchone()

    def most_repeated(self, minimum: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM utterances WHERE count >= ? ORDER BY count DESC, RANDOM() LIMIT 1",
            (minimum,),
        ).fetchone()

    def rebuild_fts(self) -> None:
        """Resync the full-text index with `utterances`.

        Call after deleting utterances directly (the maintenance scripts do).
        With an external-content FTS5 table, a bare DELETE on the content table
        leaves the index holding rows that no longer exist, so searches keep
        returning ids that resolve to nothing.
        """
        if not self.fts:
            return
        self.conn.execute("INSERT INTO utterances_fts (utterances_fts) VALUES ('rebuild')")
        self.conn.commit()

    @staticmethod
    def _fts_query(text: str) -> str:
        """Turn free-form input into an FTS5 MATCH expression.

        Two things have to happen. FTS5 joins bare terms with AND, so a whole
        sentence matches almost nothing — content words get OR-joined instead.
        And the query syntax has operators (`*`, `^`, `:`, `-`, quotes) that
        raise sqlite3.OperationalError on stray punctuation, so each term is
        phrase-quoted, which makes it a literal.
        """
        terms = []
        for word in dict.fromkeys(content_words(text)):
            cleaned = "".join(c for c in word if c not in FTS_SPECIAL).strip()
            if cleaned:
                terms.append(f'"{cleaned}"')
        return " OR ".join(terms)

    def fts_search(self, text: str, limit: int = 12) -> list[tuple[int, float]]:
        """Lexical recall via SQLite's own BM25. Returns (id, score), best first.

        bm25() returns a *negative* number where more negative is a better
        match, so it's flipped to keep the same "bigger is better" contract as
        search(). Scores aren't comparable between the two paths in absolute
        terms — Brain normalizes them to 0-1 before blending anyway.
        """
        query = self._fts_query(text)
        if not query:
            return []
        try:
            rows = self.conn.execute(
                "SELECT rowid AS uid, -bm25(utterances_fts) AS score "
                "FROM utterances_fts WHERE utterances_fts MATCH ? "
                "ORDER BY score DESC LIMIT ?",
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # Malformed MATCH despite the sanitizing above — degrade to the
            # path that can't fail rather than losing lexical recall entirely.
            return self.search(text, limit)
        return [(row["uid"], row["score"]) for row in rows]

    def lexical_search(self, text: str, limit: int = 12) -> list[tuple[int, float]]:
        """Whichever lexical path is configured and actually available."""
        if self.fts and config.FTS:
            return self.fts_search(text, limit)
        return self.search(text, limit)

    def search(self, text: str, limit: int = 12) -> list[tuple[int, float]]:
        """Find utterances sharing rare-ish words with `text`. Returns (id, score)."""
        wanted = set(content_words(text))
        if not wanted:
            return []

        total = self.conn.execute("SELECT COUNT(*) AS n FROM utterances").fetchone()["n"] or 1
        placeholders = ",".join("?" * len(wanted))
        rows = self.conn.execute(
            f"SELECT wi.utterance_id AS uid, wi.word AS word, w.df AS df "
            f"FROM word_index wi JOIN words w ON w.word = wi.word "
            f"WHERE wi.word IN ({placeholders})",
            tuple(wanted),
        ).fetchall()

        scores: dict[int, float] = {}
        for row in rows:
            # Rare shared words count for more than common ones.
            weight = (total / (1.0 + row["df"])) ** 0.5
            scores[row["uid"]] = scores.get(row["uid"], 0.0) + weight

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:limit]

    # ------------------------------------------------------------------
    # pairs
    # ------------------------------------------------------------------
    def add_pair(self, prompt_id: int, reply_id: int) -> None:
        if prompt_id == reply_id:
            return
        self.conn.execute(
            "INSERT INTO pairs (prompt_id, reply_id) VALUES (?, ?) "
            "ON CONFLICT(prompt_id, reply_id) DO UPDATE SET count = count + 1",
            (prompt_id, reply_id),
        )
        self.conn.commit()

    def replies_to(self, prompt_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT u.*, p.count AS pair_count FROM pairs p "
            "JOIN utterances u ON u.id = p.reply_id "
            "WHERE p.prompt_id = ? ORDER BY p.count DESC",
            (prompt_id,),
        ).fetchall()

    # ------------------------------------------------------------------
    # markov chain
    # ------------------------------------------------------------------
    def add_transition(self, context: tuple[str, ...], word: str) -> None:
        self.conn.execute(
            "INSERT INTO chain (context, word) VALUES (?, ?) "
            "ON CONFLICT(context, word) DO UPDATE SET count = count + 1",
            (SEP.join(context), word),
        )

    def transitions(self, context: tuple[str, ...]) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT word, count FROM chain WHERE context = ?", (SEP.join(context),)
        ).fetchall()

    def random_start(self) -> str | None:
        row = self.conn.execute(
            "SELECT word FROM chain WHERE context = ? ORDER BY RANDOM() LIMIT 1", ("",)
        ).fetchone()
        return row["word"] if row else None

    # ------------------------------------------------------------------
    # factoids
    # ------------------------------------------------------------------
    def add_factoid(self, subject: str, verb: str, obj: str) -> None:
        subject = normalize(subject).lstrip("@").lower()
        self.conn.execute(
            "INSERT INTO factoids (subject, verb, object, ts) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(subject, verb, object) DO UPDATE SET count = count + 1",
            (subject, verb.lower(), obj, time.time()),
        )
        self.conn.commit()

    def factoids_for(self, subject: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM factoids WHERE subject = ? ORDER BY count DESC, RANDOM()",
            (normalize(subject).lstrip("@").lower(),),
        ).fetchall()

    def mentions(self, subject: str, limit: int = 8) -> list[sqlite3.Row]:
        """Lines that literally contain `subject`, most-repeated first.

        Substring matching on `norm`, deliberately not the FTS/BM25 path: this
        backs the introspection commands, where "show me what you actually have"
        must not be filtered through relevance ranking or stemming. A name the
        chain can emit but that never became a factoid is only findable here.
        """
        subject = normalize(subject)
        if not subject:
            return []
        return self.conn.execute(
            "SELECT * FROM utterances WHERE norm LIKE ? "
            "ORDER BY count DESC, id DESC LIMIT ?",
            (f"%{subject}%", limit),
        ).fetchall()

    def known_subjects_in(self, text: str) -> list[str]:
        """Which known factoid subjects appear in this line? Longest first."""
        low = " " + normalize(text) + " "
        rows = self.conn.execute("SELECT DISTINCT subject FROM factoids").fetchall()
        hits = [r["subject"] for r in rows if f" {r['subject']} " in low]
        hits.sort(key=len, reverse=True)
        return hits

    # ------------------------------------------------------------------
    # phrases — recurring chunks it can drop into a sentence
    # ------------------------------------------------------------------
    def add_phrase(self, text: str, words: int) -> None:
        self.conn.execute(
            "INSERT INTO phrases (text, words, ts) VALUES (?, ?, ?) "
            "ON CONFLICT(text) DO UPDATE SET count = count + 1",
            (text, words, time.time()),
        )

    def random_phrase(self, min_count: int = 2, min_words: int = 2) -> str | None:
        row = self.conn.execute(
            "SELECT text FROM phrases WHERE count >= ? AND words >= ? "
            "ORDER BY RANDOM() LIMIT 1",
            (min_count, min_words),
        ).fetchone()
        return row["text"] if row else None

    def phrases_for(self, text: str, min_count: int = 2, limit: int = 8) -> list[str]:
        """Known phrases sharing a content word with this text."""
        wanted = set(content_words(text))
        if not wanted:
            return []
        clauses = " OR ".join(["text LIKE ?"] * len(wanted))
        params = [f"%{word}%" for word in wanted] + [min_count, limit]
        return [
            r["text"]
            for r in self.conn.execute(
                f"SELECT text FROM phrases WHERE ({clauses}) AND count >= ? "
                f"ORDER BY count DESC, RANDOM() LIMIT ?",
                params,
            )
        ]

    def top_phrases(self, limit: int = 20, min_count: int = 2) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT text, count, words FROM phrases WHERE count >= ? "
            "ORDER BY count DESC, words DESC LIMIT ?",
            (min_count, limit),
        ).fetchall()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def factoids_by_predicate(self, text: str, limit: int = 12) -> list[sqlite3.Row]:
        """Reverse lookup: 'who is dead' -> subjects whose predicate matches."""
        needle = f"%{normalize(text)}%"
        return self.conn.execute(
            "SELECT * FROM factoids WHERE "
            "  lower(verb || ' ' || object) LIKE ? OR lower(object) LIKE ? "
            "ORDER BY count DESC, subject LIMIT ?",
            (needle, needle, limit),
        ).fetchall()

    def random_factoid(self) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM factoids ORDER BY RANDOM() LIMIT 1"
        ).fetchone()

    def forget(self, subject: str) -> int:
        cur = self.conn.execute("DELETE FROM factoids WHERE subject = ?", (subject.lower(),))
        self.conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # inventory
    # ------------------------------------------------------------------
    def add_item(self, item: str, capacity: int) -> str | None:
        """Take an item. Returns the item dropped to make room, if any."""
        item = item.strip()
        if not item:
            return None
        self.conn.execute(
            "INSERT OR IGNORE INTO inventory (item, ts) VALUES (?, ?)",
            (item, time.time()),
        )
        dropped = None
        count = self.conn.execute("SELECT COUNT(*) AS n FROM inventory").fetchone()["n"]
        if count > capacity:
            row = self.conn.execute(
                "SELECT * FROM inventory WHERE item != ? ORDER BY RANDOM() LIMIT 1", (item,)
            ).fetchone()
            if row:
                dropped = row["item"]
                self.conn.execute("DELETE FROM inventory WHERE id = ?", (row["id"],))
        self.conn.commit()
        return dropped

    def remove_item(self, item: str) -> bool:
        """Discard one case-insensitive inventory item, if it is present."""
        item = item.strip()
        if not item:
            return False
        cur = self.conn.execute(
            "DELETE FROM inventory WHERE lower(item) = lower(?)", (item,)
        )
        if cur.rowcount != 1:
            self.conn.rollback()
            return False
        removed = not self.conn.execute(
            "SELECT 1 FROM inventory WHERE lower(item) = lower(?) LIMIT 1", (item,)
        ).fetchone()
        if removed:
            self.conn.commit()
        else:
            self.conn.rollback()
        return removed

    def activate_item(self, item: str) -> bool:
        """Mark one case-insensitive inventory item as currently active."""
        item = item.strip()
        if not item:
            return False
        self.conn.execute(
            "UPDATE inventory SET active = 1 WHERE lower(item) = lower(?)", (item,)
        )
        activated = bool(self.conn.execute(
            "SELECT 1 FROM inventory WHERE lower(item) = lower(?) AND active = 1 LIMIT 1",
            (item,),
        ).fetchone())
        if activated:
            self.conn.commit()
        else:
            self.conn.rollback()
        return activated

    def items(self) -> list[str]:
        return [r["item"] for r in self.conn.execute("SELECT item FROM inventory ORDER BY ts")]

    def random_item(self) -> str | None:
        row = self.conn.execute(
            "SELECT item FROM inventory ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
        return row["item"] if row else None

    def active_items(self) -> list[str]:
        return [
            row["item"]
            for row in self.conn.execute(
                "SELECT item FROM inventory WHERE active = 1 ORDER BY ts"
            )
        ]

    # ------------------------------------------------------------------
    # vectors
    # ------------------------------------------------------------------
    def add_vector(
        self, utterance_id: int, dim: int, blob: bytes, model_key: str = ""
    ) -> None:
        self.conn.execute(
            "INSERT INTO vectors (utterance_id, dim, model_key, vec) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(utterance_id) DO UPDATE SET dim = excluded.dim, "
            "model_key = excluded.model_key, vec = excluded.vec",
            (utterance_id, dim, model_key, blob),
        )
        self.conn.commit()

    def all_vectors(self, dim: int, model_key: str = "") -> list[tuple[int, bytes]]:
        return [
            (row["utterance_id"], row["vec"])
            for row in self.conn.execute(
                "SELECT utterance_id, vec FROM vectors WHERE dim = ? AND model_key = ?",
                (dim, model_key),
            )
        ]

    def unvectorized(
        self, dim: int, limit: int = 500, model_key: str = ""
    ) -> list[sqlite3.Row]:
        """Utterances with no vector, or one from a different embedding model."""
        return self.conn.execute(
            "SELECT u.id, u.text FROM utterances u "
            "LEFT JOIN vectors v ON v.utterance_id = u.id "
            "WHERE v.utterance_id IS NULL OR v.dim != ? OR v.model_key != ? LIMIT ?",
            (dim, model_key, limit),
        ).fetchall()

    def count_unvectorized(self, dim: int, model_key: str = "") -> int:
        return self.conn.execute(
            "SELECT COUNT(*) AS n FROM utterances u "
            "LEFT JOIN vectors v ON v.utterance_id = u.id "
            "WHERE v.utterance_id IS NULL OR v.dim != ? OR v.model_key != ?",
            (dim, model_key),
        ).fetchone()["n"]

    # Facts get their own index. Without one they're reachable only by
    # substring-matching the subject, so "who is violent" can never find
    # "dario is an abuser" — most of what Bucket knows stays invisible.
    def add_fact_vector(
        self, factoid_id: int, dim: int, blob: bytes, model_key: str = ""
    ) -> None:
        self.conn.execute(
            "INSERT INTO fact_vectors (factoid_id, dim, model_key, vec) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(factoid_id) DO UPDATE SET dim = excluded.dim, "
            "model_key = excluded.model_key, vec = excluded.vec",
            (factoid_id, dim, model_key, blob),
        )
        self.conn.commit()

    def all_fact_vectors(self, dim: int, model_key: str = "") -> list[tuple[int, bytes]]:
        return [
            (row["factoid_id"], row["vec"])
            for row in self.conn.execute(
                "SELECT factoid_id, vec FROM fact_vectors WHERE dim = ? AND model_key = ?",
                (dim, model_key),
            )
        ]

    def unvectorized_facts(
        self, dim: int, limit: int = 200, model_key: str = ""
    ) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT f.id, f.subject, f.verb, f.object FROM factoids f "
            "LEFT JOIN fact_vectors v ON v.factoid_id = f.id "
            "WHERE v.factoid_id IS NULL OR v.dim != ? OR v.model_key != ? LIMIT ?",
            (dim, model_key, limit),
        ).fetchall()

    def get_factoid(self, factoid_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM factoids WHERE id = ?", (factoid_id,)
        ).fetchone()

    def prune_fact_vectors(self) -> int:
        """Drop vectors whose fact has been deleted."""
        cur = self.conn.execute(
            "DELETE FROM fact_vectors WHERE factoid_id NOT IN (SELECT id FROM factoids)"
        )
        self.conn.commit()
        return cur.rowcount

    def has_vector(self, utterance_id: int, dim: int, model_key: str = "") -> bool:
        return self.conn.execute(
            "SELECT 1 FROM vectors WHERE utterance_id = ? AND dim = ? AND model_key = ?",
            (utterance_id, dim, model_key),
        ).fetchone() is not None

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, int]:
        def count(table: str) -> int:
            return self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]

        return {
            "utterances": count("utterances"),
            "pairs": count("pairs"),
            "factoids": count("factoids"),
            "transitions": count("chain"),
            "vocabulary": count("words"),
            "inventory": count("inventory"),
            "vectors": count("vectors"),
            "phrases": self.conn.execute(
                "SELECT COUNT(*) AS n FROM phrases WHERE count >= 2"
            ).fetchone()["n"],
        }

    def wipe(self) -> None:
        for table in ("utterances", "pairs", "word_index", "words", "chain",
                      "factoids", "inventory", "vectors", "fact_vectors", "phrases"):
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()
        # utterances_fts isn't in that list: it's external-content, so a plain
        # DELETE on it is an error. Rebuilding against the now-empty table is
        # how you clear it.
        self.rebuild_fts()

    def get_meta(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )
        self.conn.commit()
