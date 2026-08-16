#!/usr/bin/env python3
"""Verify that the release stores content without attribution metadata.

    python privacy_test.py
"""

import os
import sqlite3
import tempfile

os.environ["BUCKET_EMBED_BACKEND"] = "none"
os.environ["BUCKET_SEED"] = "0"

from bucket.brain import Brain  # noqa: E402
from bucket.db import BucketDB  # noqa: E402
from bucket.learn import Learner  # noqa: E402


def legacy_database(path: str) -> None:
    """Create the pre-privacy shape used to exercise the migration."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE utterances (
            id INTEGER PRIMARY KEY, text TEXT NOT NULL, norm TEXT NOT NULL UNIQUE,
            count INTEGER NOT NULL DEFAULT 1, author TEXT, chat TEXT, ts REAL NOT NULL
        );
        CREATE TABLE factoids (
            id INTEGER PRIMARY KEY, subject TEXT NOT NULL, verb TEXT NOT NULL,
            object TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 1,
            author TEXT, ts REAL NOT NULL, UNIQUE(subject, verb, object)
        );
        CREATE TABLE inventory (
            id INTEGER PRIMARY KEY, item TEXT NOT NULL UNIQUE, giver TEXT,
            ts REAL NOT NULL, active INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE aliases (alias TEXT PRIMARY KEY, canonical TEXT NOT NULL, ts REAL NOT NULL);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO utterances VALUES (1, 'the red bucket is loud', 'the red bucket is loud', 1,
                                       'sensitive-user', 'private-channel', 1);
        INSERT INTO factoids VALUES (1, 'the red bucket', 'is', 'loud', 1,
                                     'sensitive-user', 1);
        INSERT INTO inventory VALUES (1, 'a stone', 'sensitive-user', 1, 0);
        INSERT INTO aliases VALUES ('red bucket', 'the red bucket', 1);
        """
    )
    conn.commit()
    conn.close()


def main() -> int:
    failures = []
    path = os.path.join(tempfile.mkdtemp(), "legacy.sqlite3")
    legacy_database(path)

    db = BucketDB(path)
    tables = {
        row["name"]
        for row in db.conn.execute("select name from sqlite_master where type = 'table'")
    }
    for table, forbidden in (
        ("utterances", {"author", "chat"}),
        ("factoids", {"author"}),
        ("inventory", {"giver"}),
    ):
        columns = {row["name"] for row in db.conn.execute(f"pragma table_info({table})")}
        leaked = columns & forbidden
        print(f"{table}: {sorted(columns)}")
        if leaked:
            failures.append(f"{table} retained {sorted(leaked)}")

    if "aliases" in tables:
        failures.append("aliases table still exists")

    learner = Learner(db)
    uid = learner.ingest("the blue bucket is quiet")
    row = db.get_utterance(uid)
    if row is None or {"author", "chat"} & set(row.keys()):
        failures.append("new utterance exposes attribution fields")

    item = db.add_item("a feather", 2)
    if {"giver"} & {row["name"] for row in db.conn.execute("pragma table_info(inventory)")}:
        failures.append("new inventory schema exposes giver")
    if item is not None:
        failures.append("unexpected inventory eviction")

    brain = Brain(db)
    strategies = {fn.__name__ for fn in brain._plan("say something")}
    if "_from_attribution" in strategies or hasattr(brain, "_from_attribution"):
        failures.append("attribution strategy is still available")

    db.close()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("privacy schema and attribution removal verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
