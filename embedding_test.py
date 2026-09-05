#!/usr/bin/env python3
"""Regression tests for embedding identity-aware vector reuse."""

import os
import sqlite3
import tempfile

os.environ["BUCKET_EMBED_BACKEND"] = "none"
os.environ["BUCKET_FTS"] = "0"

from bucket.db import BucketDB  # noqa: E402
from bucket.embed import pack  # noqa: E402


def new_db() -> BucketDB:
    return BucketDB(os.path.join(tempfile.mkdtemp(), "bucket.sqlite3"))


def test_same_dimension_models_are_not_compatible() -> None:
    db = new_db()
    try:
        uid = db.add_utterance("the same dimension is not the same vector space")
        db.add_factoid("model", "uses", "a vector space")
        fid = db.conn.execute("SELECT id FROM factoids").fetchone()["id"]
        model_a = "ollama:model-a:latest:v1"
        model_b = "ollama:model-b:latest:v1"

        db.add_vector(uid, 3, pack([1.0, 0.0, 0.0]), model_a)
        db.add_fact_vector(fid, 3, pack([0.0, 1.0, 0.0]), model_a)

        assert db.all_vectors(3, model_a) == [(uid, pack([1.0, 0.0, 0.0]))]
        assert db.all_fact_vectors(3, model_a) == [(fid, pack([0.0, 1.0, 0.0]))]
        assert db.count_unvectorized(3, model_a) == 0
        assert len(db.unvectorized_facts(3, model_key=model_a)) == 0

        assert db.has_vector(uid, 3, model_b) is False
        assert len(db.unvectorized(3, model_key=model_b)) == 1
        assert len(db.unvectorized_facts(3, model_key=model_b)) == 1
        assert db.count_unvectorized(3, model_b) == 1
    finally:
        db.close()


def test_legacy_vectors_are_marked_for_rebuild() -> None:
    path = os.path.join(tempfile.mkdtemp(), "legacy.sqlite3")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE utterances (
            id INTEGER PRIMARY KEY, text TEXT NOT NULL, norm TEXT NOT NULL UNIQUE,
            count INTEGER NOT NULL DEFAULT 1, ts REAL NOT NULL
        );
        CREATE TABLE factoids (
            id INTEGER PRIMARY KEY, subject TEXT NOT NULL, verb TEXT NOT NULL,
            object TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 1, ts REAL NOT NULL,
            UNIQUE (subject, verb, object)
        );
        CREATE TABLE inventory (
            id INTEGER PRIMARY KEY, item TEXT NOT NULL UNIQUE,
            ts REAL NOT NULL, active INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE vectors (
            utterance_id INTEGER PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL
        );
        CREATE TABLE fact_vectors (
            factoid_id INTEGER PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL
        );
        INSERT INTO utterances(id, text, norm, ts)
            VALUES (1, 'legacy vectors have no embedding identity', 'legacy vectors', 0);
        INSERT INTO factoids(id, subject, verb, object, ts)
            VALUES (1, 'legacy', 'needs', 'rebuild', 0);
        INSERT INTO vectors(utterance_id, dim, vec) VALUES (1, 3, x'000000000000000000000000');
        INSERT INTO fact_vectors(factoid_id, dim, vec) VALUES (1, 3, x'000000000000000000000000');
        """
    )
    conn.commit()
    conn.close()

    db = BucketDB(path)
    try:
        current = "ollama:model-current:latest:v1"
        columns = {
            row["name"]
            for row in db.conn.execute("PRAGMA table_info(vectors)")
        }
        fact_columns = {
            row["name"]
            for row in db.conn.execute("PRAGMA table_info(fact_vectors)")
        }
        assert "model_key" in columns
        assert "model_key" in fact_columns
        assert len(db.unvectorized(3, model_key=current)) == 1
        assert len(db.unvectorized_facts(3, model_key=current)) == 1
    finally:
        db.close()


if __name__ == "__main__":
    test_same_dimension_models_are_not_compatible()
    test_legacy_vectors_are_marked_for_rebuild()
    print("embedding tests passed")
