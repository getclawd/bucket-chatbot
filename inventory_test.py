#!/usr/bin/env python3
"""Check directional inventory events and stateful inventory replies.

    python inventory_test.py
"""

import os
import tempfile
from unittest.mock import patch

os.environ["BUCKET_LLM_BACKEND"] = "none"
os.environ["BUCKET_EMBED_BACKEND"] = "none"
os.environ["BUCKET_SEED"] = "0"

from bucket import Bucket  # noqa: E402
from bucket.brain import Brain, Reply  # noqa: E402
from bucket.db import BucketDB  # noqa: E402
from bucket.learn import (  # noqa: E402
    INVENTORY_ACTIVATE,
    INVENTORY_DROP,
    INVENTORY_RECEIVE,
    parse_inventory_event,
)


def main() -> int:
    path = os.path.join(tempfile.mkdtemp(), "inventory.sqlite3")
    bot = Bucket(db_path=path, quiet=True)
    failures = []

    print("=== directional parser ===")
    cases = [
        ("alice gives bucket a rusty hammer", INVENTORY_RECEIVE, "rusty hammer"),
        ("alice gives a small glass owl to bucket", INVENTORY_RECEIVE, "small glass owl"),
        ("bucket picks up the knife", INVENTORY_RECEIVE, "knife"),
        ("bucket gives alice the rusty hammer", INVENTORY_DROP, "rusty hammer"),
        ("bucket drops the knife", INVENTORY_DROP, "knife"),
        ("bucket throws the owl", INVENTORY_DROP, "owl"),
        ("bucket is holding the knife", INVENTORY_ACTIVATE, "knife"),
        ("buckethead gives alice a rock", None, None),
    ]
    for text, action, item in cases:
        event = parse_inventory_event(text)
        got = (event.action, event.item) if event else None
        expected = (action, item) if action else None
        ok = got == expected
        print(f"  [{'ok ' if ok else 'FAIL'}] {text} -> {got}")
        if not ok:
            failures.append(f"parser: {text!r}: expected {expected!r}, got {got!r}")

    print("\n=== human state transitions ===")
    bot.learner.ingest("alice gives bucket a rusty hammer", author="alice", chat="t")
    bot.learner.ingest("bucket picks up the knife", author="alice", chat="t")
    if not {"rusty hammer", "knife"}.issubset(bot.db.items()):
        failures.append("receive/pick-up did not add both items")
    print(f"  after receiving: {bot.db.items()}")

    bot.learner.ingest("bucket holds the knife", author="alice", chat="t")
    active = bot.db.conn.execute(
        "SELECT active FROM inventory WHERE lower(item) = 'knife'"
    ).fetchone()
    if active is None or not active["active"]:
        failures.append("hold did not mark the knife active")
    print(f"  active knife: {bool(active and active['active'])}")

    bot.learner.ingest("bucket gives alice the rusty hammer", author="alice", chat="t")
    if "rusty hammer" in bot.db.items():
        failures.append("give did not remove the rusty hammer")
    print(f"  after giving it away: {bot.db.items()}")

    before = bot.db.items()
    bot.learner.ingest("bucket drops a nonexistent sword", author="alice", chat="t")
    if bot.db.items() != before or bot.db.remove_item("nonexistent sword"):
        failures.append("failed drop changed inventory")
    print(f"  failed drop leaves inventory unchanged: {bot.db.items() == before}")

    # Bucket's own output is stored but must never trigger inventory mutations.
    bot.learner.ingest(
        "bucket gives alice the knife",
        author="bucket",
        chat="t",
        reinforce=False,
    )
    if "knife" not in bot.db.items():
        failures.append("reinforce=False incorrectly applied an inventory action")

    print("\n=== capacity eviction ===")
    capacity_db = BucketDB(os.path.join(tempfile.mkdtemp(), "capacity.sqlite3"))
    capacity_db.add_item("one", "alice", 2)
    capacity_db.add_item("two", "alice", 2)
    dropped = capacity_db.add_item("three", "alice", 2)
    items = capacity_db.items()
    print(f"  dropped: {dropped}; remaining: {items}")
    if len(items) != 2 or dropped not in {"one", "two"}:
        failures.append(f"capacity eviction failed: dropped={dropped!r}, items={items!r}")
    capacity_db.close()

    print("\n=== generated state changes commit only after acceptance ===")
    action_db = BucketDB(os.path.join(tempfile.mkdtemp(), "action.sqlite3"))
    action_db.add_item("knife", "alice", 12)
    brain = Brain(action_db)
    brain._plan = lambda _text: [brain._from_inventory]
    with patch("bucket.brain.random.random", return_value=0.1), patch(
        "bucket.brain.random.choice", side_effect=lambda choices: choices[0]
    ):
        reply = brain.respond("say something")
    print(f"  reply: {reply.text!r}; state_changed={reply.state_changed}")
    if not reply.state_changed or "knife" in action_db.items():
        failures.append("accepted inventory action did not remove the item")

    action_db.add_item("a sword", "alice", 12)
    brain2 = Brain(action_db)
    brain2._plan = lambda _text: [brain2._from_inventory]
    with patch("bucket.brain.random.random", return_value=0.1), patch(
        "bucket.brain.random.choice", side_effect=lambda choices: choices[0]
    ), patch.object(action_db, "remove_item", return_value=False):
        failed_reply = brain2.respond("say something")
    if failed_reply.state_changed or "a sword" not in action_db.items():
        failures.append("failed inventory action was presented as committed")
    print(f"  failed action remains uncommitted: {not failed_reply.state_changed}")

    action_db.close()

    print("\n=== derived discard phrases update inventory ===")
    phrase_path = os.path.join(tempfile.mkdtemp(), "phrases.sqlite3")
    phrase_bot = Bucket(db_path=phrase_path, quiet=True)
    for index, phrase in enumerate((
        "*puts down the knife*",
        "*offers you the knife*",
        "*takes the knife*",
    )):
        phrase_bot.db.add_item("knife", "alice", 12)
        phrase_bot.brain.respond = lambda _text, context="", avoid=(), phrase=phrase: Reply(
            phrase, "markov"
        )
        phrase_bot.speak("say something", chat=f"phrase:{index}")
        if "knife" in phrase_bot.db.items():
            failures.append(f"spoken discard phrase did not remove the knife: {phrase}")
    print(f"  removed after puts/offers/takes: {'knife' not in phrase_bot.db.items()}")
    phrase_bot.close()

    bot.close()

    print()
    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("inventory state machine works. ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
