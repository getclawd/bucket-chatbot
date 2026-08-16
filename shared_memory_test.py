#!/usr/bin/env python3
"""Prove the two surfaces really share one memory, and survive concurrency.

    python shared_memory_test.py
"""

import os
import tempfile
import threading

os.environ["BUCKET_LLM_BACKEND"] = "none"  # test memory, not voice

from bucket import Bucket  # noqa: E402


def main() -> int:
    path = os.path.join(tempfile.mkdtemp(), "shared.sqlite3")
    bot = Bucket(db_path=path, quiet=True)
    failures = []

    print(f"memory backend: {bot.embedder.status()}\n")

    # --- 1. taught on discord, recalled on telegram ---------------------
    print("=== cross-surface recall ===")
    bot.handle("the server room floods every time it rains",
               chat="dc:111", is_private=True)

    # Deliberately no shared content words with what was taught.
    hits = bot._semantic("what happens during a storm", 3) if bot.index is not None else []
    recalled = [bot.db.get_utterance(uid)["text"] for uid, _ in hits]
    print("  taught on discord: the server room floods every time it rains")
    print("  asked on telegram: what happens during a storm")
    for text in recalled[:3]:
        print(f"    -> {text}")

    if bot.index is not None and not any("server room floods" in t for t in recalled):
        failures.append("discord-taught line was not recallable from telegram")

    # Factoids and inventory are shared too.
    bot.handle("a deadline is a suggestion with anxiety",
               chat="dc:111", is_private=True)
    bot.handle("gives bucket a broken keyboard",
               chat="dc:111", is_private=True)

    if not bot.db.factoids_for("a deadline"):
        failures.append("factoid taught on discord not stored")
    if not any("keyboard" in item for item in bot.db.items()):
        failures.append("item given on discord not stored")

    print(f"\n  factoid visible everywhere: {bool(bot.db.factoids_for('a deadline'))}")
    print(f"  inventory visible everywhere: {bot.db.items()}")

    # --- 2. both surfaces hammering it at once --------------------------
    print("\n=== concurrent access (2 surfaces x 25 messages) ===")
    errors: list[str] = []

    def surface(tag: str, count: int) -> None:
        for index in range(count):
            try:
                bot.handle(
                    f"{tag} message number {index} about coffee and deadlines",
                    chat=f"{tag}:999",
                    is_private=True,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{tag}: {type(exc).__name__}: {exc}")

    threads = [
        threading.Thread(target=surface, args=("tg", 25)),
        threading.Thread(target=surface, args=("dc", 25)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        failures.extend(errors[:5])
        print(f"  {len(errors)} error(s):")
        for error in errors[:5]:
            print(f"    {error}")
    else:
        print("  no errors")

    stats = bot.db.stats()
    print(f"  utterances: {stats['utterances']}  vectors: {stats['vectors']}")

    if bot.index is not None and stats["vectors"] < stats["utterances"]:
        failures.append(
            f"vector index fell behind: {stats['vectors']} vectors for "
            f"{stats['utterances']} utterances"
        )

    bot.close()

    print()
    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("one memory, two surfaces, no races. ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
