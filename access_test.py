#!/usr/bin/env python3
"""Verify that destructive commands fail closed when no admin is configured."""

import os

os.environ.setdefault("BUCKET_EMBED_BACKEND", "none")
os.environ.setdefault("BUCKET_LLM_BACKEND", "none")

from bucket import config  # noqa: E402
from bucket.auth import is_admin  # noqa: E402
from telegram_bot import BucketBot  # noqa: E402


class RecordingClient:
    def __init__(self):
        self.sent = []

    def send(self, chat_id, text, reply_to=None):
        self.sent.append((chat_id, text, reply_to))


class StubBucket:
    def __init__(self):
        self.calls = []

    def cmd_forget(self, argument):
        self.calls.append(("forget", argument))
        return "forgot"

    def cmd_chattiness(self, argument):
        self.calls.append(("chattiness", argument))
        return "changed"

    def cmd_wipe(self):
        self.calls.append(("wipe", ""))
        return "wiped"


def main() -> int:
    failures = []
    if is_admin(None, frozenset()) or is_admin(99, frozenset()):
        failures.append("empty allowlists must deny everyone")
    if not is_admin(42, frozenset({42})) or is_admin(43, frozenset({42})):
        failures.append("explicit allowlists do not match exactly")

    original = config.ADMIN_IDS
    try:
        client = RecordingClient()
        bucket = StubBucket()
        surface = BucketBot(bucket, client, {"id": 1, "username": "bucket"})

        config.ADMIN_IDS = frozenset()
        surface.handle_command("/wipe confirm", 7, {"id": 99})
        if bucket.calls or not client.sent or client.sent[-1][1] != "no":
            failures.append("Telegram /wipe was not denied with an empty allowlist")

        config.ADMIN_IDS = frozenset({42})
        surface.handle_command("/forget secret", 7, {"id": 99})
        if bucket.calls or client.sent[-1][1] != "no":
            failures.append("Telegram admin command accepted an unauthorized sender")

        surface.handle_command("/forget secret", 7, {"id": 42})
        if bucket.calls != [("forget", "secret")]:
            failures.append("Telegram admin command rejected the configured admin")
    finally:
        config.ADMIN_IDS = original

    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("admin commands fail closed on empty and non-matching allowlists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
