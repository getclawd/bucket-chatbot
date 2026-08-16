#!/usr/bin/env python3
"""How often does Bucket butt in, and does it ever flood?

Simulates a busy group and checks two things: it speaks noticeably more than the
old flat 6%, and it never fires several uninvited replies back to back.

    python chatter_test.py
"""

import os
import tempfile
import time
from unittest.mock import patch

os.environ["BUCKET_LLM_BACKEND"] = "none"
os.environ["BUCKET_EMBED_BACKEND"] = "none"  # speed; pacing is what's under test

from bucket import Bucket, config  # noqa: E402

MESSAGES = 400
SECONDS_BETWEEN = 8.0  # a chatty group: a message every 8 seconds


def main() -> int:
    path = os.path.join(tempfile.mkdtemp(), "chatter.sqlite3")
    bot = Bucket(db_path=path, quiet=True)
    failures = []

    print(f"chattiness={config.CHATTINESS}  min_gap={config.MIN_GAP}s  "
          f"boost=x{config.FOLLOWUP_BOOST} for {config.FOLLOWUP_WINDOW}s")
    print(f"\nsimulating {MESSAGES} group messages, one every {SECONDS_BETWEEN:.0f}s "
          f"({MESSAGES * SECONDS_BETWEEN / 60:.0f} minutes)\n")

    clock = [time.time()]
    spoke_at = []

    with patch("bucket.core.time.time", side_effect=lambda: clock[0]):
        for index in range(MESSAGES):
            clock[0] += SECONDS_BETWEEN
            # Nobody addresses it — this is purely uninvited chatter.
            if bot.should_speak("just some group chatter about things",
                                is_private=False, is_reply_to_bot=False, chat="tg:1"):
                spoke_at.append(clock[0])
                bot._last_spoke["tg:1"] = clock[0]

    rate = len(spoke_at) / MESSAGES
    print(f"  spoke {len(spoke_at)} times out of {MESSAGES}  ({rate:.1%} of messages)")
    print(f"  roughly once every {MESSAGES * SECONDS_BETWEEN / max(len(spoke_at), 1) / 60:.1f} minutes")

    # Never twice inside the quiet period.
    gaps = [b - a for a, b in zip(spoke_at, spoke_at[1:])]
    too_close = [g for g in gaps if g < config.MIN_GAP]
    print(f"  shortest gap between replies: {min(gaps) if gaps else 0:.0f}s "
          f"(floor is {config.MIN_GAP:.0f}s)")
    if too_close:
        failures.append(f"{len(too_close)} replies came faster than the {config.MIN_GAP}s floor")

    if rate < 0.10:
        failures.append(f"still too quiet at {rate:.1%}")
    if rate > 0.55:
        failures.append(f"far too talkative at {rate:.1%}")

    # Being spoken to must never be rate limited.
    print("\n  addressed directly, 5 times in a row:")
    addressed = [
        bot.should_speak("bucket what do you think", is_private=False,
                         is_reply_to_bot=True, chat="tg:1")
        for _ in range(5)
    ]
    print(f"    replied {sum(addressed)}/5")
    if not all(addressed):
        failures.append("ignored someone talking directly to it")

    # Persistence across a restart.
    print("\n  /chattiness persistence:")
    print("   ", bot.cmd_chattiness("0.42"))
    bot.close()
    again = Bucket(db_path=path, quiet=True)
    print(f"    after restart: {config.CHATTINESS}")
    if abs(config.CHATTINESS - 0.42) > 1e-9:
        failures.append("chattiness did not survive a restart")
    again.close()

    print()
    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("talks more, still doesn't flood. ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
