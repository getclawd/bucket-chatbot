#!/usr/bin/env python3
"""Talk to Bucket in a terminal. Useful for training and inspecting it offline.

    python console.py
"""

import sys

from bucket import Bucket, config

BANNER = r"""
  ___
 |   |   b u c k e t
 |   |   type /help for commands, /quit to leave
 \___/
"""

HELP = """commands:
  /stats             what it knows
  /literal <thing>   exactly what it believes about <thing>
  /recall <text>     what its memory retrieves for <text>, and how strongly
  /forget <thing>    make it forget <thing>
  /inventory         what it is carrying
  /say <text>        force a reply without teaching it the line
  /wipe              destroy everything and reseed
  /quit              leave
anything else is said to bucket, and learned from.
inventory examples: "gives bucket a rock", "bucket drops the rock",
or "bucket holds the rock"."""


def main() -> int:
    bot = Bucket()
    print(BANNER)
    print(f"polish layer: {bot.polisher.status()}")
    print(f"database: {config.DB_PATH}\n")

    try:
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue

            if line.startswith("/"):
                command, _, argument = line[1:].partition(" ")
                command = command.lower()
                argument = argument.strip()

                if command in ("quit", "exit", "q"):
                    break
                if command == "help":
                    print(HELP)
                elif command == "stats":
                    print(bot.cmd_stats())
                elif command == "literal":
                    print(bot.cmd_literal(argument))
                elif command == "recall":
                    print(bot.cmd_recall(argument))
                elif command == "forget":
                    print(bot.cmd_forget(argument))
                elif command == "inventory":
                    print(bot.cmd_inventory())
                elif command == "wipe":
                    if input("wipe everything? [y/N] ").strip().lower() == "y":
                        print(bot.cmd_wipe())
                elif command == "say":
                    print(f"bucket> {bot.speak(argument, chat='console')}")
                else:
                    print(f"no such command: /{command}")
                continue

            reply = bot.handle(line, author="you", chat="console", is_private=True)
            if reply:
                print(f"bucket> {reply}")
    finally:
        bot.close()

    print("bucket goes back in the cupboard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
