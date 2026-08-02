#!/usr/bin/env python3
"""Run Bucket on Telegram and Discord at once, sharing one memory.

    python run_all.py

This is the right way to run both. Two separate processes pointed at the same
database would each hold their own copy of the vector index in RAM, so neither
would be able to semantically recall anything the other learned until a restart —
they'd agree on stored facts and disagree on what they can remember.

One process, one Bucket, one index. Whatever it learns on Discord it can use on
Telegram in the next breath, and vice versa.

Runs whichever surfaces have tokens configured in .env.
"""

import sys
import threading

from bucket import Bucket, config
from bucket.lock import InstanceLock


def start_telegram(bot: Bucket) -> threading.Thread | None:
    from telegram_bot import BucketBot, TelegramClient, TelegramError

    client = TelegramClient(config.TELEGRAM_TOKEN)
    try:
        me = client.call("getMe", http_timeout=15.0)
    except TelegramError as exc:
        print(f"telegram: could not connect ({exc}) — skipping", file=sys.stderr)
        return None

    surface = BucketBot(bot, client, me)
    thread = threading.Thread(target=surface.run, name="telegram", daemon=True)
    thread.start()
    return thread


def main() -> int:
    want_telegram = bool(config.TELEGRAM_TOKEN)
    want_discord = bool(config.DISCORD_TOKEN)

    if not (want_telegram or want_discord):
        print(
            "No tokens configured. Set BUCKET_TELEGRAM_TOKEN and/or "
            "BUCKET_DISCORD_TOKEN in .env.\n"
            "See .env.example for how to get each one.",
            file=sys.stderr,
        )
        return 1

    # Refuse to start a second Bucket against the same database. Two instances
    # polling one Telegram token deadlock each other with HTTP 409 forever.
    lock = InstanceLock(f"{config.DB_PATH}.lock")
    if not lock.acquire():
        print(lock.message(), file=sys.stderr)
        return 1

    # One shared brain. Everything that touches it is serialized by a lock
    # inside Bucket, so both surfaces can call it from different threads.
    bot = Bucket()
    print(f"memory: {bot.embedder.status()}")
    print(f"polish: {bot.polisher.status()}")
    print(f"surfaces: {'telegram ' if want_telegram else ''}{'discord' if want_discord else ''}")
    print()

    httpd = None
    tunnel = None
    try:
        if config.WEBAPP:
            from webapp.server import serve

            httpd = serve(bot)
            threading.Thread(
                target=httpd.serve_forever, name="webapp", daemon=True
            ).start()
            host, port = httpd.server_address
            print(f"mini app: http://{host}:{port}")

            # Telegram needs a public https address. A free quick tunnel gets a
            # new one every launch, so start it here and use what it reports
            # rather than trusting a value pinned in .env. This must happen
            # before Telegram starts, since that registers the menu button.
            if config.TUNNEL and not config.WEBAPP_URL:
                from bucket.tunnel import Tunnel

                tunnel = Tunnel(port, host, config.CLOUDFLARED)
                print("starting tunnel...", end=" ", flush=True)
                url = tunnel.start()
                if url:
                    config.WEBAPP_URL = url
                    print(url)
                else:
                    print("failed")
                    print(f"  {tunnel.reason}", file=sys.stderr)
                    print("  the mini app will only work on this machine",
                          file=sys.stderr)
                    tunnel = None
            elif config.WEBAPP_URL:
                print(f"mini app public url: {config.WEBAPP_URL} (from .env)")

        telegram_thread = start_telegram(bot) if want_telegram else None

        if want_discord:
            # Discord owns the main thread; it runs its own asyncio loop.
            from discord_bot import build_client

            client = build_client(bot)
            client.run(config.DISCORD_TOKEN, log_handler=None)
        elif telegram_thread:
            # Telegram only — just wait on its thread.
            while telegram_thread.is_alive():
                telegram_thread.join(timeout=1.0)
        else:
            print("nothing started.", file=sys.stderr)
            return 1
    except KeyboardInterrupt:
        pass
    finally:
        if tunnel is not None:
            tunnel.stop()
        if httpd is not None:
            httpd.shutdown()
        bot.close()
        lock.release()

    print("\nbucket goes back in the cupboard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
