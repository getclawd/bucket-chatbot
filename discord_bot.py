#!/usr/bin/env python3
"""Bucket on Discord.

    python discord_bot.py          # Discord only
    python run_all.py              # Discord AND Telegram, one shared memory

Needs `pip install discord.py` and BUCKET_DISCORD_TOKEN in .env.

In the Discord developer portal, under Bot -> Privileged Gateway Intents, turn on
MESSAGE CONTENT INTENT. Without it Discord delivers empty message bodies and
Bucket learns nothing.
"""

import asyncio
import sys

try:
    import discord
except ImportError:  # pragma: no cover
    discord = None

from bucket import Bucket, config
from bucket.lock import InstanceLock

HELP_TEXT = """i am bucket. i learn from everything said near me.

talk to me by name, @mention me, reply to me, or dm me and i always answer.
otherwise i chime in on my own every so often.

`!stats`             what i know
`!about <person>`    everything i know about them
`!literal <thing>`   exactly what i believe about <thing>
`!who <thing>`       who or what matches, e.g. `!who is dead`
`!alias <a> <b>`     tell me two names are the same person
`!recall <text>`     what i dig up when you say that
`!inventory`         what i am carrying
`!say <text>`        make me answer something
`!forget <thing>`    admin: make me forget
`!chattiness <0-1>`  admin: how often i butt in (kept after restart)
`!wipe confirm`      admin: destroy everything i know

hand me things: "gives bucket a rock"; drop/use them with "bucket drops the rock"
or "bucket holds the rock"
teach me things: "a rock is a small quiet friend"
"""

PREFIXES = ("!", "/")
MAX_MESSAGE = 1900  # Discord's hard limit is 2000


class BucketDiscordClient(discord.Client if discord else object):
    def __init__(self, bot: Bucket):
        intents = discord.Intents.default()
        intents.message_content = True  # requires the privileged intent
        super().__init__(intents=intents)
        self.bot = bot

    async def on_ready(self) -> None:
        print(f"discord: connected as {self.user} ({self.user.id})")

    # ------------------------------------------------------------------
    @staticmethod
    def message_text(message) -> str:
        """Everything worth learning, not just message.content.

        Discord expands links into embeds with a title and description, so this
        is also how Bucket picks up vocabulary from pages people link.
        """
        parts = [message.content or ""]

        for sticker in getattr(message, "stickers", None) or []:
            if getattr(sticker, "name", None):
                parts.append(sticker.name)

        for embed in getattr(message, "embeds", None) or []:
            for field in (embed.title, embed.description):
                if isinstance(field, str) and field.strip():
                    parts.append(field.strip()[:400])

        return " ".join(p.strip() for p in parts if p and p.strip()).strip()

    async def on_message_edit(self, _before, after) -> None:
        # An edit is text Bucket hasn't heard; learn it without replying.
        if after.author.bot or after.author.id == self.user.id:
            return
        text = self.message_text(after)
        if text:
            await asyncio.to_thread(
                self.bot.learn_only,
                text,
                getattr(after.author, "display_name", None) or str(after.author),
                f"dc:{after.channel.id}",
            )

    async def on_message(self, message) -> None:
        if message.author.bot or message.author.id == self.user.id:
            return

        text = self.message_text(message)
        if not text:
            return

        is_private = isinstance(message.channel, discord.DMChannel)
        author = getattr(message.author, "display_name", None) or str(message.author)
        # Namespaced so a Discord channel id can never collide with a Telegram
        # chat id in the shared conversation state.
        chat = f"dc:{message.channel.id}"

        if text[0] in PREFIXES:
            handled = await self.handle_command(message, text, author)
            if handled:
                return

        mentioned = self.user in message.mentions
        replied_to_bot = False
        if message.reference and isinstance(message.reference.resolved, discord.Message):
            replied_to_bot = message.reference.resolved.author.id == self.user.id

        # Everything blocking (embedding, LLM, sqlite) goes off the event loop.
        reply = await asyncio.to_thread(
            self.bot.handle,
            text,
            author=author,
            chat=chat,
            is_private=is_private,
            is_reply_to_bot=mentioned or replied_to_bot,
        )
        if reply:
            await self.send(message.channel, reply, reference=message)

    # ------------------------------------------------------------------
    async def handle_command(self, message, text: str, author: str) -> bool:
        head, _, argument = text[1:].partition(" ")
        command = head.lower().strip()
        argument = argument.strip()
        chat = f"dc:{message.channel.id}"

        is_admin = (
            not config.DISCORD_ADMIN_IDS or message.author.id in config.DISCORD_ADMIN_IDS
        )
        bot = self.bot

        async def run(fn, *args):
            return await asyncio.to_thread(fn, *args)

        if command in ("start", "help", "bucket"):
            out = HELP_TEXT
        elif command == "stats":
            out = await run(bot.cmd_stats)
        elif command == "literal":
            out = await run(bot.cmd_literal, argument)
        elif command == "recall":
            out = await run(bot.cmd_recall, argument)
        elif command == "about":
            out = await run(bot.cmd_about, argument)
        elif command == "who":
            out = await run(bot.cmd_who, argument)
        elif command == "alias":
            out = await run(bot.cmd_alias, argument)
        elif command == "inventory":
            out = await run(bot.cmd_inventory)
        elif command == "say":
            out = await run(bot.speak, argument, chat)
        elif command == "forget":
            out = await run(bot.cmd_forget, argument) if is_admin else "no"
        elif command == "chattiness":
            out = await run(bot.cmd_chattiness, argument) if is_admin else "no"
        elif command == "wipe":
            if not is_admin:
                out = "no"
            elif argument.lower() != "confirm":
                out = ("this erases everything bucket has ever learned and cannot be "
                       "undone.\nsend: `!wipe confirm`")
            else:
                out = await run(bot.cmd_wipe)
        else:
            return False

        await self.send(message.channel, out)
        return True

    # ------------------------------------------------------------------
    @staticmethod
    async def send(channel, text: str, reference=None) -> None:
        for index in range(0, len(text), MAX_MESSAGE):
            chunk = text[index:index + MAX_MESSAGE]
            try:
                await channel.send(chunk, reference=reference if index == 0 else None)
            except discord.HTTPException as exc:
                print(f"discord: send failed: {exc}", file=sys.stderr)
                return


def build_client(bot: Bucket) -> "BucketDiscordClient":
    if discord is None:
        raise RuntimeError("discord.py is not installed (pip install discord.py)")
    return BucketDiscordClient(bot)


def main() -> int:
    if discord is None:
        print("discord.py is not installed. run: pip install discord.py", file=sys.stderr)
        return 1
    if not config.DISCORD_TOKEN:
        print(
            "No Discord token found.\n\n"
            "1. https://discord.com/developers/applications -> New Application\n"
            "2. Bot -> Reset Token -> copy it into BUCKET_DISCORD_TOKEN in .env\n"
            "3. Bot -> Privileged Gateway Intents -> enable MESSAGE CONTENT INTENT\n"
            "4. OAuth2 -> URL Generator -> scopes: bot; permissions: Send Messages,\n"
            "   Read Message History -> open the generated URL to invite it\n",
            file=sys.stderr,
        )
        return 1

    lock = InstanceLock(f"{config.DB_PATH}.lock")
    if not lock.acquire():
        print(lock.message(), file=sys.stderr)
        return 1

    bot = Bucket()
    client = build_client(bot)
    try:
        client.run(config.DISCORD_TOKEN, log_handler=None)
    except KeyboardInterrupt:
        pass
    finally:
        bot.close()
        lock.release()
    print("\nbucket goes back in the cupboard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
