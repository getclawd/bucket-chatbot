#!/usr/bin/env python3
"""Bucket on Telegram.

A thin long-polling client written straight against the Bot API — no third-party
dependency to break on a Python upgrade.

    python telegram_bot.py

Needs BUCKET_TELEGRAM_TOKEN in .env or the environment.
"""

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from bucket import Bucket, config
from bucket.lock import InstanceLock

API = "https://api.telegram.org/bot{token}/{method}"


def message_text(message: dict) -> str:
    """Everything worth learning from a message, not just its `text` field.

    A group chat is mostly not plain text. Stickers carry an emoji, photos and
    GIFs carry captions, polls carry a question and options, and a quoted forward
    carries someone else's words. All of it was previously dropped.
    """
    parts = [message.get("text") or "", message.get("caption") or ""]

    sticker = message.get("sticker") or {}
    if sticker.get("emoji"):
        parts.append(sticker["emoji"])

    poll = message.get("poll") or {}
    if poll.get("question"):
        parts.append(poll["question"])
        parts.extend(option.get("text", "") for option in poll.get("options") or [])

    # Text the sender quoted from an earlier message.
    quote = message.get("quote") or {}
    if quote.get("text"):
        parts.append(quote["text"])

    return " ".join(part.strip() for part in parts if part and part.strip()).strip()


class TelegramError(RuntimeError):
    pass


class TelegramConflict(TelegramError):
    """409 — another process is polling this same bot token."""


class TelegramClient:
    def __init__(self, token: str):
        self.token = token

    # `http_timeout` is the socket timeout; any `timeout` in params is Telegram's
    # own long-poll timeout and goes in the request body.
    def call(self, method: str, http_timeout: float = 30.0, **params) -> dict:
        url = API.format(token=self.token, method=method)
        data = urllib.parse.urlencode(
            {k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
             for k, v in params.items() if v is not None}
        ).encode("utf-8")
        request = urllib.request.Request(url, data=data)

        try:
            with urllib.request.urlopen(request, timeout=http_timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            if exc.code == 409:
                raise TelegramConflict(body) from exc
            raise TelegramError(f"{method} -> HTTP {exc.code}: {body}") from exc

        if not payload.get("ok"):
            raise TelegramError(f"{method} -> {payload.get('description')}")
        return payload["result"]

    def get_updates(self, offset: int, poll_seconds: int = 45) -> list[dict]:
        return self.call(
            "getUpdates",
            http_timeout=poll_seconds + 15,
            offset=offset,
            timeout=poll_seconds,
            # Telegram sends only what you ask for. Without edited_message an
            # edit is invisible; without message_reaction so is every reaction.
            allowed_updates=["message", "edited_message", "message_reaction"],
        )

    def send(self, chat_id: int, text: str, reply_to: int | None = None) -> None:
        # Telegram rejects messages over 4096 characters.
        for chunk in (text[i:i + 4000] for i in range(0, len(text), 4000)):
            self.call(
                "sendMessage",
                chat_id=chat_id,
                text=chunk,
                reply_to_message_id=reply_to,
                allow_sending_without_reply=True,
            )
            reply_to = None


# Telegram only shows commands a bot has registered via setMyCommands. Without
# this the "/" menu is empty and nothing autocompletes.
PUBLIC_COMMANDS = [
    ("help", "what i am and how to talk to me"),
    ("app", "open everything i know"),
    ("stats", "what i know"),
    ("about", "everything i know about a person"),
    ("literal", "exactly what i believe about a thing"),
    ("who", "who or what matches something, e.g. who is dead"),
    ("alias", "tell me two names are the same person"),
    ("recall", "what i dig up when you say something"),
    ("inventory", "what i am carrying"),
    ("say", "make me answer something"),
]

ADMIN_COMMANDS = PUBLIC_COMMANDS + [
    ("forget", "make me forget a thing"),
    ("chattiness", "how often i butt in, 0 to 1"),
    ("wipe", "destroy everything i know"),
]

SHORT_DESCRIPTION = "a chatbot that learns from whatever you say near it."

DESCRIPTION = (
    "i am bucket. i learn from everything said near me and repeat it back, "
    "badly. talk to me by name, reply to me, or just say something. "
    "teach me: \"a rock is a small quiet friend\". "
    "hand me things: \"gives bucket a rock\"; drop/use them with \"bucket drops "
    "the rock\" or \"bucket holds the rock\". "
    "send /help for the rest."
)


class BucketBot:
    def __init__(self, bot: Bucket, client: TelegramClient, me: dict):
        self.bot = bot
        self.client = client
        self.me = me
        self.username = (me.get("username") or "").lower()
        self.bot_id = me["id"]
        self.webapp_url = (
            config.WEBAPP_URL if config.WEBAPP_URL.startswith("https://") else ""
        )
        # Chats whose menu button we've already pointed at the Mini App.
        self._menu_set: set[int] = set()

    # ------------------------------------------------------------------
    def _set_menu_button(self, chat_id: int) -> bool:
        if not self.webapp_url or chat_id in self._menu_set:
            return False
        try:
            self.client.call(
                "setChatMenuButton",
                http_timeout=15.0,
                chat_id=chat_id,
                menu_button={
                    "type": "web_app",
                    "text": "memory",
                    "web_app": {"url": self.webapp_url},
                },
            )
        except (TelegramError, urllib.error.URLError, OSError):
            return False  # cosmetic; never break message handling over it
        self._menu_set.add(chat_id)
        return True

    def _ensure_menu_button(self, chat_id: int, is_private: bool) -> None:
        """Menu buttons only exist in private chats, and only stick per chat."""
        if is_private:
            self._set_menu_button(chat_id)

    def _send_app_button(self, chat_id: int, message: dict) -> None:
        """/app — open the Mini App from a button.

        A `web_app` inline button is only valid in a private chat, so in a group
        this points people at the DM instead.
        """
        if not self.webapp_url:
            self.client.send(
                chat_id,
                "i don't have a mini app running.\n"
                "start a tunnel and set BUCKET_WEBAPP_URL, then restart me.",
            )
            return

        is_private = (message.get("chat") or {}).get("type") == "private"
        if is_private:
            self._set_menu_button(chat_id)
            self.client.call(
                "sendMessage",
                http_timeout=20.0,
                chat_id=chat_id,
                text="everything i know:",
                reply_markup={
                    "inline_keyboard": [[
                        {"text": "open my memory",
                         "web_app": {"url": self.webapp_url}},
                    ]]
                },
            )
        else:
            self.client.call(
                "sendMessage",
                http_timeout=20.0,
                chat_id=chat_id,
                text="my memory only opens in a private chat with me.",
                reply_markup={
                    "inline_keyboard": [[
                        {"text": "open a chat with me",
                         "url": f"https://t.me/{self.username}?start=app"},
                    ]]
                },
            )

    # ------------------------------------------------------------------
    def register_ui(self) -> None:
        """Publish the command menu and profile text.

        Admin commands are registered only into each admin's private chat, so
        everyone else's menu stays clean and nobody is tempted by /wipe.
        """
        def commands(pairs):
            return [{"command": name, "description": text} for name, text in pairs]

        try:
            self.client.call(
                "setMyCommands",
                http_timeout=15.0,
                commands=commands(PUBLIC_COMMANDS),
                scope={"type": "default"},
            )
            for admin_id in config.ADMIN_IDS:
                self.client.call(
                    "setMyCommands",
                    http_timeout=15.0,
                    commands=commands(ADMIN_COMMANDS),
                    scope={"type": "chat", "chat_id": admin_id},
                )
            self.client.call(
                "setMyShortDescription",
                http_timeout=15.0,
                short_description=SHORT_DESCRIPTION,
            )
            self.client.call("setMyDescription", http_timeout=15.0, description=DESCRIPTION)

            # The Mini App opens from the menu button beside the message box.
            #
            # Setting it without a chat_id returns ok:true and is then silently
            # ignored — getChatMenuButton keeps reporting "commands". It only
            # sticks per chat, so seed the admins here and set it for everyone
            # else the first time they DM (see _ensure_menu_button).
            if self.webapp_url:
                for admin_id in config.ADMIN_IDS:
                    self._set_menu_button(admin_id)
                print(f"telegram: mini app -> {self.webapp_url}")
            elif config.WEBAPP_URL:
                print("telegram: BUCKET_WEBAPP_URL must be https:// — mini app skipped",
                      file=sys.stderr)
        except (TelegramError, urllib.error.URLError, OSError) as exc:
            # Cosmetic only — never stop the bot starting over this.
            print(f"telegram: could not publish the command menu ({exc})", file=sys.stderr)
            return

        count = len(PUBLIC_COMMANDS)
        extra = f" (+{len(ADMIN_COMMANDS) - count} admin)" if config.ADMIN_IDS else ""
        print(f"telegram: published {count} commands{extra}")

    # ------------------------------------------------------------------
    def run(self) -> None:
        offset = int(self.bot.db.get_meta("telegram_offset", "0"))
        backoff = 1.0

        self.register_ui()
        print(f"bucket is awake as @{self.username}")
        print(f"polish layer: {self.bot.polisher.status()}")
        print(f"chattiness: {config.CHATTINESS}  |  learning: {'on' if config.LEARN else 'off'}")
        if not config.ADMIN_IDS:
            print("telegram: BUCKET_ADMIN_IDS is unset — /forget, /chattiness and "
                  "/wipe are disabled for everyone.", file=sys.stderr)
        print("ctrl-c to stop\n")

        conflicts = 0
        while True:
            try:
                updates = self.client.get_updates(offset)
                if conflicts:
                    print("telegram: conflict cleared, polling again.", file=sys.stderr)
                    conflicts = 0
                backoff = 1.0
            except TelegramConflict:
                # Another process is polling this token. Retrying faster does not
                # help — the two just take turns evicting each other — so say it
                # plainly, once, and wait for the other one to go away.
                conflicts += 1
                if conflicts == 1:
                    print(
                        "\ntelegram: HTTP 409 — another Bucket is polling this same "
                        "bot token.\n"
                        "  Only one instance may poll a token at a time. Find the "
                        "other one and stop it:\n"
                        "    Get-CimInstance Win32_Process -Filter \"Name like "
                        "'%python%'\" | Select-Object ProcessId, CommandLine\n"
                        "    Stop-Process -Id <pid>\n"
                        "  Waiting for it to exit...\n",
                        file=sys.stderr,
                    )
                elif conflicts % 20 == 0:
                    print(
                        f"telegram: still conflicting after {conflicts} attempts.",
                        file=sys.stderr,
                    )
                time.sleep(15.0)
                continue
            except (TelegramError, urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"[poll error] {exc}", file=sys.stderr)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue

            for update in updates:
                offset = max(offset, update["update_id"] + 1)
                try:
                    self.handle_update(update)
                except Exception as exc:  # noqa: BLE001 - one bad message must not stop the bot
                    print(f"[handler error] {exc}", file=sys.stderr)

            if updates:
                self.bot.db.set_meta("telegram_offset", str(offset))

    # ------------------------------------------------------------------
    def handle_update(self, update: dict) -> None:
        # An edit is a message it hasn't heard yet, so learn from it too.
        message = update.get("message") or update.get("edited_message")
        if not message:
            return

        text = message_text(message)
        if not text.strip():
            return

        chat = message["chat"]
        chat_id = chat["id"]
        is_private = chat.get("type") == "private"
        sender = message.get("from") or {}
        author = sender.get("username") or sender.get("first_name") or str(sender.get("id", "?"))

        self._ensure_menu_button(chat_id, is_private)

        replied = message.get("reply_to_message") or {}
        is_reply_to_bot = (replied.get("from") or {}).get("id") == self.bot_id

        if text.startswith("/"):
            if self.handle_command(message, text, chat_id, sender):
                return
            # Not one of ours — fall through and treat it as ordinary chatter.

        # A mention counts as being addressed.
        if self.username and f"@{self.username}" in text.lower():
            is_reply_to_bot = True

        reply = self.bot.handle(
            text,
            author=author,
            chat=f"tg:{chat_id}",
            is_private=is_private,
            is_reply_to_bot=is_reply_to_bot,
        )
        if reply:
            self.client.send(chat_id, reply, reply_to=message["message_id"])

    # ------------------------------------------------------------------
    def handle_command(self, message: dict, text: str, chat_id: int, sender: dict) -> bool:
        """Returns True if this was a recognised command."""
        head, _, argument = text[1:].partition(" ")
        command, _, target = head.partition("@")
        command = command.lower()
        argument = argument.strip()

        # In a group, "/stats@someotherbot" isn't ours.
        if target and target.lower() != self.username:
            return False

        # Default deny. This used to be `not config.ADMIN_IDS or ...`, so leaving
        # BUCKET_ADMIN_IDS unset handed /wipe — which destroys the whole corpus
        # irreversibly — to every member of every chat the bot is in.
        is_admin = sender.get("id") in config.ADMIN_IDS

        if command in ("start", "help"):
            self.client.send(chat_id, HELP_TEXT)
        elif command in ("app", "memory"):
            self._send_app_button(chat_id, message)
        elif command == "stats":
            self.client.send(chat_id, self.bot.cmd_stats())
        elif command == "literal":
            self.client.send(chat_id, self.bot.cmd_literal(argument))
        elif command == "recall":
            self.client.send(chat_id, self.bot.cmd_recall(argument))
        elif command == "about":
            self.client.send(chat_id, self.bot.cmd_about(argument))
        elif command == "who":
            self.client.send(chat_id, self.bot.cmd_who(argument))
        elif command == "alias":
            self.client.send(chat_id, self.bot.cmd_alias(argument))
        elif command == "inventory":
            self.client.send(chat_id, self.bot.cmd_inventory())
        elif command == "say":
            self.client.send(chat_id, self.bot.speak(argument, chat=f"tg:{chat_id}"))
        elif command == "forget":
            if not is_admin:
                self.client.send(chat_id, "no")
                return True
            self.client.send(chat_id, self.bot.cmd_forget(argument))
        elif command == "chattiness":
            if not is_admin:
                self.client.send(chat_id, "no")
                return True
            self.client.send(chat_id, self.bot.cmd_chattiness(argument))
        elif command == "wipe":
            if not is_admin:
                self.client.send(chat_id, "no")
                return True
            if argument.strip().lower() != "confirm":
                self.client.send(
                    chat_id,
                    "this erases everything bucket has ever learned and cannot be undone.\n"
                    "send: /wipe confirm",
                )
                return True
            self.client.send(chat_id, self.bot.cmd_wipe())
        else:
            return False
        return True


HELP_TEXT = """i am bucket. i learn from everything said near me.

talk to me by name, reply to me, or dm me and i always answer.
otherwise i chime in on my own every so often.

/app               open everything i know
/stats             what i know
/about <person>    everything i know about them
/literal <thing>   exactly what i believe about <thing>
/who <thing>       who or what matches, e.g. /who is dead
/alias <a> <b>     tell me two names are the same person
/recall <text>     what i dig up when you say that
/inventory         what i am carrying
/say <text>        make me answer something
/forget <thing>    admin: make me forget
/chattiness <0-1>  admin: how often i butt in (kept after restart)
/wipe confirm      admin: destroy everything i know

hand me things: "gives bucket a rock"; drop/use them with "bucket drops the rock"
or "bucket holds the rock"
teach me things: "a rock is a small quiet friend"
"""


def main() -> int:
    if not config.TELEGRAM_TOKEN:
        print(
            "No bot token found.\n\n"
            "1. Message @BotFather on Telegram and send /newbot\n"
            "2. Copy .env.example to .env\n"
            "3. Put the token it gives you in BUCKET_TELEGRAM_TOKEN\n\n"
            "For groups, also send @BotFather /setprivacy -> Disable, or bucket\n"
            "will only ever see messages that mention it by name.",
            file=sys.stderr,
        )
        return 1

    lock = InstanceLock(f"{config.DB_PATH}.lock")
    if not lock.acquire():
        print(lock.message(), file=sys.stderr)
        return 1

    client = TelegramClient(config.TELEGRAM_TOKEN)
    try:
        me = client.call("getMe", http_timeout=15.0)
    except TelegramError as exc:
        print(f"could not reach telegram: {exc}", file=sys.stderr)
        lock.release()
        return 1

    bot = Bucket()
    try:
        BucketBot(bot, client, me).run()
    except KeyboardInterrupt:
        print("\nbucket goes back in the cupboard.")
    finally:
        bot.close()
        lock.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())

