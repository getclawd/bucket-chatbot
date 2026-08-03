"""HTTP server behind the Telegram Mini App.

Standard library only, same as the rest of the project. Telegram loads the page
from a public HTTPS address (a Cloudflare Tunnel pointed at this port) and signs
every request with initData, which we verify against the bot token — so there is
no login, and no way to call the API without coming through Telegram.
"""

import hashlib
import hmac
import json
import mimetypes
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bucket import Bucket, config
from bucket.core import BLOCKED

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 64 * 1024


def verify_init_data(init_data: str, token: str, max_age: int = 86400) -> dict | None:
    """Validate Telegram's signed initData. Returns the user, or None.

    Telegram's scheme: the signing key is HMAC-SHA256 of the bot token under the
    literal key "WebAppData", and the payload is every field except `hash`,
    sorted, joined with newlines.
    """
    if not init_data or not token:
        return None
    try:
        fields = dict(urllib.parse.parse_qsl(init_data, strict_parsing=True))
    except ValueError:
        return None

    received = fields.pop("hash", "")
    if not received:
        return None

    payload = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    expected = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, received):
        return None

    # Reject a replayed old payload.
    try:
        if max_age and time.time() - int(fields.get("auth_date", 0)) > max_age:
            return None
    except ValueError:
        return None

    try:
        return json.loads(fields.get("user", "{}"))
    except json.JSONDecodeError:
        return None


class Handler(BaseHTTPRequestHandler):
    bot: Bucket = None  # set by serve()
    server_version = "bucket"

    def log_message(self, *_args) -> None:  # quieter than the default
        pass

    # ------------------------------------------------------------------
    def _authorise(self) -> tuple[dict | None, str]:
        if config.WEBAPP_DEV:
            return {"id": 0, "first_name": "dev"}, ""

        user = verify_init_data(
            self.headers.get("X-Telegram-InitData", ""), config.TELEGRAM_TOKEN
        )
        if not user:
            return None, "open this from inside telegram"
        # Default deny. This used to read `and config.ADMIN_IDS`, so an unset
        # BUCKET_ADMIN_IDS — the shipped default — meant *every* Telegram user who
        # could open the bot got the whole corpus over the public tunnel, which is
        # the opposite of what .env.example promises. Opening it up is now an
        # explicit choice: set BUCKET_WEBAPP_ALLOW_ALL=1.
        if not config.WEBAPP_ALLOW_ALL and user.get("id") not in config.ADMIN_IDS:
            return None, "not on the guest list"
        return user, ""

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        # Telegram loads this in a webview; only it needs to frame us.
        self.send_header("Content-Security-Policy", "frame-ancestors https://*.telegram.org")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json; charset=utf-8")

    # ------------------------------------------------------------------
    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        query = urllib.parse.parse_qs(parsed.query)

        if not route.startswith("/api"):
            return self._static(route)

        user, error = self._authorise()
        if not user:
            return self._json({"error": error}, 403)

        first = lambda key, default="": (query.get(key) or [default])[0]  # noqa: E731

        if route == "/api/stats":
            return self._json(self._stats())
        if route == "/api/people":
            return self._json(self._people())
        if route == "/api/about":
            return self._json(self._about(first("name")))
        if route == "/api/facts":
            return self._json(self._facts(first("q"), int(first("limit", "200"))))
        if route == "/api/recall":
            return self._json(self._recall(first("q")))
        return self._json({"error": "no such endpoint"}, 404)

    def do_POST(self) -> None:
        route = urllib.parse.urlparse(self.path).path.rstrip("/")
        user, error = self._authorise()
        if not user:
            return self._json({"error": error}, 403)

        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._json({"error": "too much"}, 413)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)

        if route == "/api/chat":
            text = (payload.get("text") or "").strip()
            if not text:
                return self._json({"error": "say something"}, 400)
            return self._json(
                self.bot.explain(text, chat=f"webapp:{user.get('id')}",
                                 learn=bool(payload.get("learn")))
            )
        return self._json({"error": "no such endpoint"}, 404)

    # ------------------------------------------------------------------
    def _static(self, route: str) -> None:
        name = "index.html" if route == "/" else route.lstrip("/")
        target = (STATIC / name).resolve()
        # Never serve outside the static directory.
        if not str(target).startswith(str(STATIC.resolve())) or not target.is_file():
            return self._send(404, b"not found", "text/plain")
        kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send(200, target.read_bytes(), kind)

    # ------------------------------------------------------------------
    def _stats(self) -> dict:
        bot = self.bot
        stats = bot.db.stats()
        obsession = bot.db.most_repeated(config.OBSESSION_THRESHOLD)
        return {
            "counts": stats,
            "obsession": {"text": obsession["text"], "count": obsession["count"]}
            if obsession else None,
            "memory": bot.embedder.status(),
            "polish": bot.polisher.status(),
            "chattiness": config.CHATTINESS,
            "learning": config.LEARN,
            "inventory": bot.db.items(),
        }

    def _people(self) -> dict:
        bot = self.bot
        people = []
        for author in bot.db.known_authors():
            if author.lower() == config.NAME:
                continue
            # Dropped, not redacted. This endpoint is a list of *people*, and a
            # blocked handle is a person — rendering them as "[redacted]" would
            # still confirm they exist and publish their line and fact counts.
            if BLOCKED.blocks(author):
                continue
            canonical = bot.db.resolve(author)
            people.append({
                "name": bot.redact(author),
                "canonical": bot.redact(canonical),
                "lines": bot.db.author_total(author),
                "facts": len(bot.db.factoids_for(canonical)),
                "aliases": [bot.redact(a) for a in bot.db.aliases_of(canonical)
                            if a != author and not BLOCKED.blocks(a)],
            })
        people.sort(key=lambda p: -p["lines"])
        return {"people": people}

    def _about(self, name: str) -> dict:
        bot = self.bot
        if not name:
            return {"error": "no name"}
        canonical = bot.db.resolve(name)
        every = {canonical, name.lower(), *bot.db.aliases_of(canonical)}
        lines = []
        for one in every:
            lines.extend(
                {"text": r["text"], "count": r["count"]}
                for r in bot.db.author_lines(one, limit=20)
            )
        lines.sort(key=lambda r: -r["count"])
        return {
            "name": bot.redact(canonical),
            "aliases": sorted(bot.redact(a) for a in every if a != canonical),
            "facts": [
                {"verb": bot.redact(r["verb"]), "object": bot.redact(r["object"]),
                 "count": r["count"], "author": bot.redact(r["author"] or "?")}
                for r in bot.db.factoids_for(canonical)
            ],
            "lines": [{"text": bot.redact(r["text"]), "count": r["count"]}
                      for r in lines[:25]],
            "total": sum(bot.db.author_total(one) for one in every),
        }

    def _facts(self, query: str, limit: int) -> dict:
        bot = self.bot
        limit = max(1, min(limit, 500))
        if query.strip():
            rows = bot.db.factoids_by_predicate(query, limit)
            rows = list(rows) + list(bot.db.factoids_for(query))
        else:
            rows = bot.db.conn.execute(
                "SELECT * FROM factoids ORDER BY count DESC, subject LIMIT ?", (limit,)
            ).fetchall()
        seen, facts = set(), []
        for row in rows:
            key = (row["subject"], row["verb"], row["object"])
            if key in seen:
                continue
            seen.add(key)
            facts.append({"subject": bot.redact(row["subject"]),
                          "verb": bot.redact(row["verb"]),
                          "object": bot.redact(row["object"]),
                          "count": row["count"],
                          "author": bot.redact(row["author"] or "?")})
        return {"facts": facts[:limit]}

    def _recall(self, query: str) -> dict:
        bot = self.bot
        if not query.strip():
            return {"semantic": [], "lexical": []}
        out = {}
        for label, hits in (
            ("semantic", bot._semantic(query, 8) if bot.index is not None else []),
            ("lexical", bot.db.search(query, 8)),
        ):
            rows = []
            for uid, score in hits:
                row = bot.db.get_utterance(uid)
                if row:
                    # The web equivalent of /recall, so it redacts like /recall.
                    rows.append({"score": round(float(score), 3),
                                 "text": bot.redact(row["text"]),
                                 "author": bot.redact(row["author"] or "?")})
            out[label] = rows
        return out


def serve(bot: Bucket, host: str | None = None, port: int | None = None):
    """Start the Mini App server. Returns the ThreadingHTTPServer.

    `port=0` means "pick a free one" and must survive the default lookup —
    `port or config.WEBAPP_PORT` would silently turn it back into the configured
    port, which on Windows binds alongside an already-running instance thanks to
    SO_REUSEADDR and then steals its requests.
    """
    Handler.bot = bot
    address = (
        config.WEBAPP_HOST if host is None else host,
        config.WEBAPP_PORT if port is None else port,
    )
    httpd = ThreadingHTTPServer(address, Handler)
    httpd.daemon_threads = True
    return httpd


def main() -> int:
    bot = Bucket()
    httpd = serve(bot)
    host, port = httpd.server_address
    print(f"mini app on http://{host}:{port}")
    if config.WEBAPP_DEV:
        print("DEV MODE: signature checks are off — do not expose this port")
    if not config.WEBAPP_URL:
        print("BUCKET_WEBAPP_URL is unset; run a tunnel and set it to the https address")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        bot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
