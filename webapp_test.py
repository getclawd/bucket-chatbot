#!/usr/bin/env python3
"""The Mini App is reachable from the internet, so its auth has to hold.

Checks Telegram's initData signature scheme accepts genuine payloads and rejects
forged, tampered, stale and empty ones — then exercises every endpoint against a
real server on a loopback port.

    python webapp_test.py
"""

import hashlib
import hmac
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

os.environ["BUCKET_LLM_BACKEND"] = "none"
os.environ["BUCKET_EMBED_BACKEND"] = "none"
os.environ["BUCKET_SEED"] = "0"
os.environ["BUCKET_WEBAPP_DEV"] = "0"
os.environ["BUCKET_WEBAPP_ALLOW_ALL"] = "1"

from bucket import Bucket, config  # noqa: E402
from webapp.server import serve, verify_init_data  # noqa: E402

TOKEN = "123456:TESTTOKEN-not-a-real-secret"


def sign(fields: dict, token: str = TOKEN) -> str:
    """Build a correctly signed initData string, the way Telegram does."""
    payload = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    digest = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()
    return urllib.parse.urlencode({**fields, "hash": digest})


def main() -> int:
    failures = []
    user = json.dumps({"id": 1000000001, "first_name": "Elle"})
    now = int(time.time())

    print("=== initData signature ===")
    good = sign({"auth_date": str(now), "user": user, "query_id": "AAA"})
    cases = [
        ("genuine payload", good, TOKEN, True),
        ("wrong bot token", good, "999:OTHER", False),
        ("tampered user", good.replace("Elle", "Eve"), TOKEN, False),
        ("no hash at all", "auth_date=1&user=%7B%7D", TOKEN, False),
        ("empty string", "", TOKEN, False),
        ("stale (2 days old)", sign({"auth_date": str(now - 172800), "user": user}), TOKEN, False),
    ]
    for label, data, token, should_pass in cases:
        got = verify_init_data(data, token) is not None
        ok = got == should_pass
        print(f"  [{'ok ' if ok else 'FAIL'}] {label:<22} {'accepted' if got else 'rejected'}")
        if not ok:
            failures.append(f"signature check wrong for {label}")

    # --- live server ---------------------------------------------------
    print("\n=== endpoints ===")
    path = os.path.join(tempfile.mkdtemp(), "webapp.sqlite3")
    bot = Bucket(db_path=path, quiet=True)
    for line, who in [
        ("ana has good puh", "Ben"),
        ("cheese is the best food", "Cleo"),
        ("the server room floods when it rains", "Cleo"),
    ]:
        bot.learner.ingest(line, author=who, chat="t")

    config.TELEGRAM_TOKEN = TOKEN
    httpd = serve(bot, host="127.0.0.1", port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    base = f"http://{host}:{port}"

    # port=0 must give a free port, never the configured one — otherwise this
    # test binds alongside a running Bucket and tests that instead of itself.
    if port == config.WEBAPP_PORT:
        failures.append(f"serve(port=0) bound the configured port {port}, not a free one")
    print(f"  bound an ephemeral port: {port} (configured is {config.WEBAPP_PORT})")

    def call(path_, body=None, init=good):
        request = urllib.request.Request(
            base + path_,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"X-Telegram-InitData": init,
                     **({"Content-Type": "application/json"} if body else {})},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)

    for endpoint in ("/api/stats", "/api/people", "/api/facts",
                     "/api/about?name=ana", "/api/recall?q=rain"):
        try:
            status, data = call(endpoint)
            summary = ", ".join(list(data)[:4])
            print(f"  [ok ] {endpoint:<28} {status}  {{{summary}}}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {endpoint:<28} {exc}")
            failures.append(f"{endpoint} failed: {exc}")

    try:
        _, chat = call("/api/chat", {"text": "what about the rain", "learn": False})
        print(f"  [ok ] /api/chat                  strategy={chat['strategy']!r} "
              f"reply={chat['reply'][:32]!r}")
        for key in ("reply", "strategy", "raw", "candidates"):
            if key not in chat:
                failures.append(f"/api/chat response missing {key}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] /api/chat  {exc}")
        failures.append(f"/api/chat failed: {exc}")

    print("\n=== unauthenticated access is refused ===")
    for label, init in [("no initData", ""), ("forged", "user=%7B%7D&hash=deadbeef")]:
        try:
            call("/api/stats", init=init)
            print(f"  [FAIL] {label}: got in")
            failures.append(f"{label} was allowed through")
        except urllib.error.HTTPError as exc:
            print(f"  [{'ok ' if exc.code == 403 else 'FAIL'}] {label}: http {exc.code}")
            if exc.code != 403:
                failures.append(f"{label} returned {exc.code}, expected 403")

    print("\n=== BUCKET_LEARN=0 freezes the corpus, webapp included ===")
    # explain(learn=True) is the Mini App's learn toggle, and it was the one path
    # into the corpus that ignored BUCKET_LEARN entirely — handle() and
    # learn_only() both gate on it.
    saved_learn = config.LEARN
    for label, learn_setting, toggle, should_grow in [
        ("learning on,  toggle on ", True, True, True),
        ("learning on,  toggle off", True, False, False),
        ("learning off, toggle on ", False, True, False),
    ]:
        config.LEARN = learn_setting
        before = bot.db.stats()["utterances"]
        result = bot.explain(f"a zebra is a striped horse {label}", learn=toggle)
        grew = bot.db.stats()["utterances"] > before
        ok = grew == should_grow and result["learned"] == should_grow
        print(f"  [{'ok ' if ok else 'FAIL'}] {label} -> corpus grew: {grew}, "
              f"reported learned: {result['learned']}")
        if not ok:
            failures.append(f"{label}: grew={grew} reported={result['learned']}, "
                            f"expected {should_grow}")
    config.LEARN = saved_learn

    print("\n=== the guest list fails closed ===")
    # A correctly signed payload from someone who is not on the guest list must
    # still be refused. The guard used to read `and config.ADMIN_IDS`, so an
    # unset BUCKET_ADMIN_IDS — the shipped default — skipped the check entirely
    # and served the whole corpus to any Telegram user over the public tunnel.
    saved = (config.WEBAPP_ALLOW_ALL, config.ADMIN_IDS)
    for label, allow_all, admins, expected in [
        ("no admins configured", False, frozenset(), 403),
        ("signed but not an admin", False, frozenset({999}), 403),
        ("on the guest list", False, frozenset({1000000001}), 200),
        ("allow-all opted in", True, frozenset(), 200),
    ]:
        config.WEBAPP_ALLOW_ALL, config.ADMIN_IDS = allow_all, admins
        try:
            code = call("/api/stats")[0]
        except urllib.error.HTTPError as exc:
            code = exc.code
        ok = code == expected
        print(f"  [{'ok ' if ok else 'FAIL'}] {label:<26} http {code} (want {expected})")
        if not ok:
            failures.append(f"guest list: {label} returned {code}, expected {expected}")
    config.WEBAPP_ALLOW_ALL, config.ADMIN_IDS = saved

    print("\n=== static files and traversal ===")
    for target, expected in [("/", 200), ("/app.js", 200), ("/app.css", 200),
                             ("/../bucket.sqlite3", 404), ("/nope", 404)]:
        try:
            with urllib.request.urlopen(base + target, timeout=10) as response:
                code = response.status
        except urllib.error.HTTPError as exc:
            code = exc.code
        ok = code == expected
        print(f"  [{'ok ' if ok else 'FAIL'}] {target:<22} {code}")
        if not ok:
            failures.append(f"{target} returned {code}, expected {expected}")

    httpd.shutdown()
    bot.close()

    print()
    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("mini app serves, and only to telegram. ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
