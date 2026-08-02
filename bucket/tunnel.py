"""Run a Cloudflare quick tunnel so Telegram can reach the Mini App.

A free `trycloudflare.com` address is ephemeral: it dies with the process and
comes back different next launch. Pinning it in .env therefore breaks every
restart, and the menu button Telegram already stored points at a dead host.

So Bucket starts the tunnel itself, reads the fresh URL out of cloudflared's
output, and hands it to the Telegram surface before that registers the menu
button. Nothing to copy, nothing to go stale.
"""

import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

URL_RE = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")

# winget drops it here; PATH updates only apply to new shells, so look directly.
KNOWN_PATHS = [
    Path(r"C:\Program Files (x86)\cloudflared\cloudflared.exe"),
    Path(r"C:\Program Files\cloudflared\cloudflared.exe"),
    Path("/usr/local/bin/cloudflared"),
    Path("/usr/bin/cloudflared"),
    Path("/opt/homebrew/bin/cloudflared"),
]


def find_cloudflared(override: str = "") -> str:
    if override:
        return override if Path(override).exists() else ""
    found = shutil.which("cloudflared")
    if found:
        return found
    for candidate in KNOWN_PATHS:
        if candidate.exists():
            return str(candidate)
    return ""


class Tunnel:
    """A cloudflared quick tunnel pointed at a local port."""

    def __init__(self, port: int, host: str = "127.0.0.1", binary: str = ""):
        self.port = port
        self.host = host
        self.binary = find_cloudflared(binary)
        self.url = ""
        self.reason = ""
        self._process: subprocess.Popen | None = None
        self._found = threading.Event()

    def start(self, timeout: float = 60.0) -> str:
        """Launch the tunnel and block until it reports a URL. '' on failure."""
        if not self.binary:
            self.reason = (
                "cloudflared not found — install it with:\n"
                "  winget install --id Cloudflare.cloudflared"
            )
            return ""

        # Quick tunnels must target a loopback address cloudflared can reach.
        target = f"http://{'127.0.0.1' if self.host in ('', '0.0.0.0') else self.host}:{self.port}"
        try:
            self._process = subprocess.Popen(
                [self.binary, "tunnel", "--url", target, "--no-autoupdate"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            self.reason = f"could not start cloudflared: {exc}"
            return ""

        threading.Thread(target=self._read_output, name="tunnel", daemon=True).start()

        if not self._found.wait(timeout):
            self.reason = f"cloudflared gave no url within {timeout:.0f}s"
            self.stop()
            return ""
        return self.url

    def _read_output(self) -> None:
        assert self._process and self._process.stdout
        for line in self._process.stdout:
            if not self.url:
                match = URL_RE.search(line)
                if match:
                    self.url = match.group(0)
                    self._found.set()
        # cloudflared exited; unblock anyone still waiting.
        self._found.set()

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None


def main() -> int:
    """Run a tunnel on its own, for testing: python -m bucket.tunnel [port]"""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
    tunnel = Tunnel(port)
    print(f"cloudflared: {tunnel.binary or 'NOT FOUND'}")
    url = tunnel.start()
    if not url:
        print(tunnel.reason, file=sys.stderr)
        return 1
    print(f"tunnel up: {url}  ->  http://127.0.0.1:{port}")
    print("ctrl-c to stop")
    try:
        while tunnel.alive:
            tunnel._process.wait(timeout=1)
    except KeyboardInterrupt:
        pass
    finally:
        tunnel.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
