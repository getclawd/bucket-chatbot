"""Single-instance guard.

Running two Buckets against one bot token makes Telegram kill whichever polled
last, forever, with a 409. That failure is confusing because both processes look
alive and neither makes progress. Cheaper to refuse the second start.

Uses a real OS advisory lock rather than a PID file, so the lock is released
automatically if the process is killed or crashes — no stale lockfile to clear.
"""

import os
from pathlib import Path

try:  # Windows
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


class InstanceLock:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = None
        self.holder = ""

    # Byte 0 is the lock itself; the owner's pid is written from byte 1 on, so
    # writing it can never collide with the locked region. msvcrt.locking()
    # operates at the *current file position*, so every seek(0) here matters:
    # without them a second instance locks a different byte and gets straight in.
    _LOCK_BYTE = 1

    def acquire(self) -> bool:
        """True if we now hold the lock, False if another process does."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            try:
                handle = open(self.path, "r+")
            except FileNotFoundError:
                handle = open(self.path, "w+")
        except OSError:
            # Can't create a lockfile (read-only dir?) — don't block startup.
            return True

        try:
            handle.seek(0)
            if msvcrt is not None:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, self._LOCK_BYTE)
            elif fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:  # pragma: no cover - no locking primitive available
                handle.close()
                return True
        except OSError:
            try:
                handle.seek(self._LOCK_BYTE)
                self.holder = handle.read().strip()
            except OSError:
                pass
            handle.close()
            return False

        try:
            handle.seek(self._LOCK_BYTE)
            handle.truncate(self._LOCK_BYTE)
            handle.write(str(os.getpid()))
            handle.flush()
        except OSError:
            pass  # the pid is only there to make the error message friendlier

        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            if msvcrt is not None:
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, self._LOCK_BYTE)
            elif fcntl is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "InstanceLock":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()

    def message(self) -> str:
        who = f" (pid {self.holder})" if self.holder.isdigit() else ""
        return (
            f"Another Bucket is already running{who}.\n\n"
            "Telegram allows only one poller per bot token — a second instance "
            "makes both fail with HTTP 409 forever.\n\n"
            "Stop the other one first:\n"
            "  Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
            "Select-Object ProcessId, CommandLine\n"
            "  Stop-Process -Id <pid>\n\n"
            f"(lock file: {self.path})"
        )
