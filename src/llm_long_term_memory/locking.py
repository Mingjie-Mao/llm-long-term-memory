"""Single-writer locks for the things that cost quota.

Two processes writing the same evaluation artifact corrupt it; two processes
ingesting the same store corrupt the *usage accounting*, which is worse, because
nothing about the store looks wrong afterwards. A duplicate ingest once recorded
211 requests against roughly 320 actually spent, and the gap only showed up as the
daily quota ending early.

The lock is advisory and pid-based: it blocks a second run started from another
shell, which is the case that has actually happened here.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager, suppress
from pathlib import Path


class AlreadyRunning(RuntimeError):
    """Another process holds this lock."""


def process_alive(pid: int) -> bool:
    """Does this pid belong to a running process?

    `os.kill(pid, 0)` is the POSIX idiom and is *not* portable: on Windows signal 0
    is CTRL_C_EVENT, so the "existence check" delivers a real interrupt to the
    console group. Here that meant a lock check aimed at another run would have sent
    Ctrl+C to it — the opposite of the lock's purpose, which is to leave a running
    job alone. The Windows CI job surfaced it as a KeyboardInterrupt landing in a
    different library on each run, after every test had passed.
    """
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False  # no such pid, or it is gone
        try:
            code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                # A handle outlives the process it names, so "openable" is not
                # "running": an exited process still answers, with its exit code.
                return code.value == STILL_ACTIVE
            return True  # cannot tell; treat as live rather than steal the lock
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but is not ours
    return True


def lock_holder(lock: Path) -> int | None:
    """The pid currently holding `lock`, or None if it is free or stale.

    Read-only, so a preflight can report on a lock without taking or clearing it.
    """
    try:
        pid = int(lock.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return pid if process_alive(pid) else None


@contextmanager
def exclusive(path: Path, *, what: str = "job"):
    """Hold a lock beside `path` for the duration of the block.

    A stale lock from a killed process is reclaimed rather than being a permanent
    block: a crash during a quota-limited run is the normal case here, and a lock
    that outlives its owner would make the resume path unusable.
    """
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                raw = lock.read_text(encoding="utf-8").strip()
                owner = int(raw)
            except (OSError, ValueError):
                # O_EXCL creates an empty file before its owner writes the pid. A
                # second process must not mistake that tiny window for a stale lock.
                try:
                    age = time.time() - lock.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age < 5:
                    raise AlreadyRunning(
                        f"another process is acquiring this {what} lock on {path}"
                    ) from None
                owner = None
            if owner is not None and process_alive(owner):
                raise AlreadyRunning(
                    f"pid {owner} is already running this {what} on {path}. "
                    f"Wait for it, or kill it and delete {lock}."
                ) from None
            with suppress(FileNotFoundError):
                lock.unlink()
            continue
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
        finally:
            os.close(descriptor)
        break
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)
