"""A durable record of which namespaces were erased, and when.

An erasure that only deletes the online rows is incomplete, and the gap is not obvious:
the backups taken before it still hold the data. Restore one — after a disk failure, or
into staging — and the erased namespace is back, searchable, with nothing anywhere saying
it should not be. The person who asked for their data to be deleted has no way to know,
and neither does the operator.

So the deletion is recorded as a fact in its own right. The journal is:

**Append-only, and outside the database it describes.** A tombstone stored in the store is
restored along with everything else, which puts it back to the state it had *before* the
erasure — that is, absent. Keeping it beside the store rather than inside it is the whole
mechanism; anything else recreates the problem it solves.

**One per store.** Named after the store it sits beside: `live.db` keeps
`live.erasures.jsonl`. A single file shared by every store in a directory replayed all of
their erasures into whichever store was restored, deleting a namespace from a store that
had never been asked to delete anything.

**Content-free.** One line holds a namespace id and a timestamp. It has to survive an
erasure, so it must not contain anything the erasure was meant to remove.

**Replayed by the restore, by time rather than by trust.** `tools/backup_restore.py`
re-applies the erasures served *after* the backup was taken, and says how many it applied
and how many the backup already reflected. An erasure served before the backup is already
in the copy's past: whatever that namespace holds there was written afterwards, and
replaying it would delete a returning user's new data. A restore that skipped the replay is
reported as unsafe rather than reported as ok.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

JOURNAL_SUFFIX = ".erasures.jsonl"


def journal_path_for(store_path: str | Path) -> Path:
    """Where a store's erasure journal lives: beside the store, named after it."""
    store = Path(store_path)
    return store.with_name(store.stem + JOURNAL_SUFFIX)


@dataclass(frozen=True, slots=True)
class Erasure:
    user_id: str
    erased_at: str
    # What was removed when the request was served. Counts only — kept so an operator can
    # tell a replay that found nothing ("already gone") from one that found the rows still
    # there ("this backup predates the erasure").
    memories: int = 0
    sessions: int = 0
    turns: int = 0

    def as_line(self) -> str:
        return json.dumps(
            {
                "user_id": self.user_id,
                "erased_at": self.erased_at,
                "memories": self.memories,
                "sessions": self.sessions,
                "turns": self.turns,
            },
            ensure_ascii=False,
        )

    def erased_at_utc(self) -> datetime | None:
        """When the erasure was served, or None if this line's timestamp cannot be read."""
        try:
            moment = datetime.fromisoformat(self.erased_at)
        except ValueError:
            return None
        return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


class ErasureJournal:
    """Append-only record of served erasure requests."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def beside(cls, store_path: str | Path) -> ErasureJournal:
        """The journal for a store, next to it rather than in it."""
        return cls(journal_path_for(store_path))

    def record(self, user_id: str, *, memories: int = 0, sessions: int = 0, turns: int = 0) -> None:
        """Append one erasure and put it on disk before returning.

        Flushed and fsynced rather than left to the OS: the call that writes this line is
        the call that tells a user their data is gone, and a line still in a buffer when
        the process dies is a deletion nothing records. The cost is one fsync per erasure,
        which is a rare operation.
        """
        entry = Erasure(
            user_id=user_id,
            erased_at=datetime.now(UTC).isoformat(),
            memories=memories,
            sessions=sessions,
            turns=turns,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry.as_line() + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def entries(self) -> list[Erasure]:
        """Every erasure on record, oldest first.

        A malformed line is skipped rather than fatal. The journal is append-only and its
        purpose is to be replayed: refusing to read it because one line is corrupt would
        leave every *other* erasure unapplied, which is the failure this file exists to
        prevent.
        """
        if not self.path.is_file():
            return []
        found: list[Erasure] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                found.append(
                    Erasure(
                        user_id=data["user_id"],
                        erased_at=data["erased_at"],
                        memories=int(data.get("memories", 0)),
                        sessions=int(data.get("sessions", 0)),
                        turns=int(data.get("turns", 0)),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return found

    def namespaces(self, *, served_after: datetime | None = None) -> list[str]:
        """Namespaces that must not come back, in first-erased order.

        `served_after` keeps only erasures served at or after that moment — the start of a
        backup's copy. An entry whose timestamp cannot be read is kept in: replaying one
        erasure too many can cost that namespace what it wrote since, while skipping one
        serves erased data again.
        """
        seen: dict[str, None] = {}
        for entry in self.entries():
            if served_after is not None:
                when = entry.erased_at_utc()
                if when is not None and when < served_after:
                    continue
            seen.setdefault(entry.user_id, None)
        return list(seen)
