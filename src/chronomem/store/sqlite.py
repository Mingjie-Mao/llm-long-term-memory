"""SQLite-backed MemoryStore.

Chosen over Postgres for the first version because a single file makes the demo
trivially reproducible and FTS5 provides BM25 without a second service. The
`MemoryStore` Protocol exists so a pgvector backend can be added in V2 without
touching the retrieval or packing layers.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path

from .base import LexicalHit, Memory, MemoryStatus, MemoryType, Session

_SCHEMA = Path(__file__).with_name("schema.sql")

# FTS5 treats these as syntax. A raw user question containing one of them raises
# sqlite3.OperationalError, so queries are tokenized and re-quoted rather than
# passed through. Escaping (not stripping) keeps recall on hyphenated terms.
_FTS_TOKEN = re.compile(r"[A-Za-z0-9_]+")

_MEMORY_COLUMNS = (
    "id, user_id, type, content, subject, predicate, object, importance, confidence, "
    "event_time, valid_from, valid_to, ingested_at, replaces_previous, superseded_by, status, "
    "strength, "
    "access_count, last_accessed_at, token_count, source_session_id"
)


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class SQLiteMemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def initialize(self) -> None:
        self._conn.executescript(_SCHEMA.read_text())
        self._conn.commit()

    # ---------------------------------------------------------------- sessions

    def add_session(self, session: Session) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions(id, user_id, started_at, source) VALUES (?,?,?,?)",
                (session.id, session.user_id, _dt(session.started_at), session.source),
            )
            self._conn.executemany(
                "INSERT OR REPLACE INTO turns(id, session_id, turn_index, role, content, ts) "
                "VALUES (?,?,?,?,?,?)",
                [
                    (t.id, t.session_id, t.turn_index, t.role, t.content, _dt(t.ts))
                    for t in session.turns
                ],
            )

    # ---------------------------------------------------------------- memories

    def add_memories(self, memories: list[Memory]) -> None:
        if not memories:
            return
        rows = [
            (
                m.id,
                m.user_id,
                m.type,
                m.content,
                m.subject,
                m.predicate,
                m.object,
                m.importance,
                m.confidence,
                _dt(m.event_time),
                _dt(m.valid_from),
                _dt(m.valid_to),
                _dt(m.ingested_at or datetime.now()),
                int(m.replaces_previous),
                m.superseded_by,
                m.status,
                m.strength,
                m.access_count,
                _dt(m.last_accessed_at),
                m.token_count,
                m.source_session_id,
            )
            for m in memories
        ]
        placeholders = ",".join(["?"] * 21)
        with self._conn:
            self._conn.executemany(
                f"INSERT OR REPLACE INTO memories({_MEMORY_COLUMNS}) VALUES ({placeholders})",
                rows,
            )
            for m in memories:
                self._link_entities(m)

    def _link_entities(self, memory: Memory) -> None:
        for surface in memory.entities:
            canonical = surface.strip().lower()
            if not canonical:
                continue
            entity_id = f"{memory.user_id}:{canonical}"
            self._conn.execute(
                "INSERT OR IGNORE INTO entities(id, user_id, canonical_name, surface_form) "
                "VALUES (?,?,?,?)",
                (entity_id, memory.user_id, canonical, surface),
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO memory_entities(memory_id, entity_id) VALUES (?,?)",
                (memory.id, entity_id),
            )

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        entities = [
            r["surface_form"]
            for r in self._conn.execute(
                "SELECT e.surface_form FROM entities e "
                "JOIN memory_entities me ON me.entity_id = e.id WHERE me.memory_id = ?",
                (row["id"],),
            )
        ]
        return Memory(
            id=row["id"],
            user_id=row["user_id"],
            type=row["type"],
            content=row["content"],
            token_count=row["token_count"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            importance=row["importance"],
            confidence=row["confidence"],
            event_time=_parse(row["event_time"]),
            valid_from=_parse(row["valid_from"]),
            valid_to=_parse(row["valid_to"]),
            ingested_at=_parse(row["ingested_at"]),
            replaces_previous=bool(row["replaces_previous"]),
            superseded_by=row["superseded_by"],
            status=row["status"],
            strength=row["strength"],
            access_count=row["access_count"],
            last_accessed_at=_parse(row["last_accessed_at"]),
            source_session_id=row["source_session_id"],
            entities=entities,
        )

    def get(self, memory_id: str) -> Memory | None:
        row = self._conn.execute(
            f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def get_many(self, memory_ids: list[str]) -> list[Memory]:
        if not memory_ids:
            return []
        marks = ",".join(["?"] * len(memory_ids))
        rows = self._conn.execute(
            f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE id IN ({marks})", memory_ids
        ).fetchall()
        by_id = {r["id"]: self._row_to_memory(r) for r in rows}
        # Preserve caller ordering — retrieval passes ids in ranked order.
        return [by_id[mid] for mid in memory_ids if mid in by_id]

    def iter_active(self, user_id: str) -> list[Memory]:
        rows = self._conn.execute(
            f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE user_id = ? AND status = 'active'",
            (user_id,),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def find_by_predicate(
        self, user_id: str, subject: str, predicate: str, include_superseded: bool = False
    ) -> list[Memory]:
        """Memories sharing a (subject, predicate) key.

        `include_superseded` exists for the temporal resolver, which has to rebuild
        an entire timeline rather than compare against the current head: sessions
        are ingested in arbitrary order, so a memory arriving now can be *older*
        than one already stored and has to slot in before it.
        """
        clause = "" if include_superseded else " AND status = 'active'"
        rows = self._conn.execute(
            f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE user_id = ? AND subject = ? "
            f"AND predicate = ?{clause}",
            (user_id, subject, predicate),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def predicate_keys(self, user_id: str) -> list[tuple[str, str]]:
        """Every (subject, predicate) pair present, for a full re-resolution pass."""
        rows = self._conn.execute(
            "SELECT DISTINCT subject, predicate FROM memories "
            "WHERE user_id = ? AND subject IS NOT NULL AND predicate IS NOT NULL",
            (user_id,),
        ).fetchall()
        return [(r["subject"], r["predicate"]) for r in rows]

    # --------------------------------------------------------------- retrieval

    def search_lexical(self, user_id: str, query: str, limit: int) -> list[LexicalHit]:
        tokens = _FTS_TOKEN.findall(query)
        if not tokens:
            return []
        match = " OR ".join(f'"{t}"' for t in tokens)
        rows = self._conn.execute(
            "SELECT m.id AS id, bm25(memories_fts) AS score "
            "FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
            "WHERE memories_fts MATCH ? AND m.user_id = ? AND m.status = 'active' "
            "ORDER BY score LIMIT ?",  # bm25() is negative; more negative = better
            (match, user_id, limit),
        ).fetchall()
        return [LexicalHit(memory_id=r["id"], score=r["score"]) for r in rows]

    # ----------------------------------------------------------------- updates

    def mark_superseded(self, memory_id: str, superseded_by: str, valid_to: datetime) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE memories SET status='superseded', superseded_by=?, valid_to=? WHERE id=?",
                (superseded_by, _dt(valid_to), memory_id),
            )

    def mark_current(self, memory_id: str) -> None:
        """Reopen a memory as the live value of its key.

        Needed because resolution is not monotonic: ingesting an *older* fact for a
        key can demote the memory that was previously the head, and re-running
        resolution after a correction must be able to promote one back. Without
        this, a key could only ever accumulate superseded rows.
        """
        with self._conn:
            self._conn.execute(
                "UPDATE memories SET status='active', superseded_by=NULL, valid_to=NULL WHERE id=?",
                (memory_id,),
            )

    def set_validity(self, memory_id: str, valid_to: datetime | None) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE memories SET valid_to=? WHERE id=?", (_dt(valid_to), memory_id)
            )

    def record_access(self, memory_ids: list[str], at: datetime) -> None:
        if not memory_ids:
            return
        with self._conn:
            self._conn.executemany(
                "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? "
                "WHERE id = ?",
                [(_dt(at), mid) for mid in memory_ids],
            )

    def count(self, user_id: str | None = None, status: MemoryStatus | None = None) -> int:
        clauses, params = [], []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._conn.execute(f"SELECT COUNT(*) FROM memories{where}", params).fetchone()[0]

    def count_by_type(self, user_id: str) -> dict[MemoryType, int]:
        rows = self._conn.execute(
            "SELECT type, COUNT(*) AS n FROM memories WHERE user_id = ? AND status = 'active' "
            "GROUP BY type",
            (user_id,),
        ).fetchall()
        return {r["type"]: r["n"] for r in rows}

    def close(self) -> None:
        self._conn.close()
