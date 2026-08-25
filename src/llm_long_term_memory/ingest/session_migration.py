"""Repair legacy benchmark stores whose session primary keys were not user-scoped."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from llm_long_term_memory.evaluation.datasets.longmemeval import HaystackSession
from llm_long_term_memory.store import Session, SQLiteMemoryStore, Turn, scoped_session_id

from .extract import _parse_date


@dataclass(frozen=True, slots=True)
class SessionMigrationReport:
    sessions_archived: int
    memories_repointed: int
    legacy_sessions_removed: int
    uncheckpointed_sessions_removed: int


def migrate_scoped_sessions(
    store: SQLiteMemoryStore, pairs: list[tuple[str, HaystackSession]]
) -> SessionMigrationReport:
    """Copy completed source sessions under scoped ids and repoint provenance.

    Extraction results and the vector index are untouched. Re-archiving from the
    lossless dataset makes this suitable for salvaging a partial, quota-expensive
    ingest without retaining the namespace collision that made its archive unsafe.
    """
    pair_keys = {(namespace, session.session_id) for namespace, session in pairs}
    unresolved = store._conn.execute(
        "SELECT DISTINCT user_id, source_session_id FROM memories "
        "WHERE source_session_id IS NOT NULL AND source_session_id NOT LIKE 'scoped-session-v1:%'"
    ).fetchall()
    missing = [
        (row["user_id"], row["source_session_id"])
        for row in unresolved
        if tuple(row) not in pair_keys
    ]
    if missing:
        raise ValueError(f"store has provenance outside the completed checkpoint: {missing[0]!r}")

    repointed = 0
    legacy_ids: set[str] = set()
    for namespace, source in pairs:
        legacy_ids.add(source.session_id)
        stored_id = scoped_session_id(namespace, source.session_id)
        timestamp = _parse_date(source.date) or datetime.now()
        store.add_session(
            Session(
                id=stored_id,
                user_id=namespace,
                started_at=timestamp,
                source=f"longmemeval:{source.session_id}",
                turns=[
                    Turn(
                        id=f"{stored_id}:{index}",
                        session_id=stored_id,
                        turn_index=index,
                        role=turn.role,
                        content=turn.content,
                        ts=timestamp,
                    )
                    for index, turn in enumerate(source.turns)
                ],
            )
        )
        with store._conn:
            cursor = store._conn.execute(
                "UPDATE memories SET source_session_id = ? "
                "WHERE user_id = ? AND source_session_id = ?",
                (stored_id, namespace, source.session_id),
            )
        repointed += cursor.rowcount

    legacy_removed = 0
    with store._conn:
        for legacy_id in legacy_ids:
            cursor = store._conn.execute(
                "DELETE FROM sessions WHERE id = ? AND NOT EXISTS "
                "(SELECT 1 FROM memories WHERE source_session_id = sessions.id)",
                (legacy_id,),
            )
            legacy_removed += cursor.rowcount
        # A quota stop archives the batch before the provider call, so the batch in
        # flight may have legacy source rows without a completed checkpoint entry.
        # They have no memories and will be archived under scoped ids on resume.
        cursor = store._conn.execute(
            "DELETE FROM sessions WHERE id NOT LIKE 'scoped-session-v1:%' "
            "AND source LIKE 'longmemeval:%' AND NOT EXISTS "
            "(SELECT 1 FROM memories WHERE source_session_id = sessions.id)"
        )
        legacy_removed += cursor.rowcount

    # An older pipeline archived the in-flight batch *before* asking the model.
    # If quota stopped that call, those scoped rows were present despite never
    # reaching the checkpoint. Keep completed zero-memory sessions, but remove
    # uncheckpointed rows so diagnostics do not mistake them for extractor misses.
    completed_ids = {scoped_session_id(namespace, source.session_id) for namespace, source in pairs}
    uncheckpointed_removed = 0
    candidates = store._conn.execute(
        "SELECT id FROM sessions WHERE source LIKE 'longmemeval:%' AND NOT EXISTS "
        "(SELECT 1 FROM memories WHERE source_session_id = sessions.id)"
    ).fetchall()
    with store._conn:
        for row in candidates:
            if row["id"] in completed_ids:
                continue
            cursor = store._conn.execute("DELETE FROM sessions WHERE id = ?", (row["id"],))
            uncheckpointed_removed += cursor.rowcount

    mismatches = store._conn.execute(
        "SELECT COUNT(*) FROM memories m JOIN sessions s ON s.id=m.source_session_id "
        "WHERE m.user_id != s.user_id"
    ).fetchone()[0]
    if mismatches:
        raise RuntimeError(f"migration left {mismatches} cross-namespace provenance rows")
    return SessionMigrationReport(len(pairs), repointed, legacy_removed, uncheckpointed_removed)
