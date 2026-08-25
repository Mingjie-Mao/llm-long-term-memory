"""Read-only integrity audit for a resumable ingestion store.

Unlike a final freeze, a valid partial store may contain one raw archive batch that
has not reached a checkpoint terminal state. This happens when extraction succeeds
and the daily quota is exhausted during deduplication. The batch is safe only when
it is still pending, has no memories, and is no larger than the configured batch.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llm_long_term_memory.evaluation.datasets.longmemeval import HaystackSession
from llm_long_term_memory.ingest import fingerprint
from llm_long_term_memory.ingest.pipeline import IngestProgress, _key
from llm_long_term_memory.store import scoped_session_id


@dataclass(frozen=True, slots=True)
class IngestState:
    expected_sessions: int
    done_sessions: int
    blocked_sessions: int
    terminal_sessions: int
    pending_sessions: int
    successful_sessions: int
    empty_sessions: int
    content_blocked_sessions: int
    blocked_sessions_with_memories: int
    archived_sessions: int
    archived_pending_sessions: int
    archived_pending_with_memories: int
    checkpoint_memories: int
    database_memories: int
    index_ids: int
    unique_index_ids: int
    vector_rows: int
    vector_dimensions: int
    quick_check: str
    foreign_key_violations: int
    fingerprint_match: bool
    fingerprint_differences: tuple[str, ...]
    complete: bool
    safe_to_resume: bool
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_ingest_state(
    *,
    store_dir: Path,
    store_name: str,
    pairs: list[tuple[str, HaystackSession]],
    expected_fingerprint: dict[str, str],
    batch_size: int,
    embedding_dim: int,
) -> IngestState:
    """Inspect checkpoint, SQLite and vector artifacts without changing them."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    checkpoint_path = store_dir / f"{store_name}-ingest.json"
    database_path = store_dir / f"{store_name}.db"
    ids_path = store_dir / f"{store_name}-index.ids.json"
    vectors_path = store_dir / f"{store_name}-index.npy"
    for path in (checkpoint_path, database_path, ids_path, vectors_path):
        if not path.exists():
            raise FileNotFoundError(path)

    progress = IngestProgress.load(checkpoint_path)
    expected = {_key(namespace, session) for namespace, session in pairs}
    scoped_by_key = {
        _key(namespace, session): scoped_session_id(namespace, session.session_id)
        for namespace, session in pairs
    }
    done = progress.done_sessions
    blocked = progress.blocked_sessions
    terminal = done | blocked
    done_scoped = {scoped_by_key[key] for key in done if key in scoped_by_key}
    blocked_scoped = {scoped_by_key[key] for key in blocked if key in scoped_by_key}
    terminal_scoped = done_scoped | blocked_scoped
    expected_scoped = set(scoped_by_key.values())

    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        quick_row = connection.execute("PRAGMA quick_check").fetchone()
        quick = str(quick_row[0]) if quick_row else "missing"
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        archived = {str(row[0]) for row in connection.execute("SELECT id FROM sessions")}
        memory_sources = {
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT source_session_id FROM memories "
                "WHERE source_session_id IS NOT NULL"
            )
        }
        database_memories = int(connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
        meta_row = connection.execute(
            "SELECT value FROM meta WHERE key = 'ingest_fingerprint'"
        ).fetchone()
        stored_fingerprint = json.loads(meta_row[0]) if meta_row else {}
    finally:
        connection.close()

    index_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if not isinstance(index_ids, list):
        raise ValueError("vector index ids artifact must be a list")
    vectors = np.load(vectors_path, mmap_mode="r")
    if vectors.ndim != 2:
        raise ValueError(f"vector artifact must be two-dimensional, got {vectors.shape}")

    archived_pending = (archived & expected_scoped) - terminal_scoped
    pending_with_memories = archived_pending & memory_sources
    successful = done_scoped & memory_sources
    empty = done_scoped - memory_sources
    blocked_with_memories = blocked_scoped & memory_sources
    issues: list[str] = []
    if done & blocked:
        issues.append(f"{len(done & blocked)} checkpoint sessions are both done and blocked")
    if terminal - expected:
        issues.append(f"{len(terminal - expected)} terminal sessions are outside the manifest")
    if terminal_scoped - archived:
        issues.append(f"{len(terminal_scoped - archived)} terminal sessions lack raw archives")
    if archived - expected_scoped:
        issues.append(
            f"{len(archived - expected_scoped)} archived sessions are outside the manifest"
        )
    if len(archived_pending) > batch_size:
        issues.append(
            f"{len(archived_pending)} archived pending sessions exceed one batch of {batch_size}"
        )
    if pending_with_memories:
        issues.append(f"{len(pending_with_memories)} pending sessions already have memories")
    if blocked_with_memories:
        issues.append(f"{len(blocked_with_memories)} content-blocked sessions have memories")
    if quick != "ok":
        issues.append(f"SQLite quick_check returned {quick!r}")
    if foreign_keys:
        issues.append(f"SQLite has {len(foreign_keys)} foreign-key violations")

    moved = tuple(fingerprint.differences(stored_fingerprint, expected_fingerprint))
    if moved:
        issues.append("ingest fingerprint differs: " + "; ".join(moved))
    if progress.memories_written != database_memories:
        issues.append(
            "checkpoint/database memory mismatch: "
            f"{progress.memories_written} versus {database_memories}"
        )
    if len(index_ids) != database_memories:
        issues.append(
            f"index/database memory mismatch: {len(index_ids)} versus {database_memories}"
        )
    if len(set(index_ids)) != len(index_ids):
        issues.append(f"vector index contains {len(index_ids) - len(set(index_ids))} duplicate ids")
    if vectors.shape[0] != len(index_ids):
        issues.append(f"vector row/id mismatch: {vectors.shape[0]} versus {len(index_ids)}")
    if vectors.shape[1] != embedding_dim:
        issues.append(f"vector dimension mismatch: {vectors.shape[1]} versus {embedding_dim}")

    complete = not issues and terminal == expected and not archived_pending
    return IngestState(
        expected_sessions=len(expected),
        done_sessions=len(done),
        blocked_sessions=len(blocked),
        terminal_sessions=len(terminal),
        pending_sessions=len(expected - terminal),
        successful_sessions=len(successful),
        empty_sessions=len(empty),
        content_blocked_sessions=len(blocked_scoped),
        blocked_sessions_with_memories=len(blocked_with_memories),
        archived_sessions=len(archived),
        archived_pending_sessions=len(archived_pending),
        archived_pending_with_memories=len(pending_with_memories),
        checkpoint_memories=progress.memories_written,
        database_memories=database_memories,
        index_ids=len(index_ids),
        unique_index_ids=len(set(index_ids)),
        vector_rows=int(vectors.shape[0]),
        vector_dimensions=int(vectors.shape[1]),
        quick_check=quick,
        foreign_key_violations=len(foreign_keys),
        fingerprint_match=not moved,
        fingerprint_differences=moved,
        complete=complete,
        safe_to_resume=not issues,
        issues=tuple(issues),
    )
