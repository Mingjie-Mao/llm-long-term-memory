from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from llm_long_term_memory.config import ExperimentConfig, ModelConfig
from llm_long_term_memory.evaluation.datasets.longmemeval import HaystackSession, HaystackTurn
from llm_long_term_memory.evaluation.ingest_state import inspect_ingest_state
from llm_long_term_memory.ingest import fingerprint
from llm_long_term_memory.ingest.pipeline import IngestProgress
from llm_long_term_memory.store import (
    Memory,
    NumpyFlatIndex,
    Session,
    SQLiteMemoryStore,
    Turn,
    scoped_session_id,
)


def _source(session_id: str) -> HaystackSession:
    return HaystackSession(
        session_id,
        "2026/01/05",
        [HaystackTurn("user", f"content {session_id}")],
    )


def _build_partial(tmp_path):
    pairs = [("q1", _source(f"s{i}")) for i in range(3)]
    cfg = ExperimentConfig(models=ModelConfig(extractor="extractor-test", embedding_dim=4))
    expected = fingerprint.from_config(cfg, sessions_per_request=2).as_dict()
    store = SQLiteMemoryStore(tmp_path / "state.db")
    store.initialize()
    store.set_meta("ingest_fingerprint", json.dumps(expected))
    now = datetime(2026, 1, 5)
    for _, source in pairs:
        stored_id = scoped_session_id("q1", source.session_id)
        store.add_session(
            Session(
                stored_id,
                "q1",
                now,
                turns=[Turn(f"{stored_id}:0", stored_id, 0, "user", "content", now)],
            )
        )
    first_id = scoped_session_id("q1", "s0")
    store.add_memories([Memory("m0", "q1", "semantic", "fact", 1, source_session_id=first_id)])
    store.close()

    index = NumpyFlatIndex(tmp_path / "state-index", dim=4)
    index.add(["m0"], np.ones((1, 4), dtype=np.float32))
    index.save()
    IngestProgress(
        done_sessions={"q1:s0", "q1:s1"},
        memories_written=1,
    ).save(tmp_path / "state-ingest.json")
    return pairs, expected


def _inspect(tmp_path, pairs, expected):
    return inspect_ingest_state(
        store_dir=tmp_path,
        store_name="state",
        pairs=pairs,
        expected_fingerprint=expected,
        batch_size=2,
        embedding_dim=4,
    )


def test_one_raw_archived_pending_batch_is_safe_to_resume(tmp_path):
    pairs, expected = _build_partial(tmp_path)

    state = _inspect(tmp_path, pairs, expected)

    assert state.safe_to_resume
    assert not state.complete
    assert state.terminal_sessions == 2
    assert state.pending_sessions == 1
    assert state.successful_sessions == 1
    assert state.empty_sessions == 1
    assert state.content_blocked_sessions == 0
    assert state.archived_pending_sessions == 1
    assert state.archived_pending_with_memories == 0
    assert state.database_memories == state.index_ids == state.vector_rows == 1


def test_all_terminal_sessions_produce_a_complete_state(tmp_path):
    pairs, expected = _build_partial(tmp_path)
    IngestProgress(
        done_sessions={"q1:s0", "q1:s1", "q1:s2"},
        memories_written=1,
    ).save(tmp_path / "state-ingest.json")

    state = _inspect(tmp_path, pairs, expected)

    assert state.safe_to_resume
    assert state.complete
    assert state.pending_sessions == 0
    assert state.successful_sessions == 1
    assert state.empty_sessions == 2
    assert state.content_blocked_sessions == 0
    assert state.archived_pending_sessions == 0


def test_pending_memory_and_artifact_count_drift_are_unsafe(tmp_path):
    pairs, expected = _build_partial(tmp_path)
    store = SQLiteMemoryStore(tmp_path / "state.db")
    store.initialize()
    store.add_memories(
        [
            Memory(
                "m2",
                "q1",
                "semantic",
                "partial",
                1,
                source_session_id=scoped_session_id("q1", "s2"),
            )
        ]
    )
    store.close()

    state = _inspect(tmp_path, pairs, expected)

    assert not state.safe_to_resume
    assert state.archived_pending_with_memories == 1
    assert any("pending sessions already have memories" in issue for issue in state.issues)
    assert any("checkpoint/database memory mismatch" in issue for issue in state.issues)


def test_content_blocked_session_is_an_explicit_terminal_status(tmp_path):
    pairs, expected = _build_partial(tmp_path)
    IngestProgress(
        done_sessions={"q1:s0"},
        blocked_sessions={"q1:s1"},
        memories_written=1,
    ).save(tmp_path / "state-ingest.json")

    state = _inspect(tmp_path, pairs, expected)

    assert state.safe_to_resume
    assert not state.complete
    assert state.successful_sessions == 1
    assert state.empty_sessions == 0
    assert state.content_blocked_sessions == 1
    assert state.blocked_sessions_with_memories == 0
