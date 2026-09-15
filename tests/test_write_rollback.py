"""A write that fails after the memories are durable must leave nothing behind.

Extraction succeeding is not the write succeeding. Encoding, the index save and the
temporal resolver all run *after* `add_memories`, and a failure in any of them used to
return an error to the caller while the memories stayed in SQLite and the turn stayed in
the session. The retry then re-extracted from scratch and wrote the same facts again under
new ids, so one logical write became two sets of memories and two copies of the turn.

Resolution is the hard half: it edits rows that were already there, closing their validity
interval and pointing them at a successor, and nothing it returns says what those rows held
before. So the write path snapshots them first. These tests are that snapshot's reason for
existing.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from llm_long_term_memory.api.service import MemoryService, TurnExtractionOutcome
from llm_long_term_memory.store import Memory, NumpyFlatIndex, SQLiteMemoryStore


class _StubEncoder:
    dim = 2

    def encode(self, texts, show_progress=False):
        return np.ones((len(texts), 2), dtype=np.float32)

    def encode_one(self, text):
        return np.ones(2, dtype=np.float32)


class _Settings:
    def __init__(self, tmp_path):
        self.store_dir = tmp_path
        self.data_dir = tmp_path
        self.results_dir = tmp_path
        self.gemini_api_key = ""

    @property
    def has_api_key(self):
        return False


def _memory(memory_id: str, content: str, object_: str) -> Memory:
    return Memory(
        id=memory_id,
        user_id="alice",
        type="semantic",
        content=content,
        token_count=4,
        subject="user",
        predicate="lives_in",
        object=object_,
        ingested_at=datetime(2026, 9, 14),
    )


class _Extractor:
    """Returns one fresh memory per call, with a distinct id each time."""

    def __init__(self):
        self.calls = 0

    def extract_turn(self, **_):
        self.calls += 1
        memory = _memory(f"m_new_{self.calls}", "I live in Sydney", "sydney")
        return TurnExtractionOutcome([memory], {"requests": 1})


@pytest.fixture
def service(tmp_path):
    svc = MemoryService.__new__(MemoryService)
    svc.__init__(store_name="t", settings=_Settings(tmp_path), encoder=_StubEncoder())
    svc.store.close()
    store = SQLiteMemoryStore(tmp_path / "m.db")
    store.initialize()
    svc.store = store
    svc.index = NumpyFlatIndex(tmp_path / "idx", dim=2)
    svc.extractor = _Extractor()
    svc.resolver = None
    yield svc
    store.close()


def _state(service):
    """Everything a failed write must not have left behind."""
    turns = sum(
        len(service.store.get_session(sid).turns)
        for sid in service.store.session_ids_for_user("alice")
    )
    return turns, service.store.count("alice"), len(service.index)


def test_a_failed_index_save_leaves_no_turn_no_memory_and_no_vector(service):
    def explode():
        raise OSError("disk full")

    service.index.save = explode
    with pytest.raises(OSError):
        service.add_message("alice", "user", "I live in Sydney", session_id="s1")

    assert _state(service) == (0, 0, 0)


def test_a_failed_resolution_rolls_back_the_whole_write(service):
    class _Resolver:
        def resolve_memories(self, memories):
            raise RuntimeError("resolver blew up")

    service.resolver = _Resolver()
    with pytest.raises(RuntimeError):
        service.add_message("alice", "user", "I live in Sydney", session_id="s1")

    assert _state(service) == (0, 0, 0)


def test_a_partial_resolution_puts_the_superseded_row_back_as_it_was(service):
    """The case a rollback cannot recompute, only restore.

    The resolver closed the earlier memory's interval and pointed it at the new one, then
    failed. Deleting the new memory without undoing that leaves a row marked superseded by
    an id no longer in the table — a fact that silently stops being current, with no
    successor and nothing to say why.
    """
    earlier = _memory("m_old", "I live in Canberra", "canberra")
    service.store.add_memories([earlier])

    class _Resolver:
        def __init__(self, store):
            self.store = store

        def resolve_memories(self, memories):
            self.store.mark_superseded("m_old", memories[0].id, datetime(2026, 9, 14))
            raise RuntimeError("failed after editing an existing row")

    service.resolver = _Resolver(service.store)
    with pytest.raises(RuntimeError):
        service.add_message("alice", "user", "I live in Sydney", session_id="s1")

    restored = service.store.get("m_old")
    assert restored is not None, "the pre-existing memory was deleted by the rollback"
    assert restored.status == "active"
    assert restored.superseded_by is None
    assert restored.valid_to is None
    assert service.store.get("m_new_1") is None
    assert _state(service) == (0, 1, 0)


def test_the_retry_after_a_rolled_back_write_writes_exactly_once(service):
    """The point of the rollback: one logical write leaves one set of memories."""
    saves = {"n": 0}
    real_save = service.index.save

    def flaky():
        saves["n"] += 1
        if saves["n"] == 1:
            raise OSError("disk full")
        return real_save()

    service.index.save = flaky
    with pytest.raises(OSError):
        service.add_message(
            "alice", "user", "I live in Sydney", session_id="s1", idempotency_key="k1"
        )
    service.add_message("alice", "user", "I live in Sydney", session_id="s1", idempotency_key="k1")

    turns, memories, vectors = _state(service)
    assert (turns, memories, vectors) == (1, 1, 1)
    assert service.store.get("m_new_2") is not None, "the retry's own memory is the one kept"
    assert service.store.get("m_new_1") is None, "the failed write's memory survived"
