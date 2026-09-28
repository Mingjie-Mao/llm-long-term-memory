"""Retrieval must not fail as the whole store grows.

The flat index is shared by every namespace, and retrieval used to fetch every ranked
row in one `IN (...)` statement before keeping the namespace's. Past SQLite's
per-statement parameter limit that is an OperationalError: simulated history growth on
train150 hit it at twice the store's size.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from llm_long_term_memory.retrieve import HybridRetriever
from llm_long_term_memory.store import Memory, NumpyFlatIndex, SQLiteMemoryStore

NOW = datetime(2026, 1, 1)


def _memory(mid: str, user: str, status: str = "active") -> Memory:
    return Memory(
        id=mid,
        user_id=user,
        type="semantic",
        content=f"memory {mid}",
        token_count=2,
        status=status,
        ingested_at=NOW,
    )


@pytest.fixture
def crowded(tmp_path, monkeypatch):
    """A store whose other namespaces outnumber the (shrunk) statement limit."""
    monkeypatch.setattr(SQLiteMemoryStore, "_IN_CHUNK", 3)
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()
    memories = [_memory(f"other{i}", "u2") for i in range(20)]
    memories += [_memory("mine", "u1"), _memory("old", "u1", status="superseded")]
    store.add_memories(memories)
    index = NumpyFlatIndex(tmp_path / "idx", dim=2)
    index.add([m.id for m in memories], np.ones((len(memories), 2), dtype=np.float32))
    yield store, index
    store.close()


def test_get_many_reads_more_ids_than_one_statement_allows(crowded):
    store, _ = crowded
    ids = [f"other{i}" for i in range(20)]
    assert [m.id for m in store.get_many(ids)] == ids, "all rows, in caller order"


def test_retrieval_keeps_only_this_namespace_in_a_crowded_store(crowded):
    store, index = crowded
    retriever = HybridRetriever(store, index, weights={"semantic": 1.0}, now=NOW)

    current = retriever.retrieve(
        np.ones(2, dtype=np.float32), "memory", "u1", temporal=True, limit=10
    )
    everything = retriever.retrieve(
        np.ones(2, dtype=np.float32), "memory", "u1", temporal=False, limit=10
    )

    assert [h.memory.id for h in current] == ["mine"]
    assert {h.memory.id for h in everything} == {"mine", "old"}


def test_retrievable_ids_respects_status_and_namespace(crowded):
    store, _ = crowded
    assert store.retrievable_ids("u1", active_only=True) == {"mine"}
    assert store.retrievable_ids("u1", active_only=False) == {"mine", "old"}
