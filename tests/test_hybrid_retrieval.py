"""Deterministic tests for the five independently weighted retrieval signals."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from chronomem.retrieve import HybridRetriever
from chronomem.store import Memory, NumpyFlatIndex, SQLiteMemoryStore

NOW = datetime(2026, 1, 31)


def memory(memory_id: str, content: str, **kwargs) -> Memory:
    return Memory(
        id=memory_id,
        user_id=kwargs.pop("user_id", "u"),
        type=kwargs.pop("type", "semantic"),
        content=content,
        token_count=kwargs.pop("token_count", 8),
        ingested_at=kwargs.pop("ingested_at", NOW),
        **kwargs,
    )


@pytest.fixture
def wired(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memories.db")
    store.initialize()
    index = NumpyFlatIndex(tmp_path / "index", dim=2)
    yield store, index
    store.close()


def retrieve(
    store, index, memories, vectors, *, weights, query="where is Ada Lovelace?", temporal=True
):
    store.add_memories(memories)
    index.add([memory.id for memory in memories], np.asarray(vectors, dtype=np.float32))
    return HybridRetriever(store, index, weights=weights, now=NOW).retrieve(
        np.array([1.0, 0.0], dtype=np.float32), query, "u", temporal=temporal, limit=10
    )


def test_bm25_is_min_max_normalized_before_weighting(wired):
    store, index = wired
    hits = retrieve(
        store,
        index,
        [
            memory("semantic", "A generic fact with no matching term."),
            memory("lexical", "Ada Lovelace wrote the first algorithm."),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
        weights={"semantic": 0.0, "bm25": 1.0, "recency": 0.0, "importance": 0.0, "entity": 0.0},
    )

    assert [hit.memory.id for hit in hits] == ["lexical", "semantic"]
    assert hits[0].signals.bm25 == 1.0
    assert hits[1].signals.bm25 == 0.0


def test_recency_uses_the_configured_exponential_half_life(wired):
    store, index = wired
    recent = memory("recent", "a recent fact", event_time=NOW - timedelta(days=1))
    old = memory("old", "an old fact", event_time=NOW - timedelta(days=60))
    store.add_memories([recent, old])
    index.add(["recent", "old"], np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))

    hits = HybridRetriever(
        store,
        index,
        weights={"semantic": 0.0, "bm25": 0.0, "recency": 1.0, "importance": 0.0, "entity": 0.0},
        recency_halflife_days=30,
        now=NOW,
    ).retrieve(np.array([1.0, 0.0], dtype=np.float32), "fact", "u", temporal=True, limit=10)

    assert [hit.memory.id for hit in hits] == ["recent", "old"]
    assert hits[1].signals.recency == pytest.approx(0.25)


def test_entity_signal_matches_query_terms_against_stored_entities(wired):
    store, index = wired
    hits = retrieve(
        store,
        index,
        [
            memory("ada", "Ada was a mathematician.", entities=["Ada Lovelace"]),
            memory("other", "Grace wrote code.", entities=["Grace Hopper"]),
        ],
        [[1.0, 0.0], [1.0, 0.0]],
        weights={"semantic": 0.0, "bm25": 0.0, "recency": 0.0, "importance": 0.0, "entity": 1.0},
    )

    assert [hit.memory.id for hit in hits] == ["ada", "other"]
    assert hits[0].signals.entity == 1.0
    assert hits[1].signals.entity == 0.0


def test_candidate_union_keeps_lexical_and_semantic_hits_once_each(wired):
    store, index = wired
    hits = retrieve(
        store,
        index,
        [
            memory("semantic", "unrelated fact"),
            memory("both", "Ada Lovelace fact", entities=["Ada Lovelace"]),
        ],
        [[1.0, 0.0], [0.8, 0.2]],
        weights={"semantic": 1.0, "bm25": 1.0, "recency": 0.0, "importance": 0.0, "entity": 0.0},
    )

    assert {hit.memory.id for hit in hits} == {"semantic", "both"}
    assert len(hits) == 2


def test_temporal_filtering_excludes_superseded_and_evicted_memories(wired):
    store, index = wired
    current = memory("current", "current Ada fact")
    old = memory("old", "old Ada fact", status="superseded")
    evicted = memory("evicted", "evicted Ada fact", status="evicted")
    store.add_memories([current, old, evicted])
    index.add(
        ["current", "old", "evicted"],
        np.array([[0.7, 0.3], [1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
    )
    retriever = HybridRetriever(store, index, weights={"semantic": 1.0}, now=NOW)

    temporal = retriever.retrieve(
        np.array([1.0, 0.0], dtype=np.float32), "Ada", "u", temporal=True, limit=10
    )
    flat = retriever.retrieve(
        np.array([1.0, 0.0], dtype=np.float32), "Ada", "u", temporal=False, limit=10
    )

    assert [hit.memory.id for hit in temporal] == ["current"]
    assert {hit.memory.id for hit in flat} == {"current", "old"}
