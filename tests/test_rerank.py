"""Cross-encoder reranking, tested against a stub rather than a real model.

CI installs the `llm` extra but deliberately not `embed`, which is what pulls
torch. The reranker is therefore injected into `HybridRetriever` rather than
constructed by it, and these tests exercise the wiring — ordering, the pool bound,
score bookkeeping, tie determinism — with a scorer whose output is fixed by the
test. What a real cross-encoder scores is the experiment's question, not the
suite's.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from llm_long_term_memory.retrieve import HybridRetriever
from llm_long_term_memory.retrieve.rerank import CrossEncoderReranker
from llm_long_term_memory.store import Memory, NumpyFlatIndex, SQLiteMemoryStore

NOW = datetime(2026, 1, 31)


class StubCrossEncoder:
    """Scores by a caller-supplied table keyed on memory content."""

    def __init__(self, table: dict[str, float]) -> None:
        self.table = table
        self.seen: list[tuple[str, str]] = []

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        self.seen.extend(pairs)
        return [self.table.get(text, 0.0) for _, text in pairs]


def reranker(table: dict[str, float], **kwargs) -> CrossEncoderReranker:
    r = CrossEncoderReranker(**kwargs)
    r._model = StubCrossEncoder(table)
    return r


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


def test_reranking_promotes_a_memory_the_hybrid_ranking_buried(wired):
    """The v1 failure in miniature: the count memory outranks the duration memory
    on embedding similarity, and only a joint encoding separates them."""
    store, index = wired
    memories = [
        memory("count", "The user owns 17 vintage cameras."),
        memory("duration", "The user began collecting cameras three months ago."),
    ]
    store.add_memories(memories)
    # The count memory is the closer vector, so hybrid ranks it first.
    index.add(["count", "duration"], np.array([[1.0, 0.0], [0.7, 0.7]], dtype=np.float32))

    query = np.array([1.0, 0.0], dtype=np.float32)
    question = "How long have I been collecting vintage cameras?"

    plain = HybridRetriever(store, index, weights={"semantic": 1.0}, now=NOW).retrieve(
        query, question, "u", temporal=True, limit=2
    )
    assert [hit.memory.id for hit in plain] == ["count", "duration"]

    reranked = HybridRetriever(
        store,
        index,
        weights={"semantic": 1.0},
        now=NOW,
        reranker=reranker({memories[0].content: 2.6, memories[1].content: 7.2}),
    ).retrieve(query, question, "u", temporal=True, limit=2)

    assert [hit.memory.id for hit in reranked] == ["duration", "count"]


def test_reranking_runs_before_truncation(wired):
    """A reranker applied after the top-k cut could never rescue anything, so the
    memory that only the cross-encoder likes must survive a limit of 1."""
    store, index = wired
    memories = [
        memory("near", "A closely embedded but unhelpful fact."),
        memory("far", "The answer-bearing fact, embedded further away."),
    ]
    store.add_memories(memories)
    index.add(["near", "far"], np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32))

    hits = HybridRetriever(
        store,
        index,
        weights={"semantic": 1.0},
        now=NOW,
        reranker=reranker({memories[0].content: -5.0, memories[1].content: 9.0}),
    ).retrieve(np.array([1.0, 0.0], dtype=np.float32), "q", "u", temporal=True, limit=1)

    assert [hit.memory.id for hit in hits] == ["far"]


def test_both_scores_are_retained_for_inspection(wired):
    store, index = wired
    memories = [memory("m1", "First."), memory("m2", "Second.")]
    store.add_memories(memories)
    index.add(["m1", "m2"], np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32))

    hits = HybridRetriever(
        store,
        index,
        weights={"semantic": 1.0},
        now=NOW,
        reranker=reranker({"First.": 1.0, "Second.": 4.0}),
    ).retrieve(np.array([1.0, 0.0], dtype=np.float32), "q", "u", temporal=True, limit=2)

    assert hits[0].rerank_score == 4.0
    # The hybrid score survives reranking, which is what makes "moved from 2nd to
    # 1st" observable rather than merely asserted.
    assert hits[0].score > 0
    assert hits[0].signals.semantic > 0


def test_scores_are_none_when_reranking_is_disabled(wired):
    store, index = wired
    store.add_memories([memory("m1", "Only.")])
    index.add(["m1"], np.array([[1.0, 0.0]], dtype=np.float32))

    hits = HybridRetriever(store, index, weights={"semantic": 1.0}, now=NOW).retrieve(
        np.array([1.0, 0.0], dtype=np.float32), "q", "u", temporal=True, limit=2
    )

    assert hits[0].rerank_score is None


def test_the_rerank_pool_is_bounded(wired):
    """`candidates` caps the added latency. Anything past the cap is dropped, not
    appended: its hybrid score is not comparable to a rerank score."""
    store, index = wired
    memories = [memory(f"m{i}", f"Fact {i}.") for i in range(6)]
    store.add_memories(memories)
    index.add(
        [m.id for m in memories],
        np.array([[1.0, i * 0.01] for i in range(6)], dtype=np.float32),
    )

    r = reranker({f"Fact {i}.": float(i) for i in range(6)}, candidates=3)
    hits = HybridRetriever(store, index, weights={"semantic": 1.0}, now=NOW, reranker=r).retrieve(
        np.array([1.0, 0.0], dtype=np.float32), "q", "u", temporal=True, limit=10
    )

    assert len(r._model.seen) == 3
    assert len(hits) == 3
    assert all(hit.rerank_score is not None for hit in hits)


def test_equal_rerank_scores_break_ties_deterministically(wired):
    store, index = wired
    memories = [memory("b", "Tied."), memory("a", "Tied.")]
    store.add_memories(memories)
    index.add(["b", "a"], np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))

    orders = set()
    for _ in range(3):
        hits = HybridRetriever(
            store,
            index,
            weights={"semantic": 1.0},
            now=NOW,
            reranker=reranker({"Tied.": 1.0}),
        ).retrieve(np.array([1.0, 0.0], dtype=np.float32), "q", "u", temporal=True, limit=2)
        orders.add(tuple(hit.memory.id for hit in hits))

    assert len(orders) == 1, "tied rerank scores must not reorder between runs"


def test_empty_candidates_short_circuit():
    r = reranker({})
    assert r.rerank("q", [], limit=5) == []
    assert r._model.seen == []
