"""P5 consolidation retains evidence and cannot repeatedly summarize the same rows."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest

from llm_long_term_memory.consolidate import Consolidator
from llm_long_term_memory.store import Memory, NumpyFlatIndex, SQLiteMemoryStore


class StubEncoder:
    def encode(self, texts, show_progress=False):
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class StubClient:
    def __init__(self):
        self.prompts = []

    def generate(self, **kwargs):
        self.prompts.append(kwargs["prompt"])

        class Completion:
            text = json.dumps(
                {
                    "content": "The user regularly attends pottery classes with friends.",
                    "confidence": 0.8,
                }
            )

        return Completion()


def memory(memory_id: str, content: str) -> Memory:
    return Memory(
        id=memory_id,
        user_id="u",
        type="episodic",
        content=content,
        token_count=8,
        importance=0.8,
        strength=1.0,
        ingested_at=datetime(2026, 1, 1),
    )


@pytest.fixture
def wired(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memories.db")
    store.initialize()
    index = NumpyFlatIndex(tmp_path / "index", dim=2)
    client = StubClient()
    yield store, index, client
    store.close()


def test_consolidation_creates_a_traceable_summary_and_downweights_sources(wired):
    store, index, client = wired
    sources = [
        memory("a", "The user attended a pottery class in March."),
        memory("b", "The user went to another pottery class in April."),
        memory("c", "The user enjoyed a pottery class with friends in May."),
    ]
    store.add_memories(sources)
    consolidator = Consolidator(client, "m", StubEncoder(), store, index, min_cluster_size=3)

    report = consolidator.consolidate("u")

    assert report.clusters_found == 1
    assert len(report.memories_created) == 1
    summary_id = report.memories_created[0]
    summary = store.get(summary_id)
    assert summary.type == "semantic"
    assert summary.confidence == pytest.approx(0.8)
    assert store.evidence_for(summary_id) == ["a", "b", "c"]
    assert store.get("a").strength == pytest.approx(0.5)
    assert len(index) == 1
    assert "pottery" in client.prompts[0]


def test_consolidation_does_not_resummarize_evidence_already_folded(wired):
    store, index, client = wired
    store.add_memories(
        [
            memory("a", "The user attended a pottery class in March."),
            memory("b", "The user went to another pottery class in April."),
            memory("c", "The user enjoyed a pottery class with friends in May."),
        ]
    )
    consolidator = Consolidator(client, "m", StubEncoder(), store, index, min_cluster_size=3)

    consolidator.consolidate("u")
    second = consolidator.consolidate("u")

    assert second.clusters_found == 0
    assert second.memories_created == []
    assert len(client.prompts) == 1
