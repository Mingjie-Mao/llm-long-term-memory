"""P5 decay and eviction must be repeatable and preserve audit records."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from llm_long_term_memory.lifecycle import apply_decay, decayed_strength, evict_to_limit
from llm_long_term_memory.store import Memory, SQLiteMemoryStore

NOW = datetime(2026, 1, 1)


def memory(memory_id: str, **kwargs) -> Memory:
    return Memory(
        id=memory_id,
        user_id="u",
        type="semantic",
        content=f"fact {memory_id}",
        token_count=5,
        ingested_at=kwargs.pop("ingested_at", NOW),
        **kwargs,
    )


@pytest.fixture
def store(tmp_path):
    value = SQLiteMemoryStore(tmp_path / "memories.db")
    value.initialize()
    yield value
    value.close()


def test_strength_decays_by_one_half_life():
    item = memory("m", strength=0.8)
    assert decayed_strength(item, now=NOW + timedelta(days=30), halflife_days=30) == pytest.approx(
        0.4
    )


def test_apply_decay_is_idempotent_at_one_point_in_time(store):
    store.add_memories([memory("m", strength=1.0)])
    when = NOW + timedelta(days=60)

    first = apply_decay(store, "u", now=when, halflife_days=30)
    second = apply_decay(store, "u", now=when, halflife_days=30)

    assert first.updated == 1
    assert store.get("m").strength == pytest.approx(0.25)
    assert store.get("m").strength_updated_at == when
    assert second.updated == 0


def test_access_reinforces_strength_and_resets_the_decay_anchor(store):
    store.add_memories([memory("m", strength=0.2)])
    accessed = NOW + timedelta(days=30)
    store.record_access(["m"], accessed, reinforcement=0.5)
    item = store.get("m")

    assert item.strength == pytest.approx(0.6)
    assert item.strength_updated_at == accessed
    assert decayed_strength(
        item, now=accessed + timedelta(days=30), halflife_days=30
    ) == pytest.approx(0.3)


def test_eviction_uses_strength_times_importance_and_keeps_rows(store):
    store.add_memories(
        [
            memory("weak", strength=0.2, importance=0.5),
            memory("important", strength=0.2, importance=1.0),
            memory("strong", strength=0.9, importance=0.5),
        ]
    )

    report = evict_to_limit(store, "u", limit=2)

    assert report.evicted_ids == ["weak"]
    assert store.get("weak").status == "evicted"
    assert store.count("u") == 3
    assert [memory.id for memory in store.iter_active("u")] == ["important", "strong"]


def test_eviction_tie_break_is_stable_by_oldest_then_id(store):
    store.add_memories(
        [
            memory("later", ingested_at=NOW + timedelta(days=1)),
            memory("earlier", ingested_at=NOW),
        ]
    )

    report = evict_to_limit(store, "u", limit=1)

    assert report.evicted_ids == ["earlier"]
