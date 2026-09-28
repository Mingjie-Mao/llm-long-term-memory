"""The resolver against a written-down lifecycle gold set, in every ingestion order.

`tests/fixtures/temporal_lifecycle_gold_v1.json` states, per case, what should be
current, what each memory's final state should be, and what was in force on given
dates. The expectations were written from the lifecycle rules before this harness
ran. Every case is replayed in three ingestion orders, one memory at a time as an
ingest would deliver them, and must reach the same timeline each time; a second full
resolution must then write nothing.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from llm_long_term_memory.store import Memory, SQLiteMemoryStore
from llm_long_term_memory.temporal import TemporalResolver, as_of

GOLD = json.loads(
    (Path(__file__).parent / "fixtures" / "temporal_lifecycle_gold_v1.json").read_text(
        encoding="utf-8"
    )
)
CASES = {case["id"]: case for case in GOLD["cases"]}
ORDERS = ("given", "reversed", "rotated")


def _memory(spec: dict) -> Memory:
    when = datetime.fromisoformat(spec["date"])
    op = spec["op"]
    return Memory(
        id=spec["id"],
        user_id=spec.get("user", "u1"),
        type="semantic",
        content=spec.get("content", f"The user: {spec['predicate']} = {spec['object']}."),
        token_count=8,
        subject="user",
        predicate=spec["predicate"],
        object=spec["object"],
        target_object=spec.get("target"),
        update_op=op,
        replaces_previous=spec.get("replaces_previous", op in ("replaces", "removes")),
        observed_at=when,
        valid_from=when,
        ingested_at=datetime(2026, 1, 1),
    )


def _ordered(memories: list, order: str) -> list:
    if order == "reversed":
        return list(reversed(memories))
    if order == "rotated":
        return memories[1:] + memories[:1]
    return list(memories)


def _key_memories(store, key: str) -> list[Memory]:
    user, predicate = key.split("|")
    return store.find_by_predicate(user, "user", predicate, include_superseded=True)


def _values(memories) -> list[str]:
    return sorted({(m.object or "").strip().lower() for m in memories})


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("case_id", sorted(CASES))
def test_the_resolver_reaches_the_gold_timeline(tmp_path, case_id, order):
    case = CASES[case_id]
    store = SQLiteMemoryStore(tmp_path / "gold.db")
    store.initialize()
    resolver = TemporalResolver(store)
    try:
        for spec in _ordered(case["memories"], order):
            memory = _memory(spec)
            store.add_memories([memory])
            resolver.resolve_memories([memory])

        by_id = {m.id: m for key in case["current"] for m in _key_memories(store, key)}
        for key, expected in case["current"].items():
            current = [m for m in _key_memories(store, key) if m.is_current]
            assert _values(current) == expected, f"{case_id}/{order}: current {key}"
        for memory_id, status in case["status"].items():
            assert by_id[memory_id].status == status, f"{case_id}/{order}: status {memory_id}"
        for check in case.get("as_of", []):
            held = as_of(_key_memories(store, check["key"]), datetime.fromisoformat(check["date"]))
            assert _values(held) == check["values"], f"{case_id}/{order}: as_of {check['date']}"

        assert resolver.resolve_everything().writes == 0, f"{case_id}/{order}: not idempotent"
    finally:
        store.close()


def test_the_gold_set_covers_every_transition_kind():
    ops = {spec["op"] for case in GOLD["cases"] for spec in case["memories"]}
    assert ops == {"coexists", "replaces", "removes"}
    assert any(len({s.get("user", "u1") for s in c["memories"]}) > 1 for c in GOLD["cases"])
