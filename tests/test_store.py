from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from chronomem.store import Memory, Session, SQLiteMemoryStore, Turn

NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def store(tmp_path):
    s = SQLiteMemoryStore(tmp_path / "test.db")
    s.initialize()
    yield s
    s.close()


def mem(mid: str, content: str, **kw) -> Memory:
    kw.setdefault("ingested_at", NOW)
    return Memory(
        id=mid,
        user_id="u1",
        type=kw.pop("type", "semantic"),
        content=content,
        token_count=len(content) // 4 or 1,
        **kw,
    )


def test_roundtrip_preserves_all_fields(store):
    original = mem(
        "m1",
        "User prefers PyTorch for ML projects",
        subject="user",
        predicate="prefers_framework",
        object="PyTorch",
        importance=0.8,
        event_time=NOW,
        valid_from=NOW,
        entities=["PyTorch"],
    )
    store.add_memories([original])

    got = store.get("m1")
    assert got is not None
    assert got.content == original.content
    assert got.subject == "user"
    assert got.predicate == "prefers_framework"
    assert got.importance == pytest.approx(0.8)
    assert got.event_time == NOW
    assert got.entities == ["PyTorch"]
    assert got.is_current


def test_get_many_preserves_ranked_order(store):
    store.add_memories([mem(f"m{i}", f"fact {i}") for i in range(5)])
    ranked = ["m3", "m0", "m4"]
    assert [m.id for m in store.get_many(ranked)] == ranked


def test_lexical_search_ranks_by_bm25(store):
    store.add_memories(
        [
            mem("m1", "the user works with PyTorch every day at work"),
            mem("m2", "the user mentioned TensorFlow once in passing"),
            mem("m3", "unrelated note about weekend plans"),
        ]
    )
    hits = store.search_lexical("u1", "PyTorch", limit=10)
    assert [h.memory_id for h in hits] == ["m1"]


def test_lexical_search_survives_fts_metacharacters(store):
    """A raw user question containing FTS5 syntax must not raise. This is the bug
    that shows up the first time a real benchmark question hits the retriever."""
    store.add_memories([mem("m1", "user lives in Canberra")])
    for query in ['what "city" AND where?', "NOT a query*", "a OR b (c)", "-- ; DROP"]:
        store.search_lexical("u1", query, limit=5)  # must not raise


def test_lexical_search_ignores_other_users(store):
    store.add_memories([mem("m1", "shared keyword apples")])
    other = Memory(
        id="m2", user_id="u2", type="semantic", content="shared keyword apples", token_count=4
    )
    store.add_memories([other])
    assert [h.memory_id for h in store.search_lexical("u1", "apples", 10)] == ["m1"]


def test_supersede_marks_old_and_excludes_from_active(store):
    old = mem("m1", "User uses TensorFlow", subject="user", predicate="framework", object="TF")
    store.add_memories([old])
    new = mem("m2", "User uses PyTorch", subject="user", predicate="framework", object="PyTorch")
    store.add_memories([new])

    assert len(store.find_by_predicate("u1", "user", "framework")) == 2

    cutover = NOW + timedelta(days=200)
    store.mark_superseded("m1", superseded_by="m2", valid_to=cutover)

    survivors = store.find_by_predicate("u1", "user", "framework")
    assert [m.id for m in survivors] == ["m2"]

    stale = store.get("m1")
    assert stale.status == "superseded"
    assert stale.superseded_by == "m2"
    assert stale.valid_to == cutover
    assert not stale.is_current
    # Superseded rows are retained, not deleted — ablations need them.
    assert store.count("u1") == 2
    assert store.count("u1", status="active") == 1


def test_record_access_reinforces(store):
    store.add_memories([mem("m1", "a fact")])
    later = NOW + timedelta(days=1)
    store.record_access(["m1"], later)
    store.record_access(["m1"], later)
    got = store.get("m1")
    assert got.access_count == 2
    assert got.last_accessed_at == later


def test_content_update_keeps_fts_in_sync(store):
    store.add_memories([mem("m1", "user likes kayaking")])
    assert store.search_lexical("u1", "kayaking", 5)
    store.add_memories([mem("m1", "user likes bouldering")])
    assert not store.search_lexical("u1", "kayaking", 5)
    assert [h.memory_id for h in store.search_lexical("u1", "bouldering", 5)] == ["m1"]


def test_sessions_and_turns(store):
    session = Session(
        id="s1",
        user_id="u1",
        started_at=NOW,
        source="longmemeval:q42",
        turns=[
            Turn(id="t1", session_id="s1", turn_index=0, role="user", content="hi", ts=NOW),
            Turn(id="t2", session_id="s1", turn_index=1, role="assistant", content="hello", ts=NOW),
        ],
    )
    store.add_session(session)
    store.add_memories([mem("m1", "greeting exchange", source_session_id="s1")])
    assert store.get("m1").source_session_id == "s1"


def test_count_by_type(store):
    store.add_memories(
        [
            mem("m1", "a", type="semantic"),
            mem("m2", "b", type="semantic"),
            mem("m3", "c", type="preference"),
        ]
    )
    assert store.count_by_type("u1") == {"semantic": 2, "preference": 1}
