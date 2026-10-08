"""Raw-primary v2: fused turn ranking, dated headers, and evidence-first answers.

Registered in `results/prereg-raw-primary-dev100-v3.md`. The v1 arm must be unchanged
when none of this is switched on.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from llm_long_term_memory.answering import NotedAnswerVerdict
from llm_long_term_memory.retrieve.excerpts import archive_excerpts, fuse
from llm_long_term_memory.store import NumpyFlatIndex, Session, SQLiteMemoryStore, Turn


def _session(sid, user, day, *contents):
    started = datetime(2023, 5, day)
    return Session(
        id=sid,
        user_id=user,
        started_at=started,
        turns=[
            Turn(f"{sid}-{i}", sid, i, "user" if i % 2 == 0 else "assistant", c, started)
            for i, c in enumerate(contents)
        ],
    )


@pytest.fixture
def store(tmp_path):
    s = SQLiteMemoryStore(tmp_path / "v2.db")
    s.initialize()
    s.add_session(_session("wed", "q1", 20, "We went to my cousin's wedding.", "Lovely."))
    s.add_session(_session("gym", "q1", 2, "I joined a relatives meetup group.", "Nice."))
    s.add_session(_session("other", "q2", 3, "A relative's life event: my aunt's party.", "Fun."))
    yield s
    s.close()


@pytest.fixture
def turn_index(tmp_path):
    """Vectors chosen so the other tenant's turn is the closest to the query."""
    index = NumpyFlatIndex(tmp_path / "turns", dim=2)
    ids = ["other-0", "wed-0", "gym-0", "wed-1", "gym-1", "other-1"]
    vectors = np.array([[1, 0], [0.9, 0.1], [0.1, 0.9], [0, 1], [0, 1], [0, 1]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index.add(ids, vectors)
    return index


def test_fusion_rewards_agreement_and_keeps_first_appearance_on_ties():
    assert fuse(["a", "b"], ["b", "a"]) == ["a", "b"], "a tie keeps first appearance"
    assert fuse(["a", "b", "c"], ["b", "c"])[0] == "b", "ranked high by both beats one top"


def test_dense_ranking_never_reaches_another_tenant(store, turn_index):
    query = np.array([1.0, 0.0], dtype=np.float32)
    found = archive_excerpts(
        store, "q1", "life event of a relative", 4000, turn_index=turn_index, query_vector=query
    )
    ids = {t.id for t in found.turns}
    assert "other" not in {t.session_id for t in found.turns}
    assert "wed-0" in ids, "the turn sharing no word with the question is found"
    bm25_only = archive_excerpts(store, "q1", "life event of a relative", 4000)
    assert "wed-0" not in {t.id for t in bm25_only.turns}, "and BM25 alone misses it"


def test_personal_budget_keeps_a_user_fact_instead_of_long_assistant_advice(tmp_path):
    from llm_long_term_memory.retrieve.excerpts import bm25_rank

    db = SQLiteMemoryStore(tmp_path / "roles.db")
    db.initialize()
    stamp = datetime(2023, 5, 1)
    texts = ["fish " * 90, "My aquarium has 10 fish.", "fish " * 10]
    db.add_session(
        Session(
            "own",
            "alice",
            stamp,
            turns=[
                Turn(f"own-{i}", "own", i, "user" if i == 1 else "assistant", text, stamp)
                for i, text in enumerate(texts)
            ],
        )
    )
    db.add_session(_session("foreign", "bob", 2, "My aquarium has 9999 fish."))
    keys = {t.id: t.content for t in db.turns_for_session("own")}
    assert bm25_rank("fish", keys)[0] == "own-0"
    ordinary = archive_excerpts(db, "alice", "fish", 100, fact_keys=True)
    personal = archive_excerpts(db, "alice", "fish", 100, fact_keys=True, prefer_user=True)
    assert "own-1" not in {t.id for t in ordinary.turns}
    assert "own-1" in {t.id for t in personal.turns}
    assert personal.tokens <= 100
    assert "foreign" not in {t.session_id for t in personal.turns}
    assert any(t.role == "assistant" for t in personal.turns)
    db.close()


def test_headers_say_how_long_before_the_question():
    from llm_long_term_memory.retrieve.excerpts import ArchiveExcerpts

    turn = Turn("t", "s", 0, "user", "I started sculpting.", datetime(2023, 5, 3))
    text = ArchiveExcerpts(
        turns=[turn],
        tokens=5,
        session_dates={"s": datetime(2023, 5, 3)},
        asked_on=datetime(2023, 5, 24, 10, 0),
    ).render()
    assert "Conversation on 2023-05-03 (21 days, about 3.0 weeks, before the question):" in text


def test_without_the_v2_switches_the_v1_rendering_is_unchanged(store):
    text = archive_excerpts(store, "q1", "cousin wedding", 4000).render()
    assert "Conversation on 2023-05-20:\n" in text
    assert "before the question" not in text


def test_the_evidence_is_written_before_the_decision():
    fields = list(NotedAnswerVerdict.model_json_schema()["properties"])
    assert fields[0] == "notes"
    assert fields[1:] == ["status", "answer", "reason", "source_query"]


def test_build_turn_index_embeds_every_turn(tmp_path, monkeypatch):
    from llm_long_term_memory.commands import lifecycle
    from llm_long_term_memory.config import ExperimentConfig, Settings

    class _Encoder:
        def __init__(self, name):
            pass

        def encode(self, texts, show_progress=False):
            return np.ones((len(texts), 384), dtype=np.float32)

    monkeypatch.setattr("llm_long_term_memory.embed.Encoder", _Encoder)
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()
    store.add_session(_session("a", "u1", 1, "one", "two"))
    store.add_session(_session("b", "u2", 2, "three"))
    store.close()
    settings = Settings(LLTM_STORE_DIR=str(tmp_path))
    count = lifecycle._build_turn_index(
        ExperimentConfig.from_yaml("configs/fallback.yaml"), settings, "s"
    )
    built = NumpyFlatIndex(tmp_path / "s-turn-index", dim=384)
    assert count == 3
    assert sorted(built.ids) == ["a-0", "a-1", "b-0"]
    assert not list(tmp_path.glob(".s-turn-index-build*")), "the temporary stem is cleaned up"


def test_a_rebuilt_memory_index_can_be_opened(tmp_path, monkeypatch):
    """`rebuild-index` moved its scratch files into place under a manifest that still
    named the scratch stem, so the recovered index refused to load."""
    from llm_long_term_memory.commands import lifecycle
    from llm_long_term_memory.config import ExperimentConfig, Settings
    from llm_long_term_memory.store import Memory

    class _Encoder:
        def __init__(self, name):
            pass

        def encode(self, texts, show_progress=False):
            return np.ones((len(texts), 384), dtype=np.float32)

    monkeypatch.setattr("llm_long_term_memory.embed.Encoder", _Encoder)
    store = SQLiteMemoryStore(tmp_path / "live.db")
    store.initialize()
    store.add_memories([Memory(id="m1", user_id="u", type="semantic", content="x", token_count=1)])
    store.close()
    lifecycle._rebuild_index(
        ExperimentConfig.from_yaml("configs/fallback.yaml"),
        Settings(LLTM_STORE_DIR=str(tmp_path)),
        "live",
    )

    reopened = NumpyFlatIndex(tmp_path / "live-index", dim=384)
    assert reopened.ids == ("m1",)
    assert not list(tmp_path.glob(".live-index-rebuild*"))


def test_the_v2_arm_builds_through_the_cli(tmp_path, monkeypatch):
    """Built the way `lltm eval run` builds it, with a stub client and a real turn index."""
    from llm_long_term_memory import cli
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.store import Memory

    class _Encoder:
        dim = 384

        def __init__(self, *a, **k):
            pass

        def encode(self, texts, show_progress=False):
            return np.ones((len(texts), 384), dtype=np.float32)

        def encode_one(self, text):
            return np.ones(384, dtype=np.float32)

    monkeypatch.setattr("llm_long_term_memory.embed.Encoder", _Encoder)
    monkeypatch.setenv("LLTM_STORE_DIR", str(tmp_path))
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()
    store.add_session(_session("a", "u1", 1, "I bought a kayak.", "Nice."))
    store.add_memories(
        [Memory(id="m1", user_id="u1", type="semantic", content="kayak", token_count=1)]
    )
    store.close()
    memories = NumpyFlatIndex(tmp_path / "s-index", dim=384)
    memories.add(["m1"], np.ones((1, 384), dtype=np.float32))
    memories.save()
    turns = NumpyFlatIndex(tmp_path / "s-turn-index", dim=384)
    turns.add(["a-0", "a-1"], np.ones((2, 384), dtype=np.float32))
    turns.save()

    _, _, runner, _, _ = cli._build(
        "two_stage_raw_primary_v2",
        "configs/fallback.yaml",
        "s",
        client_override=object(),
        settings_override=Settings(LLTM_STORE_DIR=str(tmp_path)),
        read_only_store=True,
    )
    assert runner.answer_policy == "v2_notes"
    assert runner.verdict_schema is NotedAnswerVerdict
    assert runner.max_output_tokens == 1024
    assert runner.raw_primary_tokens == 4000
    assert runner.raw_primary_dated is True
    assert runner.raw_primary_turn_index is not None
    assert runner.raw_primary_memory_fusion is False, "v2 is registered without it"
    assert runner.raw_primary_fact_keys is False


def test_the_v2_arm_refuses_a_store_without_a_turn_index(tmp_path, monkeypatch):
    from llm_long_term_memory import cli
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.store import Memory

    monkeypatch.setenv("LLTM_STORE_DIR", str(tmp_path))
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()
    store.add_memories(
        [Memory(id="m1", user_id="u1", type="semantic", content="kayak", token_count=1)]
    )
    store.close()
    memories = NumpyFlatIndex(tmp_path / "s-index", dim=384)
    memories.add(["m1"], np.ones((1, 384), dtype=np.float32))
    memories.save()

    with pytest.raises(Exception, match="build-turn-index"):
        cli._build(
            "two_stage_raw_primary_v2",
            "configs/fallback.yaml",
            "s",
            client_override=object(),
            settings_override=Settings(LLTM_STORE_DIR=str(tmp_path)),
            read_only_store=True,
        )


def test_memory_led_turns_join_the_fusion_and_never_cross_tenants(store):
    """The turn a memory points at is found even when the question shares no word with it."""
    found = archive_excerpts(
        store,
        "q1",
        "family news",
        4000,
        memory_anchors=[("wed", 0), ("other", 0)],
    )
    ids = {t.id for t in found.turns}
    assert "wed-0" in ids, "led there by the memory"
    assert "other-0" not in ids, "another tenant's anchor is ignored"
    plain = archive_excerpts(store, "q1", "family news", 4000)
    assert "wed-0" not in {t.id for t in plain.turns}


def test_the_v4_arm_builds_on_the_fact_keyed_index(tmp_path, monkeypatch):
    from llm_long_term_memory import cli
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.store import Memory

    monkeypatch.setenv("LLTM_STORE_DIR", str(tmp_path))
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()
    store.add_session(_session("a", "u1", 1, "I bought a kayak.", "Nice."))
    store.add_memories(
        [Memory(id="m1", user_id="u1", type="semantic", content="kayak", token_count=1)]
    )
    store.close()
    memories = NumpyFlatIndex(tmp_path / "s-index", dim=384)
    memories.add(["m1"], np.ones((1, 384), dtype=np.float32))
    memories.save()
    settings = Settings(LLTM_STORE_DIR=str(tmp_path))

    with pytest.raises(Exception, match="--fact-keys"):
        cli._build(
            "two_stage_raw_primary_v4",
            "configs/fallback.yaml",
            "s",
            client_override=object(),
            settings_override=settings,
            read_only_store=True,
        )

    keyed = NumpyFlatIndex(tmp_path / "s-turn-key-index", dim=384)
    keyed.add(["a-0", "a-1"], np.ones((2, 384), dtype=np.float32))
    keyed.save()
    _, _, runner, _, _ = cli._build(
        "two_stage_raw_primary_v4",
        "configs/fallback.yaml",
        "s",
        client_override=object(),
        settings_override=settings,
        read_only_store=True,
    )
    assert runner.raw_primary_memory_fusion is True
    assert runner.raw_primary_fact_keys is True
    assert runner.raw_primary_dated is True
    assert runner.answer_policy == "v2_notes"
    assert runner.raw_primary_turn_index.path.name == "s-turn-key-index"


def test_fact_keys_let_a_turn_be_found_by_its_facts(store):
    """A turn whose words miss the question is found through the fact anchored to it."""
    from llm_long_term_memory.store import Memory

    store.add_memories(
        [
            Memory(
                id="f1",
                user_id="q1",
                type="semantic",
                content="The user attended a family celebration.",
                token_count=6,
                source_session_id="wed",
                source_turn_index=0,
            )
        ]
    )
    keyed = archive_excerpts(store, "q1", "family celebration", 4000, fact_keys=True)
    plain = archive_excerpts(store, "q1", "family celebration", 4000)
    assert "wed-0" in {t.id for t in keyed.turns}
    assert "wed-0" not in {t.id for t in plain.turns}
    assert "other" not in {t.session_id for t in keyed.turns}
