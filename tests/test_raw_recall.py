from __future__ import annotations

from datetime import datetime

import pytest

from llm_long_term_memory.evaluation.datasets.longmemeval import (
    HaystackSession,
    HaystackTurn,
    Instance,
)
from llm_long_term_memory.evaluation.raw_recall import (
    Probe,
    RecallReport,
    build_probes,
    disjoint_query,
    keyword_query,
    run_probes,
)
from llm_long_term_memory.store import Session, SQLiteMemoryStore, Turn, scoped_session_id

NOW = datetime(2026, 8, 22)


def _instance(qid: str = "q1") -> Instance:
    session = HaystackSession(
        "s1",
        "2026/08/20",
        [
            HaystackTurn("user", "Which posture video did you recommend?"),
            HaystackTurn(
                "assistant",
                "I recommended the Mayo Clinic workplace ergonomics video.",
                has_answer=True,
            ),
        ],
    )
    return Instance(
        qid,
        "single-session-assistant",
        "What was the workplace posture video?",
        "Mayo Clinic",
        "2026/08/22",
        [session],
        ["s1"],
    )


def test_queries_are_mechanical_and_disjoint_removes_shared_content_words():
    gold = "The Mayo Clinic workplace ergonomics video"

    assert "ergonomics" in keyword_query(gold)
    query = disjoint_query("Which Mayo Clinic video helped my posture?", gold)
    assert "mayo" not in query
    assert "clinic" not in query
    assert "posture" in query


def test_probe_builder_uses_gold_turn_annotations_and_existing_namespaces_only():
    included = build_probes([_instance("q1"), _instance("q2")], {"q1"})

    assert {probe.question_id for probe in included} == {"q1"}
    assert {probe.query_type for probe in included} == {"keyword", "natural", "disjoint"}
    assert all(probe.gold_turn_ids == {"s1:1"} for probe in included)


def test_raw_recall_matches_public_gold_ids_against_scoped_storage_ids(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "raw.db")
    store.initialize()
    stored_id = scoped_session_id("q1", "s1")
    store.add_session(
        Session(
            id=stored_id,
            user_id="q1",
            started_at=NOW,
            turns=[
                Turn(f"{stored_id}:0", stored_id, 0, "user", "Which video?", NOW),
                Turn(
                    f"{stored_id}:1",
                    stored_id,
                    1,
                    "assistant",
                    "The Mayo Clinic workplace ergonomics video.",
                    NOW,
                ),
            ],
        )
    )
    probe = Probe(
        "q1",
        "single-session-assistant",
        "natural",
        "Mayo Clinic ergonomics",
        {"s1:1"},
    )

    report = run_probes(store, [probe])

    assert report.recall_at(1) == 1.0
    assert probe.rank == 1
    store.close()


def test_recall_report_handles_misses_and_renders_all_rows():
    hit = Probe("q1", "t", "natural", "q", {"s:0"}, rank=2, latency_ms=3.0)
    miss = Probe("q2", "t", "natural", "q", {"s:0"}, rank=None, latency_ms=5.0)
    report = RecallReport([hit, miss])

    assert report.recall_at(1) == 0.0
    assert report.recall_at(3) == 0.5
    assert report.mrr() == pytest.approx(0.25)
    assert "| bm25 | natural | 2 |" in report.render()
