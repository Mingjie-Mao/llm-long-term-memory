"""Answer-coverage metric.

The tests that matter here are the ones pinning what the metric *cannot* see, so
that a future reader does not mistake a low score for a broken extractor.
"""

from __future__ import annotations

from datetime import datetime

from chronomem.evaluation.datasets.longmemeval import HaystackSession, HaystackTurn, Instance
from chronomem.ingest.coverage import (
    UNMEASURABLE_TYPES,
    CoverageCase,
    CoverageReport,
    answer_present,
    evaluate_coverage,
    source_literal_present,
)
from chronomem.store import Memory


def mem(content: str, event_time: datetime | None = None, **kw) -> Memory:
    return Memory(
        id=content[:12],
        user_id="u1",
        type="semantic",
        content=content,
        token_count=5,
        event_time=event_time,
        **kw,
    )


def test_verbatim_answer_is_found():
    assert answer_present(
        "Business Administration", [mem("The user studied Business Administration")]
    )


def test_numeric_answer_is_found_despite_different_wording():
    assert answer_present("25", [mem("The user added 25 new postcards to the collection")])
    assert answer_present("$50", [mem("The train ticket cost 50 dollars")])


def test_answer_held_only_in_event_time_is_found():
    """A temporal gold answer is often a date the content sentence never spells out
    because it lives in the structured column."""
    memories = [mem("The user launched their website", event_time=datetime(2023, 3, 1))]
    assert answer_present("March 2023", memories)


def test_structured_triple_fields_are_searched():
    memories = [
        mem("The user has a preference", subject="user", predicate="prefers", object="Kyoto")
    ]
    assert answer_present("Kyoto", memories)


def test_absent_answer_is_absent():
    assert not answer_present("The Glass Menagerie", [mem("The user auditioned for The Crucible")])
    assert not answer_present("10", [mem("The user wants to cut Netflix time")])


def test_source_literal_coverage_is_distinct_from_structured_coverage():
    inst = instance("q1", "single-session-user", "three months")
    inst.sessions = [
        HaystackSession(
            "s1",
            "2026-01-01",
            [HaystackTurn("user", "I started collecting cameras three months ago.")],
        )
    ]
    assert source_literal_present("three months", inst)
    assert not answer_present("three months", [mem("The user owns 17 vintage cameras")])


def test_empty_gold_is_never_covered():
    assert not answer_present("", [mem("anything")])
    assert not answer_present("   ", [mem("anything")])


def test_computed_answers_are_reported_as_misses_even_though_the_facts_are_present():
    """The central limitation, pinned deliberately.

    Both dates are in the store and the gold is their difference. The metric says
    miss; the extractor did nothing wrong. This is why `multi-session` is excluded
    from the headline rate and why the number must never be tuned against.
    """
    memories = [
        mem("The user sold baked goods at the Farmers' Market", event_time=datetime(2023, 2, 26)),
        mem("The user sold jewellery at the Spring Fling Market", event_time=datetime(2023, 3, 20)),
    ]
    assert not answer_present("3 weeks", memories)


def instance(qid: str, qtype: str, answer: str) -> Instance:
    return Instance(
        question_id=qid,
        question_type=qtype,
        question="q?",
        answer=answer,
        question_date="2026-01-01",
        sessions=[],
        answer_session_ids=[],
    )


def test_headline_rate_excludes_the_unmeasurable_categories():
    report = CoverageReport(
        cases=[
            CoverageCase("a", "single-session-user", "q", "gold", True, 1),
            CoverageCase("b", "temporal-reasoning", "q", "gold", False, 1),
            CoverageCase("c", "multi-session", "q", "gold", False, 1),
            CoverageCase("d", "single-session-preference", "q", "gold", False, 1),
        ]
    )
    assert len(report.measurable) == 2
    assert report.rate == 0.5, "1 of 2 measurable"
    assert report.rate_all == 0.25, "1 of 4 overall"
    assert {"multi-session", "single-session-preference"} == UNMEASURABLE_TYPES


def test_abstention_questions_are_skipped():
    """They have no gold answer to look for; scoring them would just dilute."""
    instances = [
        instance("q1", "single-session-user", "Sydney"),
        instance("q2_abs", "single-session-user", ""),
    ]
    report = evaluate_coverage(instances, lambda sessions: [mem("The user lives in Sydney")])
    assert report.n == 1
    assert report.rate == 1.0
    assert report.source_literal_rate == 0.0


def test_by_type_counts_only_answerable_questions():
    report = CoverageReport(
        cases=[
            CoverageCase("a", "temporal-reasoning", "q", "gold", True, 3),
            CoverageCase("b", "temporal-reasoning", "q", "gold", False, 2),
            CoverageCase("c", "temporal-reasoning", "q", "", False, 0),
        ]
    )
    assert report.by_type() == {"temporal-reasoning": (2, 1)}
    assert [c.question_id for c in report.misses()] == ["b"]
