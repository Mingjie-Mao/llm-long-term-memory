from __future__ import annotations

import pytest

from llm_long_term_memory.ingest.stage_b_quality import (
    ChangePair,
    ConsistencyReport,
    PredicateCase,
    PredicateReport,
    SupersedeReport,
    Supersession,
    collect_supersessions,
    judge_supersession,
)
from llm_long_term_memory.store import Memory


def memory(mid: str, *, predicate: str | None, object_: str | None, **changes) -> Memory:
    values = {
        "id": mid,
        "user_id": "alice",
        "type": "semantic",
        "content": f"content for {mid}",
        "token_count": 4,
        "subject": "user",
        "predicate": predicate,
        "object": object_,
    }
    values.update(changes)
    return Memory(**values)


def test_predicate_report_handles_empty_and_confused_cases():
    assert PredicateReport().accuracy == 0.0
    assert PredicateReport().fallthrough_rate == 0.0

    report = PredicateReport(
        cases=[
            PredicateCase("lives in Sydney", "lives_in", "lives_in"),
            PredicateCase("works at Acme", "works_at", "states"),
        ],
        fallthrough_total=1,
        memories_total=4,
    )

    assert report.accuracy == 0.5
    assert report.fallthrough_rate == 0.25
    assert report.confusions() == [("works at Acme", "works_at", "states")]


def test_consistency_distinguishes_agreement_from_temporal_resolvability():
    lives = ChangePair("Canberra", "Sydney", "home")
    generic = ChangePair("old", "new", "generic state")
    mismatch = ChangePair("engineer", "manager", "job")
    report = ConsistencyReport(
        pairs=[
            (lives, "lives_in", "lives_in"),
            (generic, "states", "states"),
            (mismatch, "works_as", "works_at"),
        ]
    )

    assert report.consistent == 2
    assert report.rate == pytest.approx(2 / 3)
    assert report.resolvable == 1
    assert report.resolvable_rate == pytest.approx(1 / 3)
    assert report.failures() == [
        ("generic state", "old", "states", "states"),
        ("job", "engineer", "works_as", "works_at"),
    ]
    assert ConsistencyReport().rate == 0.0
    assert ConsistencyReport().resolvable_rate == 0.0


def test_collect_supersessions_ignores_active_and_dangling_rows():
    later = memory("new", predicate="lives_in", object_="Sydney")
    earlier = memory(
        "old",
        predicate="lives_in",
        object_="Canberra",
        status="superseded",
        superseded_by="new",
    )
    dangling = memory(
        "dangling",
        predicate="works_at",
        object_="Acme",
        status="superseded",
        superseded_by="missing",
    )

    assert collect_supersessions([earlier, later, dangling]) == [
        Supersession(earlier=earlier, later=later)
    ]


@pytest.mark.parametrize(
    ("earlier", "later", "expected_ok", "message"),
    [
        (
            memory("a", predicate="owns", object_="bike"),
            memory("b", predicate="owns", object_="car"),
            False,
            "multi-valued",
        ),
        (
            memory("a", predicate="lives_in", object_="Sydney"),
            memory("b", predicate="lives_in", object_=" sydney "),
            False,
            "restatement",
        ),
        (
            memory("a", predicate="lives_in", object_=None),
            memory("b", predicate="lives_in", object_="Sydney"),
            False,
            "nothing was actually compared",
        ),
        (
            memory("a", predicate="lives_in", object_="Canberra"),
            memory("b", predicate="lives_in", object_="Sydney"),
            True,
            "differing values",
        ),
        (
            memory("a", predicate="owns", object_="old bike"),
            memory("b", predicate="owns", object_="new bike", replaces_previous=True),
            True,
            "differing values",
        ),
    ],
)
def test_structural_supersession_judgement(earlier, later, expected_ok, message):
    ok, reason = judge_supersession(Supersession(earlier, later))

    assert ok is expected_ok
    assert message in reason


def test_supersede_report_counts_and_renders_only_false_pairs():
    bad = Supersession(
        memory("old", predicate="owns", object_="bike"),
        memory("new", predicate="owns", object_="car"),
    )
    good = Supersession(
        memory("before", predicate="lives_in", object_="Canberra"),
        memory("after", predicate="lives_in", object_="Sydney"),
    )
    report = SupersedeReport(checked=[(bad, False, "multi-valued"), (good, True, "valid change")])

    assert report.total == 2
    assert report.false_count == 1
    assert report.false_rate == 0.5
    assert report.offenders() == [("user/owns", "bike", "car", "multi-valued")]
    assert SupersedeReport().false_rate == 0.0
