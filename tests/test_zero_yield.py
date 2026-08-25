from __future__ import annotations

from llm_long_term_memory.evaluation.datasets.longmemeval import (
    HaystackSession,
    HaystackTurn,
    Instance,
)
from llm_long_term_memory.ingest.pipeline import IngestProgress
from llm_long_term_memory.ingest.zero_yield import audit_zero_yield


def _session(sid: str, *, evidence: bool = False, date: str = "2026/01/05"):
    return HaystackSession(
        session_id=sid,
        date=date,
        turns=[HaystackTurn("user", f"fact {i}", has_answer=evidence and i == 0) for i in range(6)],
    )


def _instance(qid: str, *sessions: HaystackSession) -> Instance:
    return Instance(qid, "single-session-user", "q", "a", "2026/01/06", list(sessions), [])


def test_identical_source_that_yields_elsewhere_is_an_inconsistent_repeat():
    instances = [_instance("q1", _session("shared")), _instance("q2", _session("shared"))]
    progress = IngestProgress(done_sessions={"q1:shared", "q2:shared"})

    report = audit_zero_yield(instances, progress, {("q2", "shared"): 2})

    assert report.categories == {"inconsistent_repeat": 1}
    assert report.cases[0].positive_occurrences == 1
    assert report.cases[0].batch_position == 0


def test_a_gold_evidence_session_is_reported_as_a_real_extraction_miss():
    instances = [_instance("q1", _session("answer", evidence=True))]
    progress = IngestProgress(done_sessions={"q1:answer"})

    report = audit_zero_yield(instances, progress, {})

    assert report.categories == {"evidence_extraction_miss": 1}
    assert report.to_dict()["evidence_zero_yield"] == 1


def test_policy_format_pending_and_consistent_zero_are_not_called_random():
    instances = [
        _instance("q1", _session("blocked"), _session("malformed", date="")),
        _instance("q2", _session("waiting")),
        _instance("q3", _session("waiting")),
        _instance("q4", _session("repeat")),
        _instance("q5", _session("repeat")),
    ]
    progress = IngestProgress(
        done_sessions={"q1:malformed", "q2:waiting", "q4:repeat", "q5:repeat"},
        blocked_sessions={"q1:blocked"},
    )

    report = audit_zero_yield(instances, progress, {})

    assert report.categories == {
        "comparison_pending": 1,
        "consistent_repeat_zero": 2,
        "content_policy": 1,
        "source_format": 1,
    }
    assert not report.complete


def test_position_report_compares_zero_rate_against_every_terminal_session():
    sessions = [_session(f"s{i}") for i in range(6)]
    instances = [_instance("q1", *sessions)]
    progress = IngestProgress(done_sessions={f"q1:s{i}" for i in range(6)})
    counts = {("q1", f"s{i}"): 1 for i in range(4)}

    report = audit_zero_yield(instances, progress, counts, batch_size=6).to_dict()

    assert report["position_bands"]["front_0_3"] == {"total": 4, "zero": 0, "rate": 0.0}
    assert report["position_bands"]["later_4_end"] == {"total": 2, "zero": 2, "rate": 1.0}


def test_reused_id_with_different_source_is_data_collision_not_model_variance():
    first = _session("shared")
    second = _session("shared")
    second.turns[0].content = "different source text"
    instances = [_instance("q1", first), _instance("q2", second)]
    progress = IngestProgress(done_sessions={"q1:shared", "q2:shared"})

    report = audit_zero_yield(instances, progress, {("q2", "shared"): 2})

    assert report.categories == {"source_id_collision": 1}
    assert report.cases[0].source_variants == 2
    assert report.cases[0].source_text_variants == 2


def test_identical_text_under_different_dates_is_not_called_an_exact_repeat():
    first = _session("shared", date="2026/01/05")
    second = _session("shared", date="2026/02/05")
    instances = [_instance("q1", first), _instance("q2", second)]
    progress = IngestProgress(done_sessions={"q1:shared", "q2:shared"})

    report = audit_zero_yield(instances, progress, {("q2", "shared"): 2})

    assert report.categories == {"date_variant_outcome_difference": 1}
    assert report.cases[0].source_text_variants == 1
    assert report.cases[0].source_date_variants == 2


def test_short_zero_yields_are_accounted_for_even_when_excluded_from_substantive_audit():
    short = HaystackSession("short", "2026/01/05", [HaystackTurn("user", "hello")])
    instances = [_instance("q1", short, _session("long"))]
    progress = IngestProgress(done_sessions={"q1:short", "q1:long"})

    report = audit_zero_yield(instances, progress, {}).to_dict()

    assert report["all_zero_yield_sessions"] == 2
    assert report["short_zero_yield_sessions_below_min_turns"] == 1
    assert report["substantive_zero_yield_sessions"] == 1
    assert report["all_categories"] == {
        "short_no_durable_fact_candidate": 1,
        "unclassified_single_zero": 1,
    }
    assert len(report["all_cases"]) == 2
    assert len(report["cases"]) == 1


def test_short_policy_refusal_keeps_its_stronger_explicit_reason():
    short = HaystackSession("short", "2026/01/05", [HaystackTurn("user", "hello")])
    instances = [_instance("q1", short)]
    progress = IngestProgress(blocked_sessions={"q1:short"})

    report = audit_zero_yield(instances, progress, {}).to_dict()

    assert report["all_categories"] == {"content_policy": 1}
    assert report["short_categories"] == {"content_policy": 1}
    assert report["categories"] == {}
