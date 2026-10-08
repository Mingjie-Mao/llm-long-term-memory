from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.runtime.grounded_answering import (
    EvidenceLedger,
    EvidenceSource,
    GroundedAnswererV14,
    GroundedVerdictV7,
)


def run(
    *,
    question="How many days ago did I hike?",
    clock="2026-09-04",
    event_text="I hiked today.",
    **changes,
):
    ledger = EvidenceLedger(
        sources=[EvidenceSource("E1", "raw", "t1", event_text, "user", "2026-09-01")]
    )
    values = {
        "observations": [
            {"source": "E1", "interpretation": "Actual hike", "decision": "include"},
            {"source": "QUESTION_DATE", "interpretation": "Request clock", "decision": "include"},
        ],
        "reviewed_sources": ["E1", "QUESTION_DATE"],
        "scope_complete": True,
        "status": "answer",
        "operation": "duration",
        "operands": [
            {"source": "E1", "key": "hike"},
            {"source": "QUESTION_DATE", "key": "question-date"},
        ],
        "result_unit": "days",
        "calculation_mode": "days",
    }
    values.update(changes)
    response = GroundedVerdictV7(**values)
    request = AnswerRequest(question, clock, "alice")
    engine = object.__new__(GroundedAnswererV14)
    parsed = engine.parse_verdict(response.model_dump_json(), ledger, request)
    return engine.validate_verdict(parsed, ledger, request), parsed


def test_request_clock_observation_is_metadata_and_actual_event_still_supplies_date():
    result, parsed = run()
    assert result.computed and result.answer == "3 days"
    assert "request_clock_control_observation" in parsed.canonicalization
    assert len(parsed.observations) == 1
    assert result.detail["citations"][-1]["resolved_date"] == "2026-09-04"


def test_control_does_not_supply_missing_event_or_replace_historical_pair():
    result, _ = run(operands=[{"source": "QUESTION_DATE", "key": "question-date"}])
    assert not result.computed
    result, _ = run(question="How many days between hiking and moving?")
    assert not result.computed
    result, _ = run(clock="unknown")
    assert not result.computed


def test_control_cannot_mask_unknown_excluded_or_uncertain_evidence():
    for source, decision in [
        ("UNKNOWN", "include"),
        ("QUESTION_DATE", "exclude"),
        ("QUESTION_DATE", "uncertain"),
    ]:
        result, _ = run(
            observations=[
                {"source": "E1", "interpretation": "Actual event", "decision": "include"},
                {"source": source, "interpretation": "Clock", "decision": decision},
            ]
        )
        assert not result.computed
    result, _ = run(
        observations=[
            {"source": "QUESTION_DATE", "interpretation": "Clock", "decision": "include"}
        ],
        operands=[
            {"source": "UNKNOWN", "key": "hike"},
            {"source": "QUESTION_DATE", "key": "question-date"},
        ],
    )
    assert not result.computed
    result, _ = run(event_text="I hiked in the mountains.")
    assert not result.computed


def test_clock_cannot_be_guessed_quoted_or_mislabeled_by_provider():
    for changes in [{"key": "event"}, {"quote": "2026-09-04"}, {"value": "2026-09-04"}]:
        result, _ = run(
            operands=[
                {"source": "E1", "key": "hike"},
                {"source": "QUESTION_DATE", "key": "question-date", **changes},
            ]
        )
        assert not result.computed
