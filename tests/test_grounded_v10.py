import json

from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.runtime.grounded_answering import (
    EvidenceLedger,
    EvidenceSource,
    GroundedAnswererV10,
    GroundedVerdictV7,
)


def source(sid, text, *, kind="raw", session="s", turn=0, role="user", date="2026-09-01"):
    return EvidenceSource(sid, kind, sid, text, role, date, session_id=session, turn_index=turn)


def parse(ledger, operands, *, operation="duration", question="How many weeks?", **extra):
    engine = object.__new__(GroundedAnswererV10)
    response = GroundedVerdictV7(
        observations=[
            {"source": s.id, "interpretation": "Reviewed evidence", "decision": "include"}
            for s in ledger.sources
        ],
        reviewed_sources=[],
        scope_complete=True,
        status="answer",
        operation=operation,
        operands=operands,
        result_unit="weeks" if operation == "duration" else "",
        **extra,
    )
    request = AnswerRequest(question, "2026-10-01", "alice")
    parsed = engine.parse_verdict(response.model_dump_json(), ledger, request)
    return engine, parsed, engine.validate_verdict(parsed, ledger, request), request


def test_escaped_quote_normalization_requires_actual_declared_source_membership():
    ledger = EvidenceLedger(
        sources=[source("E1", 'I finished reading "Book One" today.'), source("E2", "Done today.")]
    )
    _, parsed, _, _ = parse(
        ledger,
        [{"source": "E1", "quote": 'I finished reading \\"Book One\\" today.', "key": "book"}],
    )
    assert json.loads(parsed.operand_records[0])["quote"] == ledger.sources[0].text
    _, parsed, _, _ = parse(
        ledger,
        [{"source": "E2", "quote": 'I finished reading \\"Book One\\" today.', "key": "book"}],
    )
    assert json.loads(parsed.operand_records[0])["quote"] != ledger.sources[0].text


def dated_memory_ledger():
    return EvidenceLedger(
        sources=[
            source("E1", "The user finished reading 'Book One'.", kind="memory"),
            source("E2", "I just finished reading 'Book One' today."),
            source("E3", "I started reading 'Book One' today.", session="start", date="2026-08-25"),
        ]
    )


def test_undated_memory_date_comes_from_exact_raw_origin_and_matching_event_phrase():
    ledger = dated_memory_ledger()
    _, parsed, result, _ = parse(
        ledger,
        [{"source": "E3", "key": "book"}, {"source": "E1", "key": "book"}],
    )
    assert result.computed and result.answer == "1 weeks"
    assert json.loads(parsed.operand_records[1])["source"] == "E2"
    assert json.loads(parsed.operand_records[1])["time"] == "today"


def test_memory_date_projection_rejects_wrong_origin_role_subject_and_ambiguous_date():
    for replacement in [
        source("E2", "I just finished reading 'Book One' today.", session="other"),
        source("E2", "I just finished reading 'Book One' today.", turn=1),
        source("E2", "I just finished reading 'Book One' today.", role="assistant"),
        source("E2", "I just finished reading 'Book Two' today."),
        source("E2", "I just finished reading 'Book One' today, after starting yesterday."),
    ]:
        ledger = dated_memory_ledger()
        ledger.sources[1] = replacement
        _, _, result, _ = parse(
            ledger,
            [{"source": "E3", "key": "book"}, {"source": "E1", "key": "book"}],
        )
        assert not result.computed


def test_date_fields_are_derived_from_actual_source_words_not_numeric_provider_fields():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "I started today."),
            source("E2", "I finished today.", date="2026-09-08"),
        ]
    )
    _, parsed, result, _ = parse(
        ledger,
        [
            {"source": "E1", "value": "1900-01-01", "unit": "days", "key": "activity"},
            {"source": "E2", "value": "9999", "key": "activity"},
        ],
    )
    assert result.computed and result.answer == "1 weeks"
    assert all(
        not json.loads(r)["value"] and not json.loads(r)["unit"] for r in parsed.operand_records
    )
    assert parsed.canonicalization


def test_undated_memory_requires_unique_raw_origin():
    ledger = dated_memory_ledger()
    ledger.sources.append(source("E4", "I just finished reading 'Book One' today."))
    _, _, result, _ = parse(
        ledger, [{"source": "E3", "key": "book"}, {"source": "E1", "key": "book"}]
    )
    assert not result.computed


def test_approximation_words_must_modify_time_not_unrelated_discussion_topic():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "I started today. Tell me about history."),
            source("E2", "I finished today.", date="2026-09-08"),
        ]
    )
    _, _, result, _ = parse(
        ledger, [{"source": "E1", "key": "activity"}, {"source": "E2", "key": "activity"}]
    )
    assert result.answer == "1 weeks" and not result.detail["approximate"]
    ledger.sources[0] = source("E1", "I started about three days ago.")
    _, _, result, _ = parse(
        ledger, [{"source": "E1", "key": "activity"}, {"source": "E2", "key": "activity"}]
    )
    assert result.answer == "About 1 weeks and 3 days" and result.detail["approximate"]


def test_redundant_question_endpoint_removed_only_for_two_actual_historical_endpoints():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "I started today."),
            source("E2", "I invested today.", date="2026-09-08"),
        ]
    )
    operands = [
        {"source": "E1", "key": "start"},
        {"source": "E2", "key": "investment"},
        {"source": "QUESTION_DATE", "key": "question-date"},
    ]
    _, _, result, _ = parse(ledger, operands, question="How many weeks had passed when I invested?")
    assert result.computed and result.answer == "1 weeks"
    _, _, result, _ = parse(ledger, operands, question="How many weeks have I been doing this now?")
    assert not result.computed


def test_explicit_between_event_and_today_retains_actual_question_date_endpoint():
    ledger = EvidenceLedger(sources=[source("E1", "I started today.")])
    _, _, result, _ = parse(
        ledger,
        [{"source": "E1", "key": "activity"}, {"source": "QUESTION_DATE", "key": "question-date"}],
        question="How many weeks between when I started and today?",
    )
    assert result.computed and result.answer == "4 weeks and 2 days"


def test_known_observation_is_review_but_exclusion_is_never_overridden():
    ledger = EvidenceLedger(sources=[source("E1", "I bought five rings.")])
    engine, parsed, result, request = parse(
        ledger,
        [{"source": "E1", "value": "five", "unit": "rings", "key": "purchase"}],
        operation="sum",
    )
    assert result.computed
    assert "E1" in parsed.reviewed_sources
    observations = [json.loads(s) for s in parsed.observations]
    observations[0]["decision"] = "exclude"
    excluded = parsed.model_copy(update={"observations": [json.dumps(s) for s in observations]})
    assert not engine.validate_verdict(excluded, ledger, request).computed


def test_location_focus_only_reorders_actual_same_session_user_context():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "I redeemed the coupon.", turn=4),
            source("E2", "I use Shop One's rewards app.", turn=2),
            source("E3", "Shop Two also sends coupons.", role="assistant", turn=5),
            source("E4", "I shop at Shop Three.", session="unrelated"),
        ]
    )
    _, _, result, _ = parse(
        ledger,
        [{"source": "E1", "value": "coupon", "key": "redemption"}],
        operation="lookup",
        question="Where did I redeem the coupon?",
        answer="You redeemed a coupon.",
    )
    assert set(result.detail["focus_sources"]) == {"E1", "E2"}
