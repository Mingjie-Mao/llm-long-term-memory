import json

from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.runtime.grounded_answering import (
    EvidenceLedger,
    EvidenceSource,
    GroundedAnswererV9,
    GroundedVerdictV7,
    event_candidate_sources,
    next_numbered_source,
)


def source(sid, text, *, session="s", turn=0, role="user", date="2026-09-01"):
    return EvidenceSource(sid, "raw", sid, text, role, date, session_id=session, turn_index=turn)


def parse(ledger, operation, operands, *, question="How many sessions?", answer="5 sessions"):
    engine = object.__new__(GroundedAnswererV9)
    response = GroundedVerdictV7(
        observations=[
            {"source": s.id, "interpretation": "Supported record", "decision": "include"}
            for s in ledger.sources
        ],
        reviewed_sources=[s.id for s in ledger.sources],
        scope_complete=True,
        status="answer",
        operation=operation,
        operands=operands,
        answer=answer,
    )
    request = AnswerRequest(question, "2026-10-01", "alice")
    parsed = engine.parse_verdict(response.model_dump_json(), ledger, request)
    return engine, parsed, engine.validate_verdict(parsed, ledger, request), request


def test_quoted_group_count_computes_quantity_and_does_not_count_phrase_once():
    ledger = EvidenceLedger(sources=[source("E1", "I attended five sessions.")])
    _, parsed, result, _ = parse(
        ledger, "count", [{"source": "E1", "value": "five", "unit": "sessions", "key": "total"}]
    )
    assert parsed.operation == "sum"
    assert result.computed and result.answer == "5 sessions"


def test_group_count_still_rejects_borrowed_units_and_mixed_members():
    ledger = EvidenceLedger(
        sources=[source("E1", "I bought five gallons and two fish."), source("E2", "I have a koi.")]
    )
    _, _, result, _ = parse(
        ledger, "count", [{"source": "E1", "value": "five", "unit": "fish", "key": "total"}]
    )
    assert not result.computed
    _, parsed, result, _ = parse(
        ledger,
        "count",
        [
            {"source": "E1", "value": "two", "unit": "fish", "key": "total"},
            {"source": "E2", "value": "koi", "key": "koi"},
        ],
    )
    assert parsed.operation == "count"
    assert not result.computed


def test_factual_number_spelling_equivalence_requires_bound_original_and_answer_units():
    ledger = EvidenceLedger(sources=[source("E1", "I attended five sessions.")])
    for value, answer, valid in [
        ("five", "5 sessions", True),
        ("5", "five sessions", True),
        ("five", "6 sessions", False),
        ("five", "5 weddings", False),
    ]:
        _, _, result, _ = parse(
            ledger,
            "lookup",
            [{"source": "E1", "value": value, "unit": "sessions", "key": "total"}],
            answer=answer,
        )
        assert (result.detail.get("cause") == "validated_lookup") == valid


def test_number_equivalence_cannot_skip_a_later_excluded_or_unkeyed_operand():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "I attended five sessions."),
            source("E2", "I attended two sessions."),
        ]
    )
    engine, parsed, _, request = parse(
        ledger,
        "lookup",
        [
            {"source": "E1", "value": "five", "unit": "sessions", "key": "new"},
            {"source": "E2", "value": "two", "unit": "sessions", "key": "old"},
        ],
        answer="5 sessions, previously two sessions.",
    )
    observations = [json.loads(s) for s in parsed.observations]
    observations[1]["decision"] = "exclude"
    excluded = parsed.model_copy(update={"observations": [json.dumps(s) for s in observations]})
    assert engine.validate_verdict(excluded, ledger, request).detail["cause"] != "validated_lookup"
    rows = [json.loads(s) for s in parsed.operand_records]
    rows[1]["key"] = ""
    unkeyed = parsed.model_copy(update={"operand_records": [json.dumps(s) for s in rows]})
    assert engine.validate_verdict(unkeyed, ledger, request).detail["cause"] != "validated_lookup"


def numbered_ledger():
    return EvidenceLedger(
        sources=[
            source("E1", "7. Open the cover", role="assistant", turn=10),
            source("E2", "8. Turn the valve", role="assistant", turn=12),
            source("E3", "9. Start the engine", role="assistant", turn=14),
        ]
    )


def test_successor_uses_original_session_turn_order_not_later_matching_entry():
    ledger = numbered_ledger()
    question = "What was your step after 7. Open the cover?"
    assert next_numbered_source(ledger, question).id == "E2"
    _, _, result, _ = parse(
        ledger,
        "lookup",
        [{"source": "E3", "value": "9. Start the engine", "key": "step"}],
        question=question,
        answer="9. Start the engine",
    )
    assert result.detail["cause"] == "sequence_successor_required"
    assert "E2" in result.detail["focus_sources"]


def test_successor_does_not_infer_when_session_role_order_or_uniqueness_missing():
    question = "What was your step after 7. Open the cover?"
    for replacement in [
        source("E2", "8. Turn the valve", role="assistant", turn=12, session="other"),
        source("E2", "8. Turn the valve", role="user", turn=12),
        source("E2", "8. Turn the valve", role="assistant", turn=8),
    ]:
        ledger = numbered_ledger()
        ledger.sources[1] = replacement
        assert next_numbered_source(ledger, question) is None
    ledger = numbered_ledger()
    ledger.sources.append(source("E4", "8. Check a different valve", role="assistant", turn=13))
    assert next_numbered_source(ledger, question) is None
    ledger = numbered_ledger()
    ledger.sources[0] = source("E1", "17. Open the cover", role="assistant", turn=10)
    assert next_numbered_source(ledger, question) is None
    ledger = numbered_ledger()
    ledger.sources.append(source("E4", "7. Open the cover", role="assistant", session="other"))
    assert next_numbered_source(ledger, question) is None


def test_duration_review_includes_completion_and_preserves_raw_relative_words():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "I started reading 'Book One' today."),
            source("E2", "I finished reading 'Book One' today.", session="t", date="2026-09-08"),
            source("E3", "I finished cleaning today.", session="u"),
        ]
    )
    assert event_candidate_sources(ledger, "How long did I read 'Book One'?") == ["E1", "E2"]
    assert ledger.sources[1].text.endswith("today.")


def test_source_repair_reorders_original_body_without_eviction_or_gold_insertion():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "First chorus: G A.", role="assistant", turn=1),
            source("E2", "Second chorus: C D.", role="assistant", turn=3),
        ]
    )
    engine, parsed, result, request = parse(
        ledger,
        "lookup",
        [{"source": "E2", "quote": "First chorus: G A.", "value": "G A", "key": "chorus"}],
        question="What was the second chorus?",
        answer="G A",
    )
    focused = engine.render_context(ledger, request, 1, result)
    assert focused.index("Second chorus: C D.") < focused.index("First chorus: G A.")
    assert all(focused.count(s.text) == 1 for s in ledger.sources)
    assert [s.id for s in ledger.sources] == ["E1", "E2"]
    assert "Second chorus: C D." not in engine.review_note(parsed, result)
    assert json.loads(parsed.operand_records[0])["source"] == "E2"
