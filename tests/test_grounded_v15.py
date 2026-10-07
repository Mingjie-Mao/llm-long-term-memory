from dataclasses import replace

import pytest

from llm_long_term_memory.runtime.grounded_answering import (
    EvidenceLedger,
    EvidenceSource,
    named_reading_duration,
)


def sources():
    return [
        EvidenceSource(
            "E1", "raw", "t1", "I started reading 'Work A' today.", "user", "2026-09-01"
        ),
        EvidenceSource(
            "E2", "raw", "t2", "I just finished reading 'Work A' today.", "user", "2026-09-08"
        ),
        EvidenceSource(
            "E3", "raw", "t3", "I began listening to 'Work B' today.", "user", "2026-09-10"
        ),
        EvidenceSource(
            "E4", "raw", "t4", "I completed listening to 'Work B' today.", "user", "2026-09-17"
        ),
    ]


def compute(raw):
    return named_reading_duration(
        EvidenceLedger(sources=raw),
        "How many weeks in total did I spend reading 'Work A' and listening to 'Work B'?",
        "2026-10-01",
    )


def test_named_original_events_compute_without_model_arithmetic_or_operation():
    result = compute(sources()[::-1])
    # Selection orders start/finish regardless of retrieval order.
    assert result.computed and result.answer == "2 weeks"
    assert len(result.detail["endpoint_provenance"]) == 4


def test_equivalent_mentions_deduplicate_but_conflicting_dates_refuse():
    raw = sources()
    raw.append(replace(raw[0], id="E5", source_id="t5"))
    result = compute(raw)
    assert result.answer == "2 weeks" and len(result.detail["endpoint_provenance"]) == 5
    raw[-1] = replace(raw[-1], conversation_date="2026-08-01")
    assert not compute(raw).computed


@pytest.mark.parametrize(
    "changes",
    [
        {"text": "I finished reading 'Work A'."},
        {"text": "I plan to finish reading 'Work A' today."},
        {"text": "I did not finish reading 'Work A' today."},
        {"text": "My friend finished reading 'Work A' today."},
        {"role": "assistant"},
        {"conversation_date": "2026-08-01"},
        {"text": "I finished reading 'Work A' today after starting yesterday."},
    ],
)
def test_missing_undated_wrong_role_subject_plan_or_reversed_events_refuse(changes):
    raw = sources()
    raw[1] = replace(raw[1], **changes)
    assert not compute(raw).computed


def test_question_selects_duration_despite_wrong_model_sum_or_guess():
    from llm_long_term_memory.conversation import AnswerRequest
    from llm_long_term_memory.runtime.grounded_answering import (
        GroundedAnswererV15,
        GroundedVerdictV7,
    )

    ledger = EvidenceLedger(sources=sources())
    engine = object.__new__(GroundedAnswererV15)
    request = AnswerRequest(
        "How many weeks in total did I spend reading 'Work A' and listening to 'Work B'?",
        "2026-10-01",
        "alice",
    )
    draft = GroundedVerdictV7(
        operation="sum",
        status="answer",
        answer="1000 weeks",
        scope_complete=True,
        operands=[{"source": "E1", "value": "1000", "unit": "weeks", "key": "guess"}],
    )
    parsed = engine.parse_verdict(draft.model_dump_json(), ledger, request)
    result = engine.validate_verdict(parsed, ledger, request)
    assert result.computed and result.answer == "2 weeks"
    assert engine.can_answer(parsed, result)
    ledger.sources.pop()
    result = engine.validate_verdict(parsed, ledger, request)
    assert not result.computed and not engine.can_answer(parsed, result)


def test_unrelated_about_word_does_not_make_exact_interval_approximate():
    raw = sources()
    raw[3] = replace(raw[3], text=raw[3].text + " Tell me about literature.")
    result = compute(raw)
    assert result.answer == "2 weeks" and not result.detail["approximate"]
    raw[0] = replace(raw[0], text="I started reading 'Work A' about three days ago.")
    result = compute(raw)
    assert result.computed and result.answer.startswith("About ")
