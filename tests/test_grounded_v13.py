from dataclasses import replace
from datetime import datetime

import pytest

from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.runtime.grounded_answering import (
    EvidenceLedger,
    EvidenceSource,
    GroundedAnswererV12,
    GroundedAnswererV13,
    GroundedVerdictV7,
    ordinal_song_progression,
)
from llm_long_term_memory.store import Session, SQLiteMemoryStore, Turn


def source(sid, text, turn=0, role="user", session="s"):
    return EvidenceSource(
        sid, "raw", sid, text, role, "2026-09-01", session_id=session, turn_index=turn
    )


def savings(ledger, rows, mode="second_minus_first"):
    engine = object.__new__(GroundedAnswererV13)
    request = AnswerRequest(
        "How much will I save by taking the bus to work instead of a ferry?", "2026-10-01", "alice"
    )
    response = GroundedVerdictV7(
        operation="difference",
        status="answer",
        scope_complete=True,
        calculation_mode=mode,
        operands=rows,
        observations=[
            {"source": s.id, "interpretation": "Fare", "decision": "include"}
            for s in ledger.sources
        ],
    )
    parsed = engine.parse_verdict(response.model_dump_json(), ledger, request)
    return engine.validate_verdict(parsed, ledger, request)


@pytest.mark.parametrize("reverse", [False, True])
def test_savings_direction_follows_question_alternatives_not_row_order_or_model_enum(reverse):
    ledger = EvidenceLedger(
        sources=[source("E1", "The ferry costs $60."), source("E2", "The bus costs $10.")]
    )
    rows = [
        {"source": "E1", "value": "60", "unit": "$", "key": "ferry"},
        {"source": "E2", "value": "10", "unit": "$", "key": "bus"},
    ]
    result = savings(ledger, rows[::-1] if reverse else rows)
    assert result.computed and result.answer == "50 $"


def test_negative_savings_remain_negative_and_approximations_are_disclosed():
    ledger = EvidenceLedger(
        sources=[source("E1", "The ferry costs around $10."), source("E2", "The bus costs $60.")]
    )
    rows = [
        {"source": "E1", "value": "10", "unit": "$", "key": "ferry"},
        {"source": "E2", "value": "60", "unit": "$", "key": "bus"},
    ]
    result = savings(ledger, rows)
    assert result.computed and result.answer == "About -50 $"


def test_comparison_rejects_ambiguous_alternatives_and_borrowed_unit():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "The ferry and bus both operate. The ferry costs $60."),
            source("E2", "The bus costs $10."),
        ]
    )
    rows = [
        {"source": "E1", "value": "60", "unit": "$", "key": "ferry"},
        {"source": "E2", "value": "10", "unit": "$", "key": "bus"},
    ]
    assert not savings(ledger, rows).computed
    ledger.sources[0] = replace(ledger.sources[0], text="The ferry costs 60 euros.")
    assert not savings(ledger, rows).computed


def songs():
    return [
        source("E1", "Create a song with notes", 0),
        source("E2", "Song one\nChorus:\nG A B\nLyrics", 1, "assistant"),
        source("E3", "Make it more romantic", 2),
        source(
            "E4", "Song two\nChorus:\nC D E F\nLyrics\nChorus:\nC D E F\nLyrics", 3, "assistant"
        ),
    ]


def question():
    return "What was your chord progression for the chorus in the second song?"


def test_ordinal_requires_original_contiguous_roles_and_unique_repeated_section():
    ledger = EvidenceLedger(sources=songs())
    result = ordinal_song_progression(ledger, question())
    assert result.computed and result.answer == "C D E F"
    assert result.detail["structural_citation"]["source"] == "E4"
    ledger.sources.pop(2)
    assert not ordinal_song_progression(ledger, question()).computed


@pytest.mark.parametrize(
    "change",
    [
        {"role": "user"},
        {"session_id": "other"},
        {"text": "Chorus:\nC D E F\nLyrics\nChorus:\nG A B\nLyrics"},
    ],
)
def test_ordinal_rejects_wrong_role_session_or_conflicting_section(change):
    ledger = EvidenceLedger(sources=songs())
    ledger.sources[3] = replace(ledger.sources[3], **change)
    assert not ordinal_song_progression(ledger, question()).computed


def test_bridge_delivers_missing_original_user_turn_and_never_crosses_tenants(
    tmp_path, monkeypatch
):
    store = SQLiteMemoryStore(tmp_path / "store.db")
    store.initialize()
    date = datetime(2026, 9, 1)
    for sid, user in [("s", "alice"), ("bob", "bob")]:
        store.add_session(
            Session(
                sid,
                user,
                date,
                turns=[
                    Turn(s.id, sid, s.turn_index, s.role, s.text, date)
                    for s in [
                        replace(s, source_id=f"{sid}-{s.turn_index}", id=f"{sid}-{s.turn_index}")
                        for s in songs()
                    ]
                ],
            )
        )
    monkeypatch.setattr(GroundedAnswererV12, "extend_ledger", lambda *args: {})
    engine = object.__new__(GroundedAnswererV13)
    engine.store = store
    try:
        for sid in ["s", "bob"]:
            raw = [
                replace(s, session_id=sid, source_id=f"{sid}-{s.turn_index}")
                for s in songs()
                if s.turn_index != 2
            ]
            ledger = EvidenceLedger(sources=list(raw))
            request = AnswerRequest(question(), "2026-10-01", "alice")
            if sid == "bob":
                with pytest.raises(ValueError, match="tenant"):
                    engine.extend_ledger(ledger, request, [], None)
            else:
                original = {s.source_id: s.text for s in ledger.sources}
                note = engine.extend_ledger(ledger, request, [], None)
                assert note["conversation_bridge_turns"] == ["s-2"]
                assert ordinal_song_progression(ledger, question()).computed
                assert all(
                    s.text == original[s.source_id]
                    for s in ledger.sources
                    if s.source_id in original
                )
                budget = EvidenceLedger(
                    sources=list(raw),
                    max_tokens=int(len(EvidenceLedger(sources=list(raw)).render()) / 4.6) + 1,
                )
                engine.extend_ledger(budget, request, [], None)
                assert (
                    len(budget.sources) == 3
                    and not ordinal_song_progression(budget, question()).computed
                )
    finally:
        store.close()


def test_third_song_counts_actual_turns_and_rejects_intervening_non_artifact():
    raw = [
        *songs(),
        source("E5", "Write one more version", 4),
        source("E6", "Song three\nChorus:\nD E F G\nLyrics", 5, "assistant"),
    ]
    ledger = EvidenceLedger(sources=raw)
    result = ordinal_song_progression(ledger, question().replace("second", "third"))
    assert result.computed and result.answer == "D E F G"
    ledger.sources[1] = replace(ledger.sources[1], text="Can you specify the genre?")
    assert not ordinal_song_progression(ledger, question()).computed


def test_ordinal_multiple_qualifying_sessions_is_ambiguous():
    raw = songs() + [replace(s, id=s.id + "b", session_id="b") for s in songs()]
    assert not ordinal_song_progression(EvidenceLedger(sources=raw), question()).computed
