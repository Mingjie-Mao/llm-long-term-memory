from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.runtime.grounded_answering import (
    Calculation,
    EvidenceLedger,
    EvidenceSource,
    GroundedAnswererV16,
    scoped_quantity_disclosure,
)


def engine():
    return object.__new__(GroundedAnswererV16)


def test_uncertain_period_returns_actionable_gap_and_original_source_identity():
    ledger = EvidenceLedger(
        sources=[
            EvidenceSource(
                "E1",
                "raw",
                "t1",
                "I attended three sessions of the art group.",
                "user",
                "2026-10-01",
                session_id="s1",
                turn_index=0,
            ),
            EvidenceSource(
                "E2",
                "raw",
                "t2",
                "The art group I attended last year helped. I remember attending five sessions.",
                "user",
                "2026-10-01",
                session_id="s2",
                turn_index=0,
            ),
        ]
    )
    request = AnswerRequest(
        "How many sessions of the art group did I attend?", "2026-10-03", "alice"
    )
    result = scoped_quantity_disclosure(ledger, request.question)
    gap = engine().completion_notes(ledger, request, None, result, result.answer)["evidence_gap"]
    assert gap["missing_fields"] == ["report_period", "overlap_between_reports"]
    assert gap["user_id"] == "alice" and gap["sources"][0]["source_id"] == "t1"
    assert "combined total" in result.answer
    assert (
        engine().completion_notes(
            ledger, request, None, Calculation("8 sessions", True, {}), "8 sessions"
        )["evidence_gap"]
        is None
    )


def test_missing_event_identity_and_dates_are_named_without_inventing_values():
    request = AnswerRequest("How many weeks did I read 'Book'?", "2026-10-03", "alice")
    notes = engine().completion_notes(
        EvidenceLedger(),
        request,
        None,
        Calculation("", False, {"cause": "named_duration_requires_start_end_pair"}),
        "I do not know.",
    )
    assert notes["evidence_gap"]["missing_fields"] == [
        "event_identity",
        "event_date",
        "start_end_pair",
    ]
    assert notes["evidence_gap"]["sources"] == []


def test_same_user_archived_clarification_resolves_duration_foreign_user_cannot(tmp_path):
    from datetime import datetime

    from llm_long_term_memory.runtime.grounded_answering import build_ledger, named_reading_duration
    from llm_long_term_memory.store import Session, SQLiteMemoryStore, Turn

    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()

    def add(sid, user, text, date):
        turn = Turn(sid + ":0", sid, 0, "user", text, datetime.fromisoformat(date))
        store.add_session(Session(sid, user, datetime.fromisoformat(date), "clarification", [turn]))
        return turn

    start = add("a", "alice", "I started reading 'Book' today.", "2026-09-01")
    foreign = add("b", "bob", "I finished reading 'Book' today.", "2026-09-15")
    other_start = add("d", "alice", "I started reading 'Other' today.", "2026-09-01")
    other_finish = add("e", "alice", "I finished reading 'Other' today.", "2026-09-08")
    question = "How many weeks in total did I spend reading 'Book' and 'Other'?"
    import pytest

    with pytest.raises(ValueError, match="tenant"):
        build_ledger(store, "alice", [], [start, foreign])
    ledger = build_ledger(store, "alice", [], [start, other_start, other_finish])
    assert not named_reading_duration(ledger, question, "2026-10-03").computed
    clarified = add("c", "alice", "I finished reading 'Book' today.", "2026-09-15")
    ledger = build_ledger(store, "alice", [], [start, other_start, other_finish, clarified])
    assert named_reading_duration(ledger, question, "2026-10-03").answer == "3 weeks"
    store.close()
