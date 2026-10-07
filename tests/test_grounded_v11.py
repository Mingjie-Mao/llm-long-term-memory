import json
from dataclasses import replace

from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.runtime.grounded_answering import (
    EvidenceLedger,
    EvidenceSource,
    GroundedAnswererV11,
    GroundedVerdictV7,
    scoped_quantity_disclosure,
)


def source(sid, text, *, session="s", turn=0, role="user", date="2026-09-01"):
    return EvidenceSource(sid, "raw", sid, text, role, date, session_id=session, turn_index=turn)


def parse(ledger, operation, operands, question, **extra):
    engine = object.__new__(GroundedAnswererV11)
    response = GroundedVerdictV7(
        observations=[
            {"source": s.id, "interpretation": "Reviewed evidence", "decision": "include"}
            for s in ledger.sources
        ],
        scope_complete=True,
        status="answer",
        operation=operation,
        operands=operands,
        **extra,
    )
    request = AnswerRequest(question, "2026-10-01", "alice")
    parsed = engine.parse_verdict(response.model_dump_json(), ledger, request)
    return engine, parsed, engine.validate_verdict(parsed, ledger, request), request


def book_ledger():
    return EvidenceLedger(
        sources=[
            source("E1", "I started reading 'Book One' today."),
            source("E2", "I finished reading 'Book One' today.", session="end1", date="2026-09-08"),
            source(
                "E3", "I started reading 'Book Two' today.", session="start2", date="2026-09-08"
            ),
            source("E4", "I finished reading 'Book Two' today.", session="end2", date="2026-09-15"),
        ]
    )


def test_duration_groups_and_phase_order_come_from_actual_quoted_objects():
    ledger = book_ledger()
    engine, parsed, result, request = parse(
        ledger,
        "duration",
        [
            {"source": "E2", "key": "book-one-end"},
            {"source": "E3", "key": "book-two-start"},
            {"source": "E1", "key": "book-one-start"},
            {"source": "E4", "key": "book-two-end"},
        ],
        "How many weeks in total did I read 'Book One' and 'Book Two'?",
        result_unit="weeks",
    )
    assert result.computed and result.answer == "2 weeks"
    assert [json.loads(s)["source"] for s in parsed.operand_records] == ["E1", "E2", "E3", "E4"]
    assert not engine.requires_review(request)


def test_duration_cannot_fabricate_zero_from_duplicate_finish_or_ambiguous_object():
    ledger = book_ledger()
    _, _, result, _ = parse(
        ledger,
        "duration",
        [
            {"source": "E1", "key": "a"},
            {"source": "E2", "key": "a"},
            {"source": "E4", "key": "b"},
            {"source": "E4", "key": "b"},
        ],
        "How many weeks in total did I read 'Book One' and 'Book Two'?",
        result_unit="weeks",
    )
    assert not result.computed
    assert result.detail["cause"] == "named_duration_requires_start_end_pair"


def test_numeric_count_needs_bound_unit_and_never_becomes_one_named_member():
    ledger = EvidenceLedger(sources=[source("E1", "I attended three sessions.")])
    _, parsed, result, _ = parse(
        ledger,
        "count",
        [{"source": "E1", "value": "three", "key": "attendance"}],
        "How many sessions did I attend?",
    )
    assert parsed.operation == "sum"
    assert result.answer == "3 sessions"
    _, _, result, _ = parse(
        ledger,
        "count",
        [{"source": "E1", "value": "three", "key": "attendance"}],
        "How many weddings did I attend?",
    )
    assert not result.computed and result.detail["cause"] == "quantity_unit_required"


def test_unique_literal_successor_is_returned_from_source_despite_wrong_model_choice():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "7. Open the cover", role="assistant", turn=10),
            source("E2", "8. Turn the valve", role="assistant", turn=12),
            source("E3", "9. Start the engine", role="assistant", turn=14),
        ]
    )
    _, _, result, _ = parse(
        ledger,
        "lookup",
        [{"source": "E3", "value": "9. Start the engine", "key": "step"}],
        "What was your step after 7. Open the cover?",
        answer="9. Start the engine",
    )
    assert result.computed and result.answer == "8. Turn the valve"
    assert result.detail["structural_citation"]["source"] == "E2"


def quantity_ledger():
    return EvidenceLedger(
        sources=[
            source("E1", "I attended three sessions of the art group."),
            source(
                "E2",
                "The art group I attended last year helped. I remember attending five sessions.",
            ),
        ]
    )


def test_per_period_quantity_disclosure_does_not_invent_one_combined_total():
    result = scoped_quantity_disclosure(
        quantity_ledger(), "How many sessions of the art group did I attend?"
    )
    assert result.computed
    assert "5 sessions" in result.answer and "3 sessions" in result.answer
    assert "last year" in result.answer and "combined total" in result.answer
    assert {r["source"] for r in result.detail["quantity_citations"]} == {"E1", "E2"}
    assert (
        scoped_quantity_disclosure(
            quantity_ledger(), "How many sessions of the art group did I attend last year?"
        )
        is None
    )


def test_quantity_disclosure_rejects_negation_and_other_subjects():
    ledger = quantity_ledger()
    ledger.sources[0] = source("E1", "I did not attend three sessions of the art group.")
    assert (
        scoped_quantity_disclosure(ledger, "How many sessions of the art group did I attend?")
        is None
    )
    ledger.sources[0] = source("E1", "I attended three sessions of the music group.")
    assert (
        scoped_quantity_disclosure(ledger, "How many sessions of the art group did I attend?")
        is None
    )


def test_factual_location_requires_independent_review_without_draft_answer_anchoring():
    engine = object.__new__(GroundedAnswererV11)
    request = AnswerRequest("Where did I redeem a coupon?", "2026-10-01", "alice")
    assert engine.requires_review(request)
    assert not engine.recover_sources


def test_named_duration_accepts_listening_and_dated_memory_but_rejects_unquoted_title():
    ledger = book_ledger()
    ledger.sources[0] = replace(
        ledger.sources[0],
        kind="memory",
        text="The user started reading 'Book One' on September 1, 2026.",
    )
    ledger.sources[2] = replace(ledger.sources[2], text="I started listening to 'Book Two' today.")
    ledger.sources[3] = replace(ledger.sources[3], text="I finished listening to 'Book Two' today.")
    operands = [{"source": s.id, "key": s.id} for s in ledger.sources]
    _, _, result, _ = parse(
        ledger,
        "duration",
        operands,
        "How many weeks in total for 'Book One' and 'Book Two'?",
        result_unit="weeks",
    )
    assert result.computed and result.answer == "2 weeks"
    ledger.sources[3] = replace(ledger.sources[3], text="I finished listening to Book Two today.")
    _, _, result, _ = parse(
        ledger,
        "duration",
        operands,
        "How many weeks in total for 'Book One' and 'Book Two'?",
        result_unit="weeks",
    )
    assert not result.computed


def test_quantity_period_and_negation_are_local_to_matching_action():
    ledger = quantity_ledger()
    ledger.sources[0] = replace(
        ledger.sources[0], text=ledger.sources[0].text + " I am not sure what to do next."
    )
    assert scoped_quantity_disclosure(
        ledger, "How many sessions of the art group did I attend?"
    ).computed
    ledger.sources[1] = replace(
        ledger.sources[1],
        text="I attended five sessions of the art group. I moved house last year.",
    )
    assert (
        scoped_quantity_disclosure(ledger, "How many sessions of the art group did I attend?")
        is None
    )


def test_ambiguous_numbered_successor_and_wrong_role_are_not_computed():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "7. Open the cover", role="assistant", turn=10),
            source("E2", "8. Turn the valve", role="assistant", turn=12),
            source("E3", "8. Start the engine", role="assistant", turn=14),
        ]
    )
    _, _, result, _ = parse(ledger, "lookup", [], "What was your step after 7. Open the cover?")
    assert not result.computed
    ledger.sources[2] = replace(ledger.sources[2], role="user")
    ledger.sources[1] = replace(ledger.sources[1], session_id="other")
    _, _, result, _ = parse(ledger, "lookup", [], "What was your step after 7. Open the cover?")
    assert not result.computed


def test_selective_repair_accepts_valid_first_result_and_blindly_repairs_invalid_one(monkeypatch):
    from types import SimpleNamespace

    import llm_long_term_memory.runtime.grounded_answering as runtime

    ledger = EvidenceLedger(sources=[source("E1", "I bought three rings.")])
    monkeypatch.setattr(runtime, "archive_excerpts", lambda *a, **k: SimpleNamespace(turns=[]))
    monkeypatch.setattr(runtime, "build_ledger", lambda *a, **k: ledger)

    def response(valid):
        return GroundedVerdictV7(
            status="answer",
            operation="sum",
            scope_complete=True,
            observations=[{"source": "E1", "interpretation": "Purchase", "decision": "include"}],
            operands=[
                {
                    "source": "E1",
                    "value": "three" if valid else "ninety",
                    "unit": "rings",
                    "key": "purchase",
                }
            ],
            answer="DRAFT ANSWER MUST NOT ANCHOR REVIEW",
            reason="DRAFT REASON",
        ).model_dump_json()

    class Client:
        def __init__(self, replies):
            self.replies = iter(replies)
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                text=next(self.replies), input_tokens=10, output_tokens=5, api_latency_ms=1
            )

    class NoRecovery:
        def recover(self, *args):
            raise AssertionError("v11 repairs must use the same pool")

    for replies, expected_calls in [([response(True)], 1), ([response(False), response(True)], 2)]:
        engine = object.__new__(GroundedAnswererV11)
        engine.client = Client(replies)
        engine.model = "same-lite"
        engine.store = engine.turn_index = engine.encoder = None
        engine.fallback = NoRecovery()
        engine.chars_per_token, engine.max_output_tokens = 4.6, 2048
        monkeypatch.setattr(engine, "extend_ledger", lambda *a: {})
        result = engine.answer(
            AnswerRequest("How many rings did I buy?", "2026-10-01", "alice"), [], None
        )
        assert result.text == "3 rings" and len(engine.client.calls) == expected_calls
        if expected_calls == 2:
            prompt = engine.client.calls[1]["prompt"]
            assert "DRAFT ANSWER" not in prompt and "DRAFT REASON" not in prompt
            assert result.prompt_tokens == 20
            assert (
                result.notes["grounded_calls"][0]["evidence"]["sources"]
                == result.notes["grounded_calls"][1]["evidence"]["sources"]
            )
