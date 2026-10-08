from __future__ import annotations

import json
import re
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.runtime.grounded_answering import (
    EvidenceLedger,
    EvidenceSource,
    GroundedAnswerer,
    GroundedAnswererV3,
    GroundedAnswererV6,
    GroundedAnswererV7,
    GroundedAnswererV8,
    GroundedVerdict,
    GroundedVerdictV6,
    GroundedVerdictV7,
    SessionEvidenceLedger,
    build_ledger,
    calculate,
    calculate_v2,
    calculate_v3,
    personal_evidence_question,
    personal_evidence_question_v5,
)
from llm_long_term_memory.store import Memory, NumpyFlatIndex, Session, SQLiteMemoryStore, Turn


def source(sid, text, date="2026-09-20", role="user"):
    return EvidenceSource(sid, "raw", sid, text, role, date)


def record(sid, quote, value="", unit="", key="event", time=""):
    return json.dumps(dict(source=sid, quote=quote, value=value, unit=unit, key=key, time=time))


def observation(sid, quote, decision="include"):
    return json.dumps(
        dict(source=sid, quote=quote, interpretation="Relevant fact", decision=decision)
    )


def review_v6(ledger, operation, records=(), **over):
    args = dict(
        observations=[observation(s.id, s.text) for s in ledger.sources],
        reviewed_sources=[s.id for s in ledger.sources],
        scope_complete=True,
        status="answer",
        operation=operation,
        operand_records=list(records),
        answer="supported",
    )
    return GroundedVerdictV6(**(args | over))


def validate_v6(ledger, response, question="What is the total?"):
    engine = object.__new__(GroundedAnswererV6)
    return engine.validate_verdict(response, ledger, AnswerRequest(question, "2026-10-01", "alice"))


def test_v6_groups_actual_session_turn_order_without_changing_bodies_or_dates():
    sources = [
        EvidenceSource(
            "E1",
            "memory",
            "m",
            "It was the second song.",
            "user",
            "2026-09-20",
            session_id="a",
            turn_index=4,
            state="active",
            valid_from="2026-09-20",
        ),
        EvidenceSource(
            "E2",
            "raw",
            "b1",
            "Different conversation.",
            "user",
            "2026-09-20",
            session_id="b",
            turn_index=1,
        ),
        EvidenceSource(
            "E3",
            "raw",
            "a4",
            "Make another sad song.",
            "user",
            "2026-09-20",
            session_id="a",
            turn_index=4,
        ),
        EvidenceSource(
            "E4", "raw", "a0", "The first song.", "user", "2026-09-20", session_id="a", turn_index=0
        ),
    ]
    ledger = SessionEvidenceLedger(sources=sources)
    text = ledger.render()
    assert text.index("[E4]") < text.index("[E3]") < text.index("[E1]") < text.index("Session S2")
    assert text.count("Session S") == 2
    assert "active[2026-09-20..open]" in text
    assert all(s.text in text for s in sources)
    assert ledger.audit()["sources"][0]["session_id"] == "a"


def test_v6_grouping_never_evicts_a_source_for_metadata():
    sources = [source("E1", "A fact."), source("E2", "Another fact.")]
    original = EvidenceLedger(sources=sources)
    ledger = SessionEvidenceLedger(sources=sources, max_tokens=len(original.render()) / 4.6)
    assert len(ledger.render()) <= ledger.max_tokens * 4.6
    assert all(s.text in ledger.render() for s in sources)
    assert ledger.dropped == []


def test_v6_schema_puts_observations_and_reason_before_answer_and_stays_flat():
    schema = GroundedVerdictV6.model_json_schema()
    assert "$defs" not in schema
    fields = list(schema["properties"])
    assert fields[0] == "observations"
    assert fields.index("reason") < fields.index("status") < fields.index("answer")


@pytest.mark.parametrize(
    "bad",
    [
        [],
        ["invalid"],
        [observation("E2", "I own a ring.")],
        [observation("E1", "I own a necklace.")],
    ],
)
def test_v6_lookup_requires_copied_evidence_from_this_pool(bad):
    ledger = EvidenceLedger(sources=[source("E1", "I own a ring.")])
    result = validate_v6(ledger, review_v6(ledger, "lookup", observations=bad))
    assert result.detail["cause"] != "validated_lookup"


def test_v6_lookup_keeps_missing_and_uncertain_facts_uncertain():
    ledger = EvidenceLedger(sources=[source("E1", "I may have bought a ring.")])
    result = validate_v6(
        ledger,
        review_v6(
            ledger, "lookup", observations=[observation("E1", ledger.sources[0].text, "uncertain")]
        ),
    )
    assert result.detail["cause"] == "scope_incomplete"


@pytest.mark.parametrize(
    "quote,value,unit,expected",
    [
        ("I attended five sessions.", "five", "sessions", "5 sessions"),
        ("I attended five sessions.", "5", "sessions", "5 sessions"),
        ("I read twenty-one books.", "21", "books", "21 books"),
        ("I paid $five.", "5", "$", "5 $"),
        ("I own a twenty-gallon tank with five fish.", "20", "fish", None),
        ("I attended five hundred sessions.", "5", "sessions", None),
        ("It changed by minus five sessions.", "5", "sessions", None),
    ],
)
def test_v6_word_quantities_bind_to_adjacent_original_units(quote, value, unit, expected):
    ledger = EvidenceLedger(sources=[source("E1", quote)])
    result = validate_v6(ledger, review_v6(ledger, "sum", [record("E1", quote, value, unit)]))
    assert result.computed == (expected is not None)
    if expected:
        assert result.answer == expected


def test_v6_question_date_can_be_reviewed_for_since_but_never_replace_between_endpoint():
    ledger = EvidenceLedger(sources=[source("E1", "I received my racket today.", "2026-09-24")])
    response = review_v6(
        ledger,
        "duration",
        [
            record("E1", ledger.sources[0].text, time="today"),
            record("QUESTION_DATE", "", key="question-date"),
        ],
        reviewed_sources=["E1", "QUESTION_DATE"],
        result_unit="weeks",
    )
    assert (
        validate_v6(ledger, response, "How many weeks since I received my racket?").answer
        == "1 weeks"
    )
    result = validate_v6(ledger, response, "How many weeks between buying and receiving my racket?")
    assert result.detail["cause"] == "historical_endpoint_required"
    assert not result.computed


def test_v6_excluded_items_cannot_be_used_as_operands():
    ledger = EvidenceLedger(sources=[source("E1", "I plan to buy it for $60.")])
    response = review_v6(
        ledger,
        "sum",
        [record("E1", ledger.sources[0].text, "60", "$")],
        observations=[observation("E1", ledger.sources[0].text, "exclude")],
    )
    assert validate_v6(ledger, response).detail["cause"] == "operand_not_included_in_review"


def test_v6_lookup_validation_repair_uses_same_pool_and_same_model_once(store, tmp_path):
    def response(kwargs):
        match = re.search(r"^\[(E\d+)\].*?\n(.+)", kwargs["prompt"], re.M)
        return GroundedVerdictV6(
            observations=[observation(match[1], match[2])],
            reviewed_sources=[match[1]],
            scope_complete=True,
            status="answer",
            answer="ring",
        ).model_dump_json()

    client = Client(
        [GroundedVerdictV6(status="answer", answer="unsupported").model_dump_json(), response]
    )
    engine = GroundedAnswererV6(
        client,
        model="same-lite",
        encoder=Encoder(),
        store=store,
        turn_index=index(tmp_path, ["a-0", "b-0", "x-0"]),
    )
    answer = engine.answer(
        AnswerRequest("What did I buy?", "2026-10-01", "alice"), [], Encoder().encode_one("q")
    )
    assert answer.text == "ring"
    assert answer.notes["fallback_level"] == "same_pool_repair"
    assert len(client.calls) == 2
    assert all(c["model"] == "same-lite" and "9999" not in c["prompt"] for c in client.calls)
    assert answer.prompt_tokens == 200


def parse_v7(ledger, operation, operands=(), **extra):
    engine = object.__new__(GroundedAnswererV7)
    response = GroundedVerdictV7(
        observations=[
            {"source": s.id, "interpretation": "Relevant", "decision": "include"}
            for s in ledger.sources
        ],
        reviewed_sources=[s.id for s in ledger.sources],
        scope_complete=True,
        status="answer",
        operation=operation,
        operands=list(operands),
        **extra,
    )
    request = AnswerRequest("What is the total?", "2026-10-01", "alice")
    parsed = engine.parse_verdict(response.model_dump_json(), ledger, request)
    return engine.validate_verdict(parsed, ledger, request), parsed


def test_v7_sdk_schema_uses_records_with_no_inner_json_or_union():
    from google.genai import _transformers

    schema = _transformers.t_schema(None, GroundedVerdictV7)
    fields = schema.properties
    assert fields["observations"].items.properties["source"].type == "STRING"
    assert fields["operands"].items.properties["value"].type == "STRING"
    assert fields["zero"].type == "ARRAY"
    assert "time" not in fields["operands"].items.properties
    assert schema.property_ordering.index("reason") < schema.property_ordering.index("answer")


def test_v7_binds_original_body_and_preserves_apostrophes_without_model_json_escaping():
    ledger = EvidenceLedger(
        sources=[source("E1", "It's $60 by taxi."), source("E2", "It's $10 by train.")]
    )
    result, parsed = parse_v7(
        ledger,
        "difference",
        [
            {"source": "E1", "value": "60", "unit": "$", "key": "taxi"},
            {"source": "E2", "value": "10", "unit": "$", "key": "train"},
        ],
        calculation_mode="first_minus_second",
    )
    assert result.answer == "50 $"
    assert json.loads(parsed.operand_records[0])["quote"] == "It's $60 by taxi."
    assert json.loads(parsed.observations[0])["quote"] == ledger.sources[0].text


def test_v7_date_resolution_uses_each_source_original_words_not_model_guessed_dates():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "I ordered it today.", "2026-09-17"),
            source("E2", "I received it today.", "2026-09-24"),
        ]
    )
    result, parsed = parse_v7(
        ledger,
        "duration",
        [
            {"source": "E1", "key": "racket"},
            {"source": "E2", "key": "racket"},
        ],
        result_unit="weeks",
    )
    assert result.answer == "1 weeks"
    assert all(json.loads(r)["time"] == "today" for r in parsed.operand_records)
    assert [c["resolved_date"] for c in result.detail["citations"]] == ["2026-09-17", "2026-09-24"]


def test_v7_ambiguous_dates_need_an_actual_narrow_clause_and_never_guess():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "I ordered it yesterday, and it arrived today.", "2026-09-24"),
            source("E2", "I returned it today.", "2026-09-25"),
        ]
    )
    result, _ = parse_v7(
        ledger,
        "duration",
        [
            {"source": "E1", "key": "racket"},
            {"source": "E2", "key": "racket"},
        ],
        result_unit="days",
    )
    assert result.detail["cause"] == "event_date_unresolved"
    result, _ = parse_v7(
        ledger,
        "duration",
        [
            {"source": "E1", "quote": "it arrived today.", "key": "racket"},
            {"source": "E2", "key": "racket"},
        ],
        result_unit="days",
    )
    assert result.answer == "1 days"


def test_v7_source_binding_cannot_use_a_foreign_or_hallucinated_quote():
    ledger = EvidenceLedger(sources=[source("E1", "I own five coins.")])
    for record in [
        {"source": "E999", "value": "5", "unit": "coins", "key": "coins"},
        {
            "source": "E1",
            "quote": "I own ten coins.",
            "value": "10",
            "unit": "coins",
            "key": "coins",
        },
    ]:
        result, _ = parse_v7(ledger, "sum", [record])
        assert not result.computed


def test_v7_sums_several_source_anchored_durations_with_named_activity_keys():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "I started A today.", "2026-01-01"),
            source("E2", "I finished A today.", "2026-01-15"),
            source("E3", "I started B today.", "2026-02-01"),
            source("E4", "I finished B today.", "2026-03-01"),
            source("E5", "I started C today.", "2026-03-06"),
            source("E6", "I finished C today.", "2026-03-20"),
        ]
    )
    result, _ = parse_v7(
        ledger,
        "duration",
        [
            {"source": s.id, "key": key}
            for s, key in zip(ledger.sources, ["a", "a", "b", "b", "c", "c"], strict=True)
        ],
        result_unit="weeks",
    )
    assert result.answer == "8 weeks"


def test_v7_explicit_zero_is_a_typed_single_source_record():
    ledger = EvidenceLedger(sources=[source("E1", "I own no coins.")])
    result, _ = parse_v7(ledger, "count", zero=[{"source": "E1", "value": "0", "key": "coins"}])
    assert result.computed and result.answer == "0"


def test_v7_named_member_count_accepts_a_redundant_label_but_quantities_require_sum():
    ledger = EvidenceLedger(sources=[source("E1", "I attended my cousin's wedding.")])
    result, _ = parse_v7(
        ledger,
        "count",
        [
            {"source": "E1", "value": "cousin's wedding", "unit": "wedding", "key": "cousin"},
        ],
    )
    assert result.computed and result.answer.startswith("1:")
    ledger = EvidenceLedger(sources=[source("E1", "I own five coins.")])
    result, _ = parse_v7(
        ledger,
        "count",
        [
            {"source": "E1", "value": "five", "unit": "coins", "key": "coins"},
        ],
    )
    assert not result.computed


def test_v7_adjacent_unit_head_noun_binds_qualifiers_without_borrowing_another_measure():
    for quote, value, unit, expected in [
        ("I planted 5 tomato plants.", "5", "plants", "5 plants"),
        ("I planted five tomato plants.", "five", "plants", "5 plants"),
        ("My tank holds 20 gallon tank fish.", "20", "fish", None),
        ("I bought 20 gallons and five fish.", "20", "fish", None),
        ("It changed by minus five tomato plants.", "5", "plants", None),
    ]:
        ledger = EvidenceLedger(sources=[source("E1", quote)])
        result, _ = parse_v7(
            ledger,
            "sum",
            [
                {"source": "E1", "value": value, "unit": unit, "key": "quantity"},
            ],
        )
        assert result.computed == (expected is not None)
        if expected:
            assert result.answer == expected


def parse_v8(ledger, operation, operands=(), question="What is the fact?", **extra):
    engine = object.__new__(GroundedAnswererV8)
    response = GroundedVerdictV7(
        observations=[
            {"source": s.id, "interpretation": "Relevant", "decision": "include"}
            for s in ledger.sources
        ],
        reviewed_sources=[s.id for s in ledger.sources],
        scope_complete=True,
        status="answer",
        operation=operation,
        operands=list(operands),
        **extra,
    )
    request = AnswerRequest(question, "2026-10-01", "alice")
    parsed = engine.parse_verdict(response.model_dump_json(), ledger, request)
    return engine.validate_verdict(parsed, ledger, request), parsed


def test_v8_rebinds_quote_only_to_raw_with_exact_memory_provenance():
    ledger = EvidenceLedger(
        sources=[
            EvidenceSource(
                "E1",
                "memory",
                "m",
                "Started book A on September 1, 2026.",
                "user",
                "2026-09-01",
                session_id="a",
                turn_index=0,
            ),
            EvidenceSource(
                "E2",
                "raw",
                "t",
                "I started book A today.",
                "user",
                "2026-09-01",
                session_id="a",
                turn_index=0,
            ),
            EvidenceSource(
                "E3",
                "raw",
                "t2",
                "I finished book A today.",
                "user",
                "2026-09-08",
                session_id="b",
                turn_index=0,
            ),
        ]
    )
    result, parsed = parse_v8(
        ledger,
        "duration",
        [
            {"source": "E1", "quote": "I started book A today.", "key": "a"},
            {"source": "E3", "key": "a"},
        ],
        result_unit="weeks",
    )
    assert result.answer == "1 weeks"
    assert json.loads(parsed.operand_records[0])["source"] == "E2"
    ledger.sources[1] = ledger.sources[1].__class__(
        "E2",
        "raw",
        "t",
        "I started book A today.",
        "user",
        "2026-09-01",
        session_id="unrelated",
        turn_index=0,
    )
    result, parsed = parse_v8(
        ledger,
        "duration",
        [
            {"source": "E1", "quote": "I started book A today.", "key": "a"},
            {"source": "E3", "key": "a"},
        ],
        result_unit="weeks",
    )
    assert not result.computed
    assert json.loads(parsed.operand_records[0])["source"] == "E1"


def test_v8_recovers_redundant_calendar_quote_only_from_independently_dated_source():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "I ordered it today.", "2026-09-17"),
            source("E2", "I received it today.", "2026-09-24"),
        ]
    )
    result, parsed = parse_v8(
        ledger,
        "duration",
        [
            {"source": "E1", "quote": "September 17, 2026", "key": "racket"},
            {"source": "E2", "key": "racket"},
        ],
        result_unit="weeks",
    )
    assert result.answer == "1 weeks"
    assert json.loads(parsed.operand_records[0])["time"] == "today"
    ledger.sources[0] = source("E1", "I ordered it.", "2026-09-17")
    result, _ = parse_v8(
        ledger,
        "duration",
        [
            {"source": "E1", "quote": "September 17, 2026", "key": "racket"},
            {"source": "E2", "key": "racket"},
        ],
        result_unit="weeks",
    )
    assert not result.computed


def test_v8_calendar_recovery_rejects_invented_event_prose_and_relative_dates():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "I ordered it on September 17, 2026.", "2026-09-18"),
            source("E2", "I received it today.", "2026-09-24"),
        ]
    )
    for quote in ("I ordered something else on September 17, 2026.", "yesterday"):
        result, _ = parse_v8(
            ledger,
            "duration",
            [
                {"source": "E1", "quote": quote, "key": "racket"},
                {"source": "E2", "key": "racket"},
            ],
            result_unit="weeks",
        )
        assert not result.computed


def test_v8_fact_values_must_come_from_selected_source_and_match_answer():
    ledger = EvidenceLedger(
        sources=[source("E1", "First chorus: G G A."), source("E2", "Second chorus: C D E.")]
    )
    for value, answer, expected in [
        ("G G A", "G G A", "fact_value_not_quoted"),
        ("C D E", "G G A", "fact_value_not_in_answer"),
        ("C D E", "The chorus is C D E.", "validated_lookup"),
    ]:
        result, _ = parse_v8(
            ledger, "lookup", [{"source": "E2", "value": value, "key": "chorus"}], answer=answer
        )
        assert result.detail["cause"] == expected


def test_v8_advice_can_synthesize_but_fact_lookup_needs_selected_records():
    ledger = EvidenceLedger(sources=[source("E1", "I like jazz.")])
    result, _ = parse_v8(
        ledger, "lookup", question="Any recommendations for tonight?", answer="Try jazz."
    )
    assert result.detail["cause"] == "validated_lookup"
    result, _ = parse_v8(ledger, "lookup", answer="Jazz.")
    assert result.detail["cause"] == "fact_records_required"


def test_v8_always_verifies_valid_arithmetic_with_same_model_without_new_sources(store, tmp_path):
    def reply(kwargs):
        match = re.search(r"^\[(E\d+)\].*?\n(.+)", kwargs["prompt"], re.M)
        return GroundedVerdictV7(
            observations=[
                {"source": match[1], "interpretation": "Actual purchase", "decision": "include"}
            ],
            reviewed_sources=[match[1]],
            scope_complete=True,
            status="answer",
            operation="sum",
            operands=[{"source": match[1], "value": "800", "unit": "$", "key": "ring"}],
            answer="$800",
        ).model_dump_json()

    client = Client([reply, reply])
    engine = GroundedAnswererV8(
        client,
        model="same-lite",
        encoder=Encoder(),
        store=store,
        turn_index=index(tmp_path, ["a-0", "b-0", "x-0"]),
    )
    answer = engine.answer(
        AnswerRequest("Total spent?", "2026-10-01", "alice"), [], Encoder().encode_one("q")
    )
    assert answer.text == "800 $"
    assert len(client.calls) == 2
    assert all(c["model"] == "same-lite" and "9999" not in c["prompt"] for c in client.calls)
    assert "VERIFICATION PASS" in client.calls[1]["prompt"]
    assert answer.prompt_tokens == 200
    assert len(answer.notes["grounded_calls"]) == 2


def verdict(ledger, operation, records, **over):
    args = dict(
        reviewed_sources=[s.id for s in ledger.sources],
        scope_complete=True,
        operation=operation,
        operand_records=records,
        status="answer",
        answer="999",
    )
    return GroundedVerdict(**(args | over))


def test_sum_counts_separate_equal_price_purchases_and_folds_repeated_mentions():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "I bought ring A for $800."),
            source("E2", "The ring A cost me $800."),
            source("E3", "I bought ring B for $800."),
            source("E4", "I bought a necklace for $900."),
        ]
    )
    records = [
        record(s.id, s.text, value, "$", key)
        for s, value, key in zip(
            ledger.sources,
            ["800", "800", "800", "900"],
            ["ring-a", "ring-a", "ring-b", "necklace"],
            strict=True,
        )
    ]
    result = calculate(verdict(ledger, "sum", records), ledger, "2026-10-01")
    assert result.answer == "2500 $"
    assert result.detail["duplicates_removed"] == 1
    assert len(result.detail["citations"]) == 4


def test_citation_alone_does_not_validate_a_wrong_number_or_unit():
    ledger = EvidenceLedger(sources=[source("E1", "My train fare is $6.")])
    for value, unit, cause in [
        ("60", "$", "number_not_in_quote"),
        ("6", "EUR", "unit_not_in_quote"),
        ("6", "", "currency_unit_missing"),
    ]:
        result = calculate(
            verdict(ledger, "sum", [record("E1", ledger.sources[0].text, value, unit)]),
            ledger,
            "2026-10-01",
        )
        assert not result.computed
        assert result.detail["cause"] == cause


def test_difference_uses_user_figures_and_explicit_direction():
    ledger = EvidenceLedger(sources=[source("E1", "Train costs $6 and taxi costs $22.")])
    records = [
        record("E1", ledger.sources[0].text, n, "$", key)
        for n, key in [("6", "train"), ("22", "taxi")]
    ]
    result = calculate(
        verdict(ledger, "difference", records, calculation_mode="second_minus_first"),
        ledger,
        "2026-10-01",
    )
    assert result.answer == "16 $"
    assert not calculate(verdict(ledger, "difference", records), ledger, "2026-10-01").computed


def test_conflicting_prices_for_the_same_event_need_resolution():
    ledger = EvidenceLedger(sources=[source("E1", "It cost $6, later corrected to $8.")])
    result = calculate(
        verdict(ledger, "sum", [record("E1", ledger.sources[0].text, n, "$") for n in ["6", "8"]]),
        ledger,
        "2026-10-01",
    )
    assert result.detail["cause"] == "duplicate_identity_conflict"


def test_count_reviews_all_sources_and_does_not_count_assistant_suggestions():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "I planted basil and mint."),
            source("E2", "Try rosemary too.", role="assistant"),
        ]
    )
    records = [
        record("E1", ledger.sources[0].text, member, key=member) for member in ["basil", "mint"]
    ]
    result = calculate(verdict(ledger, "count", records), ledger, "2026-10-01")
    assert result.answer == "2: basil, mint"
    partial = calculate(
        verdict(ledger, "count", records, reviewed_sources=["E1"]), ledger, "2026-10-01"
    )
    assert partial.detail["cause"] == "source_review_incomplete"
    suggested = calculate(
        verdict(ledger, "count", [record("E2", "Try rosemary too.", "rosemary")]),
        ledger,
        "2026-10-01",
    )
    assert suggested.detail["cause"] == "user_fact_requires_user_source"


def test_count_deduplicates_explicit_alias_keys_but_preserves_quotes():
    ledger = EvidenceLedger(sources=[source("E1", "I own basil, also called sweet basil.")])
    records = [
        record("E1", ledger.sources[0].text, v, key="basil") for v in ["basil", "sweet basil"]
    ]
    result = calculate(verdict(ledger, "count", records), ledger, "2026-10-01")
    assert result.answer == "1: basil"
    assert len(result.detail["citations"]) == 2


@pytest.mark.parametrize(
    "over,cause",
    [
        ({"reviewed_sources": ["E1", "E1"]}, "source_review_incomplete"),
        ({"scope_complete": False}, "scope_incomplete"),
        ({"uncertain_sources": ["E1"]}, "scope_incomplete"),
        ({"operand_records": ['{"source":"E1"}']}, "operand_record_invalid"),
        ({"operand_records": [record("E9", "basil", "basil")]}, "source_quote_invalid"),
        ({"operand_records": [record("E1", "basil", "rosemary")]}, "member_not_in_quote"),
    ],
)
def test_incomplete_or_ungrounded_selection_is_not_calculated(over, cause):
    ledger = EvidenceLedger(sources=[source("E1", "I own basil.")])
    result = calculate(
        verdict(ledger, "count", [record("E1", "basil", "basil")], **over), ledger, "2026-10-01"
    )
    assert not result.computed
    assert result.detail["cause"] == cause


def test_an_empty_list_is_not_zero_and_an_arbitrary_quote_is_not_zero():
    ledger = EvidenceLedger(sources=[source("E1", "I have no rings. I planted basil.")])
    assert not calculate(verdict(ledger, "count", []), ledger, "2026-10-01").computed
    bogus = verdict(ledger, "count", [], zero_record=record("E1", "I planted basil.", "0"))
    assert not calculate(bogus, ledger, "2026-10-01").computed
    zero = verdict(ledger, "count", [], zero_record=record("E1", "I have no rings.", "0"))
    assert calculate(zero, ledger, "2026-10-01").answer == "0"


def duration(quote, expression, asked="2026-10-01", **over):
    ledger = EvidenceLedger(sources=[source("E1", quote)])
    records = [
        record("E1", quote, time=expression),
        record("QUESTION_DATE", "", key="question-date"),
    ]
    return calculate(
        verdict(ledger, "duration", records, calculation_mode="days", **over), ledger, asked
    )


def test_today_is_relative_to_its_conversation_not_the_question_or_wall_clock():
    result = duration("I started today.", "today")
    assert result.answer == "11 days"
    assert result.detail["start"] == "2026-09-20"
    assert result.detail["citations"][0]["time"] == "today"


def test_explicit_event_date_wins_over_the_conversation_date():
    result = duration("I started on September 6, 2026.", "September 6, 2026")
    assert result.answer == "25 days"


def test_relative_weeks_remain_approximate_and_months_are_not_exact_days():
    result = duration("I started two weeks ago.", "two weeks ago")
    assert result.answer == "About 25 days"
    assert result.detail["approximate"] is True
    assert not duration("I started last month.", "last month").computed
    assert duration("I started about three days ago.", "three days ago").answer == "About 14 days"


@pytest.mark.parametrize(
    "quote,expression",
    [
        ("I started classes.", "2026-09-20"),
        ("I listened to Yesterday.", "Yesterday"),
        ("I started before September 6, 2026.", "September 6, 2026"),
        ("I started yesterday and finished today.", "yesterday"),
    ],
)
def test_a_header_date_title_or_ambiguous_date_cannot_be_used_as_an_event(quote, expression):
    assert not duration(quote, expression).computed


def test_duration_does_not_silently_swap_reversed_endpoints():
    result = duration("I started on October 10, 2026.", "October 10, 2026")
    assert result.detail["cause"] == "duration_end_before_start"


@pytest.fixture
def store(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()
    for sid, user, text in [
        ("a", "alice", "I bought a ring for $800."),
        ("b", "alice", "I bought a necklace for $900."),
        ("x", "bob", "I spent $9999."),
    ]:
        date = datetime(2026, 9, 20)
        store.add_session(
            Session(sid, user, date, turns=[Turn(sid + "-0", sid, 0, "user", text, date)])
        )
    yield store
    store.close()


def memory(**over):
    return Memory(
        **(
            dict(
                id="m",
                user_id="alice",
                type="semantic",
                content="I bought a ring.",
                token_count=5,
                source_session_id="a",
                source_turn_index=0,
                observed_at=datetime(2026, 9, 20),
                valid_from=datetime(2026, 9, 20),
            )
            | over
        )
    )


def test_ledger_separates_validity_mention_and_event_dates_and_rejects_other_tenants(store):
    ledger = build_ledger(
        store, "alice", [memory(event_time=datetime(2026, 9, 20))], store.turns_for_session("a")
    )
    assert ledger.sources[0].event_date is None, "legacy mention dates are not event evidence"
    assert ledger.sources[0].conversation_date == "2026-09-20"
    assert ledger.sources[0].valid_from == "2026-09-20"
    with pytest.raises(ValueError, match="tenant"):
        build_ledger(store, "alice", [memory(user_id="bob")], [])
    with pytest.raises(ValueError, match="tenant"):
        build_ledger(store, "alice", [memory(source_session_id="x")], [])
    with pytest.raises(ValueError, match="tenant"):
        build_ledger(store, "alice", [], store.turns_for_session("x"))


def test_whole_source_budget_accounts_for_metadata_and_keeps_selected_ids_auditable(store):
    ledger = build_ledger(
        store,
        "alice",
        [],
        store.turns_for_session("a") + store.turns_for_session("b"),
        max_tokens=35,
    )
    assert len(ledger.render()) <= 35 * 4.6
    assert ledger.dropped
    for s in ledger.sources:
        assert s.text.endswith(".")
    assert ledger.audit()["sources"][0]["source_id"] == "a-0"
    assert len(ledger.audit()["context_sha256"]) == 64


class Encoder:
    def encode_one(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        reply = next(self.replies)
        text = reply(kwargs) if callable(reply) else reply
        return SimpleNamespace(text=text, input_tokens=100, output_tokens=50, api_latency_ms=1.0)


def index(tmp_path, ids):
    result = NumpyFlatIndex(tmp_path / "turns", dim=2)
    result.add(ids, np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(ids), 1)))
    return result


def test_raw_recovery_keeps_first_evidence_and_uses_the_calculation_schema_again(
    store, tmp_path, monkeypatch
):
    from llm_long_term_memory.retrieve import excerpts
    from llm_long_term_memory.retrieve.excerpts import ArchiveExcerpts
    from llm_long_term_memory.retrieve.fallback import RawEvidence

    monkeypatch.setattr(excerpts, "archive_excerpts", lambda *a, **k: ArchiveExcerpts())
    # The function is imported into the shared module, patch that binding too.
    monkeypatch.setattr(
        "llm_long_term_memory.runtime.grounded_answering.archive_excerpts",
        lambda *a, **k: ArchiveExcerpts(),
    )
    calls = []

    class Fallback:
        def recover(self, user, query, memories):
            calls.append(user)
            return RawEvidence(store.turns_for_session("b"), "archive_wide")

    mem = memory(content="I bought a ring for $800.")
    first = GroundedVerdict(
        status="need_source", reason="missing necklace", operation="sum"
    ).model_dump_json()

    def second(kwargs):
        assert mem.content in kwargs["prompt"]
        assert "necklace for $900" in kwargs["prompt"]
        return GroundedVerdict(
            reviewed_sources=["E1", "E2"],
            scope_complete=True,
            operation="sum",
            operand_records=[
                record("E1", mem.content, "800", "$", "ring"),
                record("E2", "I bought a necklace for $900.", "900", "$", "necklace"),
            ],
            status="answer",
            answer="wrong model arithmetic",
        ).model_dump_json()

    client = Client([first, second])
    engine = GroundedAnswerer(
        client,
        model="same-lite",
        encoder=Encoder(),
        store=store,
        turn_index=index(tmp_path, ["a-0", "b-0", "x-0"]),
        fallback=Fallback(),
    )
    answer = engine.answer(
        AnswerRequest("Total purchases?", "2026-10-01", "alice"), [mem], Encoder().encode_one("q")
    )
    assert answer.text == "1700 $"
    assert [c["model"] for c in client.calls] == ["same-lite", "same-lite"]
    assert all(c["schema"] is GroundedVerdict for c in client.calls)
    assert (answer.prompt_tokens, answer.output_tokens, answer.latency_ms) == (200, 100, 2.0)
    assert calls == ["alice"]
    assert len(answer.notes["grounded_calls"]) == 2


def test_invalid_response_and_unverified_arithmetic_do_not_escape_as_prose(store, tmp_path):
    for reply in [
        "not json",
        GroundedVerdict(operation="sum", status="answer", answer="$9999").model_dump_json(),
    ]:
        engine = GroundedAnswerer(
            Client([reply]),
            model="same-lite",
            encoder=Encoder(),
            store=store,
            turn_index=index(tmp_path, ["a-0", "b-0", "x-0"]),
        )
        answer = engine.answer(
            AnswerRequest("Total purchases?", "2026-10-01", "alice"), [], Encoder().encode_one("q")
        )
        assert answer.text == "I do not know."
        assert "9999" not in engine.client.calls[0]["prompt"]


@pytest.mark.parametrize(
    "policy",
    [
        "grounded_v1",
        "grounded_v2",
        "grounded_v3",
        "grounded_v4",
        "grounded_v5",
        "grounded_v6",
        "grounded_v7",
        "grounded_v8",
        "grounded_v9",
        "grounded_v10",
        "grounded_v11",
        "grounded_v12",
        "grounded_v13",
        "grounded_v14",
        "grounded_v15",
        "grounded_v16",
        "grounded_v17",
        "grounded_v18",
    ],
)
def test_product_and_evaluation_use_identical_candidate_prompts_and_same_model(
    store, tmp_path, policy
):
    from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
    from llm_long_term_memory.runtime import AnswerEngine

    mem = memory()
    store.add_memories([mem])
    memories = NumpyFlatIndex(tmp_path / "memories", dim=2)
    memories.add([mem.id], np.array([[1.0, 0.0]], dtype=np.float32))
    turns = index(tmp_path, ["a-0", "b-0", "x-0"])

    def reply(kwargs):
        ids = re.findall(r"^\[(E\d+)\]", kwargs["prompt"], re.M)
        extra = {}
        from llm_long_term_memory.runtime.evidence_pages import PageSelection

        if kwargs["schema"] is PageSelection:
            return PageSelection(
                reviewed_sources=ids, selected_sources=ids, scope_complete=True
            ).model_dump_json()
        if policy in {
            "grounded_v6",
            "grounded_v7",
            "grounded_v8",
            "grounded_v9",
            "grounded_v10",
            "grounded_v11",
            "grounded_v12",
            "grounded_v13",
            "grounded_v14",
            "grounded_v15",
            "grounded_v16",
            "grounded_v17",
            "grounded_v18",
        }:
            body = re.search(r"^\[(E\d+)\].*?\n(.+)", kwargs["prompt"], re.M)
            extra["observations"] = (
                [{"source": body[1], "interpretation": "Relevant", "decision": "include"}]
                if policy
                in {
                    "grounded_v7",
                    "grounded_v8",
                    "grounded_v9",
                    "grounded_v10",
                    "grounded_v11",
                    "grounded_v12",
                    "grounded_v13",
                    "grounded_v14",
                    "grounded_v15",
                    "grounded_v16",
                    "grounded_v17",
                    "grounded_v18",
                }
                else [observation(body[1], body[2])]
            )
        return kwargs["schema"](
            reviewed_sources=ids, scope_complete=True, status="answer", answer="ring", **extra
        ).model_dump_json()

    clients = [Client([reply] * 10), Client([reply] * 10)]
    runners = [
        MemoryRunner(
            clients[0],
            model="same-lite",
            encoder=Encoder(),
            store=store,
            index=memories,
            temporal=True,
            top_k=20,
            answer_policy=policy,
            raw_primary_turn_index=turns,
            max_output_tokens=2048,
        ),
        AnswerEngine(
            clients[1],
            model="same-lite",
            encoder=Encoder(),
            store=store,
            index=memories,
            temporal=True,
            top_k=20,
            answer_policy=policy,
            raw_primary_turn_index=turns,
        ),
    ]
    request = AnswerRequest("Any suggestions based on what I bought?", "2026-10-01", "alice")
    answers = [runner.answer_request(request) for runner in runners]
    assert answers[0].text == answers[1].text == "ring"
    assert clients[0].calls == clients[1].calls
    assert "9999" not in clients[0].calls[0]["prompt"]


def test_provider_schema_keeps_records_before_answer_and_has_no_nested_definitions():
    schema = GroundedVerdict.model_json_schema()
    assert "$defs" not in schema
    assert schema["properties"]["operand_records"]["items"] == {"type": "string"}
    fields = list(schema["properties"])
    assert fields.index("operand_records") < fields.index("answer")


@pytest.mark.parametrize("version", list(range(1, 19)))
def test_candidate_builds_through_cli_without_changing_models_or_legacy_policies(
    tmp_path, monkeypatch, version
):
    from llm_long_term_memory import cli
    from llm_long_term_memory.config import Settings

    class LocalEncoder:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("llm_long_term_memory.embed.Encoder", LocalEncoder)
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()
    date = datetime(2026, 9, 20)
    store.add_session(
        Session("a", "alice", date, turns=[Turn("a-0", "a", 0, "user", "ring", date)])
    )
    store.add_memories([memory()])
    store.close()
    mem_index = NumpyFlatIndex(tmp_path / "s-index", dim=384)
    mem_index.add(["m"], np.ones((1, 384), dtype=np.float32))
    mem_index.save()
    settings = Settings(LLTM_STORE_DIR=str(tmp_path))
    variant = "two_stage_raw_primary_grounded" + (f"_v{version}" if version > 1 else "")
    with pytest.raises(Exception, match="--fact-keys"):
        cli._build(
            variant,
            "configs/fallback.yaml",
            "s",
            client_override=object(),
            settings_override=settings,
            read_only_store=True,
        )
    turns = NumpyFlatIndex(tmp_path / "s-turn-key-index", dim=384)
    turns.add(["a-0"], np.ones((1, 384), dtype=np.float32))
    turns.save()
    _, _, candidate, _, _ = cli._build(
        variant,
        "configs/fallback.yaml",
        "s",
        client_override=object(),
        settings_override=settings,
        read_only_store=True,
    )
    _, _, baseline, _, _ = cli._build(
        "two_stage_raw_primary",
        "configs/fallback.yaml",
        "s",
        client_override=object(),
        settings_override=settings,
        read_only_store=True,
    )
    assert candidate.model == baseline.model == "gemini-3.5-flash-lite"
    assert candidate.answer_prompt_version == f"memory-grounded-v{version}"
    assert candidate.verdict_schema is {
        6: GroundedVerdictV6,
        7: GroundedVerdictV7,
        8: GroundedVerdictV7,
        9: GroundedVerdictV7,
        10: GroundedVerdictV7,
        11: GroundedVerdictV7,
        12: GroundedVerdictV7,
        13: GroundedVerdictV7,
        14: GroundedVerdictV7,
        15: GroundedVerdictV7,
        16: GroundedVerdictV7,
        17: GroundedVerdictV7,
        18: GroundedVerdictV7,
    }.get(version, GroundedVerdict)
    assert candidate.max_output_tokens == 2048
    assert candidate.raw_primary_fact_keys and candidate.raw_primary_memory_fusion
    assert baseline.answer_prompt_version == "memory-aware-v2"
    assert baseline.grounded is None
    candidate.store.close()
    baseline.store.close()


def test_average_time_conversion_mixed_currencies_and_zero_baseline():
    ledger = EvidenceLedger(sources=[source("E1", "I travelled 1 hour then 30 minutes.")])
    records = [
        record("E1", ledger.sources[0].text, n, unit, key=unit)
        for n, unit in [("1", "hour"), ("30", "minutes")]
    ]
    result = calculate(verdict(ledger, "sum", records, result_unit="minutes"), ledger, "2026-10-01")
    assert result.answer == "90 minutes"
    assert (
        calculate(
            verdict(ledger, "average", records, result_unit="minutes"), ledger, "2026-10-01"
        ).answer
        == "45 minutes"
    )


def test_v2_pair_operations_do_not_require_an_irrelevant_source_census():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "Taxi costs $60."),
            source("E2", "Train costs $10."),
            source("E3", "I own a cat."),
        ]
    )
    records = [
        record("E1", "Taxi costs $60.", "60", "$", "taxi"),
        record("E2", "Train costs $10.", "10", "$", "train"),
    ]
    ev = verdict(
        ledger,
        "difference",
        records,
        reviewed_sources=["E1", "E2"],
        calculation_mode="first_minus_second",
    )
    assert not calculate(ev, ledger, "2026-10-01").computed, "v1 behavior remains replayable"
    assert calculate_v2(ev, ledger, "2026-10-01").answer == "50 $"
    ev.operation = "sum"
    assert calculate_v2(ev, ledger, "2026-10-01").detail["cause"] == "source_review_incomplete"
    ev.operation = "difference"
    ev.scope_complete = False
    assert not calculate_v2(ev, ledger, "2026-10-01").computed


@pytest.mark.parametrize(
    "start,end,unit,expected",
    [
        ("2026-01-31", "2026-02-28", "months", "1 months"),
        ("2026-01-15", "2026-03-14", "months", "1 months and 27 days"),
        ("2024-02-29", "2025-02-28", "years", "1 years"),
    ],
)
def test_calendar_duration_uses_anniversaries_not_thirty_day_months(start, end, unit, expected):
    quote = f"I started on {start}."
    ledger = EvidenceLedger(sources=[source("E1", quote), source("E2", "irrelevant")])
    ev = verdict(
        ledger,
        "duration",
        [record("E1", quote, time=start), record("QUESTION_DATE", "", key="question-date")],
        reviewed_sources=["E1"],
        result_unit=unit,
        calculation_mode="days",
    )
    assert calculate_v2(ev, ledger, end).answer == expected


def test_v2_duration_honors_requested_weeks_and_still_checks_the_quoted_expression():
    ledger = EvidenceLedger(
        sources=[
            source("E1", "I started on February 11, 2023."),
            source("E2", "I bought my tools today.", date="2023-03-04"),
            source("E3", "irrelevant"),
        ]
    )
    ev = verdict(
        ledger,
        "duration",
        [
            record("E1", ledger.sources[0].text, time="February 11, 2023"),
            record("E2", ledger.sources[1].text, time="today"),
        ],
        reviewed_sources=["E1", "E2"],
        result_unit="weeks",
        calculation_mode="days",
    )
    assert calculate_v2(ev, ledger, "2023-03-04").answer == "3 weeks"
    ev.operand_records[0] = record("E1", ledger.sources[0].text, time="2023-02-11")
    assert not calculate_v2(ev, ledger, "2023-03-04").computed


def test_calculator_rejects_mixed_currencies_and_zero_percentage_baseline():
    ledger = EvidenceLedger(sources=[source("E1", "I paid $20 and €30.")])
    records = [
        record("E1", ledger.sources[0].text, n, unit, key=unit)
        for n, unit in [("20", "$"), ("30", "€")]
    ]
    assert (
        calculate(verdict(ledger, "sum", records), ledger, "2026-10-01").detail["cause"]
        == "units_incompatible"
    )
    ledger = EvidenceLedger(sources=[source("E1", "It changed from 0 kg to 20 kg.")])
    records = [record("E1", ledger.sources[0].text, n, "kg", key=n) for n in ["0", "20"]]
    assert (
        calculate(verdict(ledger, "percentage_change", records), ledger, "2026-10-01").detail[
            "cause"
        ]
        == "percentage_baseline_zero"
    )


def test_v3_number_cannot_borrow_a_different_quantity_unit():
    ledger = EvidenceLedger(sources=[source("E1", "My 20-gallon tank contains 10 fish.")])
    ev = verdict(ledger, "sum", [record("E1", ledger.sources[0].text, "20", "fish", "fish")])
    assert calculate_v2(ev, ledger, "2026-10-01").answer == "20 fish"
    assert calculate_v3(ev, ledger, "2026-10-01").detail["cause"] == "number_unit_not_bound"
    ev.operand_records = [record("E1", ledger.sources[0].text, "10", "fish", "fish")]
    assert calculate_v3(ev, ledger, "2026-10-01").answer == "10 fish"
    ev.operand_records = [record("E1", ledger.sources[0].text, "20", "", "fish")]
    assert not calculate_v3(ev, ledger, "2026-10-01").computed


def test_v3_total_duration_validates_every_activity_pair_and_sums_days():
    dates = ["2022-01-01", "2022-01-15", "2022-02-01", "2022-03-01", "2022-03-06", "2022-03-20"]
    ledger = EvidenceLedger(
        sources=[source(f"E{i}", "I read today.", date=d) for i, d in enumerate(dates)]
    )
    records = [
        record(s.id, s.text, key=f"book-{i // 2}", time="today")
        for i, s in enumerate(ledger.sources)
    ]
    ev = verdict(ledger, "duration", records, result_unit="weeks", calculation_mode="weeks")
    assert not calculate_v2(ev, ledger, "2022-04-30").computed
    result = calculate_v3(ev, ledger, "2022-04-30")
    assert result.answer == "8 weeks"
    assert [p["days"] for p in result.detail["pairs"]] == [14, 28, 14]
    ev.operand_records = records[:-1]
    assert (
        calculate_v3(ev, ledger, "2022-04-30").detail["cause"] == "duration_group_endpoints_invalid"
    )
    ev.operand_records = [records[1], records[0], *records[2:]]
    assert calculate_v3(ev, ledger, "2022-04-30").detail["cause"] == "duration_end_before_start"
    ev.operand_records = [*records, *records[:2]]
    assert not calculate_v3(ev, ledger, "2022-04-30").computed
    ev.operand_records = records
    ev.result_unit = "months"
    assert calculate_v3(ev, ledger, "2022-04-30").detail["cause"] == "grouped_duration_unit_invalid"


@pytest.mark.parametrize("repair", [True, False])
def test_v3_same_pool_repair_is_bounded_and_preserves_actual_evidence(
    store, tmp_path, monkeypatch, repair
):
    from llm_long_term_memory.retrieve.excerpts import ArchiveExcerpts

    monkeypatch.setattr(
        "llm_long_term_memory.runtime.grounded_answering.archive_excerpts",
        lambda *a, **k: ArchiveExcerpts(),
    )
    mem = memory(content="I paid $800.")
    ledger = EvidenceLedger(sources=[source("E1", mem.content)])
    bad = verdict(
        ledger, "sum", [record("E1", mem.content, "999", "$", "purchase")]
    ).model_dump_json()

    def second(call):
        assert "number_not_in_quote" in call["prompt"]
        assert mem.content in call["prompt"]
        good = verdict(ledger, "sum", [record("E1", mem.content, "800", "$", "purchase")])
        return good.model_dump_json() if repair else bad

    client = Client([bad, second])
    engine = GroundedAnswererV3(
        client,
        model="same-lite",
        encoder=Encoder(),
        store=store,
        turn_index=index(tmp_path, ["a-0"]),
    )
    answer = engine.answer(
        AnswerRequest("How much did I pay?", "2026-10-01", "alice"),
        [mem],
        Encoder().encode_one("q"),
    )
    assert answer.text == ("800 $" if repair else "I do not know.")
    assert len(client.calls) == 2
    assert answer.notes["fallback_level"] == "same_pool_repair"
    assert (
        answer.notes["grounded_calls"][0]["evidence"]
        == answer.notes["grounded_calls"][1]["evidence"]
    )


def test_v3_lexical_recovery_respects_remaining_budget_and_tenant(store, tmp_path):
    engine = GroundedAnswererV3(
        Client([]),
        model="same-lite",
        encoder=Encoder(),
        store=store,
        turn_index=index(tmp_path, ["a-0", "b-0", "x-0"]),
    )
    request = AnswerRequest("necklace spent", "2026-10-01", "alice")
    ledger = build_ledger(store, "alice", [], store.turns_for_session("a"), max_tokens=35)
    extension = engine.extend_ledger(ledger, request, [], Encoder().encode_one("q"))
    assert extension["lexical_recovery_selected"] == ["b-0"]
    assert extension["lexical_recovery_added"] == 0
    assert ledger.dropped == ["raw:b-0"]
    assert "9999" not in ledger.render()
    ledger.max_tokens = 6000
    extension = engine.extend_ledger(ledger, request, [], Encoder().encode_one("q"))
    assert extension["lexical_recovery_added"] == 1
    assert [s.source_id for s in ledger.sources] == ["a-0", "b-0"]


@pytest.mark.parametrize(
    "question,expected",
    [
        ("How many fish are in both of my aquariums?", True),
        ("Where did I complete my degree?", True),
        ("Can you recommend something considering my interests?", True),
        ("Which movie did you recommend to me last time?", False),
        ("What recommendations from you did I follow?", False),
        ("Which URL did you previously share with me?", False),
    ],
)
def test_personal_source_route_retains_assistant_advice_questions(question, expected):
    assert personal_evidence_question(question) is expected
    assert personal_evidence_question_v5(question) is expected


def test_v5_routes_the_asked_subject_instead_of_incidental_introduction_pronouns():
    question = (
        "I was going through our previous conversation - what did the author say about the library?"
    )
    assert personal_evidence_question(question)
    assert not personal_evidence_question_v5(question)
    assert personal_evidence_question_v5("I wanted to check: where did I complete my degree?")


def test_evaluation_returns_incomplete_review_without_assuming_a_final_reader(store, tmp_path):
    from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
    from llm_long_term_memory.runtime.evidence_pages import PageSelection

    memories = NumpyFlatIndex(tmp_path / "memories", 2)
    mem = memory()
    store.add_memories([mem])
    memories.add([mem.id], np.array([[1.0, 0.0]], dtype=np.float32))
    client = Client([PageSelection(scope_complete=False).model_dump_json()])
    runner = MemoryRunner(
        client,
        model="same-lite",
        encoder=Encoder(),
        store=store,
        index=memories,
        temporal=True,
        top_k=20,
        answer_policy="grounded_v18",
        raw_primary_turn_index=index(tmp_path, ["a-0", "b-0", "x-0"]),
    )
    answer = runner.answer_request(
        AnswerRequest("How much total money did I pay?", "2026-10-03", "alice")
    )
    assert answer.notes["answer_status"] == "need_source"
    assert not answer.notes["source_session_recalled"]
    failed = answer.notes["archive_review"]["pages"][0]
    assert failed["selection_error"] == "page_review_incomplete"
    assert failed["provider_response"] == PageSelection(scope_complete=False).model_dump_json()
    assert len(client.calls) == 1
