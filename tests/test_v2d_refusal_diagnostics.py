"""A refusal has to say which failure it was, not merely that there was one.

The gate16 run refused nine of sixteen questions. Four of them recorded
`an item has no valid source label`, which is two different failures wearing one
sentence — the model returned no member text, or it cited a memory it was never
shown — and the rows carried nothing else, so the only way to tell them apart was to
spend the questions again. These tests pin the wider record, and pin that widening it
changed no decision.
"""

from __future__ import annotations

import pytest

from llm_long_term_memory.evaluation.runners.v2d import REFUSAL_CAUSES, V2DVerdict, compute


def verdict(operation, **kwargs):
    kwargs.setdefault("premise_status", "not_applicable")
    return V2DVerdict(status="answer", answer="model prose", operation=operation, **kwargs)


# (name, verdict, context labels, expected cause). Every entry is a refusal.
REFUSALS = [
    (
        "count member with no text",
        verdict("count", operand_values=["   "], operand_labels=["M1"]),
        {"M1"},
        "member_text_empty",
    ),
    (
        "count member citing a memory that was never shown",
        verdict("count", operand_values=["basil"], operand_labels=["M9"]),
        {"M1"},
        "label_not_in_context",
    ),
    (
        "count arrays of different lengths",
        verdict("count", operand_values=["basil", "mint"], operand_labels=["M1"]),
        {"M1"},
        "operand_arrays_misaligned",
    ),
    (
        "count with no operands at all",
        verdict("count"),
        {"M1"},
        "no_members",
    ),
    (
        "duration given one endpoint",
        verdict("duration", operand_values=["2026-01-01"], operand_labels=["M1"]),
        {"M1"},
        "duration_needs_two_operands",
    ),
    (
        "duration whose endpoint is not a date",
        verdict(
            "duration", operand_values=["last spring", "2026-01-01"], operand_labels=["M1", "M1"]
        ),
        {"M1"},
        "date_unparseable",
    ),
    (
        "duration citing a memory that was never shown",
        verdict(
            "duration", operand_values=["2026-01-01", "2026-02-01"], operand_labels=["M1", "M9"]
        ),
        {"M1"},
        "label_not_in_context",
    ),
    (
        "sum whose operand is not a number",
        verdict("sum", operand_values=["a few"], operand_labels=["M1"]),
        {"M1"},
        "numeric_unparseable",
    ),
    (
        "sum citing a memory that was never shown",
        verdict("sum", operand_values=["3"], operand_labels=["M9"]),
        {"M1"},
        "label_not_in_context",
    ),
    (
        "difference given only one operand",
        verdict("difference", operand_values=["3"], operand_labels=["M1"]),
        {"M1"},
        "too_few_operands",
    ),
    (
        "sum mixing units that do not convert",
        verdict(
            "sum",
            operand_values=["3", "4"],
            operand_labels=["M1", "M1"],
            operand_qualifiers=["apples", "hours"],
            result_unit="apples",
        ),
        {"M1"},
        "units_incompatible",
    ),
    (
        "percentage change from a zero baseline",
        verdict("percentage_change", operand_values=["0", "5"], operand_labels=["M1", "M1"]),
        {"M1"},
        "percentage_baseline_zero",
    ),
    (
        "a named missing operand",
        verdict(
            "sum", operand_values=["3"], operand_labels=["M1"], missing_field="the second bill"
        ),
        {"M1"},
        "missing_field_named",
    ),
    (
        "comparison given one endpoint",
        verdict("comparison", operand_values=["2026-01-01"], operand_labels=["M1"]),
        {"M1"},
        "comparison_operands_invalid",
    ),
    (
        "comparison whose endpoint is not a date",
        verdict("comparison", operand_values=["later", "2026-01-01"], operand_labels=["M1", "M1"]),
        {"M1"},
        "comparison_operands_invalid",
    ),
    (
        "an operation code never calculates",
        verdict("current_state"),
        {"M1"},
        "operation_not_computable",
    ),
]


@pytest.mark.parametrize(
    ("case", "given", "labels", "cause"),
    REFUSALS,
    ids=[name for name, _, _, _ in REFUSALS],
)
def test_each_refusal_names_its_own_cause(case, given, labels, cause):
    result = compute(given, labels)

    assert not result.computed, case
    assert result.detail["cause"] == cause
    assert result.detail["cause"] in REFUSAL_CAUSES


def test_the_two_failures_that_shared_one_sentence_are_now_distinguishable():
    empty = compute(verdict("count", operand_values=[""], operand_labels=["M1"]), {"M1"})
    uncited = compute(verdict("count", operand_values=["basil"], operand_labels=["M9"]), {"M1"})

    # The sentence the gate16 rows recorded is identical, which is the whole problem.
    assert empty.detail["reason"] == uncited.detail["reason"] == "an item has no valid source label"
    assert empty.detail["cause"] != uncited.detail["cause"]


def test_a_refusal_records_what_the_model_supplied_and_what_it_could_have_cited():
    result = compute(
        verdict("count", operand_values=["basil", "mint"], operand_labels=["M1", "M9"]),
        {"M1", "M2"},
    )

    assert result.detail["member"] == "mint"
    assert result.detail["label_supplied"] == "M9"
    assert result.detail["labels_available"] == ["M1", "M2"]
    assert [item["value"] for item in result.detail["operands_supplied"]] == ["basil", "mint"]


def test_a_numeric_refusal_names_the_operand_that_failed():
    result = compute(
        verdict("sum", operand_values=["12", "a few"], operand_labels=["M1", "M1"]),
        {"M1"},
    )

    assert result.detail["cause"] == "numeric_unparseable"
    assert result.detail["index"] == 2
    assert result.detail["value"] == "a few"


def test_a_duration_refusal_says_which_endpoint_failed():
    result = compute(
        verdict("duration", operand_values=["2026-01-01", "soon"], operand_labels=["M1", "M1"]),
        {"M1"},
    )

    endpoints = {item["name"]: item for item in result.detail["endpoints"]}
    assert endpoints["start"]["parsed"] == "2026-01-01"
    assert endpoints["end"]["parsed"] is None
    assert endpoints["end"]["cited_in_context"] is True


# (name, verdict, labels, computed, answer). Widening the record must not move any of
# these: the gate16 decision rests on exactly which questions were computed.
DECISIONS = [
    (
        "count",
        verdict(
            "count",
            operand_values=["basil", "mint", "the basil"],
            operand_labels=["M1", "M1", "[M2]"],
        ),
        {"M1", "M2"},
        True,
        "2 distinct: basil, mint",
    ),
    (
        "duration",
        verdict(
            "duration",
            operand_values=["2026-01-01", "2026-01-15"],
            operand_labels=["M1", "M2"],
            calculation_mode="days",
        ),
        {"M1", "M2"},
        True,
        "14 days",
    ),
    (
        "sum",
        verdict("sum", operand_values=["17", "5", "1"], operand_labels=["M1", "M2", "M4"]),
        {"M1", "M2", "M4"},
        True,
        "23 total",
    ),
    (
        "lookup is left to the model's prose",
        verdict("lookup"),
        {"M1"},
        False,
        "",
    ),
    (
        "an uncited count is refused",
        verdict("count", operand_values=["basil"], operand_labels=["M9"]),
        {"M1"},
        False,
        "",
    ),
]


@pytest.mark.parametrize(
    ("case", "given", "labels", "computed", "answer"),
    DECISIONS,
    ids=[name for name, _, _, _, _ in DECISIONS],
)
def test_the_decision_and_the_answer_are_unchanged(case, given, labels, computed, answer):
    result = compute(given, labels)

    assert result.computed is computed, case
    assert result.answer == answer
