from llm_long_term_memory.evaluation.runners.v2d import V2DVerdict, compute


def verdict(operation, **kwargs):
    kwargs.setdefault("premise_status", "not_applicable")
    return V2DVerdict(
        status="answer",
        answer="model prose",
        operation=operation,
        **kwargs,
    )


def test_count_deduplicates_members_but_allows_several_from_one_memory():
    result = compute(
        verdict(
            "count",
            operand_values=["Basil", "mint", "the basil"],
            operand_labels=["M1", "M1", "[M2]"],
        ),
        {"M1", "M2"},
    )
    assert result.computed
    assert result.answer == "2 distinct: Basil, mint"
    assert result.detail["count"] == 2
    assert len(result.detail["citations"]) == 3


def test_count_refuses_an_invented_source_label():
    result = compute(
        verdict("count", operand_values=["basil"], operand_labels=["M9"]),
        {"M1"},
    )
    assert not result.computed
    assert "source label" in result.detail["reason"]


def test_duration_is_computed_in_requested_weeks_without_model_arithmetic():
    result = compute(
        verdict(
            "duration",
            operand_values=["started 2026-01-01", "2026-01-17"],
            operand_labels=["M1", "QUESTION_DATE"],
            calculation_mode="weeks",
        ),
        {"M1"},
    )
    assert result.answer == "2 weeks and 2 days"
    assert result.detail["days"] == 16


def test_sum_uses_decimal_arithmetic_and_requires_matching_units():
    result = compute(
        verdict(
            "sum",
            operand_values=["$19.95", "10.05"],
            operand_qualifiers=["USD", "USD"],
            operand_labels=["M1", "M2"],
        ),
        {"M1", "M2"},
    )
    assert result.answer == "30 USD total"
    assert result.detail["result"] == "30"


def test_sum_converts_mixed_time_units_to_the_requested_unit():
    result = compute(
        verdict(
            "sum",
            result_unit="minutes",
            operand_values=["1", "30"],
            operand_qualifiers=["hour", "minutes"],
            operand_labels=["M1", "M2"],
        ),
        {"M1", "M2"},
    )
    assert result.answer == "90 minutes total"


def test_average_is_calculated_from_all_cited_operands():
    result = compute(
        verdict(
            "average",
            operand_values=["20", "40", "60"],
            operand_qualifiers=["years", "years", "years"],
            operand_labels=["M1", "M2", "M3"],
        ),
        {"M1", "M2", "M3"},
    )
    assert result.answer == "40 years average"


def test_grouped_count_returns_each_time_slice_separately():
    result = compute(
        verdict(
            "count",
            operand_values=["Alice", "Alice", "Bob"],
            operand_labels=["M1", "M2", "M2"],
            operand_qualifiers=["then", "now", "now"],
        ),
        {"M1", "M2"},
    )
    assert result.detail["groups"] == {"then": 1, "now": 2}
    assert "then: 1 distinct" in result.answer
    assert "now: 2 distinct" in result.answer


def test_duration_refuses_dates_without_provenance():
    result = compute(
        verdict(
            "duration",
            operand_values=["2026-01-01", "2026-01-02"],
            operand_labels=["", ""],
        ),
        set(),
    )
    assert not result.computed
    assert "citation" in result.detail["reason"]


def test_difference_preserves_direction():
    result = compute(
        verdict(
            "difference",
            calculation_mode="first_minus_second",
            operand_values=["80", "72.5"],
            operand_qualifiers=["kg", "kg"],
            operand_labels=["M1", "M2"],
        ),
        {"M1", "M2"},
    )
    assert result.answer == "7.5 kg difference"


def test_percentage_change_uses_first_operand_as_baseline():
    result = compute(
        verdict(
            "percentage_change",
            operand_values=["80", "60"],
            operand_qualifiers=["USD", "USD"],
            operand_labels=["M1", "M2"],
        ),
        {"M1", "M2"},
    )
    assert result.answer == "25% decrease"


def test_percentage_change_refuses_a_zero_baseline():
    result = compute(
        verdict(
            "percentage_change",
            operand_values=["0", "20"],
            operand_labels=["M1", "M2"],
        ),
        {"M1", "M2"},
    )
    assert not result.computed
    assert "zero" in result.detail["reason"]


def test_a_named_missing_operand_blocks_a_confident_calculation():
    result = compute(
        verdict(
            "sum",
            missing_field="second price",
            operand_values=["20"],
            operand_labels=["M1"],
        ),
        {"M1"},
    )
    assert not result.computed
    assert result.detail["missing_field"] == "second price"


def test_a_cited_false_premise_returns_the_correction_before_calculating():
    result = compute(
        verdict(
            "sum",
            premise_status="contradicted",
            premise_claim="the user bought both tickets",
            premise_evidence_labels=["[M2]"],
            premise_correction="The record says the second ticket was only considered, not bought.",
            operand_values=["20"],
            operand_labels=["M1"],
        ),
        {"M1", "M2"},
    )
    assert result.computed
    assert "only considered" in result.answer
    assert result.detail["operation"] == "false_premise"


def test_an_uncited_claim_of_contradiction_is_not_presented_as_fact():
    result = compute(
        verdict(
            "lookup",
            premise_status="contradicted",
            premise_claim="the user visited Rome",
            premise_evidence_labels=["M99"],
            premise_correction="The user never visited Rome.",
        ),
        {"M1"},
    )
    assert result.answer == "I do not know."
    assert "valid citation" in result.detail["reason"]


def test_an_unknown_premise_abstains_instead_of_using_a_nearby_fact():
    result = compute(
        verdict(
            "lookup",
            premise_status="unknown",
            premise_claim="the user completed the marathon",
        ),
        {"M1"},
    )
    assert result.answer == "I do not know whether the user completed the marathon."
    assert result.detail["premise_status"] == "unknown"


def test_operand_schema_is_flat_and_uses_only_string_arrays_for_gemini():
    schema = V2DVerdict.model_json_schema()
    assert "$defs" not in schema
    for field in (
        "operand_values",
        "operand_labels",
        "operand_qualifiers",
    ):
        assert schema["properties"][field]["items"] == {"type": "string"}


def test_misaligned_operand_arrays_are_rejected():
    count = compute(verdict("count", operand_values=["basil"], operand_labels=[]), {"M1"})
    numeric = compute(verdict("sum", operand_values=["1", "2"], operand_labels=["M1"]), {"M1"})
    assert not count.computed
    assert not numeric.computed
