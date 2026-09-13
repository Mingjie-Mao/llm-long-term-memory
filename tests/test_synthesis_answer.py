"""The code must do the arithmetic, and must refuse when it has nothing to work with.

v3.3 scored 4.0% on the count probes and 42.5% on duration, while `current_state` — the
one operation needing no arithmetic — scored 86.7%. The split in `synthesis.py` follows
that: the model picks the operands, Python performs the operation.

The second half of that contract matters as much as the first. A number derived from
absent operands would be worse than the model's own guess, because it would arrive with
a derivation that looks checked.
"""

from __future__ import annotations

from llm_long_term_memory.evaluation.runners.synthesis import (
    SynthesisVerdict,
    compute,
    deduplicate,
)


def _verdict(**kwargs) -> SynthesisVerdict:
    kwargs.setdefault("status", "answer")
    # `answer` is required on the v4 schema — the provider must supply it rather than be
    # asked to. These tests exercise `compute`, which reads the operand fields, so the
    # default here keeps their subject unchanged; a test that cares about the reply text
    # passes its own.
    kwargs.setdefault("answer", "")
    return SynthesisVerdict(**kwargs)


def test_counting_is_done_by_the_code_not_the_model():
    """The observed failure: the model enumerates correctly and then states a wrong
    total. Here it says nothing about the total at all."""
    result = compute(
        _verdict(
            operation="count",
            items=["engagement ring", "emerald earrings", "gold bracelet"],
            answer="You acquired two pieces of jewellery.",
        )
    )
    assert result.computed
    assert result.detail["count"] == 3
    assert result.answer.startswith("3")


def test_repeated_members_are_folded_before_counting():
    """The other direction of the same failure: five babies reported as six."""
    result = compute(
        _verdict(
            operation="count",
            items=["Max", "max", "  Max  ", "Ava", "the twins"],
        )
    )
    assert result.detail["items_stated"] == 5
    assert result.detail["count"] == 3


def test_deduplication_does_not_merge_genuinely_different_members():
    """Over-merging would turn an under-count into the failure it was meant to fix."""
    assert len(deduplicate(["gold bracelet", "silver bracelet"])) == 2
    assert len(deduplicate(["Uber Eats", "Uber"])) == 2


def test_duration_is_subtracted_not_generated():
    result = compute(_verdict(operation="duration", start_date="2022-01-01", end_date="2022-01-19"))
    assert result.computed
    assert result.detail["days"] == 18
    assert result.answer == "18 days"


def test_duration_operands_may_arrive_in_either_order():
    """Nothing guarantees the model puts the earlier date in `start_date`, and a
    negative duration is never the right answer."""
    result = compute(_verdict(operation="duration", start_date="2022-01-19", end_date="2022-01-01"))
    assert result.detail["days"] == 18


def test_a_timestamp_copied_out_of_the_context_still_parses():
    result = compute(
        _verdict(
            operation="duration",
            start_date="2022-01-01T09:30:00",
            end_date="2022-01-19",
        )
    )
    assert result.detail["days"] == 18


def test_comparison_reports_which_operand_is_earlier():
    result = compute(
        _verdict(operation="comparison", start_date="2023-05-20", end_date="2023-03-10")
    )
    assert result.computed
    assert result.detail["earlier"] == "2023-03-10"
    assert result.detail["start_is_earlier"] is False


def test_missing_operands_refuse_rather_than_invent():
    """`computed=False` is the contract that keeps the caller on the model's prose."""
    assert not compute(_verdict(operation="count", items=[])).computed
    assert not compute(_verdict(operation="duration", start_date="2022-01-01")).computed
    assert not compute(
        _verdict(operation="duration", start_date="not a date", end_date="also not")
    ).computed


def test_an_impossible_date_is_treated_as_missing():
    """A 13th month parses as three integers and is not a date."""
    assert not compute(
        _verdict(operation="duration", start_date="2022-13-01", end_date="2022-01-19")
    ).computed


def test_lookup_and_current_state_keep_the_model_answer():
    """The two operations with nothing to compute must not be touched."""
    for operation in ("lookup", "current_state"):
        result = compute(_verdict(operation=operation, answer="Spotify"))
        assert not result.computed, f"{operation} has no arithmetic to do"


def test_the_count_answer_is_readable_by_the_probe_grader():
    """A contract between two components that are edited independently.

    The grader has to find the asserted total among every number in the reply, and a
    member can contain one of its own. The first format tried here — `3 (2 concert
    tickets, a book, a scarf)` — was read as **2**, because the grader's fallback takes
    the last number. That would have produced silent false errors in the v4 measurement
    and looked like a counting failure.
    """
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "tools" / "run_synthesis_probes.py"
    spec = importlib.util.spec_from_file_location("run_synthesis_probes_for_tests", path)
    assert spec and spec.loader
    grader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = grader
    spec.loader.exec_module(grader)

    for items in (
        ["engagement ring", "emerald earrings", "gold bracelet"],
        ["2 concert tickets", "a book", "a scarf"],
        ["a trip in 2023", "a lamp", "a rug", "a chair"],
        ["Max", "max", "Ava"],
    ):
        result = compute(_verdict(operation="count", items=items))
        read_back = grader._int_for("count", result.answer)
        assert read_back == result.detail["count"], (
            f"grader read {read_back} from {result.answer!r}, code computed "
            f"{result.detail['count']}"
        )


def test_the_duration_answer_is_readable_by_the_probe_grader():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "tools" / "run_synthesis_probes.py"
    spec = importlib.util.spec_from_file_location("run_synthesis_probes_for_duration", path)
    assert spec and spec.loader
    grader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = grader
    spec.loader.exec_module(grader)

    result = compute(_verdict(operation="duration", start_date="2022-01-01", end_date="2022-01-19"))
    assert grader._int_for("duration", result.answer) == 18


# --------------------------------------------------------------- missing operands
#
# v4.0's most likely way of making things worse is converting an abstention into a
# confidently wrong computed number: the model names two wrong dates and Python
# subtracts them exactly, producing an answer that arrives with a derivation attached.
# `missing_field` is the declared guard against that, and until these tests it was a
# third write-only field — defined in the schema, named in the prompt, read by nothing.


def test_a_named_missing_operand_blocks_computation():
    """The model's own words win. It said something was absent."""
    result = compute(
        _verdict(
            operation="duration",
            start_date="2022-01-01",
            end_date="2022-01-19",
            missing_field="no end date was recorded",
            answer="I do not know when it finished.",
        )
    )
    assert not result.computed, "computed from operands the verdict called incomplete"
    assert result.detail["missing_field"] == "no end date was recorded"


def test_the_guard_applies_to_counting_too():
    result = compute(
        _verdict(
            operation="count",
            items=["a ring", "earrings"],
            missing_field="the memories may not list every purchase",
        )
    )
    assert not result.computed


def test_whitespace_is_not_a_missing_operand():
    """A model that fills the field with a space has said nothing, and blocking on it
    would silently disable the arithmetic for every question."""
    result = compute(
        _verdict(
            operation="duration",
            start_date="2022-01-01",
            end_date="2022-01-19",
            missing_field="   ",
        )
    )
    assert result.computed
    assert result.detail["days"] == 18


def test_the_guard_is_recorded_so_the_stop_condition_can_count_it():
    """A stop condition cannot be evaluated from a field nothing writes down."""
    result = compute(
        _verdict(operation="count", items=["a"], missing_field="not all purchases are listed")
    )
    assert "missing_field" in result.detail
    assert result.detail["operation"] == "count"


def test_what_this_cannot_catch():
    """The dangerous path is invisible offline, and saying so is part of the test.

    A model that supplies two *wrong* dates and leaves `missing_field` empty produces a
    confidently wrong number, and nothing in the verdict distinguishes it from a right
    one. Only grading against ground truth catches it, which is why the registered stop
    condition is measured on the run rather than asserted here.
    """
    result = compute(_verdict(operation="duration", start_date="1999-01-01", end_date="1999-01-19"))
    assert result.computed, "the code cannot know these are the wrong dates"
    assert result.detail["days"] == 18


# ------------------------------------------------------- the answer must be prose
#
# On the first paid v4.0 run, 55 of 142 rows reached the grader as raw JSON: the model
# filled `operation` and left `answer` empty, so the runner fell back to the raw
# completion, which under a structured schema is the structure itself. `comparison` lost
# 31 of 33 rows that way and scored -30 points — a plumbing failure that read exactly
# like a reasoning failure.


def test_the_prompt_requires_an_answer_for_every_operation():
    """The prompt previously described operands in detail and never said `answer` was
    mandatory, so for operations needing no computation the model filled neither."""
    from llm_long_term_memory.evaluation.runners.synthesis import SYNTHESIS_ANSWER_SYSTEM

    assert "Always write your reply in prose in `answer`" in SYNTHESIS_ANSWER_SYSTEM
    assert "whatever the operation is" in SYNTHESIS_ANSWER_SYSTEM


def test_comparison_still_keeps_the_model_wording_and_needs_it():
    """`compute` returns no sentence for comparison — knowing which date is earlier does
    not say which described event it belongs to — so `answer` is the only source of a
    reply, which is exactly why the empty-answer case destroyed this operation."""
    result = compute(
        _verdict(operation="comparison", start_date="2023-05-20", end_date="2023-03-10")
    )
    assert result.computed
    assert result.answer == "", "a sentence here would assert a mapping the code lacks"
    assert result.detail["earlier"] == "2023-03-10"


# --------------------------------------------- missing_field: name it, do not narrate
#
# On v4.0-flat the field was populated on five rows and three of them were the model
# thinking out loud inside it — "none needed for current_state unless absent, which it
# is not here, so leave empty or valid string per schema guidelines..." — which then
# blocked a computation that should have run. The field asks for the name of an absent
# operand; these tests keep the two behaviours separable.


def test_a_short_operand_name_is_a_real_report():
    from llm_long_term_memory.evaluation.runners.synthesis import missing_field_is_narration

    for value in ("end date", "start and end date", "volunteer start date", "no items"):
        assert not missing_field_is_narration(value), value


def test_deliberation_inside_the_field_is_detected():
    from llm_long_term_memory.evaluation.runners.synthesis import missing_field_is_narration

    observed = (
        "none needed for current_state lookup here unless specific field is empty, but "
        "this is handled by answer/lookup style fallback. wait, rule says missing_field "
        "only when operation cannot be completed"
    )
    assert missing_field_is_narration(observed)
    assert missing_field_is_narration("none needed unless absent, which it is not here")


def test_narration_does_not_block_a_computation():
    """The failure being fixed: deliberation in the field stopped the arithmetic."""
    result = compute(
        _verdict(
            operation="duration",
            start_date="2022-01-01",
            end_date="2022-01-19",
            missing_field="none needed here, since both dates are present, so leave empty",
        )
    )
    assert result.computed, "narration blocked a computation that had its operands"
    assert result.detail["days"] == 18


def test_a_real_report_still_blocks():
    result = compute(
        _verdict(
            operation="duration", start_date="2022-01-01", end_date="", missing_field="end date"
        )
    )
    assert not result.computed
    assert result.detail["missing_field"] == "end date"


def test_the_field_description_asks_for_a_name_not_a_sentence():
    fields = SynthesisVerdict.model_fields
    description = fields["missing_field"].description
    assert "NAME" in description
    assert "reasoning" in description.lower()
