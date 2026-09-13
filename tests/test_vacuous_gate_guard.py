"""A gate that cannot fail must not be allowed to report as a pass.

The registered `dev60` check `no_new_confident_errors` counted rows whose `confidence`
was `high`. The answerer was never sent that schema field, so `confidence` was `medium`
— its default — on 100% of rows in every v3 phase, and the count was zero by
construction. The check passed. It had never been able to do anything else.
"""

from __future__ import annotations

import pytest

from llm_long_term_memory.evaluation.validation import (
    VacuousGateError,
    assert_gate_input_observed,
)


def test_a_field_pinned_at_its_default_is_refused():
    """The exact v3 shape: 60 rows, all 'medium', which is the default."""
    with pytest.raises(VacuousGateError, match="vacuous"):
        assert_gate_input_observed("no_new_confident_errors", ["medium"] * 60, default="medium")


def test_a_field_that_varies_is_accepted():
    assert_gate_input_observed(
        "no_new_confident_errors",
        ["medium"] * 58 + ["high", "low"],
        default="medium",
    )


def test_a_field_constant_at_something_other_than_the_default_is_accepted():
    """A run where every row genuinely is 'high' is a real observation, not a defect.
    Only constant-at-the-default is indistinguishable from never being asked."""
    assert_gate_input_observed("g", ["high"] * 40, default="medium")


def test_no_rows_is_refused_rather_than_silently_passing():
    with pytest.raises(VacuousGateError, match="no rows"):
        assert_gate_input_observed("g", [], default="medium")


def test_none_is_a_default_like_any_other():
    """The v2 control carries no confidence field at all, which reads as None."""
    with pytest.raises(VacuousGateError):
        assert_gate_input_observed("g", [None] * 30, default=None)
