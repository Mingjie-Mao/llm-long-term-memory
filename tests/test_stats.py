"""One exact-McNemar implementation, and proof it is the one the others computed.

Four copies had grown: `ingest/zero_yield_report.py`, `scripts/batch_position_analyse.py`,
`tools/analyze_v2b_gate16_qa.py` and `tools/batch_size_curve.py`. Collapsing them is only
safe if they agreed, so the four formulas as they were written are reproduced here and
compared against the survivor over the whole range the gates operate in. A p-value that
decides whether an experiment is promoted must not depend on which file computed it.
"""

from __future__ import annotations

from math import comb

import pytest

from llm_long_term_memory.stats import exact_mcnemar, paired_outcomes


def _zero_yield_report_copy(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _batch_position_copy(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)


def _gate16_qa_copy(wins: int, losses: int) -> float:
    discordant = wins + losses
    if not discordant:
        return 1.0
    tail = sum(comb(discordant, k) for k in range(min(wins, losses) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def _batch_size_curve_copy(gained: int, lost: int) -> float:
    n = gained + lost
    if not n:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(gained, lost) + 1))
    return min(1.0, 2 * tail / 2**n)


COPIES = (
    _zero_yield_report_copy,
    _batch_position_copy,
    _gate16_qa_copy,
    _batch_size_curve_copy,
)


@pytest.mark.parametrize("wins", range(0, 13))
@pytest.mark.parametrize("losses", range(0, 13))
def test_the_survivor_returns_what_all_four_copies_returned(wins, losses):
    expected = exact_mcnemar(wins, losses)

    for copy in COPIES:
        assert copy(wins, losses) == expected


def test_no_discordant_pairs_is_not_a_significant_result():
    assert exact_mcnemar(0, 0) == 1.0


def test_the_p_value_is_two_sided_and_never_exceeds_one():
    assert exact_mcnemar(3, 0) == exact_mcnemar(0, 3)
    assert exact_mcnemar(1, 1) == 1.0
    assert exact_mcnemar(10, 0) == pytest.approx(2 / 2**10)


def test_counts_cannot_be_negative():
    with pytest.raises(ValueError):
        exact_mcnemar(-1, 2)


def test_paired_outcomes_decomposes_the_comparison_the_gates_are_judged_on():
    baseline = {"a": True, "b": False, "c": True, "d": False}
    candidate = {"a": True, "b": True, "c": False, "d": False}

    result = paired_outcomes(baseline, candidate, ["a", "b", "c", "d"])

    assert result["wins"] == ["b"]
    assert result["losses"] == ["c"]
    assert result["ties"] == 2
    assert result["net"] == 0
    assert result["old_correct"] == 2
    assert result["new_correct"] == 2
    assert result["exact_mcnemar_p"] == exact_mcnemar(1, 1)


def test_a_comparison_over_ids_one_arm_is_missing_is_an_error_not_a_dropped_pair():
    with pytest.raises(KeyError, match="absent from an arm"):
        paired_outcomes({"a": True}, {"a": True, "b": False}, ["a", "b"])
