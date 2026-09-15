"""Clustered paired comparison, against a test that quietly overstates its own n."""

from __future__ import annotations

from math import sqrt

import pytest

from llm_long_term_memory.evaluation.clustered import (
    design_effect,
    minimum_detectable_effect,
    sign_flip_p_value,
)


def test_independent_questions_have_no_design_effect():
    assert design_effect(20, 0.0) == 1.0
    assert design_effect(20, 0.05) == pytest.approx(1.95)


def test_the_detectable_effect_is_the_paired_null_inflated_by_clustering():
    independent = minimum_detectable_effect(660, 0.04, 20, 0.0)
    assert independent == pytest.approx(2 * sqrt(660 * 0.04) / 660)
    assert minimum_detectable_effect(660, 0.04, 20, 0.15) > independent
    assert minimum_detectable_effect(1_320, 0.04, 20, 0.0) < independent


def test_six_conversations_all_one_way_is_the_smallest_exact_two_sided_signal():
    assert sign_flip_p_value([1] * 6) == pytest.approx(2 / 64)
    assert sign_flip_p_value([1] * 5) == pytest.approx(2 / 32)


def test_no_net_difference_is_no_evidence():
    assert sign_flip_p_value([]) == 1.0
    assert sign_flip_p_value([2, -2]) == 1.0


def test_the_sampled_test_is_seeded_and_never_reports_zero():
    nets = [1.0] * 30
    p = sign_flip_p_value(nets, permutations=2_000, seed=3)
    assert p == sign_flip_p_value(nets, permutations=2_000, seed=3)
    assert 0 < p < 0.01


@pytest.mark.parametrize(
    "args",
    [(0, 0.1, 20, 0.0), (10, 1.5, 20, 0.0), (10, 0.1, 20, 1.5), (10, 0.1, 0, 0.0)],
)
def test_impossible_inputs_are_refused(args):
    with pytest.raises(ValueError):
        minimum_detectable_effect(*args)
