from __future__ import annotations

import pytest

from llm_long_term_memory.evaluation.context_selection import (
    ContextCandidate,
    NoEligibleContext,
    choose_context_candidate,
)


def candidate(name: str, **overrides) -> ContextCandidate:
    values = {
        "name": name,
        "aggregate": "mean",
        "window_radius": 1,
        "max_total_memories": 30,
        "rank_top3": 0.90,
        "assembled_recall": 0.88,
        "median_context_tokens": 150,
        "flat_median_tokens": 360,
        "truncated_questions": 1,
    }
    values.update(overrides)
    return ContextCandidate(**values)


def test_assembled_recall_is_primary_after_the_top3_gate():
    lower = candidate("lower", rank_top3=0.99, assembled_recall=0.90)
    higher = candidate("higher", rank_top3=0.91, assembled_recall=0.95)

    assert choose_context_candidate([lower, higher]) == higher


def test_equal_recall_prefers_the_smaller_hard_cap():
    cap40 = candidate("cap40", max_total_memories=40, truncated_questions=0)
    cap30 = candidate("cap30", max_total_memories=30, truncated_questions=1)

    assert choose_context_candidate([cap40, cap30]) == cap30


def test_candidate_must_clear_rank_assembly_and_context_gates():
    candidates = [
        candidate("rank", rank_top3=0.79),
        candidate("assembly", assembled_recall=0.79),
        candidate("context", median_context_tokens=541, flat_median_tokens=360),
    ]

    with pytest.raises(NoEligibleContext):
        choose_context_candidate(candidates)


def test_context_ratio_with_no_flat_baseline_is_not_eligible():
    with pytest.raises(NoEligibleContext):
        choose_context_candidate([candidate("missing-baseline", flat_median_tokens=0)])
