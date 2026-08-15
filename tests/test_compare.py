"""Paired comparison and run-to-run variability.

These exist because the same configuration, re-run unchanged, scored 48.0%, 50.0%
and 56.0%. Any claim resting on a smaller gap than that is unsupported, and the
headline-accuracy comparison cannot tell the difference.
"""

from __future__ import annotations

import pytest

from llm_long_term_memory.evaluation.compare import Variability, compare
from llm_long_term_memory.evaluation.harness import QuestionResult, RunReport


def report(variant: str, outcomes: dict[str, bool]) -> RunReport:
    return RunReport(
        variant=variant,
        results=[
            QuestionResult(
                question_id=qid,
                question_type="temporal-reasoning",
                is_abstention=False,
                correct=ok,
                hypothesis="h",
                gold="g",
                judge_reason="r",
                context_tokens=10,
                prompt_tokens=12,
                output_tokens=2,
                latency_ms=100.0,
            )
            for qid, ok in outcomes.items()
        ],
    )


def test_shared_outcomes_are_discarded():
    """Questions both variants get right, or both get wrong, say nothing about
    which is better — that is the whole point of pairing."""
    a = report("a", {"q1": True, "q2": False, "q3": True, "q4": False})
    b = report("b", {"q1": True, "q2": False, "q3": False, "q4": True})

    c = compare(a, b)
    assert c.both_right == 1
    assert c.both_wrong == 1
    assert c.discordant == 2
    assert c.b_wins == 1 and c.b_losses == 1
    assert c.net == 0


def test_a_lopsided_result_is_significant():
    outcomes_a = {f"q{i}": False for i in range(12)}
    outcomes_b = {f"q{i}": True for i in range(12)}
    c = compare(report("a", outcomes_a), report("b", outcomes_b))

    assert c.b_wins == 12 and c.b_losses == 0
    assert c.p_value < 0.001
    assert c.significant
    assert "better" in c.verdict()


def test_a_small_edge_is_reported_as_no_detectable_difference():
    """The case that matters: B looks 4 points better on headline accuracy and the
    paired test refuses to call it."""
    a = {f"q{i}": i % 2 == 0 for i in range(50)}
    b = dict(a)
    b["q1"] = True  # one flip in B's favour
    b["q3"] = True
    c = compare(report("a", a), report("b", b))

    assert c.accuracy_delta == pytest.approx(0.04)
    assert not c.significant
    assert "no detectable difference" in c.verdict()


def test_identical_runs_have_no_discordant_pairs():
    outcomes = {f"q{i}": i % 3 == 0 for i in range(20)}
    c = compare(report("a", outcomes), report("b", dict(outcomes)))
    assert c.discordant == 0
    assert c.p_value == 1.0
    assert "identically" in c.verdict()


def test_only_questions_present_in_both_runs_are_paired():
    """A partially-completed run must not be silently compared against a full one."""
    a = report("a", {"q1": True, "q2": True, "q3": True})
    b = report("b", {"q1": False, "q2": True})

    c = compare(a, b)
    assert c.n_paired == 2
    assert c.b_losses == 1


def test_empty_overlap_is_handled():
    c = compare(report("a", {"q1": True}), report("b", {"q9": True}))
    assert c.n_paired == 0
    assert c.verdict() == "no questions in common"


def test_win_and_loss_ids_are_recoverable_for_inspection():
    """Knowing *which* questions flipped is how a change gets diagnosed, not just
    scored."""
    a = report("a", {"q1": False, "q2": True})
    b = report("b", {"q1": True, "q2": False})
    c = compare(a, b)
    assert c.win_ids == ["q1"]
    assert c.loss_ids == ["q2"]


def test_variability_reports_the_observed_spread():
    v = Variability("naive_rag", [0.48, 0.50, 0.56])
    assert v.n_runs == 3
    assert v.mean == pytest.approx(0.5133, abs=1e-3)
    assert v.spread == pytest.approx(0.08)
    assert v.stdev > 0.03
    assert "not evidence" in v.summary()


def test_a_single_run_has_no_measurable_spread():
    v = Variability("naive_rag", [0.48])
    assert v.spread == 0.0
    assert v.stdev == 0.0


def test_bootstrap_interval_is_deterministic_and_contains_the_mean():
    v = Variability("naive_rag", [0.48, 0.50, 0.56])
    low, high = v.bootstrap_interval
    assert low <= v.mean <= high
    assert (low, high) == v.bootstrap_interval
