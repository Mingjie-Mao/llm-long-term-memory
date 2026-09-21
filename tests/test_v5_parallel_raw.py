"""v5.0 attaches raw turns before the first answer call instead of after a refusal.

The mechanism itself is `RawFallback.recover_local_detail`, unchanged: it ranks turns
inside the sessions structured retrieval already selected and never falls through to
another conversation. What changes is that it is no longer conditional. These tests pin
the two things that decide whether the arm means anything — that the turns actually
reach the prompt, and that the budget is the one the offline replay justified.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_long_term_memory.evaluation.runners.memory import MemoryRunner


def _runner(*, windows: int, planned: bool) -> MemoryRunner:
    runner = MemoryRunner.__new__(MemoryRunner)
    runner.parallel_raw_windows = windows
    runner.parallel_raw_planned = planned
    runner.parallel_raw = SimpleNamespace(max_turns=windows) if windows else None
    return runner


def test_the_fixed_arm_spends_the_same_budget_on_every_question():
    runner = _runner(windows=3, planned=False)

    for question in (
        "What is my dentist called?",
        "How many bikes did I service in March?",
        "How long was it between the two appointments?",
    ):
        assert runner._parallel_raw_budget(question) == 3


def test_an_aggregation_gets_the_wider_budget_the_replay_justified():
    """Six, not five.

    "up to 50 hours per week" is an assistant summary that ranks sixth among the turns
    of the twelve sessions its question selected, so a five-window budget misses it and
    the arm would be measured without the evidence it exists to supply.
    """
    runner = _runner(windows=3, planned=True)

    assert runner._parallel_raw_budget("How many bikes did I service in March?") == 6


def test_a_lookup_is_not_charged_for_an_aggregation_s_budget():
    runner = _runner(windows=3, planned=True)

    assert runner._parallel_raw_budget("What is my dentist called?") == 2


def test_the_arm_is_off_unless_it_is_asked_for():
    runner = _runner(windows=0, planned=False)

    assert runner.parallel_raw is None
    assert runner._parallel_raw_budget("anything") == 0


@pytest.mark.parametrize("variant", ["two_stage_v5_fixed", "two_stage_v5_planned"])
def test_both_arms_keep_v2c_s_answer_policy_so_only_the_evidence_differs(variant):
    """A comparison against v2c is only meaningful while everything else is v2c."""
    import inspect

    from llm_long_term_memory import cli

    source = inspect.getsource(cli._build)
    assert f'"{variant}": "v2c",' in source


def test_the_recovered_turns_reach_the_prompt_and_are_recorded():
    """A mechanism whose evidence never reaches the model is not being measured."""
    import inspect

    source = inspect.getsource(MemoryRunner.answer_request)

    assert "parallel_evidence = self.parallel_raw.recover_local_detail(" in source
    assert "Verbatim turns from those same conversations" in source
    assert "parallel_raw_turns" in inspect.getsource(MemoryRunner.answer_request)
