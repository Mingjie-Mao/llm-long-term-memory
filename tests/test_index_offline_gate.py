"""Budget filling and coverage in `tools/index_offline_gate.py`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from index_offline_gate import covers_all, covers_any, fill, paired


def test_fill_takes_units_in_order_until_the_budget():
    cost = {("s", 0): 10, ("s", 1): 10, ("s", 2): 10}
    assert fill([[("s", 0)], [("s", 1)], [("s", 2)]], cost, 20) == [("s", 0), ("s", 1)]


def test_fill_skips_an_overflowing_unit_and_keeps_going():
    cost = {("s", 0): 5, ("s", 1): 50, ("s", 2): 5}
    assert fill([[("s", 0)], [("s", 1)], [("s", 2)]], cost, 12) == [("s", 0), ("s", 2)]


def test_overlapping_windows_are_charged_only_for_new_turns():
    cost = {("s", i): 10 for i in range(4)}
    windows = [[("s", 0), ("s", 1), ("s", 2)], [("s", 1), ("s", 2), ("s", 3)]]
    assert fill(windows, cost, 40) == [("s", 0), ("s", 1), ("s", 2), ("s", 3)]


def test_turns_outside_the_session_are_ignored_not_charged():
    """A ±1 window at a session's edge names a turn that does not exist."""
    cost = {("s", 0): 10}
    assert fill([[("s", -1), ("s", 0), ("s", 1)]], cost, 10) == [("s", 0)]


def test_coverage_needs_every_gold_turn_for_all_and_one_for_any():
    gold = {("s", 1), ("t", 2)}
    assert covers_any([("s", 1)], gold) and not covers_all([("s", 1)], gold)
    assert covers_all([("s", 1), ("t", 2), ("u", 0)], gold)
    assert not covers_all([], set()), "no gold is not full coverage"


def test_paired_counts_flips_over_shared_questions():
    assert paired({"a": True, "b": False, "c": True}, {"a": False, "b": True, "d": True}) == (
        1,
        1,
    )
