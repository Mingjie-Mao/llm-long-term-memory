"""Session ranking by a DCG over matched turns."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from session_retrieval_offline import session_order


def test_a_session_with_several_high_turns_outranks_one_top_hit():
    ranked = [("a", 0), ("b", 0), ("b", 1), ("b", 2), ("c", 0)]
    assert session_order(ranked)[0] == "b"


def test_ties_break_by_the_first_appearance():
    assert session_order([("x", 0), ("y", 0)], top=1) == ["x"]
    assert session_order([("x", 0), ("y", 5)]) == ["x", "y"]


def test_only_the_top_turns_count():
    ranked = [("a", 0)] + [("b", i) for i in range(10)]
    assert session_order(ranked, top=1) == ["a"]
