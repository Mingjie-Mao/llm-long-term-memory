"""The measurement that decides whether the dedup fix is an amendment or a second arm."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

diff_tool = pytest.importorskip("dedup_fix_behavioral_diff")


def arm(memories, reaching, crossings=0):
    return {
        "memories": memories,
        "memory_count": sum(len(v) for v in memories.values()),
        "reaching": reaching,
        "cross_namespace_adjudications": crossings,
    }


REACHED = {"q1": {"ranked": ["m1", "m2"], "context": ["m1"], "context_tokens": 10}}


def test_identical_stores_and_contexts_read_as_a_measurement_fix():
    same = arm({"ns": ["a", "b"]}, REACHED)
    result = diff_tool.diff(same, same)
    assert result["identical"] is True
    assert result["verdict"].startswith("measurement fix")


def test_a_memory_the_fix_saves_is_a_system_change_even_with_an_unchanged_context():
    """The strict rule is about which memories are deduplicated, not only about what was
    retrieved on the questions that happened to be asked."""
    before = arm({"ns": ["a"]}, REACHED)
    after = arm({"ns": ["a", "b"]}, REACHED)
    result = diff_tool.diff(before, after)
    assert result["memories_the_fix_saved"] == 1
    assert result["questions_with_a_different_context"] == 0
    assert result["identical"] is False
    assert result["verdict"].startswith("system change")


def test_a_changed_context_is_reported_per_question():
    before = arm({"ns": ["a"]}, {"q1": {"ranked": ["m1"], "context": ["m1"]}})
    after = arm({"ns": ["a"]}, {"q1": {"ranked": ["m1", "m2"], "context": ["m2"]}})
    result = diff_tool.diff(before, after)
    assert result["questions_with_different_ranked_memory_ids"] == 1
    assert result["questions_with_a_different_context"] == 1


def test_the_scoped_neighbour_search_drops_only_foreign_memories():
    class Memory:
        def __init__(self, user_id):
            self.user_id = user_id

    mine, theirs = Memory("ns-a"), Memory("ns-b")
    patched = diff_tool.namespace_scoped(lambda self, c, v, sv, sm: [mine, theirs])
    assert patched(None, Memory("ns-a"), None, None, None) == [mine]


def test_the_worst_case_probe_only_forces_cross_namespace_pairs():
    class Memory:
        def __init__(self, user_id):
            self.user_id = user_id

    def original(self, old, new):
        return "UNTOUCHED"

    patched = diff_tool.force_cross_namespace_duplicates(original)
    assert patched(None, Memory("a"), Memory("a")) == "UNTOUCHED"
    assert patched(None, Memory("a"), Memory("b")).verdict == "DUPLICATE"
