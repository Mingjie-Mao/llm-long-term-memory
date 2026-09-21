"""The pilot's registered gates, including the one the scorer did not evaluate.

`results/prereg-specificity-repair-pilot.md` registers six conditions. Five were
computed by `score`; the third — that every accepted repair fact is anchored to an
exact span of a user turn, and that a baseline-missing specificity occurs in both that
span and the fact's content — was enforced at write time and never reported. A gate
enforced silently is, in the artifact, indistinguishable from one nobody ran.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent


def _module():
    path = REPO / "tools/specificity_repair_pilot.py"
    spec = importlib.util.spec_from_file_location("specificity_repair_pilot_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TURN = "I walked 416 steps to the Museum of Contemporary Art."


def _session():
    return SimpleNamespace(
        session_id="s1",
        turns=[
            SimpleNamespace(role="assistant", content="How was your day?"),
            SimpleNamespace(role="user", content=TURN),
        ],
    )


def _checkpoint(**overrides):
    memory = {
        "id": "r1",
        "content": "The user walked 416 steps.",
        "source_turn_index": 1,
        "source_char_start": 0,
        "source_char_end": len(TURN),
    }
    memory.update(overrides)
    return {
        "repairs": {
            "s1": {"called": True, "missing": ["416"], "memories": [memory]},
        }
    }


def _failures(**overrides):
    return _module()._grounding_failures(_checkpoint(**overrides), [_session()])


def test_a_properly_grounded_repair_passes():
    assert _failures() == []


def test_a_repair_with_no_source_offsets_is_a_failure():
    problems = _failures(source_char_start=None, source_char_end=None)

    assert len(problems) == 1
    assert "offsets" in problems[0]["problem"]


def test_a_repair_anchored_to_an_assistant_turn_is_a_failure():
    """The mechanism repairs what the *user* said; an assistant turn is a different claim."""
    problems = _failures(source_turn_index=0)

    assert len(problems) == 1
    assert "assistant turn" in problems[0]["problem"]


def test_a_repair_pointing_outside_the_session_is_a_failure():
    problems = _failures(source_turn_index=9)

    assert len(problems) == 1
    assert "outside the session" in problems[0]["problem"]


def test_a_repair_whose_span_does_not_contain_the_recovered_specific_is_a_failure():
    """The span has to be the evidence, not merely a citation next to it."""
    problems = _failures(source_char_start=0, source_char_end=8)

    assert len(problems) == 1
    assert "span" in problems[0]["problem"]


def test_a_repair_whose_content_drops_the_specific_is_a_failure():
    problems = _failures(content="The user went to a museum.")

    assert len(problems) == 1
    assert "span and the content" in problems[0]["problem"]


def test_the_scorer_reports_every_gate_the_preregistration_names():
    import inspect

    source = inspect.getsource(_module().score)

    for gate in (
        "fidelity_gain_at_least_10pp",
        "at_least_10_gains",
        "zero_losses",
        "every_repair_is_grounded_in_a_user_span",
        "repair_calls_at_most_half",
        "added_memories_at_most_one_per_session",
    ):
        assert f'"{gate}"' in source
