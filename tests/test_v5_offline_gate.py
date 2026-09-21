"""The gate that decides whether v5.0 and v5.1 are worth paying for.

Its two jobs are to locate a gold answer honestly and to refuse to locate one it
cannot. Both were got wrong on the first pass, and both mistakes pointed the decision
the wrong way, so they are pinned here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ROWS = REPO / "results/raw/two_stage_v2c.reasoning48-v2c8.jsonl"
STORE = REPO / "stores/v2c-reasoning48.db"


def _module():
    path = REPO / "tools/v5_offline_gate.py"
    spec = importlib.util.spec_from_file_location("v5_offline_gate_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _memories(*contents: str):
    from llm_long_term_memory.store import Memory

    return [
        Memory(
            id=f"m{index}",
            user_id="u",
            type="semantic",
            content=content,
            token_count=len(content.split()),
        )
        for index, content in enumerate(contents)
    ]


def test_an_exact_phrase_is_reported_as_an_exact_match():
    assert _module().match_kind("Patagonia", _memories("The assistant named Patagonia.")) == "exact"


def test_a_numeric_gold_satisfied_only_by_stray_digits_is_reported_as_weak():
    """`answer_present` says yes here, and saying only yes is what hid a real hit.

    "1 week" was scored as already present in a store of 2,949 memories because the
    digit 1 occurs in it, which pre-empted the rung that was actually true — the
    sentence was in the raw turns of a session retrieval had already selected.
    """
    module = _module()
    haystack = _memories("The user has 1 sibling.", "The user drank 2 coffees.")

    assert module.match_kind("1 week", haystack) == "numeric"

    from llm_long_term_memory.evaluation.extraction_coverage import answer_present

    assert answer_present("1 week", haystack) is True


def test_a_gold_that_is_absent_matches_nothing():
    assert _module().match_kind("Patagonia", _memories("The user likes Reebok.")) is None


def test_the_hand_audit_must_describe_this_run_and_only_this_run():
    """A stale attribution is worse than none: it would read as evidence."""
    module = _module()
    entries = [
        {"question_id": "q1", "correct": False},
        {"question_id": "q2", "correct": True},
    ]

    with pytest.raises(SystemExit, match="unattributed"):
        module._audit(entries)


@pytest.mark.skipif(not STORE.is_file(), reason="the v2c store is a local research artifact")
@pytest.mark.skipif(not ROWS.is_file(), reason="the v2c rows are not in this checkout")
def test_the_gate_reproduces_the_attribution_the_decision_rests_on():
    result = _module().analyse(ROWS, STORE)

    assert result["questions"] == 48
    assert result["correct"] == 37
    # Raw text reached the reader in ten questions; the other thirty-eight saw
    # structured memory only. That is the opportunity v5.0 is registered against.
    assert result["raw_never_reached_the_reader"] == 38
    audit = result["audit"]
    assert audit["reachable_by_v5_0"] == 3
    assert audit["reachable_by_v5_1"] == 1
    assert audit["not_reachable_by_either"] == 7
