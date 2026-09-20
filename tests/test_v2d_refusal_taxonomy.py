"""The taxonomy must classify what it can and refuse to invent what it cannot."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GATE16 = REPO / "results/raw/two_stage_v2d.v2d-gate16.jsonl"


def _module():
    path = REPO / "tools/v2d_refusal_taxonomy.py"
    spec = importlib.util.spec_from_file_location("v2d_refusal_taxonomy_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(question_id: str, computation: dict | None, correct: bool = False) -> dict:
    return {
        "question_id": question_id,
        "correct": correct,
        "notes": {"synthesis_computation": computation},
    }


def test_a_row_written_with_a_cause_is_classified_by_it():
    entry = _module().classify(
        _row(
            "q1",
            {
                "operation": "count",
                "cause": "label_not_in_context",
                "reason": "an item has no valid source label",
                "member": "mint",
                "label_supplied": "M9",
                "labels_available": ["M1", "M2"],
            },
        )
    )

    assert entry["outcome"] == "refused"
    assert entry["cause"] == "label_not_in_context"
    assert entry["evidence"]["label_supplied"] == "M9"


def test_an_ambiguous_legacy_sentence_is_reported_as_unclassifiable_not_guessed():
    entry = _module().classify(
        _row("q1", {"operation": "count", "reason": "an item has no valid source label"})
    )

    assert entry["cause"] == "unclassifiable_legacy_row"
    assert entry["could_be"] == ["member_text_empty", "label_not_in_context"]


def test_a_legacy_sentence_that_can_only_mean_one_thing_is_still_classified():
    entry = _module().classify(
        _row("q1", {"operation": "count", "reason": "count operand arrays are misaligned"})
    )

    assert entry["cause"] == "operand_arrays_misaligned"
    assert "could_be" not in entry


def test_a_lookup_is_not_counted_as_a_calculation_code_performed():
    module = _module()

    assert module.classify(_row("q1", {"operation": "lookup"}))["outcome"] == (
        "no_calculation_requested"
    )
    assert (
        module.classify(_row("q2", {"operation": "sum", "result": "23", "operands": []}))["outcome"]
        == "computed"
    )
    assert module.classify(_row("q3", None))["outcome"] == "no_computation_record"


def test_the_gate16_rows_reproduce_the_figures_the_decision_was_written_from():
    if not GATE16.is_file():
        pytest.skip(f"{GATE16.name} is not in this checkout")
    result = _module().analyse(GATE16)

    # `results/analysis/v2d-gate16.md` records 2 computed and 9 refused. If this
    # disagrees, one of the two documents is describing a different run.
    assert result["questions"] == 16
    assert result["computed"] == 2
    assert result["refused"] == 9
    # And none of them can be split further, which is the finding that decides v2e.
    assert result["diagnosable"] == 0
    assert result["unclassifiable"] == 9
