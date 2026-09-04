from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from llm_long_term_memory.evaluation.reproducibility import FreezeError
from llm_long_term_memory.evaluation.validation import ArmAggregate, UsageAggregate


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_v3_dev60.py"
    spec = importlib.util.spec_from_file_location("run_v3_dev60_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dev60 = _load_script()


def _arm(name: str, context: int, answer_output: int, selected_recall: float) -> ArmAggregate:
    return ArmAggregate(
        arm=name,
        questions=60,
        runs=3,
        run_accuracies=[0.7, 0.7, 0.7],
        mean_accuracy=0.7,
        stdev_accuracy=0.0,
        spread_accuracy=0.0,
        majority_accuracy=0.7,
        unanimous_agreement=1.0,
        median_context_tokens=context,
        recall_by_stage={"selected": selected_recall},
        majority_accuracy_by_type={},
        majority_failure_types={},
        fallback_trigger_rate=0.0,
        correct_with_raw_fallback_rate=0.0,
        usage=UsageAggregate(by_role={"answerer": {"output_tokens": answer_output}}),
    )


def _slices(*, target: float, temporal: float, multi: float) -> dict[str, float | int]:
    return {
        "target_accuracy": target,
        "temporal_accuracy": temporal,
        "multi_session_accuracy": multi,
        "knowledge_update_accuracy": 0.7,
        "ordinary_accuracy": 0.8,
        "high_confidence_majority_wrong": 2,
    }


def test_dev60_protocol_is_exactly_two_arms_three_repeats_and_360_rows():
    assert dev60.RUNS == 3
    assert dev60.ARMS == (
        ("v2-control", "two_stage_fallback"),
        ("v3.3-compact", "two_stage_reasoned_evidence"),
    )
    assert sum(dev60.EXPECTED_TYPES.values()) == 60
    assert len(dev60.ARMS) * dev60.RUNS * sum(dev60.EXPECTED_TYPES.values()) == 360


def test_dev60_gate_uses_registered_2x_context_boundary():
    baseline_slices = _slices(target=0.60, temporal=0.60, multi=0.60)
    candidate_slices = _slices(target=0.66, temporal=0.61, multi=0.71)
    gate = dev60._gate(
        _arm("v2", context=600, answer_output=1000, selected_recall=0.9),
        _arm("v3.3", context=1200, answer_output=1500, selected_recall=0.9),
        baseline_slices,
        candidate_slices,
    )

    assert gate["passed"] is True
    assert gate["checks"]["context_within_2x_v2"] is True


def test_dev60_gate_stops_context_above_2x_or_target_gain_below_5_points():
    baseline_slices = _slices(target=0.60, temporal=0.60, multi=0.60)
    candidate_slices = _slices(target=0.64, temporal=0.64, multi=0.64)
    gate = dev60._gate(
        _arm("v2", context=600, answer_output=1000, selected_recall=0.9),
        _arm("v3.3", context=1201, answer_output=1000, selected_recall=0.9),
        baseline_slices,
        candidate_slices,
    )

    assert gate["checks"]["target_gain"] is False
    assert gate["checks"]["context_within_2x_v2"] is False
    assert gate["passed"] is False


def test_dev60_artifacts_without_ledger_are_refused(tmp_path: Path):
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}", encoding="utf-8")
    orphan = tmp_path / "orphan.jsonl"
    orphan.write_text("row\n", encoding="utf-8")

    with pytest.raises(FreezeError, match="without the one-shot ledger"):
        dev60._prepare_ledger(
            tmp_path / "one-shot.json",
            freeze,
            [orphan],
            tmp_path / "aggregate.json",
            create=False,
            forbid_complete=False,
        )


def test_dev60_ledger_is_written_before_calls_and_binds_freeze(tmp_path: Path):
    freeze = tmp_path / "freeze.json"
    freeze.write_text("frozen", encoding="utf-8")
    ledger_path = tmp_path / "one-shot.json"
    ledger = dev60._prepare_ledger(
        ledger_path,
        freeze,
        [],
        tmp_path / "aggregate.json",
        create=True,
        forbid_complete=True,
    )

    assert ledger_path.is_file()
    assert ledger["status"] == "started"
    assert ledger["expected_rows"] == 360
    dev60._validate_ledger(ledger, freeze)
