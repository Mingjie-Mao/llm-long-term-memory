from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from llm_long_term_memory.evaluation.validation import ArmAggregate, UsageAggregate


def _load_script(name: str, relative: str):
    path = Path(__file__).resolve().parent.parent / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pilot = _load_script("run_v3_answer_pilot_for_tests", "scripts/run_v3_answer_pilot.py")
selector = _load_script(
    "create_v3_reasoning_manifest_for_tests", "tools/create_v3_reasoning_manifest.py"
)


def _arm(name: str, accuracy: float, failures: int, output_tokens: int) -> ArmAggregate:
    return ArmAggregate(
        arm=name,
        questions=48,
        runs=3,
        run_accuracies=[accuracy] * 3,
        mean_accuracy=accuracy,
        stdev_accuracy=0.0,
        spread_accuracy=0.0,
        majority_accuracy=accuracy,
        unanimous_agreement=1.0,
        median_context_tokens=500.0,
        recall_by_stage={},
        majority_accuracy_by_type={},
        majority_failure_types={"wrong_despite_gold_session_context": failures},
        fallback_trigger_rate=0.0,
        correct_with_raw_fallback_rate=0.0,
        usage=UsageAggregate(output_tokens=output_tokens),
    )


def test_selector_is_deterministic_and_honours_type_quotas():
    instances = []
    for question_type, count in selector.QUOTAS.items():
        instances.extend(
            SimpleNamespace(question_id=f"{question_type}-{index}", question_type=question_type)
            for index in range(count + 3)
        )
    train_ids = tuple(instance.question_id for instance in instances)

    first = selector.select_ids(instances, train_ids)
    second = selector.select_ids(instances, train_ids)

    assert first == second
    assert len(first) == sum(selector.QUOTAS.values()) == 48


def test_registered_pilot_is_two_arms_and_three_repeats():
    assert pilot.ARMS == (
        ("v2-control", "two_stage_fallback"),
        ("v3-reasoned", "two_stage_reasoned"),
    )
    assert pilot.RUNS == 3


def test_gate_accepts_accuracy_gain_without_cost_or_guard_regression():
    baseline = _arm("v2-control", 0.60, failures=20, output_tokens=1000)
    candidate = _arm("v3-reasoned", 0.65, failures=18, output_tokens=1200)
    baseline_slices = {
        "target_majority_accuracy": 0.55,
        "ordinary_majority_accuracy": 0.80,
        "high_confidence_majority_wrong": 3,
    }
    candidate_slices = {
        "target_majority_accuracy": 0.62,
        "ordinary_majority_accuracy": 0.80,
        "high_confidence_majority_wrong": 2,
    }

    decision = pilot._gate(baseline, candidate, baseline_slices, candidate_slices)

    assert decision["passed"] is True
    assert all(decision["checks"].values())


def test_gate_rejects_an_ordinary_question_regression():
    baseline = _arm("v2-control", 0.60, failures=20, output_tokens=1000)
    candidate = _arm("v3-reasoned", 0.65, failures=18, output_tokens=1200)
    baseline_slices = {
        "target_majority_accuracy": 0.55,
        "ordinary_majority_accuracy": 0.80,
        "high_confidence_majority_wrong": 3,
    }
    candidate_slices = {
        "target_majority_accuracy": 0.70,
        "ordinary_majority_accuracy": 0.70,
        "high_confidence_majority_wrong": 2,
    }

    decision = pilot._gate(baseline, candidate, baseline_slices, candidate_slices)

    assert decision["passed"] is False
    assert decision["checks"]["ordinary_regression"] is False
