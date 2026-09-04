from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from llm_long_term_memory.evaluation.harness import QuestionResult, RunReport
from llm_long_term_memory.evaluation.validation import ArmAggregate, UsageAggregate


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_v3_phase4_tune2.py"
    spec = importlib.util.spec_from_file_location("run_v3_phase4_tune2_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tune2 = _load_script()


def _report(name: str, correct_by_type: dict[str, int]) -> RunReport:
    rows = []
    for question_type, correct_count in correct_by_type.items():
        total = tune2.EXPECTED_TYPES[question_type]
        rows.extend(
            QuestionResult(
                question_id=f"{name}-{question_type}-{index}",
                question_type=question_type,
                is_abstention=False,
                correct=index < correct_count,
                hypothesis="answer",
                gold="gold",
                judge_reason="test",
                context_tokens=100,
                prompt_tokens=100,
                output_tokens=10,
                latency_ms=1.0,
                notes={
                    "reasoning_kind": "temporal"
                    if question_type == "temporal-reasoning"
                    else "direct",
                    "hydration_applied": question_type == "temporal-reasoning",
                    "recall_coverage": {"selected": 1.0},
                },
            )
            for index in range(total)
        )
    return RunReport(variant=name, results=rows)


def _arm(name: str, accuracy_by_type: dict[str, float], context: int, output: int):
    return ArmAggregate(
        arm=name,
        questions=42,
        runs=1,
        run_accuracies=[0.7],
        mean_accuracy=0.7,
        stdev_accuracy=0.0,
        spread_accuracy=0.0,
        majority_accuracy=0.7,
        unanimous_agreement=1.0,
        median_context_tokens=context,
        recall_by_stage={},
        majority_accuracy_by_type=accuracy_by_type,
        majority_failure_types={},
        fallback_trigger_rate=0.0,
        correct_with_raw_fallback_rate=0.0,
        usage=UsageAggregate(by_role={"answerer": {"output_tokens": output}}),
    )


def test_candidate_variant_adds_only_adaptive_reasoning_evidence():
    assert tune2.VARIANT == "two_stage_reasoned_evidence"
    assert tune2.CANDIDATE_ROWS.name == "v3.2-adaptive.jsonl"


def test_gate_requires_target_gain_without_type_regression():
    baseline_counts = {
        "temporal-reasoning": 4,
        "multi-session": 7,
        "knowledge-update": 3,
        "single-session-user": 9,
        "single-session-assistant": 5,
        "single-session-preference": 0,
    }
    candidate_counts = {**baseline_counts, "temporal-reasoning": 5, "multi-session": 8}
    baseline_report = _report("baseline", baseline_counts)
    candidate_report = _report("candidate", candidate_counts)
    baseline_rates = {
        key: value / tune2.EXPECTED_TYPES[key] for key, value in baseline_counts.items()
    }
    candidate_rates = {
        key: value / tune2.EXPECTED_TYPES[key] for key, value in candidate_counts.items()
    }

    gate = tune2._gate(
        _arm("baseline", baseline_rates, context=600, output=1000),
        _arm("candidate", candidate_rates, context=900, output=1200),
        baseline_report,
        candidate_report,
    )

    assert gate["passed"] is True
    assert round(gate["target_accuracy_delta"], 10) == 0.10


def test_gate_rejects_hydration_on_a_direct_question():
    counts = {
        "temporal-reasoning": 6,
        "multi-session": 8,
        "knowledge-update": 4,
        "single-session-user": 9,
        "single-session-assistant": 5,
        "single-session-preference": 1,
    }
    baseline = _report("baseline", counts)
    candidate = _report("candidate", counts)
    candidate.results[-1].notes["hydration_applied"] = True
    rates = {key: value / tune2.EXPECTED_TYPES[key] for key, value in counts.items()}

    gate = tune2._gate(
        _arm("baseline", rates, 600, 1000),
        _arm("candidate", rates, 600, 1000),
        baseline,
        candidate,
    )

    assert gate["checks"]["hydration_is_targeted"] is False
    assert gate["passed"] is False
