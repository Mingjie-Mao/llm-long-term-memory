from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from llm_long_term_memory.evaluation.harness import QuestionResult, RunReport
from llm_long_term_memory.evaluation.validation import ArmAggregate, UsageAggregate


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_v3_phase5_tune3.py"
    spec = importlib.util.spec_from_file_location("run_v3_phase5_tune3_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tune3 = _load_script()


def _report(name: str) -> RunReport:
    rows = []
    for question_type, total in tune3.EXPECTED_TYPES.items():
        hydrate = question_type in {"temporal-reasoning", "multi-session"}
        rows.extend(
            QuestionResult(
                question_id=f"{name}-{question_type}-{index}",
                question_type=question_type,
                is_abstention=False,
                correct=True,
                hypothesis="answer",
                gold="gold",
                judge_reason="test",
                context_tokens=100,
                prompt_tokens=100,
                output_tokens=10,
                latency_ms=1.0,
                notes={
                    "reasoning_kind": "temporal" if hydrate else "direct",
                    "hydration_applied": hydrate,
                    "hydration_session_coverage": 1.0 if hydrate else None,
                    "hydration_redundant_anchors": 0,
                    "answer_confidence": "high",
                    "recall_coverage": {"hydrated": 1.0 if hydrate else None},
                },
            )
            for index in range(total)
        )
    return RunReport(variant=name, results=rows)


def _arm(name: str, context: int, output: int) -> ArmAggregate:
    return ArmAggregate(
        arm=name,
        questions=42,
        runs=1,
        run_accuracies=[0.74],
        mean_accuracy=0.74,
        stdev_accuracy=0.0,
        spread_accuracy=0.0,
        majority_accuracy=0.74,
        unanimous_agreement=1.0,
        median_context_tokens=context,
        recall_by_stage={},
        majority_accuracy_by_type={key: 1.0 for key in tune3.EXPECTED_TYPES},
        majority_failure_types={},
        fallback_trigger_rate=0.0,
        correct_with_raw_fallback_rate=0.0,
        usage=UsageAggregate(by_role={"answerer": {"output_tokens": output}}),
    )


def test_tune3_reuses_both_references_and_creates_only_42_candidate_rows():
    assert tune3.VARIANT == "two_stage_reasoned_evidence"
    assert tune3.CANDIDATE_ROWS.name == "v3.3-compact.jsonl"
    assert sum(tune3.EXPECTED_TYPES.values()) == 42


def test_tune3_gate_passes_when_accuracy_coverage_and_cost_are_preserved():
    reference = _report("v3.2")
    candidate = _report("v3.3")

    gate = tune3._gate(
        _arm("v2", context=600, output=900),
        _arm("v3.2", context=1500, output=1000),
        _arm("v3.3", context=1100, output=1050),
        reference,
        candidate,
    )

    assert gate["passed"] is True


def test_tune3_gate_stops_a_candidate_above_twice_v2_context():
    reference = _report("v3.2")
    candidate = _report("v3.3")

    gate = tune3._gate(
        _arm("v2", context=600, output=900),
        _arm("v3.2", context=1500, output=1000),
        _arm("v3.3", context=1201, output=1000),
        reference,
        candidate,
    )

    assert gate["checks"]["context_within_v2_limit"] is False
    assert gate["passed"] is False
