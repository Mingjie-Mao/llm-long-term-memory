from __future__ import annotations

import json

import pytest

from llm_long_term_memory.evaluation.harness import QuestionResult, RunReport
from llm_long_term_memory.evaluation.reproducibility import sha256_file
from llm_long_term_memory.evaluation.validation import (
    ArmAggregate,
    UsageAggregate,
    ValidationArtifactError,
    aggregate_arm,
    bind_artifacts,
    load_dev_decision,
    load_usage,
    paired_majority,
    render_final_markdown,
    render_markdown,
    select_dev_candidate,
    single_run_arm_dict,
)


def row(
    qid: str,
    correct: bool,
    *,
    stages: tuple[bool, bool, bool] | None = (True, True, True),
    fallback: str = "none",
    qtype: str = "multi-session",
) -> QuestionResult:
    notes = {"fallback_level": fallback}
    if stages is not None:
        notes["recall_stages"] = dict(
            zip(("candidates", "ranked", "selected"), stages, strict=True)
        )
    return QuestionResult(
        question_id=qid,
        question_type=qtype,
        is_abstention=False,
        correct=correct,
        hypothesis="must stay sealed",
        gold="must stay sealed too",
        judge_reason="private reason",
        context_tokens=100,
        prompt_tokens=120,
        output_tokens=10,
        latency_ms=500.0,
        notes=notes,
    )


def reports(name: str, outcomes: list[dict[str, bool]], **row_kwargs) -> list[RunReport]:
    return [
        RunReport(name, [row(qid, correct, **row_kwargs) for qid, correct in run.items()])
        for run in outcomes
    ]


def arm(name: str, accuracy: float, context: float) -> ArmAggregate:
    return ArmAggregate(
        arm=name,
        questions=100,
        runs=3,
        run_accuracies=[accuracy] * 3,
        mean_accuracy=accuracy,
        stdev_accuracy=0.0,
        spread_accuracy=0.0,
        majority_accuracy=accuracy,
        unanimous_agreement=1.0,
        median_context_tokens=context,
        recall_by_stage={},
        majority_accuracy_by_type={},
        majority_failure_types={},
        fallback_trigger_rate=0.0,
        correct_with_raw_fallback_rate=0.0,
        usage=UsageAggregate(),
    )


def test_registered_dev_rule_adopts_coherent_only_when_both_gates_pass():
    decision = select_dev_candidate(
        [
            arm("flat20", 0.70, 100),
            arm("coherent-auto", 0.70, 150),
            arm("coherent-oracle", 0.75, 150),
        ]
    )

    assert decision.selected_variant == "two_stage_coherent"
    assert decision.accuracy_gate_passed
    assert decision.context_gate_passed


@pytest.mark.parametrize(
    ("coherent_accuracy", "oracle_accuracy", "context", "diagnostic"),
    [
        (0.69, 0.72, 120, "session_selection_gap"),
        (0.69, 0.68, 120, "coherent_shape_not_supported"),
        (0.72, 0.75, 151, "context_budget_gate_failed"),
        (0.69, 0.75, 151, "accuracy_and_context_gates_failed"),
    ],
)
def test_registered_dev_rule_falls_back_with_an_explicit_diagnosis(
    coherent_accuracy, oracle_accuracy, context, diagnostic
):
    decision = select_dev_candidate(
        [
            arm("flat20", 0.70, 100),
            arm("coherent-auto", coherent_accuracy, context),
            arm("coherent-oracle", oracle_accuracy, context),
        ]
    )

    assert decision.selected_variant == "two_stage_fallback"
    assert decision.diagnostic == diagnostic


def test_dev_decision_is_cryptographically_bound_to_its_aggregate(tmp_path):
    decision = select_dev_candidate(
        arms := [
            arm("flat20", 0.70, 100),
            arm("coherent-auto", 0.71, 120),
            arm("coherent-oracle", 0.72, 120),
            arm("naive_rag", 0.99, 13_000),
            arm("memory-only", 0.98, 90),
        ]
    ).to_dict()
    results = tmp_path / "results"
    sealed = results / "sealed" / "dev100"
    sealed.mkdir(parents=True)
    artifacts = []
    for name in (
        "flat20",
        "coherent-auto",
        "coherent-oracle",
        "naive_rag",
        "memory-only",
    ):
        for number in range(1, 4):
            for suffix in ("jsonl", "usage.json"):
                path = sealed / f"{name}.rep{number}.{suffix}"
                path.write_text(f"{path.name}\n", encoding="utf-8")
                artifacts.append(path)
    aggregate = results / "validation" / "dev100-aggregate.json"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_text(
        json.dumps(
            {
                "arms": [item.to_dict() for item in arms],
                "registered_decision": decision,
                "sealed_artifacts": bind_artifacts(artifacts),
            }
        ),
        encoding="utf-8",
    )

    decision_path = aggregate.with_name("dev100-decision.json")
    decision_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                **decision,
                "aggregate_sha256": sha256_file(aggregate),
            }
        ),
        encoding="utf-8",
    )

    loaded = load_dev_decision(decision_path, aggregate)
    assert loaded["selected_variant"] == "two_stage_coherent"

    original = artifacts[0].read_text(encoding="utf-8")
    artifacts[0].write_text("changed", encoding="utf-8")
    with pytest.raises(ValidationArtifactError, match="changed after aggregation"):
        load_dev_decision(decision_path, aggregate)
    artifacts[0].write_text(original, encoding="utf-8")

    forged = json.loads(aggregate.read_text(encoding="utf-8"))
    forged["registered_decision"] = {
        **decision,
        "selected_arm": "flat20",
        "selected_variant": "two_stage_fallback",
    }
    aggregate.write_text(json.dumps(forged), encoding="utf-8")
    forged_decision = json.loads(decision_path.read_text(encoding="utf-8"))
    forged_decision.update(forged["registered_decision"])
    forged_decision["aggregate_sha256"] = sha256_file(aggregate)
    decision_path.write_text(json.dumps(forged_decision), encoding="utf-8")
    with pytest.raises(ValidationArtifactError, match="disagrees with its aggregate"):
        load_dev_decision(decision_path, aggregate)

    aggregate.write_text("{}", encoding="utf-8")
    with pytest.raises(ValidationArtifactError, match="aggregate report hash"):
        load_dev_decision(decision_path, aggregate)


def test_final_report_uses_single_run_terms_without_fake_agreement_statistics():
    result = arm("v2", 0.70, 100)
    result.runs = 1
    result.run_accuracies = [0.70]

    payload = single_run_arm_dict(result)
    text = render_final_markdown([result], [], UsageAggregate())

    assert payload["accuracy"] == 0.70
    assert "majority_accuracy" not in payload
    assert "unanimous_agreement" not in payload
    assert "Each frozen arm ran exactly once" in text
    assert "mean ± sd" not in text


def test_aggregate_reports_repeats_majority_agreement_and_stage_recall():
    runs = reports(
        "candidate",
        [
            {"secret-a": True, "secret-b": False},
            {"secret-a": True, "secret-b": False},
            {"secret-a": True, "secret-b": True},
        ],
    )

    result = aggregate_arm("candidate", runs, {"secret-a", "secret-b"})

    assert result.run_accuracies == [0.5, 0.5, 1.0]
    assert result.majority_accuracy == 0.5
    assert result.unanimous_agreement == 0.5
    assert result.recall_by_stage == {"candidates": 1.0, "ranked": 1.0, "selected": 1.0}
    assert result.majority_failure_types == {"wrong_despite_gold_session_context": 1}


@pytest.mark.parametrize(
    ("stages", "fallback", "expected"),
    [
        ((False, False, False), "none", "no_gold_session_in_candidates"),
        ((True, False, False), "none", "gold_session_lost_before_top_k"),
        ((True, True, False), "none", "gold_session_lost_in_context"),
        ((True, True, True), "session", "wrong_after_raw_fallback"),
        ((True, True, True), "none", "wrong_despite_gold_session_context"),
        (None, "none", "stage_trace_unavailable"),
    ],
)
def test_failure_stage_buckets_are_non_overlapping(stages, fallback, expected):
    runs = reports(
        "candidate",
        [{"sealed-id": False}, {"sealed-id": False}, {"sealed-id": True}],
        stages=stages,
        fallback=fallback,
    )

    result = aggregate_arm("candidate", runs, {"sealed-id"})

    assert result.majority_failure_types == {expected: 1}


def test_manifest_mismatch_reports_counts_without_leaking_identifiers():
    runs = reports("candidate", [{"private-a": True}] * 3)

    with pytest.raises(ValidationArtifactError) as caught:
        aggregate_arm("candidate", runs, {"private-a", "private-b"})

    message = str(caught.value)
    assert "1 missing" in message
    assert "private-a" not in message
    assert "private-b" not in message


def test_even_number_of_repeats_is_refused():
    runs = reports("candidate", [{"q": True}, {"q": True}])

    with pytest.raises(ValidationArtifactError, match="odd number"):
        aggregate_arm("candidate", runs, {"q"})


def test_majority_pairing_uses_same_frozen_questions_without_returning_ids():
    baseline = reports(
        "baseline",
        [
            {"private-a": False, "private-b": True},
            {"private-a": False, "private-b": True},
            {"private-a": True, "private-b": True},
        ],
    )
    candidate = reports(
        "candidate",
        [
            {"private-a": True, "private-b": False},
            {"private-a": True, "private-b": False},
            {"private-a": True, "private-b": True},
        ],
    )

    comparison = paired_majority(
        "baseline", baseline, "candidate", candidate, {"private-a", "private-b"}
    )

    assert comparison.candidate_wins == 1
    assert comparison.candidate_losses == 1
    assert "private-a" not in json.dumps(comparison.to_dict())


def test_usage_is_recomputed_from_calls_and_merged_into_arm(tmp_path):
    usage_path = tmp_path / "run.usage.json"
    usage_path.write_text(
        json.dumps(
            {
                "summary": {"total_requests": 999},
                "calls": [
                    {
                        "role": "answerer",
                        "model": "m",
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "latency_ms": 1,
                        "ok": True,
                        "error": None,
                    },
                    {
                        "role": "judge",
                        "model": "m",
                        "input_tokens": 20,
                        "output_tokens": 2,
                        "latency_ms": 1,
                        "ok": False,
                        "error": "quota",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    parsed = load_usage(usage_path)
    runs = reports("candidate", [{"q": True}] * 3)

    result = aggregate_arm("candidate", runs, {"q"}, usage=[parsed, parsed, parsed])

    assert parsed.requests == 2, "do not trust the stale cached summary"
    assert result.usage.requests == 6
    assert result.usage.failed_requests == 3
    assert result.usage.total_tokens == 396
    assert result.usage.by_role["answerer"]["requests"] == 3


def test_markdown_contains_only_aggregate_material():
    runs = reports("candidate", [{"hidden-qid": True}] * 3)
    arm = aggregate_arm("candidate", runs, {"hidden-qid"})

    rendered = render_markdown([arm], [])

    assert "hidden-qid" not in rendered
    assert "must stay sealed" not in rendered
    assert "majority" in rendered
    assert "failure stages" in rendered


def test_markdown_includes_shared_ingestion_and_total_usage():
    runs = reports("candidate", [{"hidden-qid": True}] * 3)
    arm = aggregate_arm("candidate", runs, {"hidden-qid"})
    arm.usage.requests = 6
    arm.usage.input_tokens = 600
    shared = UsageAggregate(requests=4, input_tokens=400, output_tokens=40)

    rendered = render_markdown([arm], [], shared)

    assert "shared ingestion" in rendered
    assert "**10**" in rendered
    assert "provider billing" in rendered


def test_reported_baselines_do_not_enter_the_dev_decision():
    """The 2026-08-25 amendment adds naive_rag and memory-only as reported arms.

    They exist so the final table can answer "compared with what?". A baseline that
    could move the product choice would be a decision rule rewritten after the arms
    were chosen, so the selection must ignore them entirely — including when a
    baseline scores higher than every decision arm.
    """
    decision = select_dev_candidate(
        [
            arm("flat20", 0.70, 300.0),
            arm("coherent-auto", 0.70, 320.0),
            arm("coherent-oracle", 0.74, 320.0),
            arm("naive_rag", 0.99, 13000.0),
            arm("memory-only", 0.98, 250.0),
        ]
    )
    assert decision.selected_arm == "coherent-auto"
    assert decision.baseline_majority_accuracy == 0.70


def test_an_unregistered_dev_arm_is_refused():
    with pytest.raises(ValidationArtifactError):
        select_dev_candidate(
            [
                arm("flat20", 0.70, 300.0),
                arm("coherent-auto", 0.70, 320.0),
                arm("coherent-oracle", 0.74, 320.0),
                arm("coherent-r2", 0.80, 320.0),
            ]
        )
