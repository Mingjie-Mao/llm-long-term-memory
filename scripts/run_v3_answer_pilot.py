"""Run the frozen, train-only v3 answer-policy pilot.

The command is a preflight unless ``--run`` is supplied. Individual rows are kept
under ``results/sealed/v3-answer-pilot`` for resume and audit; console and final
reports expose aggregate results only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.cli import _build  # noqa: E402
from llm_long_term_memory.config import Settings  # noqa: E402
from llm_long_term_memory.evaluation.datasets import longmemeval as lme  # noqa: E402
from llm_long_term_memory.evaluation.harness import (  # noqa: E402
    QuestionResult,
    RunReport,
    run_eval,
)
from llm_long_term_memory.evaluation.manifest import load_manifest  # noqa: E402
from llm_long_term_memory.evaluation.report import load_report  # noqa: E402
from llm_long_term_memory.evaluation.reproducibility import (  # noqa: E402
    FreezeError,
    FreezeRequest,
    capture_system,
    differences,
)
from llm_long_term_memory.evaluation.validation import (  # noqa: E402
    ArmAggregate,
    ValidationArtifactError,
    aggregate_arm,
    bind_artifacts,
    load_usage,
    paired_majority,
)

FREEZE_NAME = "v3-answer-pilot"
CONFIG = Path("configs/v3-answer.yaml")
TRAIN_MANIFEST = Path("results/manifests/train150.json")
PILOT_MANIFEST = Path("results/manifests/v3-reasoning48.json")
STORE_NAME = "train150"
RUNS = 3
ARMS = (
    ("v2-control", "two_stage_fallback"),
    ("v3-reasoned", "two_stage_reasoned"),
)
PROTOCOL_PATHS = (
    Path("results/prereg-v3-answer.md"),
    PILOT_MANIFEST,
    Path("results/archive/v2-final/conclusion.json"),
)
EXPECTED_TYPES = {
    "temporal-reasoning": 12,
    "multi-session": 12,
    "knowledge-update": 8,
    "single-session-user": 6,
    "single-session-assistant": 6,
    "single-session-preference": 4,
}
TARGET_TYPES = {
    "temporal-reasoning",
    "multi-session",
    "knowledge-update",
    "single-session-preference",
}
ORDINARY_TYPES = {"single-session-user", "single-session-assistant"}


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def _freeze_request(settings: Settings) -> FreezeRequest:
    return FreezeRequest(
        repo=REPO,
        config_path=_rooted(CONFIG),
        manifest_path=_rooted(TRAIN_MANIFEST),
        data_dir=_rooted(settings.data_dir),
        store_dir=_rooted(settings.store_dir),
        results_dir=_rooted(settings.results_dir),
        variants=tuple(variant for _, variant in ARMS),
        store_name=STORE_NAME,
        protocol_paths=tuple(_rooted(path) for path in PROTOCOL_PATHS),
    )


def _verify_freeze(settings: Settings) -> dict:
    path = REPO / "results/frozen" / FREEZE_NAME / "freeze.json"
    if not path.is_file():
        raise FreezeError(f"missing v3 pilot freeze: {path}")
    try:
        frozen = json.loads(path.read_text(encoding="utf-8"))["system"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FreezeError(f"invalid v3 pilot freeze: {exc}") from exc
    current = capture_system(_freeze_request(settings))
    moved = differences(frozen, current)
    if moved:
        raise FreezeError(
            f"v3 pilot changed after freeze ({len(moved)} differences): " + "; ".join(moved[:5])
        )
    if current["config"]["resolved"]["decay"]["enabled"]:
        raise FreezeError("v3 repeats refuse a store-mutating decay configuration")
    return frozen


def _instances(settings: Settings):
    pilot = load_manifest(_rooted(PILOT_MANIFEST))
    train = load_manifest(_rooted(TRAIN_MANIFEST))
    if pilot.name != "v3-reasoning48" or len(pilot) != 48:
        raise ValidationArtifactError("v3 pilot requires the fixed 48-question manifest")
    if pilot.variant != train.variant or not set(pilot.question_ids) <= set(train.question_ids):
        raise ValidationArtifactError("v3 pilot must be a strict train150-only subset")
    every = lme.load(train.variant, _rooted(settings.data_dir))
    by_id = {instance.question_id: instance for instance in every}
    missing = set(pilot.question_ids) - set(by_id)
    if missing:
        raise ValidationArtifactError(f"v3 pilot has {len(missing)} unknown question(s)")
    selected = [by_id[question_id] for question_id in pilot.question_ids]
    counts = Counter(instance.question_type for instance in selected)
    if dict(counts) != EXPECTED_TYPES:
        raise ValidationArtifactError(f"v3 pilot type allocation changed: {dict(counts)}")
    return pilot, selected


def _majority(rows: list[QuestionResult]) -> bool:
    return sum(row.correct for row in rows) >= (len(rows) // 2 + 1)


def _group(reports: list[RunReport], expected: set[str]) -> dict[str, list[QuestionResult]]:
    grouped: dict[str, list[QuestionResult]] = {question_id: [] for question_id in expected}
    for report in reports:
        seen: set[str] = set()
        for row in report.results:
            if row.question_id not in grouped or row.question_id in seen:
                raise ValidationArtifactError("v3 pilot rows differ from the fixed manifest")
            grouped[row.question_id].append(row)
            seen.add(row.question_id)
    if any(len(rows) != RUNS for rows in grouped.values()):
        raise ValidationArtifactError("v3 pilot does not contain three rows per question")
    return grouped


def _slice_metrics(reports: list[RunReport], expected: set[str]) -> dict[str, float | int]:
    grouped = _group(reports, expected)
    slice_counts = Counter()
    slice_correct = Counter()
    high_confidence_wrong = 0
    for rows in grouped.values():
        question_type = rows[0].question_type
        slice_name = "target" if question_type in TARGET_TYPES else "ordinary"
        slice_counts[slice_name] += 1
        correct = _majority(rows)
        slice_correct[slice_name] += int(correct)
        confidence = Counter(row.notes.get("answer_confidence") for row in rows).most_common(1)[0][
            0
        ]
        if not correct and confidence == "high":
            high_confidence_wrong += 1
    return {
        "target_majority_accuracy": slice_correct["target"] / slice_counts["target"],
        "ordinary_majority_accuracy": slice_correct["ordinary"] / slice_counts["ordinary"],
        "high_confidence_majority_wrong": high_confidence_wrong,
    }


def _answer_stage_failures(arm: ArmAggregate) -> int:
    return sum(
        arm.majority_failure_types.get(name, 0)
        for name in ("wrong_after_raw_fallback", "wrong_despite_gold_session_context")
    )


def _gate(
    baseline: ArmAggregate,
    candidate: ArmAggregate,
    baseline_slices: dict[str, float | int],
    candidate_slices: dict[str, float | int],
) -> dict[str, object]:
    baseline_failures = _answer_stage_failures(baseline)
    candidate_failures = _answer_stage_failures(candidate)
    failure_reduction = (
        (baseline_failures - candidate_failures) / baseline_failures if baseline_failures else 0.0
    )
    accuracy_delta = candidate.majority_accuracy - baseline.majority_accuracy
    ordinary_delta = float(candidate_slices["ordinary_majority_accuracy"]) - float(
        baseline_slices["ordinary_majority_accuracy"]
    )
    context_ratio = (
        candidate.median_context_tokens / baseline.median_context_tokens
        if baseline.median_context_tokens
        else 1.0
    )
    output_ratio = (
        candidate.usage.output_tokens / baseline.usage.output_tokens
        if baseline.usage.output_tokens
        else 1.0
    )
    checks = {
        "target_gain": accuracy_delta >= 0.03 or failure_reduction >= 0.30,
        "ordinary_regression": ordinary_delta >= -0.02,
        "high_confidence_wrong": int(candidate_slices["high_confidence_majority_wrong"])
        <= int(baseline_slices["high_confidence_majority_wrong"]),
        "context": context_ratio <= 1.0,
        "generation_cost": output_ratio <= 1.5,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "majority_accuracy_delta": accuracy_delta,
        "answer_stage_failure_reduction": failure_reduction,
        "ordinary_accuracy_delta": ordinary_delta,
        "context_ratio": context_ratio,
        "output_token_ratio": output_ratio,
        "baseline_slices": baseline_slices,
        "candidate_slices": candidate_slices,
    }


def _aggregate(manifest, sealed_dir: Path) -> tuple[dict, str]:
    expected = set(manifest.question_ids)
    reports_by_name: dict[str, list[RunReport]] = {}
    aggregates: list[ArmAggregate] = []
    sealed_paths: list[Path] = []
    for name, _ in ARMS:
        paths = [sealed_dir / f"{name}.rep{number}.jsonl" for number in range(1, RUNS + 1)]
        usage_paths = [path.with_suffix(".usage.json") for path in paths]
        reports = [load_report(path, variant=name) for path in paths]
        reports_by_name[name] = reports
        aggregates.append(
            aggregate_arm(name, reports, expected, [load_usage(path) for path in usage_paths])
        )
        sealed_paths.extend(item for pair in zip(paths, usage_paths, strict=True) for item in pair)
    comparison = paired_majority(
        ARMS[0][0],
        reports_by_name[ARMS[0][0]],
        ARMS[1][0],
        reports_by_name[ARMS[1][0]],
        expected,
    )
    gate = _gate(
        aggregates[0],
        aggregates[1],
        _slice_metrics(reports_by_name[ARMS[0][0]], expected),
        _slice_metrics(reports_by_name[ARMS[1][0]], expected),
    )
    payload = {
        "schema_version": 1,
        "manifest": {"name": manifest.name, "questions": len(manifest)},
        "arms": [arm.to_dict() for arm in aggregates],
        "paired_majority": comparison.to_dict(),
        "gate": gate,
        "shared_store": {
            "name": STORE_NAME,
            "ingestion_reused": True,
            "ingestion_cost_attributed_to_pilot": False,
        },
        "sealed_artifacts": bind_artifacts(sealed_paths),
        "privacy": {"aggregate_only": True, "contains_question_ids_or_text": False},
    }
    lines = [
        "# v3 train-only answer pilot",
        "",
        "Two answer policies used the same v2 store, retrieval, context and fallback.",
        "No question ids, question text, answers or judge reasons are reported.",
        "",
        "| arm | runs | majority accuracy | target accuracy | ordinary accuracy | "
        "median context | output tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    slices = [gate["baseline_slices"], gate["candidate_slices"]]
    for arm, slice_result in zip(aggregates, slices, strict=True):
        lines.append(
            f"| `{arm.arm}` | {arm.runs} | {arm.majority_accuracy:.1%} | "
            f"{slice_result['target_majority_accuracy']:.1%} | "
            f"{slice_result['ordinary_majority_accuracy']:.1%} | "
            f"{arm.median_context_tokens:,.0f} | {arm.usage.output_tokens:,} |"
        )
    lines.extend(
        [
            "",
            "## Pre-registered gate",
            "",
            f"Overall: **{'PASS' if gate['passed'] else 'STOP'}**",
            "",
            "| check | pass |",
            "|---|---:|",
            *[
                f"| `{name}` | {'yes' if passed else 'no'} |"
                for name, passed in gate["checks"].items()
            ],
            "",
            f"Paired delta: {comparison.accuracy_delta:+.1%}; exact p={comparison.p_value:.4f}.",
        ]
    )
    return payload, "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    try:
        manifest, instances = _instances(settings)
        _verify_freeze(settings)
    except (OSError, ValueError, FreezeError, ValidationArtifactError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    print(
        f"PASS: frozen v3 train-only pilot · {len(manifest)} questions · "
        f"{len(ARMS)} arms x {RUNS} repeats · rows sealed"
    )
    if not args.run:
        print("Preflight only. Add --run to start or resume provider calls.")
        return 0

    sealed_dir = _rooted(settings.results_dir) / "sealed/v3-answer-pilot"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    for name, variant in ARMS:
        for number in range(1, RUNS + 1):
            path = sealed_dir / f"{name}.rep{number}.jsonl"
            _, _, runner, judge, usage = _build(variant, str(_rooted(CONFIG)), STORE_NAME)
            runner.name = f"{name}.rep{number}"
            try:
                report = run_eval(runner, judge, instances, path, usage=usage, resume=True)
            finally:
                store = getattr(runner, "store", None)
                if store is not None:
                    store.close()
            if not report.completed:
                print(
                    f"PAUSED: {name} repeat {number} has {report.n}/{len(manifest)} rows; "
                    "resume the same command after quota reset. No scores exposed."
                )
                return 2
            print(f"COMPLETE: {name} repeat {number} ({report.n}/{len(manifest)} rows sealed)")

    try:
        _verify_freeze(settings)
        payload, markdown = _aggregate(manifest, sealed_dir)
    except (OSError, ValueError, FreezeError, ValidationArtifactError) as exc:
        print(f"STOP: cannot aggregate v3 pilot: {exc}", file=sys.stderr)
        return 2
    destination = _rooted(settings.results_dir) / "validation/v3-answer-pilot-aggregate.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    destination.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"Aggregate report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
