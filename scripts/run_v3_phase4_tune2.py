"""Run the frozen v3.2 adaptive-evidence tune42 iteration.

The completed tune1 v2 rows are the fixed baseline. This command creates only the
42 candidate rows and is a preflight unless ``--run`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.cli import _build  # noqa: E402
from llm_long_term_memory.config import Settings  # noqa: E402
from llm_long_term_memory.evaluation.datasets import longmemeval as lme  # noqa: E402
from llm_long_term_memory.evaluation.harness import RunReport, run_eval  # noqa: E402
from llm_long_term_memory.evaluation.manifest import load_manifest  # noqa: E402
from llm_long_term_memory.evaluation.report import load_report  # noqa: E402
from llm_long_term_memory.evaluation.reproducibility import (  # noqa: E402
    FreezeError,
    FreezeRequest,
    capture_system,
    differences,
    sha256_file,
)
from llm_long_term_memory.evaluation.validation import (  # noqa: E402
    ArmAggregate,
    ValidationArtifactError,
    aggregate_arm,
    load_usage,
    paired_majority,
)

FREEZE_NAME = "v3-phase4-tune2"
CONFIG = Path("configs/v3-phase4-adaptive.yaml")
TRAIN_MANIFEST = Path("results/manifests/train150.json")
TUNE_MANIFEST = Path("results/manifests/v3-reasoning-tune42.json")
BASELINE_ROWS = Path("results/sealed/v3-phase3-tune1/v2-control.jsonl")
BASELINE_USAGE = Path("results/sealed/v3-phase3-tune1/v2-control.usage.json")
TUNE1_AGGREGATE = Path("results/validation/v3-phase3-tune1.json")
CANDIDATE_ROWS = Path("results/sealed/v3-phase4-tune2/v3.2-adaptive.jsonl")
STORE_NAME = "train150"
VARIANT = "two_stage_reasoned_evidence"
PROTOCOL_PATHS = (
    Path("results/prereg-v3-phase4-adaptive.md"),
    TUNE_MANIFEST,
    Path("results/archive/v3-phase3-tune1/conclusion.json"),
)
EXPECTED_TYPES = {
    "temporal-reasoning": 10,
    "multi-session": 10,
    "knowledge-update": 5,
    "single-session-user": 10,
    "single-session-assistant": 6,
    "single-session-preference": 1,
}
TARGET_TYPES = {"temporal-reasoning", "multi-session"}
ORDINARY_TYPES = {
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
}
HYDRATED_OPERATIONS = {"temporal", "multi_session_aggregation", "current_state"}


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
        variants=(VARIANT,),
        store_name=STORE_NAME,
        protocol_paths=tuple(_rooted(path) for path in PROTOCOL_PATHS),
    )


def _verify_freeze(settings: Settings) -> None:
    path = REPO / "results/frozen" / FREEZE_NAME / "freeze.json"
    if not path.is_file():
        raise FreezeError(f"missing v3.2 tune freeze: {path}")
    try:
        frozen = json.loads(path.read_text(encoding="utf-8"))["system"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FreezeError(f"invalid v3.2 tune freeze: {exc}") from exc
    current = capture_system(_freeze_request(settings))
    if moved := differences(frozen, current):
        raise FreezeError(f"v3.2 tune changed after freeze ({len(moved)}): " + "; ".join(moved[:5]))
    if current["config"]["resolved"]["decay"]["enabled"]:
        raise FreezeError("v3.2 tune refuses store-mutating decay")


def _verify_frozen_baseline() -> None:
    try:
        aggregate = json.loads(_rooted(TUNE1_AGGREGATE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationArtifactError(f"cannot read tune1 aggregate: {exc}") from exc
    sealed = aggregate.get("sealed_artifacts", {})
    for relative in (BASELINE_ROWS, BASELINE_USAGE):
        record = sealed.get(relative.name)
        path = _rooted(relative)
        if (
            not isinstance(record, dict)
            or not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValidationArtifactError(f"frozen tune1 baseline changed: {relative}")


def _instances(settings: Settings):
    tune = load_manifest(_rooted(TUNE_MANIFEST))
    train = load_manifest(_rooted(TRAIN_MANIFEST))
    if tune.name != "v3-reasoning-tune42" or len(tune) != 42:
        raise ValidationArtifactError("v3.2 tune requires the fixed tune42 manifest")
    if tune.variant != train.variant or not set(tune.question_ids) <= set(train.question_ids):
        raise ValidationArtifactError("tune42 must remain a train150 subset")
    all_instances = lme.load(train.variant, _rooted(settings.data_dir))
    by_id = {instance.question_id: instance for instance in all_instances}
    if missing := set(tune.question_ids) - set(by_id):
        raise ValidationArtifactError(f"tune42 has {len(missing)} unknown question(s)")
    selected = [by_id[question_id] for question_id in tune.question_ids]
    if dict(Counter(instance.question_type for instance in selected)) != EXPECTED_TYPES:
        raise ValidationArtifactError("tune42 type allocation changed")
    return tune, selected


def _slice_accuracy(report: RunReport, types: set[str]) -> float:
    rows = [row for row in report.results if row.question_type in types]
    return sum(row.correct for row in rows) / len(rows)


def _high_confidence_wrong(report: RunReport) -> int:
    return sum(
        not row.correct and row.notes.get("answer_confidence") == "high" for row in report.results
    )


def _candidate_diagnostics(report: RunReport) -> dict[str, float | int]:
    applied = [row for row in report.results if row.notes.get("hydration_applied")]
    unexpected = [
        row for row in applied if row.notes.get("reasoning_kind") not in HYDRATED_OPERATIONS
    ]
    coverage = [
        float(value)
        for row in report.results
        if isinstance((value := row.notes.get("recall_coverage", {}).get("selected")), float)
    ]
    return {
        "hydrated_questions": len(applied),
        "unexpected_hydration": len(unexpected),
        "mean_selected_session_coverage": mean(coverage) if coverage else 0.0,
        "full_selected_session_coverage_rate": (
            sum(value == 1.0 for value in coverage) / len(coverage) if coverage else 0.0
        ),
    }


def _gate(
    baseline: ArmAggregate,
    candidate: ArmAggregate,
    baseline_report: RunReport,
    candidate_report: RunReport,
) -> dict[str, object]:
    target_delta = _slice_accuracy(candidate_report, TARGET_TYPES) - _slice_accuracy(
        baseline_report, TARGET_TYPES
    )
    ordinary_delta = _slice_accuracy(candidate_report, ORDINARY_TYPES) - _slice_accuracy(
        baseline_report, ORDINARY_TYPES
    )
    temporal_delta = (
        candidate.majority_accuracy_by_type["temporal-reasoning"]
        - baseline.majority_accuracy_by_type["temporal-reasoning"]
    )
    multi_delta = (
        candidate.majority_accuracy_by_type["multi-session"]
        - baseline.majority_accuracy_by_type["multi-session"]
    )
    context_ratio = candidate.median_context_tokens / baseline.median_context_tokens
    baseline_answer_tokens = baseline.usage.by_role["answerer"]["output_tokens"]
    candidate_answer_tokens = candidate.usage.by_role["answerer"]["output_tokens"]
    output_ratio = candidate_answer_tokens / baseline_answer_tokens
    diagnostics = _candidate_diagnostics(candidate_report)
    checks = {
        "target_gain": target_delta >= 0.10 - 1e-12,
        "temporal_no_regression": temporal_delta >= 0.0,
        "multi_session_no_regression": multi_delta >= 0.0,
        "overall_no_regression": candidate.majority_accuracy >= baseline.majority_accuracy,
        "ordinary_no_regression": ordinary_delta >= 0.0,
        "no_new_confident_errors": _high_confidence_wrong(candidate_report)
        <= _high_confidence_wrong(baseline_report),
        "hydration_is_targeted": diagnostics["unexpected_hydration"] == 0,
        "context_budget": context_ratio <= 2.0,
        "generation_cost": output_ratio <= 1.5,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "target_accuracy_delta": target_delta,
        "temporal_accuracy_delta": temporal_delta,
        "multi_session_accuracy_delta": multi_delta,
        "ordinary_accuracy_delta": ordinary_delta,
        "overall_accuracy_delta": candidate.majority_accuracy - baseline.majority_accuracy,
        "context_ratio": context_ratio,
        "answer_output_token_ratio": output_ratio,
        "candidate_diagnostics": diagnostics,
    }


def _artifact(relative: Path) -> dict[str, int | str]:
    path = _rooted(relative)
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _aggregate(manifest) -> tuple[dict, str]:
    expected = set(manifest.question_ids)
    baseline_report = load_report(_rooted(BASELINE_ROWS), variant="v2-control")
    candidate_report = load_report(_rooted(CANDIDATE_ROWS), variant="v3.2-adaptive")
    baseline = aggregate_arm(
        "v2-control",
        [baseline_report],
        expected,
        [load_usage(_rooted(BASELINE_USAGE))],
    )
    candidate_usage = CANDIDATE_ROWS.with_suffix(".usage.json")
    candidate = aggregate_arm(
        "v3.2-adaptive",
        [candidate_report],
        expected,
        [load_usage(_rooted(candidate_usage))],
    )
    paired = paired_majority(
        baseline.arm, [baseline_report], candidate.arm, [candidate_report], expected
    )
    gate = _gate(baseline, candidate, baseline_report, candidate_report)
    artifact_paths = (BASELINE_ROWS, BASELINE_USAGE, CANDIDATE_ROWS, candidate_usage)
    payload = {
        "schema_version": 1,
        "manifest": {"name": manifest.name, "questions": len(manifest)},
        "arms": [baseline.to_dict(), candidate.to_dict()],
        "paired": paired.to_dict(),
        "gate": gate,
        "artifacts": {path.as_posix(): _artifact(path) for path in artifact_paths},
        "privacy": {"aggregate_only": True, "contains_question_ids_or_text": False},
    }
    lines = [
        "# v3.2 adaptive source-evidence tune42",
        "",
        "The candidate adds compact raw spans only for registered multi-step operations.",
        "The v2 baseline is the hash-verified tune1 artifact; it was not rerun.",
        "",
        "| arm | accuracy | temporal | multi-session | ordinary | median context | answer output |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, report in ((baseline, baseline_report), (candidate, candidate_report)):
        lines.append(
            f"| `{arm.arm}` | {arm.majority_accuracy:.1%} | "
            f"{arm.majority_accuracy_by_type['temporal-reasoning']:.1%} | "
            f"{arm.majority_accuracy_by_type['multi-session']:.1%} | "
            f"{_slice_accuracy(report, ORDINARY_TYPES):.1%} | "
            f"{arm.median_context_tokens:,.0f} | "
            f"{arm.usage.by_role['answerer']['output_tokens']:,} |"
        )
    lines.extend(
        [
            "",
            f"Tune promotion signal: **{'PASS' if gate['passed'] else 'STOP'}**",
            "",
            "| check | pass |",
            "|---|---:|",
            *[
                f"| `{name}` | {'yes' if passed else 'no'} |"
                for name, passed in gate["checks"].items()
            ],
            "",
            f"Paired difference: {paired.accuracy_delta:+.1%}; exact p={paired.p_value:.4f}.",
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
        _verify_frozen_baseline()
        _verify_freeze(settings)
    except (OSError, ValueError, FreezeError, ValidationArtifactError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    print("PASS: frozen v3.2 adaptive tune42 · 42 new rows · frozen v2 baseline reused")
    if not args.run:
        print("Preflight only. Add --run after explicit permission for this v3.2 API use.")
        return 0

    path = _rooted(CANDIDATE_ROWS)
    path.parent.mkdir(parents=True, exist_ok=True)
    _, _, runner, judge, usage = _build(VARIANT, str(_rooted(CONFIG)), STORE_NAME)
    runner.name = "v3.2-adaptive"
    try:
        report = run_eval(runner, judge, instances, path, usage=usage, resume=True)
    finally:
        store = getattr(runner, "store", None)
        if store is not None:
            store.close()
    if not report.completed:
        print(f"PAUSED: v3.2 has {report.n}/{len(manifest)} rows; resume after quota reset")
        return 2

    try:
        _verify_freeze(settings)
        payload, markdown = _aggregate(manifest)
    except (OSError, ValueError, FreezeError, ValidationArtifactError) as exc:
        print(f"STOP: cannot aggregate v3.2 tune: {exc}", file=sys.stderr)
        return 2
    destination = _rooted(settings.results_dir) / "validation/v3-phase4-tune2.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    destination.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"Aggregate report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
