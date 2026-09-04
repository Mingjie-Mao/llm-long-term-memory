"""Preflight or run the frozen v3.3 compact-evidence tune42 iteration.

The completed v2 and v3.2 rows are immutable references. This command creates only
the 42 v3.3 candidate rows, and does nothing provider-facing without ``--run``.
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

FREEZE_NAME = "v3-phase5-tune3"
CONFIG = Path("configs/v3-phase5-compact.yaml")
TRAIN_MANIFEST = Path("results/manifests/train150.json")
TUNE_MANIFEST = Path("results/manifests/v3-reasoning-tune42.json")
PHASE4_CONCLUSION = Path("results/archive/v3-phase4-tune2/conclusion.json")
PHASE4_AGGREGATE = Path("results/validation/v3-phase4-tune2.json")
PROJECTION = Path("results/analysis/v3-phase5-context-projection.json")
V2_ROWS = Path("results/sealed/v3-phase3-tune1/v2-control.jsonl")
V2_USAGE = Path("results/sealed/v3-phase3-tune1/v2-control.usage.json")
V32_ROWS = Path("results/sealed/v3-phase4-tune2/v3.2-adaptive.jsonl")
V32_USAGE = Path("results/sealed/v3-phase4-tune2/v3.2-adaptive.usage.json")
CANDIDATE_ROWS = Path("results/sealed/v3-phase5-tune3/v3.3-compact.jsonl")
STORE_NAME = "train150"
VARIANT = "two_stage_reasoned_evidence"
PROTOCOL_PATHS = (
    Path("results/prereg-v3-phase5-compact.md"),
    TUNE_MANIFEST,
    PHASE4_CONCLUSION,
    PROJECTION,
)
EXPECTED_TYPES = {
    "temporal-reasoning": 10,
    "multi-session": 10,
    "knowledge-update": 5,
    "single-session-user": 10,
    "single-session-assistant": 6,
    "single-session-preference": 1,
}
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
        raise FreezeError(f"missing v3.3 tune freeze: {path}")
    try:
        frozen = json.loads(path.read_text(encoding="utf-8"))["system"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FreezeError(f"invalid v3.3 tune freeze: {exc}") from exc
    current = capture_system(_freeze_request(settings))
    if moved := differences(frozen, current):
        raise FreezeError(f"v3.3 tune changed after freeze ({len(moved)}): " + "; ".join(moved[:5]))
    hydration = current["config"]["resolved"]["hydration"]
    if hydration != {
        "allocation": "session_fair",
        "max_tokens": 425,
        "neighbouring_sentences": 0,
    }:
        raise FreezeError("v3.3 hydration settings differ from the pre-registration")
    if current["config"]["resolved"]["decay"]["enabled"]:
        raise FreezeError("v3.3 tune refuses store-mutating decay")


def _verify_file(path: Path, record: dict) -> None:
    rooted = _rooted(path)
    if (
        not rooted.is_file()
        or rooted.stat().st_size != record.get("bytes")
        or sha256_file(rooted) != record.get("sha256")
    ):
        raise ValidationArtifactError(f"archived reference changed: {path}")


def _verify_archived_references() -> None:
    try:
        conclusion = json.loads(_rooted(PHASE4_CONCLUSION).read_text(encoding="utf-8"))
        aggregate = json.loads(_rooted(PHASE4_AGGREGATE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationArtifactError(f"cannot read phase4 archive: {exc}") from exc
    inventory = {Path(record["path"]): record for record in conclusion.get("files", [])}
    for path in (PHASE4_AGGREGATE, V2_ROWS, V2_USAGE, V32_ROWS, V32_USAGE):
        record = inventory.get(path)
        if not isinstance(record, dict):
            raise ValidationArtifactError(f"phase4 conclusion does not bind {path}")
        _verify_file(path, record)
    if aggregate.get("manifest") != {"name": "v3-reasoning-tune42", "questions": 42}:
        raise ValidationArtifactError("phase4 aggregate is not the fixed tune42 result")
    if [arm.get("arm") for arm in aggregate.get("arms", [])] != [
        "v2-control",
        "v3.2-adaptive",
    ]:
        raise ValidationArtifactError("phase4 reference arms changed")
    for path in (V2_ROWS, V2_USAGE, V32_ROWS, V32_USAGE):
        record = aggregate.get("artifacts", {}).get(path.as_posix())
        if not isinstance(record, dict):
            raise ValidationArtifactError(f"phase4 aggregate does not bind {path}")
        _verify_file(path, record)


def _instances(settings: Settings):
    tune = load_manifest(_rooted(TUNE_MANIFEST))
    train = load_manifest(_rooted(TRAIN_MANIFEST))
    if tune.name != "v3-reasoning-tune42" or len(tune) != 42:
        raise ValidationArtifactError("v3.3 tune requires the fixed tune42 manifest")
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


def _hydration_diagnostics(report: RunReport) -> dict[str, float | int]:
    applied = [row for row in report.results if row.notes.get("hydration_applied")]
    unexpected = [
        row for row in applied if row.notes.get("reasoning_kind") not in HYDRATED_OPERATIONS
    ]
    gold_coverage = [
        float(value)
        for row in applied
        if isinstance((value := row.notes.get("recall_coverage", {}).get("hydrated")), float)
    ]
    return {
        "hydrated_questions": len(applied),
        "unexpected_hydration": len(unexpected),
        "mean_hydrated_gold_session_coverage": (mean(gold_coverage) if gold_coverage else 0.0),
        "full_hydrated_gold_session_coverage_rate": (
            sum(value == 1.0 for value in gold_coverage) / len(gold_coverage)
            if gold_coverage
            else 0.0
        ),
        "mean_selected_session_coverage": mean(
            float(row.notes.get("hydration_session_coverage") or 0.0) for row in applied
        )
        if applied
        else 0.0,
        "deduplicated_source_anchors": sum(
            int(row.notes.get("hydration_redundant_anchors") or 0) for row in applied
        ),
    }


def _gate(
    v2: ArmAggregate,
    reference: ArmAggregate,
    candidate: ArmAggregate,
    reference_report: RunReport,
    candidate_report: RunReport,
) -> dict[str, object]:
    reference_diagnostics = _hydration_diagnostics(reference_report)
    candidate_diagnostics = _hydration_diagnostics(candidate_report)
    reference_output = reference.usage.by_role["answerer"]["output_tokens"]
    candidate_output = candidate.usage.by_role["answerer"]["output_tokens"]
    checks = {
        "overall_no_regression": candidate.majority_accuracy >= reference.majority_accuracy,
        "temporal_no_regression": (
            candidate.majority_accuracy_by_type["temporal-reasoning"]
            >= reference.majority_accuracy_by_type["temporal-reasoning"]
        ),
        "multi_session_no_regression": (
            candidate.majority_accuracy_by_type["multi-session"]
            >= reference.majority_accuracy_by_type["multi-session"]
        ),
        "ordinary_no_regression": (
            _slice_accuracy(candidate_report, ORDINARY_TYPES)
            >= _slice_accuracy(reference_report, ORDINARY_TYPES)
        ),
        "no_new_confident_errors": _high_confidence_wrong(candidate_report)
        <= _high_confidence_wrong(reference_report),
        "hydration_is_targeted": candidate_diagnostics["unexpected_hydration"] == 0,
        "mean_gold_coverage_no_regression": (
            candidate_diagnostics["mean_hydrated_gold_session_coverage"]
            >= reference_diagnostics["mean_hydrated_gold_session_coverage"]
        ),
        "full_gold_coverage_no_regression": (
            candidate_diagnostics["full_hydrated_gold_session_coverage_rate"]
            >= reference_diagnostics["full_hydrated_gold_session_coverage_rate"]
        ),
        "context_within_v2_limit": candidate.median_context_tokens
        <= 2.0 * v2.median_context_tokens,
        "context_lower_than_v3_2": (
            candidate.median_context_tokens < reference.median_context_tokens
        ),
        "generation_cost": candidate_output <= 1.1 * reference_output,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "accuracy_delta_vs_v3_2": (candidate.majority_accuracy - reference.majority_accuracy),
        "temporal_delta_vs_v3_2": (
            candidate.majority_accuracy_by_type["temporal-reasoning"]
            - reference.majority_accuracy_by_type["temporal-reasoning"]
        ),
        "multi_session_delta_vs_v3_2": (
            candidate.majority_accuracy_by_type["multi-session"]
            - reference.majority_accuracy_by_type["multi-session"]
        ),
        "ordinary_delta_vs_v3_2": (
            _slice_accuracy(candidate_report, ORDINARY_TYPES)
            - _slice_accuracy(reference_report, ORDINARY_TYPES)
        ),
        "context_ratio_vs_v2": candidate.median_context_tokens / v2.median_context_tokens,
        "context_ratio_vs_v3_2": (
            candidate.median_context_tokens / reference.median_context_tokens
        ),
        "answer_output_ratio_vs_v3_2": candidate_output / reference_output,
        "reference_diagnostics": reference_diagnostics,
        "candidate_diagnostics": candidate_diagnostics,
    }


def _artifact(relative: Path) -> dict[str, int | str]:
    path = _rooted(relative)
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _aggregate(manifest) -> tuple[dict, str]:
    expected = set(manifest.question_ids)
    reports = {
        "v2-control": load_report(_rooted(V2_ROWS), variant="v2-control"),
        "v3.2-adaptive": load_report(_rooted(V32_ROWS), variant="v3.2-adaptive"),
        "v3.3-compact": load_report(_rooted(CANDIDATE_ROWS), variant="v3.3-compact"),
    }
    arms = {
        "v2-control": aggregate_arm(
            "v2-control",
            [reports["v2-control"]],
            expected,
            [load_usage(_rooted(V2_USAGE))],
        ),
        "v3.2-adaptive": aggregate_arm(
            "v3.2-adaptive",
            [reports["v3.2-adaptive"]],
            expected,
            [load_usage(_rooted(V32_USAGE))],
        ),
        "v3.3-compact": aggregate_arm(
            "v3.3-compact",
            [reports["v3.3-compact"]],
            expected,
            [load_usage(_rooted(CANDIDATE_ROWS.with_suffix(".usage.json")))],
        ),
    }
    paired = paired_majority(
        "v3.2-adaptive",
        [reports["v3.2-adaptive"]],
        "v3.3-compact",
        [reports["v3.3-compact"]],
        expected,
    )
    gate = _gate(
        arms["v2-control"],
        arms["v3.2-adaptive"],
        arms["v3.3-compact"],
        reports["v3.2-adaptive"],
        reports["v3.3-compact"],
    )
    artifact_paths = (
        V2_ROWS,
        V2_USAGE,
        V32_ROWS,
        V32_USAGE,
        CANDIDATE_ROWS,
        CANDIDATE_ROWS.with_suffix(".usage.json"),
    )
    payload = {
        "schema_version": 1,
        "manifest": {"name": manifest.name, "questions": len(manifest)},
        "arms": [arms[name].to_dict() for name in arms],
        "paired": paired.to_dict(),
        "gate": gate,
        "artifacts": {path.as_posix(): _artifact(path) for path in artifact_paths},
        "privacy": {"aggregate_only": True, "contains_question_ids_or_text": False},
    }
    lines = [
        "# v3.3 compact session-fair evidence tune42",
        "",
        "v2 and v3.2 are hash-verified references and were not rerun.",
        "",
        "| arm | accuracy | temporal | multi-session | ordinary | median context | answer output |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in arms:
        arm = arms[name]
        report = reports[name]
        lines.append(
            f"| `{name}` | {arm.majority_accuracy:.1%} | "
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
            f"Paired v3.3 vs v3.2: {paired.accuracy_delta:+.1%}; exact p={paired.p_value:.4f}.",
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
        _verify_archived_references()
        _verify_freeze(settings)
    except (OSError, ValueError, FreezeError, ValidationArtifactError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    print("PASS: frozen v3.3 compact tune42 · 42 new rows · v2/v3.2 reused")
    if not args.run:
        print("Preflight only. --run requires separate permission for 42 provider results.")
        return 0

    path = _rooted(CANDIDATE_ROWS)
    path.parent.mkdir(parents=True, exist_ok=True)
    _, _, runner, judge, usage = _build(VARIANT, str(_rooted(CONFIG)), STORE_NAME)
    runner.name = "v3.3-compact"
    try:
        report = run_eval(runner, judge, instances, path, usage=usage, resume=True)
    finally:
        store = getattr(runner, "store", None)
        if store is not None:
            store.close()
    if not report.completed:
        print(f"PAUSED: v3.3 has {report.n}/{len(manifest)} rows; resume from checkpoint")
        return 2

    try:
        _verify_freeze(settings)
        payload, markdown = _aggregate(manifest)
    except (OSError, ValueError, FreezeError, ValidationArtifactError) as exc:
        print(f"STOP: cannot aggregate v3.3 tune: {exc}", file=sys.stderr)
        return 2
    destination = _rooted(settings.results_dir) / "validation/v3-phase5-tune3.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    destination.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"Aggregate report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
