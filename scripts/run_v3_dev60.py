"""One-shot, resumable, aggregate-only v3.3 dev60 validation.

Without ``--run`` this command performs only a local preflight and never loads the
question text. With ``--run`` it writes a durable ledger before loading the sealed
instances or making a provider call. Completed runs cannot be restarted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from contextlib import ExitStack
from datetime import UTC, datetime
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
    sha256_file,
)
from llm_long_term_memory.evaluation.validation import (  # noqa: E402
    ArmAggregate,
    ValidationArtifactError,
    aggregate_arm,
    bind_artifacts,
    load_usage,
    paired_majority,
    validate_bound_artifacts,
)
from llm_long_term_memory.locking import AlreadyRunning, exclusive  # noqa: E402

FREEZE_NAME = "v3-candidate-dev60"
CONFIG = Path("configs/v3-phase5-compact.yaml")
TRAIN_MANIFEST = Path("results/manifests/train150.json")
DEV_MANIFEST = Path("results/manifests/v3-reasoning-dev60.json")
PHASE5_CONCLUSION = Path("results/archive/v3-phase5-tune3/conclusion.json")
PREREG = Path("results/prereg-v3-dev60.md")
STORE_NAME = "train150"
RUNS = 3
ARMS = (
    ("v2-control", "two_stage_fallback"),
    ("v3.3-compact", "two_stage_reasoned_evidence"),
)
EXPECTED_TYPES = {
    "temporal-reasoning": 18,
    "multi-session": 18,
    "knowledge-update": 10,
    "single-session-user": 5,
    "single-session-assistant": 5,
    "single-session-preference": 4,
}
TARGET_TYPES = {"temporal-reasoning", "multi-session"}
ORDINARY_TYPES = {
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
}
SEALED_DIR = Path("results/sealed/v3-dev60")
AGGREGATE = Path("results/validation/v3-dev60.json")


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def _freeze_path() -> Path:
    return REPO / "results/frozen" / FREEZE_NAME / "freeze.json"


def _ledger_path() -> Path:
    return _freeze_path().with_name("one-shot.json")


def _freeze_request(settings: Settings) -> FreezeRequest:
    return FreezeRequest(
        repo=REPO,
        config_path=_rooted(CONFIG),
        # The shared store contains all train150 sessions. dev60 is a question
        # subset, so using it as the store manifest would falsely report the other
        # train sessions as extras. Bind the subset separately as protocol input.
        manifest_path=_rooted(TRAIN_MANIFEST),
        data_dir=_rooted(settings.data_dir),
        store_dir=_rooted(settings.store_dir),
        results_dir=_rooted(settings.results_dir),
        variants=tuple(variant for _, variant in ARMS),
        store_name=STORE_NAME,
        protocol_paths=(
            _rooted(PREREG),
            _rooted(DEV_MANIFEST),
            _rooted(PHASE5_CONCLUSION),
        ),
    )


def _verify_archive() -> None:
    try:
        conclusion = json.loads(_rooted(PHASE5_CONCLUSION).read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise FreezeError(f"invalid v3.3 conclusion archive: {exc}") from exc
    if conclusion.get("experiment_id") != "v3-phase5-tune3":
        raise FreezeError("v3.3 conclusion archive has the wrong experiment id")
    if conclusion.get("outcome") != "promoted_to_dev60":
        raise FreezeError("v3.3 tune result did not permit dev60")
    files = conclusion.get("files")
    if not isinstance(files, list) or not files:
        raise FreezeError("v3.3 conclusion has no evidence inventory")
    seen: set[str] = set()
    for record in files:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise FreezeError("v3.3 conclusion has an invalid evidence record")
        name = str(record["path"])
        path = _rooted(Path(name))
        if name in seen or not path.is_file():
            raise FreezeError(f"v3.3 evidence is duplicate or missing: {name}")
        seen.add(name)
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise FreezeError(f"v3.3 evidence changed: {name}")


def _verify_freeze(settings: Settings) -> dict:
    path = _freeze_path()
    if not path.is_file():
        raise FreezeError(f"missing frozen v3.3 dev60 candidate: {path}")
    try:
        frozen = json.loads(path.read_text(encoding="utf-8"))["system"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FreezeError(f"invalid v3.3 dev60 freeze: {exc}") from exc
    current = capture_system(_freeze_request(settings))
    if moved := differences(frozen, current):
        raise FreezeError(
            f"v3.3 dev60 system changed after freeze ({len(moved)}): " + "; ".join(moved[:5])
        )
    if current["config"]["resolved"]["hydration"] != {
        "allocation": "session_fair",
        "max_tokens": 425,
        "neighbouring_sentences": 0,
    }:
        raise FreezeError("dev60 candidate is not the promoted compact hydrator")
    if current["config"]["resolved"]["decay"]["enabled"]:
        raise FreezeError("dev60 refuses a store-mutating decay configuration")
    return frozen


def _manifest_only():
    dev = load_manifest(_rooted(DEV_MANIFEST))
    train = load_manifest(_rooted(TRAIN_MANIFEST))
    if dev.name != "v3-reasoning-dev60" or len(dev) != 60:
        raise ValidationArtifactError("dev60 requires the fixed 60-question manifest")
    if dev.variant != train.variant or not set(dev.question_ids) <= set(train.question_ids):
        raise ValidationArtifactError("dev60 must remain a train150-only subset")
    return dev


def _instances(settings: Settings, manifest):
    every = lme.load(manifest.variant, _rooted(settings.data_dir))
    by_id = {instance.question_id: instance for instance in every}
    if missing := set(manifest.question_ids) - set(by_id):
        raise ValidationArtifactError(f"dev60 has {len(missing)} unknown question(s)")
    selected = [by_id[question_id] for question_id in manifest.question_ids]
    counts = dict(Counter(instance.question_type for instance in selected))
    if counts != EXPECTED_TYPES:
        raise ValidationArtifactError("dev60 type allocation changed")
    return selected


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _artifact_paths(sealed_dir: Path, destination: Path) -> list[Path]:
    paths = [destination, destination.with_suffix(".md")]
    for name, _ in ARMS:
        for number in range(1, RUNS + 1):
            rows = sealed_dir / f"{name}.rep{number}.jsonl"
            paths.extend((rows, rows.with_suffix(".usage.json")))
    return paths


def _new_ledger(freeze_path: Path) -> dict:
    row_names = [f"{name}.rep{number}" for name, _ in ARMS for number in range(1, RUNS + 1)]
    return {
        "schema_version": 1,
        "status": "started",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": None,
        "freeze_sha256": sha256_file(freeze_path),
        "arms": [{"name": name, "variant": variant} for name, variant in ARMS],
        "runs_per_arm": RUNS,
        "questions": 60,
        "expected_rows": 60 * RUNS * len(ARMS),
        "rows": {name: 0 for name in row_names},
        "rule": "one resumable dev60 execution; never fresh, repeated, or tuned",
    }


def _validate_ledger(ledger: dict, freeze_path: Path) -> None:
    expected_arms = [{"name": name, "variant": variant} for name, variant in ARMS]
    if ledger.get("schema_version") != 1:
        raise FreezeError("dev60 ledger has an unsupported schema")
    if ledger.get("freeze_sha256") != sha256_file(freeze_path):
        raise FreezeError("dev60 ledger does not match the frozen candidate")
    if ledger.get("arms") != expected_arms or ledger.get("runs_per_arm") != RUNS:
        raise FreezeError("dev60 ledger does not match the registered arms and repeats")
    if ledger.get("questions") != 60 or ledger.get("expected_rows") != 360:
        raise FreezeError("dev60 ledger has the wrong fixed size")


def _prepare_ledger(
    ledger_path: Path,
    freeze_path: Path,
    artifacts: list[Path],
    destination: Path,
    *,
    create: bool,
    forbid_complete: bool,
) -> dict:
    if ledger_path.exists():
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            raise FreezeError(f"invalid dev60 ledger: {exc}") from exc
        _validate_ledger(ledger, freeze_path)
        if ledger.get("status") not in {"started", "complete"}:
            raise FreezeError("dev60 ledger has an invalid status")
    else:
        present = sum(path.exists() for path in artifacts)
        if present:
            raise FreezeError(f"{present} dev60 artifact(s) exist without the one-shot ledger")
        ledger = _new_ledger(freeze_path)
        if create:
            _atomic_json(ledger_path, ledger)

    if ledger.get("status") == "complete":
        if forbid_complete:
            raise FreezeError("dev60 is already complete; a second run is forbidden")
        if not destination.is_file() or ledger.get("aggregate_sha256") != sha256_file(destination):
            raise FreezeError("completed dev60 ledger does not match its aggregate")
        try:
            aggregate = json.loads(destination.read_text(encoding="utf-8"))
            expected = {
                filename
                for name, _ in ARMS
                for number in range(1, RUNS + 1)
                for filename in (
                    f"{name}.rep{number}.jsonl",
                    f"{name}.rep{number}.usage.json",
                )
            }
            validate_bound_artifacts(aggregate, _rooted(SEALED_DIR), expected)
        except (OSError, TypeError, ValidationArtifactError, json.JSONDecodeError) as exc:
            raise FreezeError(f"completed dev60 artifacts are invalid: {exc}") from exc
    return ledger


def _group(reports: list[RunReport], expected: set[str]) -> dict[str, list[QuestionResult]]:
    grouped: dict[str, list[QuestionResult]] = {question_id: [] for question_id in expected}
    for report in reports:
        seen: set[str] = set()
        for row in report.results:
            if row.question_id not in grouped or row.question_id in seen:
                raise ValidationArtifactError("dev60 rows differ from the frozen manifest")
            grouped[row.question_id].append(row)
            seen.add(row.question_id)
    if any(len(rows) != RUNS for rows in grouped.values()):
        raise ValidationArtifactError("dev60 requires three rows per question and arm")
    return grouped


def _majority(rows: list[QuestionResult]) -> bool:
    return sum(row.correct for row in rows) >= 2


def _slice_metrics(reports: list[RunReport], expected: set[str]) -> dict[str, float | int]:
    grouped = _group(reports, expected)
    totals = Counter()
    correct = Counter()
    high_confidence_wrong = 0
    for rows in grouped.values():
        question_type = rows[0].question_type
        totals[question_type] += 1
        correct[question_type] += int(_majority(rows))
        confidence = Counter(row.notes.get("answer_confidence") for row in rows).most_common(1)[0][
            0
        ]
        if not _majority(rows) and confidence == "high":
            high_confidence_wrong += 1

    def accuracy(types: set[str]) -> float:
        denominator = sum(totals[name] for name in types)
        return sum(correct[name] for name in types) / denominator

    return {
        "target_accuracy": accuracy(TARGET_TYPES),
        "temporal_accuracy": accuracy({"temporal-reasoning"}),
        "multi_session_accuracy": accuracy({"multi-session"}),
        "knowledge_update_accuracy": accuracy({"knowledge-update"}),
        "ordinary_accuracy": accuracy(ORDINARY_TYPES),
        "high_confidence_majority_wrong": high_confidence_wrong,
    }


def _gate(
    baseline: ArmAggregate,
    candidate: ArmAggregate,
    baseline_slices: dict[str, float | int],
    candidate_slices: dict[str, float | int],
) -> dict[str, object]:
    target_delta = float(candidate_slices["target_accuracy"]) - float(
        baseline_slices["target_accuracy"]
    )
    selected_baseline = baseline.recall_by_stage.get("selected")
    selected_candidate = candidate.recall_by_stage.get("selected")
    recall_ok = (
        selected_baseline is not None
        and selected_candidate is not None
        and selected_candidate >= selected_baseline
    )
    context_ratio = (
        candidate.median_context_tokens / baseline.median_context_tokens
        if baseline.median_context_tokens
        else float("inf")
    )
    baseline_output = baseline.usage.by_role.get("answerer", {}).get("output_tokens", 0)
    candidate_output = candidate.usage.by_role.get("answerer", {}).get("output_tokens", 0)
    output_ratio = candidate_output / baseline_output if baseline_output else float("inf")
    checks = {
        "target_gain": target_delta >= 0.05,
        "temporal_no_regression": (
            candidate_slices["temporal_accuracy"] >= baseline_slices["temporal_accuracy"]
        ),
        "multi_session_no_regression": (
            candidate_slices["multi_session_accuracy"] >= baseline_slices["multi_session_accuracy"]
        ),
        "knowledge_update_no_regression": (
            candidate_slices["knowledge_update_accuracy"]
            >= baseline_slices["knowledge_update_accuracy"]
        ),
        "ordinary_no_regression": (
            candidate_slices["ordinary_accuracy"] >= baseline_slices["ordinary_accuracy"]
        ),
        "no_new_confident_errors": (
            candidate_slices["high_confidence_majority_wrong"]
            <= baseline_slices["high_confidence_majority_wrong"]
        ),
        "selected_recall_no_regression": recall_ok,
        "context_within_2x_v2": context_ratio <= 2.0,
        "generation_cost": output_ratio <= 1.5,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "target_accuracy_delta": target_delta,
        "context_ratio": context_ratio,
        "answer_output_ratio": output_ratio,
        "baseline_slices": baseline_slices,
        "candidate_slices": candidate_slices,
    }


def _aggregate(manifest, sealed_dir: Path) -> tuple[dict, str]:
    expected = set(manifest.question_ids)
    reports_by_name: dict[str, list[RunReport]] = {}
    aggregates: list[ArmAggregate] = []
    artifact_paths: list[Path] = []
    for name, _ in ARMS:
        rows = [sealed_dir / f"{name}.rep{number}.jsonl" for number in range(1, RUNS + 1)]
        usages = [path.with_suffix(".usage.json") for path in rows]
        reports = [
            load_report(path, variant=f"{name}.rep{number}") for number, path in enumerate(rows, 1)
        ]
        reports_by_name[name] = reports
        aggregates.append(
            aggregate_arm(name, reports, expected, [load_usage(path) for path in usages])
        )
        artifact_paths.extend(item for pair in zip(rows, usages, strict=True) for item in pair)

    baseline_slices = _slice_metrics(reports_by_name[ARMS[0][0]], expected)
    candidate_slices = _slice_metrics(reports_by_name[ARMS[1][0]], expected)
    gate = _gate(aggregates[0], aggregates[1], baseline_slices, candidate_slices)
    paired = paired_majority(
        ARMS[0][0],
        reports_by_name[ARMS[0][0]],
        ARMS[1][0],
        reports_by_name[ARMS[1][0]],
        expected,
    )
    payload = {
        "schema_version": 1,
        "protocol": "one frozen dev60 execution; two arms x three repeats",
        "manifest": {"name": manifest.name, "questions": len(manifest)},
        "arms": [arm.to_dict() for arm in aggregates],
        "paired_majority": paired.to_dict(),
        "gate": gate,
        "shared_store": {"name": STORE_NAME, "ingestion_reused": True},
        "sealed_artifacts": bind_artifacts(artifact_paths),
        "privacy": {"aggregate_only": True, "contains_question_ids_or_text": False},
    }
    lines = [
        "# Frozen v3.3 dev60 validation",
        "",
        "Two frozen policies used the same train150 store; individual content is sealed.",
        "",
        "| arm | runs | majority accuracy | target | temporal | multi-session | "
        "update | ordinary | selected recall | median context | answer output |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, slices in zip(aggregates, (baseline_slices, candidate_slices), strict=True):
        recall = arm.recall_by_stage.get("selected")
        recall_text = "n/a" if recall is None else f"{recall:.1%}"
        lines.append(
            f"| `{arm.arm}` | {arm.runs} | {arm.majority_accuracy:.1%} | "
            f"{slices['target_accuracy']:.1%} | {slices['temporal_accuracy']:.1%} | "
            f"{slices['multi_session_accuracy']:.1%} | "
            f"{slices['knowledge_update_accuracy']:.1%} | "
            f"{slices['ordinary_accuracy']:.1%} | {recall_text} | "
            f"{arm.median_context_tokens:,.0f} | "
            f"{arm.usage.by_role['answerer']['output_tokens']:,} |"
        )
    lines.extend(
        [
            "",
            f"Registered dev60 gate: **{'PASS' if gate['passed'] else 'STOP'}**",
            "",
            "| check | pass |",
            "|---|---:|",
            *[
                f"| `{name}` | {'yes' if passed else 'no'} |"
                for name, passed in gate["checks"].items()
            ],
            "",
            f"Paired majority delta: {paired.accuracy_delta:+.1%}; exact p={paired.p_value:.4f}.",
        ]
    )
    return payload, "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="spend the registered dev60 shot")
    args = parser.parse_args()
    settings = Settings()
    try:
        _verify_archive()
        _verify_freeze(settings)
        manifest = _manifest_only()
    except (OSError, ValueError, FreezeError, ValidationArtifactError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2

    sealed_dir = _rooted(SEALED_DIR)
    destination = _rooted(AGGREGATE)
    freeze_path = _freeze_path()
    ledger_path = _ledger_path()
    artifacts = _artifact_paths(sealed_dir, destination)
    try:
        ledger = _prepare_ledger(
            ledger_path,
            freeze_path,
            artifacts,
            destination,
            create=False,
            forbid_complete=args.run,
        )
    except FreezeError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2

    print("PASS: frozen v3.3 dev60 · 60 questions · 2 arms x 3 repeats · rows sealed")
    if not args.run:
        state = "already complete" if ledger.get("status") == "complete" else "not started"
        print(
            f"Preflight only ({state}). --run requires explicit permission to send "
            "dev60 text and retrieved context."
        )
        return 0

    lock_stack = ExitStack()
    try:
        lock_stack.enter_context(exclusive(ledger_path, what="dev60 validation"))
    except AlreadyRunning as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    with lock_stack:
        try:
            ledger = _prepare_ledger(
                ledger_path,
                freeze_path,
                artifacts,
                destination,
                create=True,
                forbid_complete=True,
            )
            instances = _instances(settings, manifest)
        except (OSError, ValueError, FreezeError, ValidationArtifactError) as exc:
            print(f"STOP: {exc}", file=sys.stderr)
            return 2

        sealed_dir.mkdir(parents=True, exist_ok=True)
        for name, variant in ARMS:
            for number in range(1, RUNS + 1):
                run_name = f"{name}.rep{number}"
                path = sealed_dir / f"{run_name}.jsonl"
                _, _, runner, judge, usage = _build(variant, str(_rooted(CONFIG)), STORE_NAME)
                runner.name = run_name
                try:
                    report = run_eval(runner, judge, instances, path, usage=usage, resume=True)
                finally:
                    store = getattr(runner, "store", None)
                    if store is not None:
                        store.close()
                ledger["rows"][run_name] = report.n
                _atomic_json(ledger_path, ledger)
                if not report.completed:
                    print(
                        f"PAUSED: {run_name} has {report.n}/{len(manifest)} rows; "
                        "resume after quota reset. No scores exposed."
                    )
                    return 2
                print(f"COMPLETE: {run_name} ({report.n}/{len(manifest)} rows sealed)")

        try:
            _verify_archive()
            _verify_freeze(settings)
            payload, markdown = _aggregate(manifest, sealed_dir)
        except (OSError, ValueError, FreezeError, ValidationArtifactError) as exc:
            print(f"STOP: cannot aggregate dev60: {exc}", file=sys.stderr)
            return 2
        destination.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(destination, payload)
        destination.with_suffix(".md").write_text(markdown, encoding="utf-8")
        ledger["status"] = "complete"
        ledger["completed_at_utc"] = datetime.now(UTC).isoformat()
        ledger["aggregate_sha256"] = sha256_file(destination)
        _atomic_json(ledger_path, ledger)

    print(markdown, end="")
    print(f"Aggregate report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
