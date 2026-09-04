"""Run the frozen, one-repeat v3.1 tune42 comparison.

This is a preflight unless ``--run`` is supplied. It never prints question content.
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
from llm_long_term_memory.evaluation.harness import run_eval  # noqa: E402
from llm_long_term_memory.evaluation.manifest import load_manifest  # noqa: E402
from llm_long_term_memory.evaluation.report import load_report  # noqa: E402
from llm_long_term_memory.evaluation.reproducibility import (  # noqa: E402
    FreezeError,
    FreezeRequest,
    capture_system,
    differences,
)
from llm_long_term_memory.evaluation.validation import (  # noqa: E402
    ValidationArtifactError,
    aggregate_arm,
    bind_artifacts,
    load_usage,
    paired_majority,
)

FREEZE_NAME = "v3-phase3-tune1"
CONFIG = Path("configs/v3-phase3.yaml")
TRAIN_MANIFEST = Path("results/manifests/train150.json")
TUNE_MANIFEST = Path("results/manifests/v3-reasoning-tune42.json")
STORE_NAME = "train150"
RUNS = 1
ARMS = (
    ("v2-control", "two_stage_fallback"),
    ("v3.1-reasoned", "two_stage_reasoned"),
)
PROTOCOL_PATHS = (
    Path("results/prereg-v3-phase3.md"),
    TUNE_MANIFEST,
    Path("results/archive/v3-answer-pilot/conclusion.json"),
)
EXPECTED_TYPES = {
    "temporal-reasoning": 10,
    "multi-session": 10,
    "knowledge-update": 5,
    "single-session-user": 10,
    "single-session-assistant": 6,
    "single-session-preference": 1,
}


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


def _verify_freeze(settings: Settings) -> None:
    path = REPO / "results/frozen" / FREEZE_NAME / "freeze.json"
    if not path.is_file():
        raise FreezeError(f"missing phase-3 tune freeze: {path}")
    try:
        frozen = json.loads(path.read_text(encoding="utf-8"))["system"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FreezeError(f"invalid phase-3 tune freeze: {exc}") from exc
    current = capture_system(_freeze_request(settings))
    if moved := differences(frozen, current):
        raise FreezeError(
            f"phase-3 tune configuration changed after freeze ({len(moved)}): "
            + "; ".join(moved[:5])
        )
    if current["config"]["resolved"]["decay"]["enabled"]:
        raise FreezeError("tune run refuses a store-mutating decay configuration")


def _instances(settings: Settings):
    tune = load_manifest(_rooted(TUNE_MANIFEST))
    train = load_manifest(_rooted(TRAIN_MANIFEST))
    if tune.name != "v3-reasoning-tune42" or len(tune) != 42:
        raise ValidationArtifactError("phase-3 tune requires the fixed tune42 manifest")
    if tune.variant != train.variant or not set(tune.question_ids) <= set(train.question_ids):
        raise ValidationArtifactError("tune42 must be a train150-only subset")
    all_instances = lme.load(train.variant, _rooted(settings.data_dir))
    by_id = {instance.question_id: instance for instance in all_instances}
    if missing := set(tune.question_ids) - set(by_id):
        raise ValidationArtifactError(f"tune42 has {len(missing)} unknown question(s)")
    selected = [by_id[question_id] for question_id in tune.question_ids]
    counts = Counter(instance.question_type for instance in selected)
    if dict(counts) != EXPECTED_TYPES:
        raise ValidationArtifactError(f"tune42 allocation changed: {dict(counts)}")
    return tune, selected


def _aggregate(manifest, sealed_dir: Path) -> tuple[dict, str]:
    expected = set(manifest.question_ids)
    reports = {}
    arms = []
    artifacts: list[Path] = []
    for name, _ in ARMS:
        row_path = sealed_dir / f"{name}.jsonl"
        usage_path = row_path.with_suffix(".usage.json")
        report = load_report(row_path, variant=name)
        reports[name] = [report]
        arms.append(aggregate_arm(name, [report], expected, [load_usage(usage_path)]))
        artifacts.extend((row_path, usage_path))
    paired = paired_majority(
        ARMS[0][0], reports[ARMS[0][0]], ARMS[1][0], reports[ARMS[1][0]], expected
    )
    payload = {
        "schema_version": 1,
        "manifest": {"name": manifest.name, "questions": len(manifest)},
        "arms": [arm.to_dict() for arm in arms],
        "paired": paired.to_dict(),
        "shared_store": {"name": STORE_NAME, "ingestion_reused": True},
        "sealed_artifacts": bind_artifacts(artifacts),
        "privacy": {"aggregate_only": True, "contains_question_ids_or_text": False},
    }
    types = tuple(EXPECTED_TYPES)
    lines = [
        "# v3.1 phase-3 tune42 comparison",
        "",
        "One diagnostic repeat on the inspectable tune split; this is not the dev60 gate.",
        "No question ids, text, answers or judge reasons are reported here.",
        "",
        "| arm | accuracy | temporal | multi-session | update | median context | output tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        values = arm.majority_accuracy_by_type
        lines.append(
            f"| `{arm.arm}` | {arm.majority_accuracy:.1%} | "
            f"{values.get(types[0], 0.0):.1%} | {values.get(types[1], 0.0):.1%} | "
            f"{values.get(types[2], 0.0):.1%} | {arm.median_context_tokens:,.0f} | "
            f"{arm.usage.output_tokens:,} |"
        )
    lines.extend(
        [
            "",
            f"Paired difference: {paired.accuracy_delta:+.1%}; exact p={paired.p_value:.4f}.",
            "A tune result may guide one candidate; it is not permission to inspect dev60.",
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
    print("PASS: frozen phase-3 tune42 · 2 arms x 1 repeat · resume-safe")
    if not args.run:
        print("Preflight only. Add --run after explicit provider-data permission.")
        return 0

    sealed_dir = _rooted(settings.results_dir) / "sealed/v3-phase3-tune1"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    for name, variant in ARMS:
        path = sealed_dir / f"{name}.jsonl"
        _, _, runner, judge, usage = _build(variant, str(_rooted(CONFIG)), STORE_NAME)
        runner.name = name
        try:
            report = run_eval(runner, judge, instances, path, usage=usage, resume=True)
        finally:
            store = getattr(runner, "store", None)
            if store is not None:
                store.close()
        if not report.completed:
            print(f"PAUSED: {name} has {report.n}/{len(manifest)} rows; resume after quota reset")
            return 2
        print(f"COMPLETE: {name} ({report.n}/{len(manifest)} rows)")

    try:
        _verify_freeze(settings)
        payload, markdown = _aggregate(manifest, sealed_dir)
    except (OSError, ValueError, FreezeError, ValidationArtifactError) as exc:
        print(f"STOP: cannot aggregate tune42: {exc}", file=sys.stderr)
        return 2
    destination = _rooted(settings.results_dir) / "validation/v3-phase3-tune1.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    destination.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"Aggregate report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
