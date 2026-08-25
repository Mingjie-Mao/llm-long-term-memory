"""Resume a frozen dev100 experiment while exposing aggregate results only.

The command is a preflight unless ``--run`` is present.  Raw rows are written under
``results/sealed/dev100`` solely for checkpoint/resume and audit.  Progress output
contains counts, never question ids, answers, correctness, or failure details.

The first ``--arm`` is the paired baseline.  The arm list, source, config, data and
completed store must already be captured by ``scripts/freeze_v2.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
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
    formal_protocol_paths,
    sha256_file,
)
from llm_long_term_memory.evaluation.validation import (  # noqa: E402
    ValidationArtifactError,
    aggregate_arm,
    bind_artifacts,
    load_usage,
    paired_majority,
    render_markdown,
    select_dev_candidate,
)

_SAFE_ARM = re.compile(r"^[A-Za-z0-9_-]+$")
REGISTERED_DEV_ARMS = (
    ("flat20", "two_stage_fallback"),
    ("coherent-auto", "two_stage_coherent"),
    ("coherent-oracle", "two_stage_coherent_oracle"),
)


def _arm(raw: str) -> tuple[str, str]:
    try:
        name, variant = (part.strip() for part in raw.split("=", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("arm must be NAME=VARIANT") from exc
    if not _SAFE_ARM.fullmatch(name) or not variant:
        raise argparse.ArgumentTypeError(
            "arm name may contain only letters, digits, underscore and hyphen"
        )
    return name, variant


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def _validate_protocol(args) -> None:
    if args.runs != 3:
        raise ValidationArtifactError("the pre-registered dev100 protocol requires exactly 3 runs")
    if tuple(args.arm) != REGISTERED_DEV_ARMS:
        rendered = ", ".join(f"{name}={variant}" for name, variant in REGISTERED_DEV_ARMS)
        raise ValidationArtifactError(
            f"dev100 arms must exactly match the pre-registration: {rendered}"
        )
    canonical_manifest = REPO / "results" / "manifests" / "dev100.json"
    if _rooted(args.manifest).resolve() != canonical_manifest.resolve():
        raise ValidationArtifactError(
            "dev100 validation requires results/manifests/dev100.json exactly"
        )
    if args.store_name != "dev100":
        raise ValidationArtifactError("dev100 validation requires the frozen store name 'dev100'")
    if args.freeze_name != "v2-candidate":
        raise ValidationArtifactError(
            "dev100 validation requires the registered freeze name 'v2-candidate'"
        )


def _load_and_verify_freeze(
    name: str,
    config: Path,
    manifest: Path,
    store_name: str,
    variants: tuple[str, ...],
    settings: Settings,
) -> dict:
    path = REPO / "results" / "frozen" / name / "freeze.json"
    if not path.exists():
        raise FreezeError(f"candidate freeze does not exist: {path}")
    try:
        frozen = json.loads(path.read_text(encoding="utf-8"))["system"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FreezeError(f"invalid candidate freeze: {exc}") from exc
    request = FreezeRequest(
        repo=REPO,
        config_path=_rooted(config),
        manifest_path=_rooted(manifest),
        data_dir=_rooted(settings.data_dir),
        store_dir=_rooted(settings.store_dir),
        results_dir=_rooted(settings.results_dir),
        variants=variants,
        store_name=store_name,
        protocol_paths=formal_protocol_paths(REPO, "dev100"),
    )
    current = capture_system(request)
    moved = differences(frozen, current)
    if moved:
        summary = "; ".join(moved[:5])
        raise FreezeError(
            f"system changed after candidate freeze ({len(moved)} differences): {summary}"
        )
    if frozen["config"]["resolved"]["decay"]["enabled"]:
        raise FreezeError(
            "validation refuses decay/reinforcement because it mutates the shared store "
            "between repeats"
        )
    return frozen


def _instances(manifest_path: Path, settings: Settings):
    manifest = load_manifest(manifest_path)
    if manifest.name.lower() != "dev100":
        raise ValidationArtifactError(
            "this validation runner accepts only the frozen dev100 manifest"
        )
    if len(manifest) != 100:
        raise ValidationArtifactError(
            f"the frozen dev100 manifest must contain exactly 100 questions, got {len(manifest)}"
        )
    every = lme.load(manifest.variant, _rooted(settings.data_dir))
    by_id = {instance.question_id: instance for instance in every}
    missing = set(manifest.question_ids) - set(by_id)
    if missing:
        raise ValidationArtifactError(
            f"manifest has {len(missing)} question(s) outside its declared dataset"
        )
    return manifest, [by_id[qid] for qid in manifest.question_ids]


def _aggregate(
    manifest,
    arms: list[tuple[str, str]],
    runs: int,
    sealed_dir: Path,
    ingest_usage_path: Path,
) -> tuple[dict, str, dict]:
    expected_ids = set(manifest.question_ids)
    reports_by_arm = {}
    aggregates = []
    sealed_paths: list[Path] = []
    for name, _ in arms:
        paths = [sealed_dir / f"{name}.rep{number}.jsonl" for number in range(1, runs + 1)]
        usage_paths = [path.with_suffix(".usage.json") for path in paths]
        sealed_paths.extend(item for pair in zip(paths, usage_paths, strict=True) for item in pair)
        reports = [load_report(path, variant=name) for path in paths]
        reports_by_arm[name] = reports
        aggregates.append(
            aggregate_arm(name, reports, expected_ids, [load_usage(path) for path in usage_paths])
        )
    baseline = arms[0][0]
    comparisons = [
        paired_majority(
            baseline,
            reports_by_arm[baseline],
            name,
            reports_by_arm[name],
            expected_ids,
        )
        for name, _ in arms[1:]
    ]
    shared_ingestion = load_usage(ingest_usage_path)
    decision = select_dev_candidate(aggregates).to_dict()
    payload = {
        "schema_version": 2,
        "manifest": {"name": manifest.name, "questions": len(manifest)},
        "arms": [item.to_dict() for item in aggregates],
        "paired_majority": [item.to_dict() for item in comparisons],
        "shared_ingestion_usage": shared_ingestion.to_dict(),
        "sealed_artifacts": bind_artifacts(sealed_paths),
        "registered_decision": decision,
        "privacy": {
            "aggregate_only": True,
            "contains_question_ids_or_text": False,
        },
    }
    markdown = render_markdown(aggregates, comparisons, shared_ingestion)
    markdown += (
        "\n## Registered dev100 decision\n\n"
        f"- selected arm: `{decision['selected_arm']}`\n"
        f"- selected product variant: `{decision['selected_variant']}`\n"
        f"- diagnostic: `{decision['diagnostic']}`\n"
        f"- coherent accuracy gate: {decision['accuracy_gate_passed']}\n"
        f"- coherent context ratio: {decision['coherent_context_ratio']:.2f} "
        f"(gate passed: {decision['context_gate_passed']})\n"
    )
    return payload, markdown, decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="spend API quota after all gates pass")
    parser.add_argument("--freeze-name", default="v2-candidate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("results/manifests/dev100.json"))
    parser.add_argument("--store-name", default="dev100")
    parser.add_argument("--arm", type=_arm, action="append", required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    try:
        _validate_protocol(args)
    except ValidationArtifactError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2

    settings = Settings()
    manifest_path = _rooted(args.manifest)
    try:
        manifest, instances = _instances(manifest_path, settings)
        _load_and_verify_freeze(
            args.freeze_name,
            args.config,
            args.manifest,
            args.store_name,
            tuple(variant for _, variant in args.arm),
            settings,
        )
    except (OSError, ValueError, FreezeError, ValidationArtifactError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2

    sealed_dir = _rooted(settings.results_dir) / "sealed" / "dev100"
    print(
        f"PASS: frozen dev100 candidate · {len(manifest)} questions · "
        f"{len(args.arm)} arms x {args.runs} repeats · individual rows sealed"
    )
    if not args.run:
        print("Preflight only. Re-run with --run to start or resume aggregate-only validation.")
        return 0

    sealed_dir.mkdir(parents=True, exist_ok=True)
    for name, variant in args.arm:
        for number in range(1, args.runs + 1):
            path = sealed_dir / f"{name}.rep{number}.jsonl"
            cfg, _, runner, judge, usage = _build(
                variant, str(_rooted(args.config)), args.store_name
            )
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
                    "resume after quota reset. No scores were exposed."
                )
                return 2
            print(f"COMPLETE: {name} repeat {number} ({report.n}/{len(manifest)} rows sealed)")
            # cfg is intentionally constructed and thereby validated even though
            # the freeze, not this local name, is the report's provenance.
            del cfg

    try:
        _load_and_verify_freeze(
            args.freeze_name,
            args.config,
            args.manifest,
            args.store_name,
            tuple(variant for _, variant in args.arm),
            settings,
        )
        ingest_usage_path = (
            _rooted(settings.results_dir) / "raw" / f"{args.store_name}.ingest.usage.json"
        )
        if not ingest_usage_path.exists():
            raise ValidationArtifactError(
                "shared ingestion usage is missing, so total experiment cost cannot be reported"
            )
        payload, markdown, decision = _aggregate(
            manifest, args.arm, args.runs, sealed_dir, ingest_usage_path
        )
    except (OSError, ValueError, FreezeError, ValidationArtifactError) as exc:
        print(f"STOP: cannot build aggregate report: {exc}", file=sys.stderr)
        return 2
    destination = _rooted(settings.results_dir) / "validation" / "dev100-aggregate.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    destination.with_suffix(".md").write_text(markdown, encoding="utf-8")
    decision_path = destination.with_name("dev100-decision.json")
    decision_record = {
        "schema_version": 2,
        "rule": (
            "select coherent-auto only when majority accuracy is not below flat20 and "
            "median context is at most 1.50x flat20"
        ),
        **decision,
        "aggregate_path": str(destination.relative_to(REPO)),
        "aggregate_sha256": sha256_file(destination),
    }
    decision_path.write_text(
        json.dumps(decision_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(markdown, end="")
    print(f"Aggregate report: {destination}")
    print(f"Registered decision: {decision_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
