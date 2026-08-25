"""One-shot, resumable, aggregate-only execution of the frozen test100 protocol.

"One shot" means one run of each arm named in the final freeze. The first arm is
the dev-selected product candidate and must be named ``v2``; later arms are the
pre-registered baselines. Quota pauses do
not create new runs: every arm resumes its same JSONL checkpoint.  A durable ledger
is written *before* the first API call and refuses a second completed execution.

The test store must already have been ingested under a pre-ingest freeze, then
captured with ``scripts/freeze_v2.py`` as ``v2-final``.  This runner never prints
question ids, questions, answers, per-question correctness, or judge reasons.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import ExitStack
from datetime import UTC, datetime
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
    load_dev_decision,
    load_usage,
    paired_majority,
    render_final_markdown,
    single_run_arm_dict,
    validate_bound_artifacts,
)
from llm_long_term_memory.locking import AlreadyRunning, exclusive  # noqa: E402

_SAFE_ARM = re.compile(r"^[A-Za-z0-9_-]+$")
_COMMON_FINAL_ARMS = (
    ("full_context", "full_context"),
    ("naive_rag", "naive_rag"),
)


def _arm(raw: str) -> tuple[str, str]:
    try:
        name, variant = (part.strip() for part in raw.split("=", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("arm must be NAME=VARIANT") from exc
    if not _SAFE_ARM.fullmatch(name) or not variant:
        raise argparse.ArgumentTypeError("invalid arm name or empty variant")
    return name, variant


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def _validate_final_arms(arms: list[tuple[str, str]]) -> None:
    """Require exactly the arms and order recorded before dev/test access."""
    if not arms or arms[0][0] != "v2":
        raise FreezeError("the first final arm must be the frozen product candidate named 'v2'")
    candidate = arms[0][1]
    if candidate not in {"two_stage_coherent", "two_stage_fallback"}:
        raise FreezeError("v2 must be the coherent or flat-memory candidate selected on dev100")
    expected = [("v2", candidate), *_COMMON_FINAL_ARMS]
    if candidate == "two_stage_coherent":
        expected.append(("flat_memory_fallback", "two_stage_fallback"))
    if arms != expected:
        rendered = ", ".join(f"{name}={variant}" for name, variant in expected)
        raise FreezeError(f"final arms must exactly match the pre-registration: {rendered}")


def _verify_freeze(args, settings: Settings) -> tuple[dict, Path]:
    freeze_path = REPO / "results" / "frozen" / args.freeze_name / "freeze.json"
    if not freeze_path.exists():
        raise FreezeError(f"final freeze does not exist: {freeze_path}")
    try:
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))["system"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FreezeError(f"invalid final freeze: {exc}") from exc
    request = FreezeRequest(
        repo=REPO,
        config_path=_rooted(args.config),
        manifest_path=_rooted(args.manifest),
        data_dir=_rooted(settings.data_dir),
        store_dir=_rooted(settings.store_dir),
        results_dir=_rooted(settings.results_dir),
        variants=tuple(variant for _, variant in args.arm),
        store_name=args.store_name,
        protocol_paths=formal_protocol_paths(REPO, "test100"),
    )
    moved = differences(frozen, capture_system(request))
    if moved:
        raise FreezeError(
            f"system changed after final freeze ({len(moved)} differences): " + "; ".join(moved[:5])
        )
    if frozen["store"] is None:
        raise FreezeError(
            "final freeze is only a pre-ingest lock; the complete test store is not frozen"
        )
    if frozen["config"]["resolved"]["decay"]["enabled"]:
        raise FreezeError("final evaluation refuses a store that mutates between frozen arms")
    return frozen, freeze_path


def _instances(manifest_path: Path, settings: Settings):
    manifest = load_manifest(manifest_path)
    if manifest.name.lower() != "test100":
        raise ValidationArtifactError("the final runner accepts only the sealed test100 manifest")
    if len(manifest) != 100:
        raise ValidationArtifactError(
            f"the sealed test100 manifest must contain exactly 100 questions, got {len(manifest)}"
        )
    every = lme.load(manifest.variant, _rooted(settings.data_dir))
    by_id = {instance.question_id: instance for instance in every}
    missing = set(manifest.question_ids) - set(by_id)
    if missing:
        raise ValidationArtifactError(
            f"manifest has {len(missing)} question(s) outside its declared dataset"
        )
    return manifest, [by_id[qid] for qid in manifest.question_ids]


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _new_ledger(args, freeze_path: Path) -> dict:
    return {
        "schema_version": 2,
        "status": "started",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": None,
        "freeze_sha256": sha256_file(freeze_path),
        "arms": [{"name": name, "variant": variant} for name, variant in args.arm],
        "rows": {name: 0 for name, _ in args.arm},
        "rule": "one resumable run per pre-registered arm; never fresh/repeated",
    }


def _validate_ledger(ledger: dict, args, freeze_path: Path) -> None:
    expected = [{"name": name, "variant": variant} for name, variant in args.arm]
    if ledger.get("freeze_sha256") != sha256_file(freeze_path) or ledger.get("arms") != expected:
        raise FreezeError("one-shot ledger does not match this freeze and arm list")


def _final_artifacts(sealed_dir: Path, destination: Path, args) -> list[Path]:
    artifacts = [destination, destination.with_suffix(".md")]
    for name, _ in args.arm:
        rows = sealed_dir / f"{name}.jsonl"
        artifacts.extend((rows, rows.with_suffix(".usage.json")))
    return artifacts


def _prepare_ledger(
    ledger_path: Path,
    args,
    freeze_path: Path,
    artifacts: list[Path],
    aggregate_path: Path,
    *,
    create: bool,
    forbid_complete: bool,
) -> dict:
    """Read or create the ledger; callers repeat this after taking the lock."""
    if ledger_path.exists():
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            raise FreezeError(f"invalid one-shot ledger: {exc}") from exc
        _validate_ledger(ledger, args, freeze_path)
        if ledger.get("status") not in {"started", "complete"}:
            raise FreezeError("one-shot ledger has an invalid status")
    else:
        present = sum(path.exists() for path in artifacts)
        if present:
            raise FreezeError(
                f"{present} final-test artifact(s) exist without the durable one-shot ledger"
            )
        ledger = _new_ledger(args, freeze_path)
        if create:
            _atomic_json(ledger_path, ledger)

    if ledger.get("status") == "complete":
        if forbid_complete:
            raise FreezeError("test100 protocol is already complete; a second run is forbidden")
        if not aggregate_path.exists() or ledger.get("aggregate_sha256") != sha256_file(
            aggregate_path
        ):
            raise FreezeError("completed one-shot ledger does not match its aggregate report")
        try:
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            expected_names = {
                filename
                for name, _ in args.arm
                for filename in (f"{name}.jsonl", f"{name}.usage.json")
            }
            validate_bound_artifacts(
                aggregate,
                aggregate_path.parent.parent / "sealed" / "test100",
                expected_names,
            )
        except (OSError, TypeError, ValidationArtifactError, json.JSONDecodeError) as exc:
            raise FreezeError(f"completed final artifacts are invalid: {exc}") from exc
    return ledger


def _aggregate(manifest, arms, sealed_dir: Path, ingest_usage_path: Path) -> tuple[dict, str]:
    expected_ids = set(manifest.question_ids)
    reports = {}
    aggregates = []
    sealed_paths: list[Path] = []
    for name, _ in arms:
        path = sealed_dir / f"{name}.jsonl"
        report = load_report(path, variant=name)
        sealed_paths.extend((path, path.with_suffix(".usage.json")))
        reports[name] = [report]
        aggregates.append(
            aggregate_arm(
                name,
                [report],
                expected_ids,
                [load_usage(path.with_suffix(".usage.json"))],
            )
        )
    candidate = arms[0][0]
    comparisons = [
        paired_majority(name, reports[name], candidate, reports[candidate], expected_ids)
        for name, _ in arms[1:]
    ]
    shared_ingestion = load_usage(ingest_usage_path)
    payload = {
        "schema_version": 1,
        "protocol": "single frozen run per pre-registered arm",
        "manifest": {"name": "test100", "questions": len(manifest)},
        "arms": [single_run_arm_dict(item) for item in aggregates],
        "paired_single_run": [item.to_dict() for item in comparisons],
        "shared_ingestion_usage": shared_ingestion.to_dict(),
        "sealed_artifacts": bind_artifacts(sealed_paths),
        "privacy": {"aggregate_only": True, "contains_question_ids_or_text": False},
    }
    return payload, render_final_markdown(aggregates, comparisons, shared_ingestion)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="spend the registered one shot")
    parser.add_argument("--freeze-name", default="v2-final")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("results/manifests/test100.json"))
    parser.add_argument("--store-name", default="test100")
    parser.add_argument("--arm", type=_arm, action="append", required=True)
    args = parser.parse_args()
    canonical_manifest = REPO / "results" / "manifests" / "test100.json"
    try:
        _validate_final_arms(args.arm)
        if _rooted(args.manifest).resolve() != canonical_manifest.resolve():
            raise FreezeError("the final run requires results/manifests/test100.json exactly")
        if args.store_name != "test100":
            raise FreezeError("the final run requires the frozen store name 'test100'")
        if args.freeze_name != "v2-final":
            raise FreezeError("the final run requires the registered freeze name 'v2-final'")
        validation_dir = REPO / "results" / "validation"
        decision = load_dev_decision(
            validation_dir / "dev100-decision.json",
            validation_dir / "dev100-aggregate.json",
        )
        if args.arm[0][1] != decision["selected_variant"]:
            raise FreezeError("the v2 arm does not match the registered dev100 decision")
    except FreezeError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    except ValidationArtifactError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2

    settings = Settings()
    try:
        manifest, instances = _instances(_rooted(args.manifest), settings)
        _, freeze_path = _verify_freeze(args, settings)
    except (OSError, ValueError, FreezeError, ValidationArtifactError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2

    sealed_dir = _rooted(settings.results_dir) / "sealed" / "test100"
    ledger_path = freeze_path.with_name("one-shot.json")
    destination = _rooted(settings.results_dir) / "final" / "test100-aggregate.json"
    artifacts = _final_artifacts(sealed_dir, destination, args)
    try:
        ledger = _prepare_ledger(
            ledger_path,
            args,
            freeze_path,
            artifacts,
            destination,
            create=False,
            forbid_complete=args.run,
        )
    except FreezeError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2

    print(
        f"PASS: frozen test100 · {len(manifest)} questions · {len(args.arm)} registered arms · "
        "individual rows sealed"
    )
    if not args.run:
        state = "already complete" if ledger.get("status") == "complete" else "not started"
        print(f"Preflight only ({state}). Add --run only when ready to spend the one shot.")
        return 0

    lock_stack = ExitStack()
    try:
        lock_stack.enter_context(exclusive(ledger_path, what="final test"))
    except AlreadyRunning as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    sealed_dir.mkdir(parents=True, exist_ok=True)
    with lock_stack:
        try:
            ledger = _prepare_ledger(
                ledger_path,
                args,
                freeze_path,
                artifacts,
                destination,
                create=True,
                forbid_complete=True,
            )
        except FreezeError as exc:
            print(f"STOP: {exc}", file=sys.stderr)
            return 2
        for name, variant in args.arm:
            path = sealed_dir / f"{name}.jsonl"
            _, _, runner, judge, usage = _build(variant, str(_rooted(args.config)), args.store_name)
            runner.name = name
            try:
                report = run_eval(runner, judge, instances, path, usage=usage, resume=True)
            finally:
                store = getattr(runner, "store", None)
                if store is not None:
                    store.close()
            ledger["rows"][name] = report.n
            _atomic_json(ledger_path, ledger)
            if not report.completed:
                print(
                    f"PAUSED: {name} has {report.n}/{len(manifest)} rows; resume the same "
                    "one-shot run after quota reset. No score was exposed."
                )
                return 2
            print(f"COMPLETE: {name} ({report.n}/{len(manifest)} rows sealed)")

        try:
            _verify_freeze(args, settings)
            ingest_usage_path = (
                _rooted(settings.results_dir) / "raw" / f"{args.store_name}.ingest.usage.json"
            )
            if not ingest_usage_path.exists():
                raise ValidationArtifactError(
                    "shared ingestion usage is missing, so total final cost cannot be reported"
                )
            payload, markdown = _aggregate(manifest, args.arm, sealed_dir, ingest_usage_path)
        except (OSError, ValueError, FreezeError, ValidationArtifactError) as exc:
            print(f"STOP: final aggregate failed: {exc}", file=sys.stderr)
            return 2
        destination.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(destination, payload)
        destination.with_suffix(".md").write_text(markdown, encoding="utf-8")
        ledger["status"] = "complete"
        ledger["completed_at_utc"] = datetime.now(UTC).isoformat()
        ledger["aggregate_sha256"] = sha256_file(destination)
        _atomic_json(ledger_path, ledger)

    print(markdown, end="")
    print(f"FINAL aggregate report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
