"""Capture or verify a v2 content-addressed experiment freeze.

Capture only after the selected store is complete::

    python scripts/freeze_v2.py --capture --name v2-candidate \
      --config configs/v2.yaml --manifest results/manifests/dev100.json \
      --store-name dev100 --variant two_stage_fallback --variant two_stage_coherent

Run the same command without ``--capture`` before every resumed validation or
final-test process.  The working tree may be dirty: the record hashes content, so
no commit is required.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.config import Settings  # noqa: E402
from llm_long_term_memory.evaluation.manifest import load_manifest  # noqa: E402
from llm_long_term_memory.evaluation.reproducibility import (  # noqa: E402
    FreezeError,
    FreezeRequest,
    capture_system,
    differences,
    formal_protocol_paths,
)
from llm_long_term_memory.evaluation.validation import (  # noqa: E402
    ValidationArtifactError,
    load_dev_decision,
)

_DEV_VARIANTS = (
    "two_stage_fallback",
    "two_stage_coherent",
    "two_stage_coherent_oracle",
    # Amendment 2026-08-25: reported baselines. They read the same store and cannot
    # move the decision; see results/prereg-context-shape.md.
    "naive_rag",
    "two_stage_memory_only",
)


def _validate_formal_freeze(args, manifest, *, selected_variant: str | None = None) -> None:
    name = manifest.name.lower()
    if name not in {"dev100", "test100"}:
        return
    canonical = REPO / "results" / "manifests" / f"{name}.json"
    manifest_path = args.manifest if args.manifest.is_absolute() else REPO / args.manifest
    if manifest_path.resolve() != canonical.resolve() or len(manifest) != 100:
        raise FreezeError(f"formal {name} freeze requires its canonical 100-question manifest")
    if args.replace:
        raise FreezeError(f"formal {name} freezes can never be replaced")
    pre = args.pre_ingest_store_name is not None
    target = args.pre_ingest_store_name if pre else args.store_name
    if target != name:
        raise FreezeError(f"formal {name} freeze requires store name {name!r}")
    expected_name = (
        "v2-candidate-preingest"
        if name == "dev100" and pre
        else "v2-candidate"
        if name == "dev100"
        else "v2-final-preingest"
        if pre
        else "v2-final"
    )
    if args.name != expected_name:
        raise FreezeError(f"formal {name} freeze requires name {expected_name!r}")
    if name == "dev100":
        expected_variants = _DEV_VARIANTS
    else:
        if selected_variant is None:
            raise FreezeError("formal test100 freeze requires the registered dev100 decision")
        expected_variants = (selected_variant, "full_context", "naive_rag")
        if selected_variant == "two_stage_coherent":
            expected_variants += ("two_stage_fallback",)
    if tuple(args.variant) != expected_variants:
        raise FreezeError(f"formal {name} freeze variants do not match the pre-registration")


def _validate_pre_to_post_continuity(args, manifest, current: dict) -> None:
    """Require a formal post-ingest freeze to extend, not replace, its pre-ingest lock."""
    name = manifest.name.lower()
    if name not in {"dev100", "test100"} or args.store_name is None:
        return
    pre_name = "v2-candidate-preingest" if name == "dev100" else "v2-final-preingest"
    path = REPO / "results" / "frozen" / pre_name / "freeze.json"
    try:
        pre = json.loads(path.read_text(encoding="utf-8"))["system"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FreezeError(
            f"valid pre-ingest freeze is required before post-ingest freeze: {exc}"
        ) from exc
    comparable = copy.deepcopy(current)
    comparable["store"] = None
    comparable["pre_ingest_store_name"] = args.store_name
    moved = differences(pre, comparable)
    if moved:
        raise FreezeError(
            f"system changed between pre- and post-ingest freezes ({len(moved)} differences): "
            + "; ".join(moved[:5])
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--name", required=True, help="freeze directory name")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--store-name", help="completed store stem; omit only for a pre-ingest lock"
    )
    parser.add_argument(
        "--pre-ingest-store-name",
        help="planned store stem; capture refuses if any store artifact already exists",
    )
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="evaluation runner identity; repeat for every frozen arm",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing freeze; never use after reading validation/test results",
    )
    parser.add_argument(
        "--protocol-path",
        action="append",
        type=Path,
        default=[],
        help="additional pre-registered artifact to hash for a non-formal research freeze",
    )
    args = parser.parse_args()
    settings = Settings()
    destination = REPO / "results" / "frozen" / args.name / "freeze.json"

    def rooted(path: Path) -> Path:
        return path if path.is_absolute() else REPO / path

    if args.store_name and args.pre_ingest_store_name:
        parser.error("--store-name and --pre-ingest-store-name are mutually exclusive")
    try:
        manifest = load_manifest(rooted(args.manifest))
        selected_variant = None
        if manifest.name.lower() == "test100":
            validation_dir = REPO / "results" / "validation"
            decision = load_dev_decision(
                validation_dir / "dev100-decision.json",
                validation_dir / "dev100-aggregate.json",
            )
            selected_variant = str(decision["selected_variant"])
        _validate_formal_freeze(args, manifest, selected_variant=selected_variant)
    except (OSError, ValueError, FreezeError, ValidationArtifactError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    if args.capture and args.pre_ingest_store_name:
        store_root = rooted(settings.store_dir)
        existing = [
            store_root / f"{args.pre_ingest_store_name}.db",
            store_root / f"{args.pre_ingest_store_name}-ingest.json",
            store_root / f"{args.pre_ingest_store_name}-index.ids.json",
            store_root / f"{args.pre_ingest_store_name}-index.npy",
            rooted(settings.results_dir)
            / "raw"
            / f"{args.pre_ingest_store_name}.ingest.usage.json",
            rooted(settings.results_dir) / "sealed" / args.pre_ingest_store_name,
        ]
        present = sum(path.exists() for path in existing)
        if present:
            print(
                f"STOP: pre-ingest freeze requires an untouched store name; "
                f"{present} artifact(s) already exist",
                file=sys.stderr,
            )
            return 2
    request = FreezeRequest(
        repo=REPO,
        config_path=rooted(args.config),
        manifest_path=rooted(args.manifest),
        data_dir=rooted(settings.data_dir),
        store_dir=rooted(settings.store_dir),
        results_dir=rooted(settings.results_dir),
        variants=tuple(args.variant),
        store_name=args.store_name,
        pre_ingest_store_name=args.pre_ingest_store_name,
        protocol_paths=(
            *formal_protocol_paths(REPO, manifest.name),
            *(rooted(path) for path in args.protocol_path),
        ),
    )
    try:
        current = capture_system(request)
    except (OSError, ValueError, FreezeError) as exc:
        print(f"STOP: cannot freeze this system: {exc}", file=sys.stderr)
        return 2

    if args.capture:
        try:
            _validate_pre_to_post_continuity(args, manifest, current)
        except FreezeError as exc:
            print(f"STOP: {exc}", file=sys.stderr)
            return 2
        if destination.exists() and not args.replace:
            print(
                f"STOP: {destination} already exists. Refreezing after seeing a result "
                "would destroy the experimental boundary.",
                file=sys.stderr,
            )
            return 2
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "note": (
                "Content-addressed v2 experiment freeze. Run this command without "
                "--capture to verify the same system before resuming or reporting."
            ),
            "system": current,
        }
        destination.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        store = current.get("store")
        store_text = (
            "pre-ingest lock"
            if store is None
            else (
                f"{store['terminal']['expected']:,} terminal sessions, "
                f"{store['database']['counts']['memories']:,} memories"
            )
        )
        print(f"FROZEN: {destination}")
        print(f"  source {current['source']['tree_sha256']}")
        print(f"  config {current['config']['sha256']}")
        print(f"  data   {current['data']['manifest']['questions']} questions (ids sealed)")
        print(f"  store  {store_text}")
        return 0

    if not destination.exists():
        print(f"STOP: no freeze record at {destination}", file=sys.stderr)
        return 2
    try:
        frozen = json.loads(destination.read_text(encoding="utf-8"))["system"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"STOP: invalid freeze record: {exc}", file=sys.stderr)
        return 2
    moved = differences(frozen, current)
    if moved:
        print("STOP: the system changed after it was frozen:", file=sys.stderr)
        for change in moved[:50]:
            print(f"  {change}", file=sys.stderr)
        if len(moved) > 50:
            print(f"  ... and {len(moved) - 50} more", file=sys.stderr)
        return 2
    print(f"PASS: unchanged since {destination}")
    print(f"  source {current['source']['tree_sha256']}")
    print(f"  manifest {current['data']['manifest']['questions']} questions (ids sealed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
