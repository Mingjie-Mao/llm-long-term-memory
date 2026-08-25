"""Start or resume dev100/test100 ingestion under an immutable pre-ingest freeze.

The freeze is captured while the planned store name is untouched.  Every resume
re-hashes source, prompts, models, config, data and manifest before invoking the
normal checkpointed ingestion command.  Progress is aggregate-only.

Typical flow::

    python scripts/freeze_v2.py --capture --name v2-candidate-preingest \
      --config configs/v2.yaml --manifest results/manifests/dev100.json \
      --pre-ingest-store-name dev100 --variant two_stage_fallback \
      --variant two_stage_coherent --variant two_stage_coherent_oracle

    python scripts/run_frozen_ingest.py --run --freeze-name v2-candidate-preingest \
      --config configs/v2.yaml --manifest results/manifests/dev100.json \
      --store-name dev100 --variant two_stage_fallback \
      --variant two_stage_coherent --variant two_stage_coherent_oracle

There is deliberately no ``--fresh`` option.  A quota pause resumes the same
checkpoint; it never silently becomes a new ingest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
)


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def _verify_pre_ingest(args, settings: Settings) -> Path:
    freeze = REPO / "results" / "frozen" / args.freeze_name / "freeze.json"
    if not freeze.exists():
        raise FreezeError(f"pre-ingest freeze does not exist: {freeze}")
    try:
        frozen = json.loads(freeze.read_text(encoding="utf-8"))["system"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FreezeError(f"invalid pre-ingest freeze: {exc}") from exc
    if frozen.get("store") is not None:
        raise FreezeError("this is a post-ingest freeze, not a pre-ingest lock")
    if frozen.get("pre_ingest_store_name") != args.store_name:
        raise FreezeError("pre-ingest freeze names a different planned store")
    request = FreezeRequest(
        repo=REPO,
        config_path=_rooted(args.config),
        manifest_path=_rooted(args.manifest),
        data_dir=_rooted(settings.data_dir),
        store_dir=_rooted(settings.store_dir),
        results_dir=_rooted(settings.results_dir),
        variants=tuple(args.variant),
        pre_ingest_store_name=args.store_name,
        protocol_paths=formal_protocol_paths(REPO, args.store_name),
    )
    moved = differences(frozen, capture_system(request))
    if moved:
        raise FreezeError(
            f"system changed after pre-ingest freeze ({len(moved)} differences): "
            + "; ".join(moved[:5])
        )
    return freeze


def _post_ingest_check(args, settings: Settings) -> dict:
    request = FreezeRequest(
        repo=REPO,
        config_path=_rooted(args.config),
        manifest_path=_rooted(args.manifest),
        data_dir=_rooted(settings.data_dir),
        store_dir=_rooted(settings.store_dir),
        results_dir=_rooted(settings.results_dir),
        variants=tuple(args.variant),
        store_name=args.store_name,
        protocol_paths=formal_protocol_paths(REPO, args.store_name),
    )
    return capture_system(request)


def _validate_formal_target(args, manifest, *, selected_variant: str | None = None) -> None:
    name = manifest.name.lower()
    canonical = REPO / "results" / "manifests" / f"{name}.json"
    if _rooted(args.manifest).resolve() != canonical.resolve():
        raise FreezeError(f"frozen ingestion requires the canonical {name} manifest")
    if len(manifest) != 100:
        raise FreezeError(f"the {name} manifest must contain exactly 100 questions")
    if args.store_name != name:
        raise FreezeError(f"the {name} ingest requires store name {name!r}")
    if name == "dev100":
        if args.freeze_name != "v2-candidate-preingest":
            raise FreezeError("dev100 ingest requires freeze 'v2-candidate-preingest'")
        if tuple(args.variant) != _DEV_VARIANTS:
            raise FreezeError("dev100 ingest variants do not match the pre-registration")
        return
    if selected_variant is None:
        raise FreezeError("test100 ingest requires the registered dev100 decision")
    if args.freeze_name != "v2-final-preingest":
        raise FreezeError("test100 ingest requires freeze 'v2-final-preingest'")
    expected = (selected_variant, "full_context", "naive_rag")
    if selected_variant == "two_stage_coherent":
        expected += ("two_stage_fallback",)
    if tuple(args.variant) != expected:
        raise FreezeError("test100 ingest variants do not match the pre-registered final arms")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="spend extractor quota")
    parser.add_argument("--freeze-name", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--store-name", required=True)
    parser.add_argument("--variant", action="append", required=True)
    args = parser.parse_args()
    settings = Settings()
    try:
        manifest = load_manifest(_rooted(args.manifest))
        if manifest.name.lower() not in {"dev100", "test100"}:
            raise FreezeError("frozen ingestion accepts only dev100 or test100")
        selected_variant = None
        if manifest.name.lower() == "test100":
            validation_dir = REPO / "results" / "validation"
            decision = load_dev_decision(
                validation_dir / "dev100-decision.json",
                validation_dir / "dev100-aggregate.json",
            )
            selected_variant = str(decision["selected_variant"])
        _validate_formal_target(args, manifest, selected_variant=selected_variant)
        freeze = _verify_pre_ingest(args, settings)
        final_ledger = REPO / "results" / "frozen" / "v2-final" / "one-shot.json"
        if manifest.name.lower() == "test100" and final_ledger.exists():
            ledger = json.loads(final_ledger.read_text(encoding="utf-8"))
            if ledger.get("status") == "complete":
                raise FreezeError("test100 final protocol is complete; its store is immutable")
    except (
        OSError,
        ValueError,
        FreezeError,
        ValidationArtifactError,
        json.JSONDecodeError,
    ) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2

    print(
        f"PASS: {manifest.name} pre-ingest freeze · {len(manifest)} questions · "
        f"planned store {args.store_name} · ids sealed"
    )
    print(f"  freeze: {freeze}")
    if not args.run:
        print("Preflight only. Add --run to start or resume the frozen ingest.")
        return 0

    environment = dict(os.environ)
    environment["HF_HUB_OFFLINE"] = "1"
    command = [
        str(REPO / ".venv" / "bin" / "python"),
        "-m",
        "llm_long_term_memory.cli",
        "ingest",
        "run",
        "--config",
        str(_rooted(args.config)),
        "--store-name",
        args.store_name,
        "--questions",
        str(_rooted(args.manifest)),
    ]
    result = subprocess.run(command, cwd=REPO, env=environment)
    if result.returncode != 0:
        print(
            f"PAUSED: frozen {manifest.name} ingest returned {result.returncode}; "
            "resume this same command after quota reset."
        )
        return result.returncode

    try:
        system = _post_ingest_check(args, settings)
    except (OSError, ValueError, FreezeError) as exc:
        print(f"STOP: ingest exited successfully but completeness check failed: {exc}")
        return 2
    terminal = system["store"]["terminal"]
    print(
        f"COMPLETE: {terminal['done']:,} done + {terminal['blocked']:,} blocked = "
        f"{terminal['expected']:,}/{terminal['expected']:,} terminal"
    )
    print("Next: capture the post-ingest freeze with scripts/freeze_v2.py --store-name")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
