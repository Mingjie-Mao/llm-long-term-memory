"""Build the only normal human-facing report for ``dev100``.

Example after three independently checkpointed repeats per arm::

    python scripts/aggregate_validation.py \
      --manifest results/manifests/dev100.json \
      --arm flat20=<three-comma-separated-JSONL-paths> \
      --arm coherent=<three-comma-separated-JSONL-paths>

Raw rows remain available for quota resume and an external audit, but this command
never prints or writes question-level material.  ``test100`` is refused: its single
final execution has a separate one-shot gate so a validation helper cannot become
an accidental second test runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.evaluation.manifest import load_manifest  # noqa: E402
from llm_long_term_memory.evaluation.report import load_report  # noqa: E402
from llm_long_term_memory.evaluation.validation import (  # noqa: E402
    ValidationArtifactError,
    aggregate_arm,
    load_usage,
    paired_majority,
    render_markdown,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_arm(raw: str) -> tuple[str, list[Path]]:
    try:
        name, paths = raw.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "arm must be NAME=run1.jsonl,run2.jsonl,run3.jsonl"
        ) from exc
    name = name.strip()
    files = [Path(value.strip()) for value in paths.split(",") if value.strip()]
    if not name or not files:
        raise argparse.ArgumentTypeError("arm must include a name and at least one JSONL path")
    return name, files


def build_report(
    manifest_path: Path,
    arm_specs: list[tuple[str, list[Path]]],
    ingest_usage_path: Path,
) -> tuple[dict, str]:
    manifest = load_manifest(manifest_path)
    if manifest.name.lower() == "test100" or "test100" in manifest_path.stem.lower():
        raise ValidationArtifactError(
            "test100 is sealed and cannot be read by the validation reporter"
        )
    if not arm_specs:
        raise ValidationArtifactError("at least one --arm is required")

    expected_ids = set(manifest.question_ids)
    reports_by_arm = {}
    aggregates = []
    for name, paths in arm_specs:
        if name in reports_by_arm:
            raise ValidationArtifactError(f"duplicate arm name {name!r}")
        missing = sum(not path.exists() for path in paths)
        if missing:
            raise ValidationArtifactError(f"arm {name!r} is missing {missing} result file(s)")
        usage_paths = [path.with_suffix(".usage.json") for path in paths]
        missing_usage = sum(not path.exists() for path in usage_paths)
        if missing_usage:
            raise ValidationArtifactError(
                f"arm {name!r} is missing {missing_usage} usage file(s); cost cannot be reported"
            )
        reports = [load_report(path, variant=name) for path in paths]
        reports_by_arm[name] = reports
        aggregates.append(
            aggregate_arm(name, reports, expected_ids, [load_usage(path) for path in usage_paths])
        )

    baseline = arm_specs[0][0]
    comparisons = [
        paired_majority(
            baseline,
            reports_by_arm[baseline],
            name,
            reports_by_arm[name],
            expected_ids,
        )
        for name, _ in arm_specs[1:]
    ]
    if not ingest_usage_path.exists():
        raise ValidationArtifactError(
            "shared ingestion usage is missing, so total experiment cost cannot be reported"
        )
    shared_ingestion = load_usage(ingest_usage_path)
    payload = {
        "schema_version": 1,
        "protocol": "aggregate-only validation; first arm is the paired baseline",
        "manifest": {
            "name": manifest.name,
            "questions": len(manifest),
            "sha256": _sha256(manifest_path),
        },
        "arms": [arm.to_dict() for arm in aggregates],
        "paired_majority": [item.to_dict() for item in comparisons],
        "shared_ingestion_usage": shared_ingestion.to_dict(),
        "privacy": {
            "contains_question_ids": False,
            "contains_questions_or_answers": False,
            "contains_hypotheses_or_judge_reasons": False,
        },
    }
    return payload, render_markdown(aggregates, comparisons, shared_ingestion)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", action="append", type=_parse_arm, default=[])
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "results" / "validation" / "dev100-aggregate.json",
        help="aggregate JSON destination; Markdown is written beside it",
    )
    parser.add_argument(
        "--ingest-usage",
        type=Path,
        default=REPO / "results" / "raw" / "dev100.ingest.usage.json",
    )
    args = parser.parse_args()
    try:
        payload, markdown = build_report(args.manifest, args.arm, args.ingest_usage)
    except (OSError, ValueError, ValidationArtifactError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = args.out.with_suffix(".md")
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"Aggregate artifacts: {args.out} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
