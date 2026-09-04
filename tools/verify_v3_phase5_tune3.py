"""Capture or verify the immutable v3.3 tune42 promotion package."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
ARCHIVE = Path("results/archive/v3-phase5-tune3")
FREEZE = Path("results/frozen/v3-phase5-tune3/freeze.json")
AGGREGATE = Path("results/validation/v3-phase5-tune3.json")
REPORT = Path("results/validation/v3-phase5-tune3.md")
SNAPSHOT = ARCHIVE / "source.tar.gz"
README = ARCHIVE / "README.md"
CONCLUSION = ARCHIVE / "conclusion.json"
VERIFIER = Path("tools/verify_v3_phase5_tune3.py")


class VerificationError(RuntimeError):
    """The frozen tune result is missing, inconsistent, or changed."""


def _read_json(repo: Path, relative: Path) -> dict[str, Any]:
    try:
        value = json.loads((repo / relative).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read valid JSON from {relative}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"expected one JSON object in {relative}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(repo: Path, relative: Path, record: dict[str, Any]) -> None:
    path = repo / relative
    if not path.is_file() or path.stat().st_size != record.get("bytes"):
        raise VerificationError(f"file missing or size changed: {relative}")
    if _sha256(path) != record.get("sha256"):
        raise VerificationError(f"file hash changed: {relative}")


def _verify_live_source(repo: Path, freeze: dict[str, Any]) -> None:
    for name, record in freeze["system"]["source"]["files"].items():
        _verify_file(repo, Path(name), record)


def _capture_snapshot(repo: Path, freeze: dict[str, Any]) -> None:
    destination = repo / SNAPSHOT
    if destination.exists():
        raise VerificationError("source snapshot already exists and cannot be replaced")
    _verify_live_source(repo, freeze)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as handle:
        for name in sorted(freeze["system"]["source"]["files"]):
            handle.add(repo / name, arcname=name, recursive=False)


def _verify_snapshot(repo: Path, freeze: dict[str, Any]) -> None:
    expected = freeze["system"]["source"]["files"]
    try:
        with tarfile.open(repo / SNAPSHOT, "r:gz") as handle:
            members = handle.getmembers()
            names = [member.name.removeprefix("./") for member in members]
            if len(names) != len(set(names)) or set(names) != set(expected):
                raise VerificationError("source snapshot inventory differs from freeze")
            for member, name in zip(members, names, strict=True):
                if not member.isfile():
                    raise VerificationError(f"non-file snapshot member: {name}")
                stream = handle.extractfile(member)
                if stream is None:
                    raise VerificationError(f"cannot read snapshot member: {name}")
                content = stream.read()
                record = expected[name]
                if len(content) != record["bytes"]:
                    raise VerificationError(f"snapshot member size changed: {name}")
                if hashlib.sha256(content).hexdigest() != record["sha256"]:
                    raise VerificationError(f"snapshot member hash changed: {name}")
    except (OSError, tarfile.TarError) as exc:
        raise VerificationError(f"cannot read source snapshot: {exc}") from exc


def _verify_chain(repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = _read_json(repo, FREEZE)
    aggregate = _read_json(repo, AGGREGATE)
    system = freeze["system"]
    if system.get("variants") != ["two_stage_reasoned_evidence"]:
        raise VerificationError("frozen variant changed")
    if system.get("store", {}).get("name") != "train150":
        raise VerificationError("tune did not use train150")

    config = system["config"]
    config_path = Path(config["path"])
    if config_path != Path("configs/v3-phase5-compact.yaml"):
        raise VerificationError("unexpected tune config")
    if _sha256(repo / config_path) != config["sha256"]:
        raise VerificationError("tune config changed")
    for name, record in system["protocol"]["files"].items():
        _verify_file(repo, Path(name), record)

    if aggregate.get("manifest") != {"name": "v3-reasoning-tune42", "questions": 42}:
        raise VerificationError("aggregate is not tune42")
    if aggregate.get("privacy") != {
        "aggregate_only": True,
        "contains_question_ids_or_text": False,
    }:
        raise VerificationError("aggregate privacy declaration changed")

    arms = aggregate.get("arms", [])
    if [arm.get("arm") for arm in arms] != [
        "v2-control",
        "v3.2-adaptive",
        "v3.3-compact",
    ]:
        raise VerificationError("aggregate arms changed")
    if [arm.get("majority_accuracy") for arm in arms] != [2 / 3, 31 / 42, 31 / 42]:
        raise VerificationError("recorded accuracy changed")
    if any(arm.get("questions") != 42 or arm.get("runs") != 1 for arm in arms):
        raise VerificationError("aggregate is not one 42-question run per arm")
    if [arm.get("median_context_tokens") for arm in arms] != [585.5, 1476.5, 1124.0]:
        raise VerificationError("recorded context sizes changed")

    gate = aggregate.get("gate", {})
    checks = gate.get("checks", {})
    expected_checks = {
        "context_lower_than_v3_2",
        "context_within_v2_limit",
        "full_gold_coverage_no_regression",
        "generation_cost",
        "hydration_is_targeted",
        "mean_gold_coverage_no_regression",
        "multi_session_no_regression",
        "no_new_confident_errors",
        "ordinary_no_regression",
        "overall_no_regression",
        "temporal_no_regression",
    }
    if gate.get("passed") is not True or set(checks) != expected_checks:
        raise VerificationError("promotion decision changed")
    if any(value is not True for value in checks.values()):
        raise VerificationError("a registered promotion check no longer passes")
    if gate.get("context_ratio_vs_v2") != 1124.0 / 585.5:
        raise VerificationError("v2 context ratio changed")
    if gate.get("context_ratio_vs_v3_2") != 1124.0 / 1476.5:
        raise VerificationError("v3.2 context ratio changed")

    paired = aggregate.get("paired", {})
    if paired.get("accuracy_delta") != 0 or paired.get("p_value") != 1.0:
        raise VerificationError("paired result changed")
    if paired.get("candidate_wins") != 1 or paired.get("candidate_losses") != 1:
        raise VerificationError("paired outcomes changed")

    expected_artifacts = {
        "results/sealed/v3-phase3-tune1/v2-control.jsonl",
        "results/sealed/v3-phase3-tune1/v2-control.usage.json",
        "results/sealed/v3-phase4-tune2/v3.2-adaptive.jsonl",
        "results/sealed/v3-phase4-tune2/v3.2-adaptive.usage.json",
        "results/sealed/v3-phase5-tune3/v3.3-compact.jsonl",
        "results/sealed/v3-phase5-tune3/v3.3-compact.usage.json",
    }
    artifacts = aggregate.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise VerificationError("aggregate does not bind exactly six artifacts")
    for name, record in artifacts.items():
        relative = Path(name)
        _verify_file(repo, relative, record)
        if name.endswith(".jsonl"):
            rows = sum(1 for line in (repo / relative).open(encoding="utf-8") if line.strip())
            if rows != 42:
                raise VerificationError(f"sealed result must have 42 rows: {relative}")
    _verify_snapshot(repo, freeze)
    return freeze, aggregate


def _paths(freeze: dict[str, Any], aggregate: dict[str, Any]) -> list[Path]:
    paths = {FREEZE, AGGREGATE, REPORT, SNAPSHOT, README, VERIFIER}
    paths.add(Path(freeze["system"]["config"]["path"]))
    paths.update(Path(name) for name in freeze["system"]["protocol"]["files"])
    paths.update(Path(name) for name in aggregate["artifacts"])
    return sorted(paths, key=Path.as_posix)


def _capture_conclusion(
    repo: Path, freeze: dict[str, Any], aggregate: dict[str, Any]
) -> None:
    if (repo / CONCLUSION).exists():
        raise VerificationError("tune3 conclusion already exists and cannot be replaced")
    files = []
    for relative in _paths(freeze, aggregate):
        path = repo / relative
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload = {
        "schema_version": 1,
        "experiment_id": "v3-phase5-tune3",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": freeze["system"]["source"]["tree_sha256"],
        "files": files,
        "summary_file": AGGREGATE.as_posix(),
        "outcome": "promoted_to_dev60",
        "outcome_reason": (
            "v3.3 retained v3.2's 73.8% overall accuracy and all registered slice "
            "scores while reducing median context from 1,476.5 to 1,124 tokens "
            "(1.92x v2), so every pre-registered tune gate passed."
        ),
    }
    (repo / CONCLUSION).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _verify_conclusion(repo: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    conclusion = _read_json(repo, CONCLUSION)
    if conclusion.get("schema_version") != 1:
        raise VerificationError("unsupported conclusion schema")
    if conclusion.get("experiment_id") != "v3-phase5-tune3":
        raise VerificationError("unexpected conclusion experiment id")
    if conclusion.get("source_sha256") != freeze["system"]["source"]["tree_sha256"]:
        raise VerificationError("conclusion source hash changed")
    if conclusion.get("outcome") != "promoted_to_dev60":
        raise VerificationError("conclusion outcome changed")
    files = conclusion.get("files")
    if not isinstance(files, list) or not files:
        raise VerificationError("conclusion has no evidence inventory")
    seen: set[str] = set()
    for record in files:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise VerificationError("invalid conclusion record")
        if record["path"] in seen:
            raise VerificationError(f"duplicate conclusion file: {record['path']}")
        seen.add(record["path"])
        _verify_file(repo, Path(record["path"]), record)
    return conclusion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        freeze = _read_json(repo, FREEZE)
        if args.capture:
            _capture_snapshot(repo, freeze)
        freeze, aggregate = _verify_chain(repo)
        if args.capture:
            _capture_conclusion(repo, freeze, aggregate)
            print(f"CAPTURED: {CONCLUSION}")
        conclusion = _verify_conclusion(repo, freeze)
    except VerificationError as exc:
        print(f"STOP: {exc}")
        return 2
    print("PASS: preserved v3.3 tune42 promotion · 42 new rows · 73.8% accuracy")
    print(f"  source snapshot: {len(freeze['system']['source']['files'])} files")
    print(f"  evidence inventory: {len(conclusion['files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
