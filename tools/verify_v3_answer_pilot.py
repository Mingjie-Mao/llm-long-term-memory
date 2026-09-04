"""Capture or verify the immutable, aggregate-only v3 answer pilot conclusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
ARCHIVE = Path("results/archive/v3-answer-pilot")
FREEZE = Path("results/frozen/v3-answer-pilot/freeze.json")
AGGREGATE = Path("results/validation/v3-answer-pilot-aggregate.json")
REPORT = Path("results/validation/v3-answer-pilot-aggregate.md")
SNAPSHOT = ARCHIVE / "source.tar.gz"
CONCLUSION = ARCHIVE / "conclusion.json"
README = ARCHIVE / "README.md"
VERIFIER = Path("tools/verify_v3_answer_pilot.py")
SEALED_DIR = Path("results/sealed/v3-answer-pilot")


class VerificationError(RuntimeError):
    """The preserved pilot evidence is missing, inconsistent or changed."""


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
        raise VerificationError(f"file size changed or file is missing: {relative}")
    if _sha256(path) != record.get("sha256"):
        raise VerificationError(f"file hash changed: {relative}")


def _verify_snapshot(repo: Path, freeze: dict[str, Any]) -> None:
    expected = freeze["system"]["source"]["files"]
    try:
        with tarfile.open(repo / SNAPSHOT, "r:gz") as handle:
            members = handle.getmembers()
            names = [member.name.removeprefix("./") for member in members]
            if len(names) != len(set(names)) or set(names) != set(expected):
                raise VerificationError("source snapshot inventory differs from the freeze")
            for member, name in zip(members, names, strict=True):
                if not member.isfile():
                    raise VerificationError(f"non-file in source snapshot: {name}")
                stream = handle.extractfile(member)
                if stream is None:
                    raise VerificationError(f"cannot read source snapshot member: {name}")
                content = stream.read()
                record = expected[name]
                if len(content) != record["bytes"]:
                    raise VerificationError(f"source snapshot size changed: {name}")
                if hashlib.sha256(content).hexdigest() != record["sha256"]:
                    raise VerificationError(f"source snapshot hash changed: {name}")
    except (OSError, tarfile.TarError) as exc:
        raise VerificationError(f"cannot read source snapshot: {exc}") from exc


def _verify_chain(repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = _read_json(repo, FREEZE)
    aggregate = _read_json(repo, AGGREGATE)
    system = freeze.get("system", {})
    if system.get("variants") != ["two_stage_fallback", "two_stage_reasoned"]:
        raise VerificationError("frozen pilot variants changed")
    if system.get("store", {}).get("name") != "train150":
        raise VerificationError("pilot did not use the shared train150 store")

    config = system.get("config", {})
    config_path = Path(config.get("path", ""))
    if config_path != Path("configs/v3-answer.yaml"):
        raise VerificationError("unexpected frozen pilot config")
    _verify_file(
        repo,
        config_path,
        {"bytes": (repo / config_path).stat().st_size, "sha256": config.get("sha256")},
    )
    for name, record in system.get("protocol", {}).get("files", {}).items():
        _verify_file(repo, Path(name), record)

    if aggregate.get("manifest") != {"name": "v3-reasoning48", "questions": 48}:
        raise VerificationError("aggregate is not the fixed 48-question pilot")
    privacy = aggregate.get("privacy", {})
    if privacy != {"aggregate_only": True, "contains_question_ids_or_text": False}:
        raise VerificationError("aggregate privacy declaration changed")
    arms = aggregate.get("arms", [])
    if [arm.get("arm") for arm in arms] != ["v2-control", "v3-reasoned"]:
        raise VerificationError("aggregate arms or order changed")
    if any(arm.get("questions") != 48 or arm.get("runs") != 3 for arm in arms):
        raise VerificationError("aggregate does not contain 48 questions x 3 repeats per arm")

    expected_names = {
        f"{arm}.{suffix}"
        for arm in ("v2-control", "v3-reasoned")
        for repeat in range(1, 4)
        for suffix in (f"rep{repeat}.jsonl", f"rep{repeat}.usage.json")
    }
    sealed = aggregate.get("sealed_artifacts")
    if not isinstance(sealed, dict) or set(sealed) != expected_names:
        raise VerificationError("aggregate does not bind exactly 12 sealed artifacts")
    for name, record in sealed.items():
        relative = SEALED_DIR / name
        _verify_file(repo, relative, record)
        if name.endswith(".jsonl"):
            rows = sum(1 for line in (repo / relative).open(encoding="utf-8") if line.strip())
            if rows != 48:
                raise VerificationError(f"sealed result must contain 48 rows: {relative}")
    _verify_snapshot(repo, freeze)
    return freeze, aggregate


def _inventory_paths(freeze: dict[str, Any], aggregate: dict[str, Any]) -> list[Path]:
    paths = {FREEZE, AGGREGATE, REPORT, SNAPSHOT, README, VERIFIER}
    paths.add(Path(freeze["system"]["config"]["path"]))
    paths.update(Path(name) for name in freeze["system"]["protocol"]["files"])
    paths.update(SEALED_DIR / name for name in aggregate["sealed_artifacts"])
    return sorted(paths, key=Path.as_posix)


def _capture(repo: Path, freeze: dict[str, Any], aggregate: dict[str, Any]) -> None:
    destination = repo / CONCLUSION
    if destination.exists():
        raise VerificationError("v3 pilot conclusion already exists and cannot be replaced")
    files = []
    for relative in _inventory_paths(freeze, aggregate):
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
        "experiment_id": "v3-answer-pilot-train48",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": freeze["system"]["source"]["tree_sha256"],
        "files": files,
        "summary_file": AGGREGATE.as_posix(),
        "outcome": "positive_development_signal",
        "outcome_reason": (
            "The pre-registered gate passed and v3 gained 4.2 percentage points, but the "
            "48-question paired result was not statistically conclusive (p=0.6875)."
        ),
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _verify_conclusion(repo: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    conclusion = _read_json(repo, CONCLUSION)
    if conclusion.get("schema_version") != 1:
        raise VerificationError("unsupported conclusion schema")
    if conclusion.get("experiment_id") != "v3-answer-pilot-train48":
        raise VerificationError("unexpected conclusion experiment id")
    if conclusion.get("source_sha256") != freeze["system"]["source"]["tree_sha256"]:
        raise VerificationError("conclusion source fingerprint differs from freeze")
    files = conclusion.get("files")
    if not isinstance(files, list) or not files:
        raise VerificationError("conclusion has no file inventory")
    seen: set[str] = set()
    for record in files:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise VerificationError("invalid conclusion file record")
        relative = record["path"]
        if relative in seen:
            raise VerificationError(f"duplicate conclusion file: {relative}")
        seen.add(relative)
        _verify_file(repo, Path(relative), record)
    return conclusion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        freeze, aggregate = _verify_chain(repo)
        if args.capture:
            _capture(repo, freeze, aggregate)
            print(f"CAPTURED: {CONCLUSION}")
        conclusion = _verify_conclusion(repo, freeze)
    except VerificationError as exc:
        print(f"STOP: {exc}")
        return 2
    print(
        "PASS: preserved v3 answer pilot · 288 rows · "
        "v3 72.9% / v2 68.8% · positive development signal"
    )
    print(f"  source snapshot: {len(freeze['system']['source']['files'])} files")
    print(f"  evidence inventory: {len(conclusion['files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
