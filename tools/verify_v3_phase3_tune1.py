"""Capture or verify the immutable v3.1 tune42 negative-result package."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
ARCHIVE = Path("results/archive/v3-phase3-tune1")
FREEZE = Path("results/frozen/v3-phase3-tune1/freeze.json")
AGGREGATE = Path("results/validation/v3-phase3-tune1.json")
REPORT = Path("results/validation/v3-phase3-tune1.md")
SEALED = Path("results/sealed/v3-phase3-tune1")
SNAPSHOT = ARCHIVE / "source.tar.gz"
README = ARCHIVE / "README.md"
CONCLUSION = ARCHIVE / "conclusion.json"
VERIFIER = Path("tools/verify_v3_phase3_tune1.py")


class VerificationError(RuntimeError):
    """The frozen tune result is missing, inconsistent or changed."""


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
    if system.get("variants") != ["two_stage_fallback", "two_stage_reasoned"]:
        raise VerificationError("frozen variants changed")
    if system.get("store", {}).get("name") != "train150":
        raise VerificationError("tune did not use train150")
    config = system["config"]
    config_path = Path(config["path"])
    if config_path != Path("configs/v3-phase3.yaml"):
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
    if [arm.get("arm") for arm in arms] != ["v2-control", "v3.1-reasoned"]:
        raise VerificationError("aggregate arms changed")
    if [arm.get("majority_accuracy") for arm in arms] != [2 / 3, 2 / 3]:
        raise VerificationError("recorded no-gain outcome changed")
    expected = {
        "v2-control.jsonl",
        "v2-control.usage.json",
        "v3.1-reasoned.jsonl",
        "v3.1-reasoned.usage.json",
    }
    sealed = aggregate.get("sealed_artifacts")
    if not isinstance(sealed, dict) or set(sealed) != expected:
        raise VerificationError("aggregate does not bind exactly four artifacts")
    for name, record in sealed.items():
        relative = SEALED / name
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
    paths.update(SEALED / name for name in aggregate["sealed_artifacts"])
    return sorted(paths, key=Path.as_posix)


def _capture(repo: Path, freeze: dict[str, Any], aggregate: dict[str, Any]) -> None:
    if (repo / CONCLUSION).exists():
        raise VerificationError("tune1 conclusion already exists and cannot be replaced")
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
        "experiment_id": "v3-phase3-tune1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": freeze["system"]["source"]["tree_sha256"],
        "files": files,
        "summary_file": AGGREGATE.as_posix(),
        "outcome": "negative",
        "outcome_reason": (
            "v3.1 tied v2 at 66.7% overall and regressed by 10 points on both "
            "temporal and multi-session tune slices, so it cannot proceed to dev60."
        ),
    }
    (repo / CONCLUSION).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _verify_conclusion(repo: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    conclusion = _read_json(repo, CONCLUSION)
    if conclusion.get("schema_version") != 1:
        raise VerificationError("unsupported tune conclusion schema")
    if conclusion.get("experiment_id") != "v3-phase3-tune1":
        raise VerificationError("unexpected tune conclusion id")
    if conclusion.get("source_sha256") != freeze["system"]["source"]["tree_sha256"]:
        raise VerificationError("conclusion source hash changed")
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
        freeze, aggregate = _verify_chain(repo)
        if args.capture:
            _capture(repo, freeze, aggregate)
            print(f"CAPTURED: {CONCLUSION}")
        conclusion = _verify_conclusion(repo, freeze)
    except VerificationError as exc:
        print(f"STOP: {exc}")
        return 2
    print("PASS: preserved v3.1 tune42 negative result · 84 rows · 66.7% tie")
    print(f"  source snapshot: {len(freeze['system']['source']['files'])} files")
    print(f"  evidence inventory: {len(conclusion['files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
