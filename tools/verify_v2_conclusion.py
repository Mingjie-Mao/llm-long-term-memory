"""Capture or verify the immutable, aggregate-only v2 conclusion package.

This tool is deliberately outside ``src/`` and ``scripts/``: those paths are part of the
answer-affecting v2 source inventory. Adding a post-experiment verifier must not pretend
that it existed when v2 was run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = Path("results/archive/v2-final")
CONCLUSION = ARCHIVE_DIR / "conclusion.json"
SOURCE_SNAPSHOT = ARCHIVE_DIR / "source.tar.gz"
FINAL_FREEZE = Path("results/frozen/v2-final/freeze.json")
PREINGEST_FREEZE = Path("results/frozen/v2-final-preingest/freeze.json")
LEDGER = Path("results/frozen/v2-final/one-shot.json")
FINAL_AGGREGATE = Path("results/final/test100-aggregate.json")
FINAL_MARKDOWN = Path("results/final/test100-aggregate.md")
CONFIG = Path("configs/v2.yaml")
INGEST_USAGE = Path("results/raw/test100.ingest.usage.json")
README = ARCHIVE_DIR / "README.md"

_ALLOWED_MANIFEST_KEYS = {
    "schema_version",
    "experiment_id",
    "created_at_utc",
    "producer",
    "fingerprints",
    "files",
    "summary_file",
    "outcome",
    "outcome_reason",
    "evidence",
}


class ConclusionError(RuntimeError):
    """The preserved v2 conclusion is incomplete, inconsistent or changed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_input_sha256(freeze: dict[str, Any]) -> str:
    data = freeze["system"]["data"]
    digest = hashlib.sha256()
    for value in (data["manifest"]["sha256"], data["dataset"]["sha256"]):
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConclusionError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConclusionError(f"expected one JSON object in {path}")
    return value


def _verify_source_snapshot(repo: Path, freeze: dict[str, Any]) -> None:
    archive = repo / SOURCE_SNAPSHOT
    if not archive.is_file():
        raise ConclusionError(f"missing v2 source snapshot: {SOURCE_SNAPSHOT}")
    expected = freeze["system"]["source"]["files"]
    try:
        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getmembers()
            names = [member.name.removeprefix("./") for member in members]
            if len(names) != len(set(names)):
                raise ConclusionError("source snapshot contains duplicate paths")
            if set(names) != set(expected):
                missing = sorted(set(expected) - set(names))
                extra = sorted(set(names) - set(expected))
                raise ConclusionError(
                    f"source snapshot inventory differs: {len(missing)} missing, "
                    f"{len(extra)} extra"
                )
            for member, name in zip(members, names, strict=True):
                if not member.isfile():
                    raise ConclusionError(f"source snapshot member is not a file: {name}")
                stream = handle.extractfile(member)
                if stream is None:
                    raise ConclusionError(f"cannot read source snapshot member: {name}")
                content = stream.read()
                record = expected[name]
                if len(content) != record["bytes"]:
                    raise ConclusionError(f"source snapshot size changed: {name}")
                if hashlib.sha256(content).hexdigest() != record["sha256"]:
                    raise ConclusionError(f"source snapshot hash changed: {name}")
    except (OSError, tarfile.TarError) as exc:
        raise ConclusionError(f"cannot read source snapshot: {exc}") from exc


def _verify_final_chain(
    repo: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze_path = repo / FINAL_FREEZE
    ledger_path = repo / LEDGER
    aggregate_path = repo / FINAL_AGGREGATE
    freeze = _read_json(freeze_path)
    ledger = _read_json(ledger_path)
    aggregate = _read_json(aggregate_path)
    system = freeze.get("system", {})

    if ledger.get("status") != "complete":
        raise ConclusionError("final one-shot ledger is not complete")
    if ledger.get("freeze_sha256") != _sha256_file(freeze_path):
        raise ConclusionError("one-shot ledger does not bind the final freeze")
    if ledger.get("aggregate_sha256") != _sha256_file(aggregate_path):
        raise ConclusionError("one-shot ledger does not bind the final aggregate")

    arms = ledger.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ConclusionError("one-shot ledger has no registered arms")
    arm_names = [item.get("name") for item in arms]
    variants = [item.get("variant") for item in arms]
    if arm_names != ["v2", "full_context", "naive_rag"]:
        raise ConclusionError("final arm names or order changed")
    if variants != system.get("variants"):
        raise ConclusionError("final arm variants differ from the freeze")
    if ledger.get("rows") != {name: 100 for name in arm_names}:
        raise ConclusionError("final ledger must contain exactly 100 rows per arm")

    aggregate_arms = aggregate.get("arms")
    if not isinstance(aggregate_arms, list):
        raise ConclusionError("final aggregate has no arm list")
    if [item.get("arm") for item in aggregate_arms] != arm_names:
        raise ConclusionError("final aggregate arms differ from the ledger")
    if aggregate.get("manifest") != {"name": "test100", "questions": 100}:
        raise ConclusionError("final aggregate is not the sealed test100 run")
    privacy = aggregate.get("privacy", {})
    if privacy.get("aggregate_only") is not True or privacy.get(
        "contains_question_ids_or_text"
    ) is not False:
        raise ConclusionError("final aggregate does not declare aggregate-only privacy")

    sealed = aggregate.get("sealed_artifacts")
    expected_sealed = {
        f"{name}.{suffix}"
        for name in arm_names
        for suffix in ("jsonl", "usage.json")
    }
    if not isinstance(sealed, dict) or set(sealed) != expected_sealed:
        raise ConclusionError("final aggregate does not bind exactly six sealed artifacts")
    for name, record in sealed.items():
        path = repo / "results" / "sealed" / "test100" / name
        if not path.is_file():
            raise ConclusionError(f"missing sealed artifact: {path.relative_to(repo)}")
        if path.stat().st_size != record.get("bytes"):
            raise ConclusionError(f"sealed artifact size changed: {path.relative_to(repo)}")
        if _sha256_file(path) != record.get("sha256"):
            raise ConclusionError(f"sealed artifact hash changed: {path.relative_to(repo)}")

    config = system.get("config", {})
    if config.get("path") != CONFIG.as_posix():
        raise ConclusionError("final freeze names an unexpected v2 config")
    if _sha256_file(repo / CONFIG) != config.get("sha256"):
        raise ConclusionError("v2 config differs from the final freeze")
    return freeze, ledger, aggregate


def _artifact(
    repo: Path, relative: Path, artifact_class: str
) -> dict[str, str | int]:
    path = repo / relative
    if not path.is_file():
        raise ConclusionError(f"missing conclusion artifact: {relative}")
    return {
        "path": relative.as_posix(),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "class": artifact_class,
    }


def _artifact_paths(freeze: dict[str, Any], aggregate: dict[str, Any]) -> dict[Path, str]:
    paths: dict[Path, str] = {
        SOURCE_SNAPSHOT: "audit",
        README: "audit",
        Path("tools/verify_v2_conclusion.py"): "audit",
        FINAL_FREEZE: "audit",
        PREINGEST_FREEZE: "audit",
        LEDGER: "outcome",
        CONFIG: "input",
        FINAL_AGGREGATE: "summary",
        FINAL_MARKDOWN: "summary",
        INGEST_USAGE: "usage",
    }
    for name in freeze["system"]["protocol"]["files"]:
        paths.setdefault(Path(name), "registry")
    for name in aggregate["sealed_artifacts"]:
        artifact_class = "usage" if name.endswith(".usage.json") else "raw"
        paths[Path("results/sealed/test100") / name] = artifact_class
    return paths


def _capture(repo: Path) -> dict[str, Any]:
    destination = repo / CONCLUSION
    if destination.exists():
        raise ConclusionError("v2 conclusion already exists and cannot be replaced")
    freeze, _, aggregate = _verify_final_chain(repo)
    _verify_source_snapshot(repo, freeze)
    system = freeze["system"]
    for path, record in system["protocol"]["files"].items():
        actual = repo / path
        if not actual.is_file() or actual.stat().st_size != record["bytes"]:
            raise ConclusionError(f"frozen protocol artifact changed: {path}")
        if _sha256_file(actual) != record["sha256"]:
            raise ConclusionError(f"frozen protocol artifact changed: {path}")

    files = [
        _artifact(repo, path, artifact_class)
        for path, artifact_class in sorted(
            _artifact_paths(freeze, aggregate).items(), key=lambda item: item[0].as_posix()
        )
    ]
    manifest = {
        "schema_version": 1,
        "experiment_id": "v2-final-test100",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "producer": {
            "command": "tools/verify_v2_conclusion.py --capture",
            "source_commit": system["git_head_for_reference_only"],
        },
        "fingerprints": {
            "source_sha256": system["source"]["tree_sha256"],
            "config_sha256": system["config"]["sha256"],
            "inputs_sha256": _combined_input_sha256(freeze),
            "store_sha256": system["store"]["database"]["logical_sha256"],
        },
        "files": files,
        "summary_file": FINAL_AGGREGATE.as_posix(),
        "outcome": "inconclusive",
        "outcome_reason": (
            "v2 was stable at 72% and used far less context, but its +7 point result "
            "over naive RAG was not statistically conclusive and full context was "
            "significantly more accurate."
        ),
        "evidence": [
            FINAL_FREEZE.as_posix(),
            LEDGER.as_posix(),
            FINAL_AGGREGATE.as_posix(),
            Path("results/validation/dev100-decision.json").as_posix(),
            CONFIG.as_posix(),
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _verify_manifest(repo: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    manifest = _read_json(repo / CONCLUSION)
    if set(manifest) != _ALLOWED_MANIFEST_KEYS:
        raise ConclusionError("v2 conclusion manifest has unexpected or missing fields")
    if manifest.get("schema_version") != 1:
        raise ConclusionError("unsupported v2 conclusion schema version")
    if manifest.get("experiment_id") != "v2-final-test100":
        raise ConclusionError("unexpected v2 conclusion experiment id")
    if manifest.get("outcome") not in {"positive", "negative", "inconclusive", "invalid"}:
        raise ConclusionError("invalid v2 conclusion outcome")
    system = freeze["system"]
    expected_fingerprints = {
        "source_sha256": system["source"]["tree_sha256"],
        "config_sha256": system["config"]["sha256"],
        "inputs_sha256": _combined_input_sha256(freeze),
        "store_sha256": system["store"]["database"]["logical_sha256"],
    }
    if manifest.get("fingerprints") != expected_fingerprints:
        raise ConclusionError("v2 conclusion fingerprints differ from the final freeze")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ConclusionError("v2 conclusion contains no file inventory")
    seen: set[str] = set()
    for record in files:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "bytes", "class"}:
            raise ConclusionError("invalid v2 conclusion file record")
        relative = record["path"]
        if relative in seen:
            raise ConclusionError(f"duplicate v2 conclusion file: {relative}")
        seen.add(relative)
        path = repo / relative
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ConclusionError(f"v2 conclusion file changed: {relative}")
        if _sha256_file(path) != record["sha256"]:
            raise ConclusionError(f"v2 conclusion file changed: {relative}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        if args.capture:
            manifest = _capture(repo)
            print(f"CAPTURED: {CONCLUSION} ({len(manifest['files'])} files)")
        freeze, _, aggregate = _verify_final_chain(repo)
        _verify_source_snapshot(repo, freeze)
        manifest = _verify_manifest(repo, freeze)
    except (ConclusionError, KeyError, TypeError, ValueError) as exc:
        print(f"STOP: {exc}")
        return 2
    accuracies = {item["arm"]: item["accuracy"] for item in aggregate["arms"]}
    print(
        "PASS: preserved v2 conclusion · "
        f"{len(manifest['files'])} files · "
        f"v2 {accuracies['v2']:.0%} / full context {accuracies['full_context']:.0%} / "
        f"naive RAG {accuracies['naive_rag']:.0%}"
    )
    print(f"  source snapshot: {len(freeze['system']['source']['files'])} files")
    print(f"  outcome: {manifest['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
