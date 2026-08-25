"""Content-addressed freeze records for v2 validation and final evaluation.

Git commits are convenient labels, but they are not required for a reproducible
run.  A freeze is stronger when it hashes the files and store that actually affect
the result.  This module works on a dirty tree and records enough information to
detect a changed prompt, config field, dependency lock, dataset, manifest, database
row, or vector index.

Hosted model ids are recorded but cannot be content-hashed by the client.  The
freeze calls that limitation out explicitly instead of pretending a mutable remote
alias is a local artifact.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llm_long_term_memory.config import ExperimentConfig
from llm_long_term_memory.evaluation.datasets import longmemeval as lme
from llm_long_term_memory.evaluation.manifest import load_manifest
from llm_long_term_memory.ingest import fingerprint, namespaced_sessions
from llm_long_term_memory.ingest.extract import Extractor
from llm_long_term_memory.ingest.fingerprint import schema_version
from llm_long_term_memory.ingest.pipeline import resolved_sessions_per_request
from llm_long_term_memory.ingest.two_stage import TwoStageExtractor
from llm_long_term_memory.llm import Limits, QuotaManager
from llm_long_term_memory.llm.usage import UsageTracker
from llm_long_term_memory.store import scoped_session_id

_STORE_TABLES = (
    "entities",
    "evidence",
    "memories",
    "memory_entities",
    "meta",
    "sessions",
    "turns",
)


class FreezeError(RuntimeError):
    """The requested system cannot be frozen as a complete, coherent artifact."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _relative(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def file_inventory(repo: Path, patterns: tuple[str, ...]) -> dict[str, dict[str, int | str]]:
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in repo.glob(pattern) if path.is_file())
    inventory = {
        _relative(path, repo): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(files)
    }
    if not inventory:
        raise FreezeError("the answer-affecting source inventory is empty")
    return inventory


def artifact_inventory(repo: Path, paths: tuple[Path, ...]) -> dict[str, dict[str, int | str]]:
    """Hash an explicit set of decision/protocol artifacts without reading them into reports."""
    resolved = tuple(path.resolve() for path in paths)
    if len(resolved) != len(set(resolved)):
        raise FreezeError("protocol artifact paths contain duplicates")
    missing = [path for path in resolved if not path.is_file()]
    if missing:
        raise FreezeError(f"{len(missing)} protocol artifact(s) do not exist")
    return {
        _relative(path, repo): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(resolved)
    }


def formal_protocol_paths(repo: Path, manifest_name: str) -> tuple[Path, ...]:
    """Decision-affecting records that must stay fixed through each formal phase."""
    train_grid = tuple(
        repo / "results" / "analysis" / "train150-context-grid-final" / f"{name}.json"
        for name in (
            "max-whole-cap20",
            "mean-whole-cap20",
            "mean-r1-cap20",
            "mean-r1-cap30",
            "mean-r1-cap40",
            "mean-r2-cap30",
            "max-r1-cap30",
        )
    )
    shared = (
        repo / "results" / "prereg-context-shape.md",
        repo / "results" / "prereg-v2-final.md",
        repo / "results" / "analysis" / "train150-ingest-state.final.json",
        repo / "results" / "analysis" / "train150-zero-yield.final.json",
        repo / "results" / "analysis" / "train150-zero-yield.final.md",
        repo / "results" / "analysis" / "train150-context-selection.final.json",
        repo / "results" / "analysis" / "train150-session-recall.final.json",
        *train_grid,
    )
    if manifest_name.lower() == "dev100":
        return shared
    if manifest_name.lower() == "test100":
        return (
            *shared,
            repo / "results" / "validation" / "dev100-aggregate.json",
            repo / "results" / "validation" / "dev100-decision.json",
        )
    return ()


def _tree_sha(inventory: dict[str, dict[str, int | str]]) -> str:
    return sha256_text(json.dumps(inventory, sort_keys=True, separators=(",", ":")))


def _cell(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, float):
        # JSON's representation is deterministic on supported Python versions;
        # using repr avoids locale-dependent formatting.
        return {"float": repr(value)}
    return value


def logical_sqlite_fingerprint(path: str | Path) -> dict[str, Any]:
    """Hash semantic rows, including WAL contents, without relying on file layout."""
    db = Path(path)
    if not db.exists():
        raise FreezeError(f"store database does not exist: {db}")
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    digest = hashlib.sha256()
    counts: dict[str, int] = {}
    meta: dict[str, str] = {}
    try:
        quick = connection.execute("PRAGMA quick_check").fetchone()
        if not quick or quick[0] != "ok":
            raise FreezeError(f"SQLite quick_check failed: {quick}")
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_keys:
            raise FreezeError(f"SQLite has {len(foreign_keys)} foreign-key violation(s)")
        existing = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table in _STORE_TABLES:
            if table not in existing:
                raise FreezeError(f"store is missing required table {table!r}")
            columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
            if not columns:
                raise FreezeError(f"store table {table!r} has no columns")
            quoted = ", ".join(f'"{name}"' for name in columns)
            rows = connection.execute(f'SELECT {quoted} FROM "{table}" ORDER BY rowid')
            count = 0
            digest.update(table.encode("utf-8") + b"\0")
            digest.update(json.dumps(columns, separators=(",", ":")).encode("utf-8") + b"\0")
            for row in rows:
                digest.update(
                    json.dumps([_cell(value) for value in row], ensure_ascii=False).encode("utf-8")
                    + b"\n"
                )
                count += 1
            counts[table] = count
        meta = dict(connection.execute("SELECT key, value FROM meta ORDER BY key"))
    finally:
        connection.close()
    return {
        "logical_sha256": digest.hexdigest(),
        "counts": counts,
        "meta": meta,
        "quick_check": "ok",
        "foreign_key_violations": 0,
    }


def _git_head(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            # Without this the locale codec decodes git's output, which is an
            # EncodingWarning the suite turns into an error (P7-0). A commit sha is
            # ASCII either way; the point is that no text-mode I/O site is exempt.
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _store_artifacts(
    repo: Path,
    store_dir: Path,
    store_name: str,
    manifest_path: Path,
    data_dir: Path,
    results_dir: Path,
    expected_fingerprint: dict[str, str],
    embedding_dim: int,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    dataset_path = data_dir / lme.VARIANTS[manifest.variant]
    if not dataset_path.exists():
        raise FreezeError(f"dataset file does not exist: {dataset_path}")
    every = lme.load(manifest.variant, data_dir)
    by_id = {instance.question_id: instance for instance in every}
    if set(manifest.question_ids) - set(by_id):
        raise FreezeError("manifest contains questions outside its declared dataset")
    pairs = namespaced_sessions([by_id[qid] for qid in manifest.question_ids])
    expected = {f"{namespace}:{session.session_id}" for namespace, session in pairs}
    expected_scoped = {
        scoped_session_id(namespace, session.session_id) for namespace, session in pairs
    }

    checkpoint = store_dir / f"{store_name}-ingest.json"
    if not checkpoint.exists():
        raise FreezeError(f"ingest checkpoint does not exist: {checkpoint}")
    progress = json.loads(checkpoint.read_text(encoding="utf-8"))
    done = set(progress.get("done_sessions", []))
    blocked = set(progress.get("blocked_sessions", []))
    terminal = done | blocked
    missing = expected - terminal
    extra = terminal - expected
    if missing or extra or len(done) + len(blocked) != len(terminal):
        raise FreezeError(
            "store is not exactly terminal for its manifest: "
            f"{len(missing)} missing, {len(extra)} extra, "
            f"{len(done & blocked)} both done and blocked"
        )

    db = store_dir / f"{store_name}.db"
    ids = store_dir / f"{store_name}-index.ids.json"
    vectors = store_dir / f"{store_name}-index.npy"
    for artifact in (ids, vectors):
        if not artifact.exists():
            raise FreezeError(f"store artifact does not exist: {artifact}")
    index_ids = json.loads(ids.read_text(encoding="utf-8"))
    if not isinstance(index_ids, list):
        raise FreezeError("vector index id artifact is not a list")
    if any(not isinstance(item, str) for item in index_ids):
        raise FreezeError("vector index id artifact contains a non-string id")
    if len(index_ids) != len(set(index_ids)):
        raise FreezeError("vector index id artifact contains duplicate ids")
    try:
        vector_rows = np.load(vectors, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise FreezeError(f"vector artifact cannot be loaded: {exc}") from exc
    if vector_rows.ndim != 2:
        raise FreezeError(f"vector artifact must be two-dimensional, got {vector_rows.shape}")
    database = logical_sqlite_fingerprint(db)
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        database_sessions = {str(row[0]) for row in connection.execute("SELECT id FROM sessions")}
        memory_rows = connection.execute("SELECT id, source_session_id FROM memories").fetchall()
    finally:
        connection.close()
    database_memory_ids = {str(row[0]) for row in memory_rows}
    memory_sources = {str(row[1]) for row in memory_rows if row[1] is not None}
    if database_sessions != expected_scoped:
        raise FreezeError(
            "database/manifest session mismatch: "
            f"{len(database_sessions - expected_scoped)} extra and "
            f"{len(expected_scoped - database_sessions)} missing"
        )
    checkpoint_memories = int(progress.get("memories_written", 0))
    if checkpoint_memories != database["counts"]["memories"]:
        raise FreezeError(
            "checkpoint/database memory mismatch: "
            f"{checkpoint_memories} versus {database['counts']['memories']}"
        )
    if len(index_ids) != database["counts"]["memories"]:
        raise FreezeError(
            "vector/database memory mismatch: "
            f"{len(index_ids)} index ids versus {database['counts']['memories']} memories"
        )
    if set(index_ids) != database_memory_ids:
        raise FreezeError("vector index ids do not match database memory ids")
    if vector_rows.shape[0] != len(index_ids):
        raise FreezeError(f"vector row/id mismatch: {vector_rows.shape[0]} versus {len(index_ids)}")
    if vector_rows.shape[1] != embedding_dim:
        raise FreezeError(
            f"vector dimension mismatch: {vector_rows.shape[1]} versus {embedding_dim}"
        )
    blocked_scoped = {scoped_session_id(*key.split(":", 1)) for key in blocked if ":" in key}
    blocked_with_memories = blocked_scoped & memory_sources
    if blocked_with_memories:
        raise FreezeError(
            f"{len(blocked_with_memories)} content-blocked sessions unexpectedly have memories"
        )
    try:
        stored_fingerprint = json.loads(database["meta"].get("ingest_fingerprint", "{}"))
    except (TypeError, json.JSONDecodeError) as exc:
        raise FreezeError(f"store ingest fingerprint is invalid: {exc}") from exc
    moved = fingerprint.differences(stored_fingerprint, expected_fingerprint)
    if moved:
        raise FreezeError("store/config ingest fingerprint mismatch: " + "; ".join(moved))

    usage_path = results_dir / "raw" / f"{store_name}.ingest.usage.json"
    if not usage_path.exists():
        raise FreezeError(f"ingest usage artifact does not exist: {usage_path}")
    try:
        usage_payload = json.loads(usage_path.read_text(encoding="utf-8"))
        usage_records = UsageTracker._load_records(usage_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FreezeError(f"invalid ingest usage artifact: {exc}") from exc
    usage = UsageTracker(records=usage_records)
    if usage_payload.get("summary") != usage.summary():
        raise FreezeError("ingest usage summary disagrees with its call records")
    checkpoint_requests = int(progress.get("extraction_requests", 0)) + int(
        progress.get("adjudication_requests", 0)
    )
    if usage.total_requests < checkpoint_requests:
        raise FreezeError(
            "ingest usage undercounts checkpoint requests: "
            f"{usage.total_requests} versus at least {checkpoint_requests}"
        )

    return {
        "name": store_name,
        "terminal": {
            "expected": len(expected),
            "done": len(done),
            "blocked": len(blocked),
        },
        "checkpoint": {
            "path": _relative(checkpoint, repo),
            "sha256": sha256_file(checkpoint),
        },
        "usage": {
            "path": _relative(usage_path, repo),
            "sha256": sha256_file(usage_path),
            "requests": usage.total_requests,
            "tokens": usage.total_tokens,
            "failures": sum(not record.ok for record in usage.records),
        },
        "database": database,
        "index": {
            "ids": len(index_ids),
            "unique_ids": len(set(index_ids)),
            "ids_sha256": sha256_file(ids),
            "vectors_sha256": sha256_file(vectors),
            "vectors_bytes": vectors.stat().st_size,
            "vectors_shape": list(vector_rows.shape),
        },
    }


@dataclass(frozen=True, slots=True)
class FreezeRequest:
    repo: Path
    config_path: Path
    manifest_path: Path
    data_dir: Path
    store_dir: Path
    results_dir: Path
    variants: tuple[str, ...]
    store_name: str | None = None
    pre_ingest_store_name: str | None = None
    protocol_paths: tuple[Path, ...] = ()


def capture_system(request: FreezeRequest) -> dict[str, Any]:
    repo = request.repo.resolve()
    config_path = request.config_path.resolve()
    manifest_path = request.manifest_path.resolve()
    cfg = ExperimentConfig.from_yaml(config_path)
    manifest = load_manifest(manifest_path)
    if not request.variants or len(set(request.variants)) != len(request.variants):
        raise FreezeError("freeze variants must be a non-empty list without duplicates")
    if request.store_name and request.pre_ingest_store_name:
        raise FreezeError("a freeze cannot be both pre-ingest and post-ingest")
    dataset_path = request.data_dir / lme.VARIANTS[manifest.variant]
    if not dataset_path.exists():
        raise FreezeError(f"dataset file does not exist: {dataset_path}")

    inventory = file_inventory(
        repo,
        (
            "src/llm_long_term_memory/**/*.py",
            "src/llm_long_term_memory/**/*.sql",
            "scripts/*.py",
            "pyproject.toml",
            "uv.lock",
        ),
    )
    protocol_inventory = artifact_inventory(repo, request.protocol_paths)
    extractor = TwoStageExtractor if cfg.ingest.two_stage else Extractor
    quota = QuotaManager(
        state_dir=request.store_dir / "quota",
        default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
    )
    quota.load_learned()
    sessions_per_request = resolved_sessions_per_request(
        cfg.ingest.sessions_per_request,
        quota.for_model(cfg.models.extractor).limits.tpm,
    )
    expected_fingerprint = fingerprint.from_config(
        cfg, sessions_per_request=sessions_per_request
    ).as_dict()
    from llm_long_term_memory.evaluation.judge import JUDGE_PROMPT_VERSION, JUDGE_SYSTEM
    from llm_long_term_memory.evaluation.runners.base import ANSWER_PROMPT_VERSION, ANSWER_SYSTEM

    system = {
        "schema_version": 5,
        "variants": list(request.variants),
        "pre_ingest_store_name": request.pre_ingest_store_name,
        "git_head_for_reference_only": _git_head(repo),
        "runtime": {"python": sys.version.split()[0]},
        "source": {"tree_sha256": _tree_sha(inventory), "files": inventory},
        "protocol": {
            "tree_sha256": _tree_sha(protocol_inventory),
            "files": protocol_inventory,
        },
        "config": {
            "path": _relative(config_path, repo),
            "sha256": sha256_file(config_path),
            "resolved": cfg.model_dump(mode="json"),
        },
        "prompts": {
            "extractor_version": extractor.version,
            "extractor_sha256": sha256_text(*extractor.prompt_texts()),
            "answer_version": ANSWER_PROMPT_VERSION,
            "answer_system_sha256": sha256_text(ANSWER_SYSTEM),
            "judge_version": JUDGE_PROMPT_VERSION,
            "judge_system_sha256": sha256_text(JUDGE_SYSTEM),
            "store_schema_sha256": schema_version(),
        },
        "models": {
            **cfg.models.model_dump(mode="json"),
            "limitation": (
                "Hosted provider model ids are recorded aliases; their remote weights "
                "cannot be content-hashed by this client."
            ),
        },
        "ingest_runtime": {
            "sessions_per_request": sessions_per_request,
            "fingerprint": expected_fingerprint,
        },
        "data": {
            "manifest": {
                "path": _relative(manifest_path, repo),
                "name": manifest.name,
                "variant": manifest.variant,
                "questions": len(manifest),
                "sha256": sha256_file(manifest_path),
            },
            "dataset": {
                "path": _relative(dataset_path, repo),
                "sha256": sha256_file(dataset_path),
                "bytes": dataset_path.stat().st_size,
            },
        },
        "store": None,
    }
    if request.store_name:
        system["store"] = _store_artifacts(
            repo,
            request.store_dir,
            request.store_name,
            manifest_path,
            request.data_dir,
            request.results_dir,
            expected_fingerprint,
            cfg.models.embedding_dim,
        )
    return system


def differences(frozen: Any, current: Any, path: str = "") -> list[str]:
    """Name every changed leaf while keeping secrets and dataset rows out."""
    if isinstance(frozen, dict) and isinstance(current, dict):
        changes: list[str] = []
        for key in sorted(set(frozen) | set(current)):
            here = f"{path}.{key}" if path else key
            if key not in frozen:
                changes.append(f"{here}: added")
            elif key not in current:
                changes.append(f"{here}: removed")
            else:
                changes.extend(differences(frozen[key], current[key], here))
        return changes
    if frozen != current:
        return [f"{path}: changed"]
    return []
