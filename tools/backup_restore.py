"""Back up a store and index, and prove the restore by rebuilding into an empty directory.

A backup nobody has restored is a belief, not a capability. So this does both, and the
drill is the point: `verify` restores into a fresh directory and checks that what comes
back is what went in — row counts, the database's own integrity check, and the agreement
between the vector index and the memories it claims to index.

**Why the index is rebuilt rather than trusted.** The index is a separate file from the
database and nothing makes them atomic. A backup taken mid-write can hold a database with
memories the index has never seen, and that asymmetry is invisible until a search silently
misses them. `--rebuild-index` re-encodes from the restored rows, which is the only way to
know the two agree; without it the check reports the drift rather than hiding it.

SQLite is copied through its own backup API, not `cp`. A live database has a WAL sidecar,
and a file copy of one can land mid-transaction — the failure mode that produced an
unreadable snapshot in the v4.2 freeze until it was fixed there too.

Zero provider calls: the encoder used for a rebuild is the local ONNX/MiniLM one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in ("memories", "sessions", "turns")
    }


def snapshot(store: Path, destination: Path) -> dict:
    """Copy one store and its index into `destination`, and describe what was copied."""
    if not store.is_file():
        raise FileNotFoundError(f"no store at {store}")
    destination.mkdir(parents=True, exist_ok=True)

    target = destination / store.name
    with closing(sqlite3.connect(f"file:{store}?mode=ro", uri=True)) as source:
        original = counts(source)
        with closing(sqlite3.connect(target)) as copy:
            source.backup(copy)
            # A WAL-mode source leaves -wal/-shm sidecars beside the copy. Finishing it as
            # one closed file in DELETE mode is what makes the backup a single artifact
            # whose hash means something.
            copy.execute("PRAGMA journal_mode=DELETE").fetchone()
            if copy.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RuntimeError("the copied database failed its integrity check")

    files = [{"path": target.name, "sha256": sha256(target), "bytes": target.stat().st_size}]
    stem = store.with_suffix("").name
    for suffix in ("-index.npy", "-index.ids.json"):
        companion = store.parent / f"{stem}{suffix}"
        if companion.is_file():
            shutil.copyfile(companion, destination / companion.name)
            files.append(
                {
                    "path": companion.name,
                    "sha256": sha256(destination / companion.name),
                    "bytes": companion.stat().st_size,
                }
            )

    manifest = {
        "schema_version": 1,
        "taken_at_utc": datetime.now(UTC).isoformat(),
        "source": str(store),
        "counts": original,
        "files": files,
        "index_present": len(files) > 1,
    }
    (destination / "backup.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def restore(backup: Path, into: Path, *, rebuild_index: bool = False) -> dict:
    """Restore into an empty directory and report whether it came back intact."""
    manifest = json.loads((backup / "backup.json").read_text(encoding="utf-8"))
    if into.exists() and any(into.iterdir()):
        raise RuntimeError(f"{into} is not empty; a drill must restore into a clean place")
    into.mkdir(parents=True, exist_ok=True)

    damaged = []
    for item in manifest["files"]:
        source = backup / item["path"]
        if not source.is_file():
            damaged.append({"path": item["path"], "problem": "missing from the backup"})
            continue
        if sha256(source) != item["sha256"]:
            damaged.append({"path": item["path"], "problem": "checksum changed since backup"})
            continue
        shutil.copyfile(source, into / item["path"])

    database = into / Path(manifest["source"]).name
    restored: dict[str, int] = {}
    integrity = "not checked"
    if database.is_file():
        with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as connection:
            integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
            restored = counts(connection)

    report = {
        "restored_into": str(into),
        "integrity": integrity,
        "counts_before": manifest["counts"],
        "counts_after": restored,
        "counts_match": restored == manifest["counts"],
        "damaged": damaged,
        "index_rebuilt": False,
    }
    report["index"] = _check_index(into, database, rebuild=rebuild_index, report=report)
    report["ok"] = (
        not damaged
        and integrity == "ok"
        and report["counts_match"]
        and report["index"]["consistent"]
    )
    return report


def _check_index(folder: Path, database: Path, *, rebuild: bool, report: dict) -> dict:
    """Do the vector index and the restored memories agree?

    Reported rather than repaired by default. A restore that silently rebuilt would hide
    the asymmetry that makes a mid-write backup dangerous, and the whole point of a drill
    is to find out whether the pair survived together.
    """
    # A damaged or uncopied database is already reported by the caller; reaching past it
    # to read the index would raise instead, and turn a clear report into a traceback.
    if not database.is_file():
        return {"present": False, "consistent": False, "detail": "no database was restored"}
    ids_file = folder / f"{database.with_suffix('').name}-index.ids.json"
    vectors_file = folder / f"{database.with_suffix('').name}-index.npy"
    if not (ids_file.is_file() and vectors_file.is_file()):
        return {"present": False, "consistent": not rebuild, "detail": "no index in the backup"}

    import numpy as np

    indexed = json.loads(ids_file.read_text(encoding="utf-8"))
    # Read, do not memory-map. Windows refuses to write a file that is still mapped, so
    # `--rebuild-index` failed with EINVAL on the very file it was trying to replace —
    # on the real recovery path, not only in a test. A full read is proportionate: the
    # drilled store's index is under 10MB, and the restore has just hashed and copied it.
    vector_count = int(np.load(vectors_file).shape[0])
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as connection:
        stored = {row[0] for row in connection.execute("SELECT id FROM memories")}

    missing = sorted(stored - set(indexed))
    orphans = sorted(set(indexed) - stored)
    detail = {
        "present": True,
        "indexed": len(indexed),
        "in_database": len(stored),
        "rows_never_indexed": len(missing),
        "vectors_without_a_row": len(orphans),
        "shape_matches_ids": vector_count == len(indexed),
    }
    if rebuild and (missing or orphans or not detail["shape_matches_ids"]):
        _rebuild(folder, database, stored)
        report["index_rebuilt"] = True
        detail["rebuilt"] = True
        detail["consistent"] = True
        return detail
    detail["consistent"] = not missing and not orphans and detail["shape_matches_ids"]
    return detail


def _rebuild(folder: Path, database: Path, stored: set[str]) -> None:
    """Re-encode every restored memory. Local encoder only; no provider call."""
    import numpy as np

    from llm_long_term_memory.embed import Encoder

    encoder = Encoder()
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as connection:
        rows = connection.execute("SELECT id, content FROM memories ORDER BY id").fetchall()
    ids = [row[0] for row in rows]
    vectors = encoder.encode([row[1] for row in rows]) if rows else np.zeros((0, 384))
    stem = database.with_suffix("").name
    np.save(folder / f"{stem}-index.npy", np.asarray(vectors, dtype=np.float32))
    (folder / f"{stem}-index.ids.json").write_text(json.dumps(ids), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    take = sub.add_parser("backup", help="copy a store and its index into a folder")
    take.add_argument("--store", type=Path, required=True)
    take.add_argument("--into", type=Path, required=True)

    drill = sub.add_parser("verify", help="restore into an empty folder and check it")
    drill.add_argument("--backup", type=Path, required=True)
    drill.add_argument("--into", type=Path, required=True)
    drill.add_argument(
        "--rebuild-index",
        action="store_true",
        help="re-encode the index from restored rows when it disagrees with them",
    )

    args = parser.parse_args()
    if args.command == "backup":
        manifest = snapshot(args.store, args.into)
        print(f"backed up {manifest['source']}")
        for table, n in manifest["counts"].items():
            print(f"  {table:10s} {n}")
        print(f"  files      {len(manifest['files'])}")
        print(f"\nwrote {args.into / 'backup.json'}")
        return 0

    report = restore(args.backup, args.into, rebuild_index=args.rebuild_index)
    print(f"restored into {report['restored_into']}")
    print(f"  integrity      {report['integrity']}")
    print(f"  counts match   {report['counts_match']}  {report['counts_after']}")
    index = report["index"]
    if index.get("present"):
        print(
            f"  index          {index['indexed']} vectors / {index['in_database']} rows, "
            f"never indexed {index['rows_never_indexed']}, "
            f"orphans {index['vectors_without_a_row']}"
            + ("  (rebuilt)" if index.get("rebuilt") else "")
        )
    for item in report["damaged"]:
        print(f"  DAMAGED        {item['path']}: {item['problem']}")
    print(f"\n{'OK' if report['ok'] else 'FAILED'}: restore drill")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
