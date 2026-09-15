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
import sys
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore  # noqa: E402
from llm_long_term_memory.store.erasure import (  # noqa: E402
    JOURNAL_SUFFIX,
    ErasureJournal,
    journal_path_for,
)


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
    # Stamped before the copy starts, not after it ends. The copy is the database as it stood
    # when the read began, and the restore replays only erasures served after this moment:
    # stamped at the end of a long copy, an erasure committed mid-copy would carry an earlier
    # time, be judged already reflected in the backup, and be skipped with its data inside.
    taken_at = datetime.now(UTC).isoformat()
    with (
        closing(sqlite3.connect(f"file:{store}?mode=ro", uri=True)) as source,
        closing(sqlite3.connect(target)) as copy,
    ):
        source.backup(copy)
        # A WAL-mode source leaves -wal/-shm sidecars beside the copy. Finishing it as
        # one closed file in DELETE mode is what makes the backup a single artifact
        # whose hash means something.
        copy.execute("PRAGMA journal_mode=DELETE").fetchone()
        if copy.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("the copied database failed its integrity check")
        original = counts(copy)

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
        # 2: `taken_at_utc` marks the start of the copy. Version 1 stamped its end, which the
        # erasure replay cannot use as a cutoff.
        "schema_version": 2,
        "taken_at_utc": taken_at,
        "source": str(store),
        "counts": original,
        "files": files,
        "index_present": len(files) > 1,
    }
    (destination / "backup.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def prune(folder: Path, keep: int) -> dict:
    """Keep the newest `keep` backups under `folder` and delete the rest.

    A retention policy is part of an erasure story, not housekeeping: every backup kept is
    another copy that a restore could bring an erased namespace back from, and "we keep
    everything forever" and "we honour deletion requests" cannot both be true. Keeping a
    stated number makes the exposure a number too.

    Newest is decided by each backup's own `taken_at_utc`, not by file mtime: a copied or
    rsynced backup directory has whatever mtime the copy gave it, and pruning by that
    would delete by accident of transport.
    """
    if keep < 1:
        raise ValueError("keep at least one backup; pruning to zero is not a retention policy")
    dated: list[tuple[str, Path]] = []
    for candidate in sorted(folder.iterdir()):
        manifest = candidate / "backup.json"
        if not manifest.is_file():
            continue
        taken = json.loads(manifest.read_text(encoding="utf-8")).get("taken_at_utc") or ""
        dated.append((taken, candidate))
    dated.sort(key=lambda pair: pair[0], reverse=True)

    removed = []
    for _, stale in dated[keep:]:
        shutil.rmtree(stale)
        removed.append(stale.name)
    return {
        "kept": [path.name for _, path in dated[:keep]],
        "removed": removed,
        # Stated because it is the number that matters for an erasure: this is how far
        # back a restore could reach, and therefore how long deleted data remains
        # recoverable from a backup.
        "oldest_kept": dated[min(keep, len(dated)) - 1][0] if dated else None,
    }


def restore(
    backup: Path,
    into: Path,
    *,
    rebuild_index: bool = False,
    erasures: Path | None = None,
) -> dict:
    """Restore into an empty directory and report whether it came back intact.

    `erasures` is the store's append-only journal of served deletion requests. A backup
    taken before an erasure still contains that namespace, so restoring one puts erased data
    back — searchable, with nothing saying it should not be there. Replaying the journal is
    what closes that; a restore run without one is reported as `ok: False` rather than
    quietly succeeding, because "intact" and "lawful to serve" are different claims.

    The replay runs before the index check, so the check describes the copy that would be
    served: its rows and vectors after the erasures, not before them.
    """
    started = time.monotonic()
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
    report["erasures"] = _replay_erasures(into, database, erasures, manifest)
    report["index"] = _check_index(into, database, rebuild=rebuild_index, report=report)
    # Measured, not estimated. An operator quoting a recovery objective needs a number
    # from a drill that actually ran, and the data window is what the backup's own
    # timestamp says was lost — everything written after it is not in this copy.
    report["rto_seconds"] = round(time.monotonic() - started, 3)
    report["data_window"] = _data_window(manifest)
    report["ok"] = (
        not damaged
        and integrity == "ok"
        and report["counts_match"]
        and report["index"]["consistent"]
        and report["erasures"]["replayed"]
    )
    return report


def _data_window(manifest: dict) -> dict:
    """How much time this copy is behind now: the RPO this backup actually offers."""
    taken = manifest.get("taken_at_utc")
    if not taken:
        return {"backup_taken_at": None, "behind_seconds": None}
    try:
        moment = datetime.fromisoformat(taken)
    except ValueError:
        return {"backup_taken_at": taken, "behind_seconds": None}
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return {
        "backup_taken_at": taken,
        "behind_seconds": round((datetime.now(UTC) - moment).total_seconds(), 3),
    }


def _replay_erasures(
    folder: Path, database: Path, journal_path: Path | None, manifest: dict
) -> dict:
    """Re-apply to the restored copy every erasure served after the backup was taken.

    The counts are worth reading. `rows_removed` above zero means the backup predates an
    erasure and that data really was about to come back. `skipped_before_backup` counts
    erasures the copy already reflects: whatever those namespaces hold in it was written
    after they were erased — a user who deleted their data and carried on — and replaying
    them deleted that new data while reporting it as erased rows correctly removed.

    Three states for the journal, and the third is the one that matters. A journal that is
    given is replayed, and a given path that does not exist fails rather than reading as
    empty: a typo in the one argument that keeps erased data erased must not pass the drill.
    Given none, the store's own directory is checked: if it is reachable and holds no
    journal, no erasure has ever been served and there is nothing to replay. If it is *not*
    reachable — restoring onto a new machine, which is exactly when a restore happens — the
    absence of a journal proves nothing, and saying "nothing to replay" there would be a
    guess dressed as a check. That case is reported as not replayed.

    The replay is the store's own erasure run against the copy, not a copy of its SQL. A
    hand-written list of DELETEs missed the entity table, which holds the names a user
    mentioned, and ran without foreign keys, so no cascade removed entity links or evidence
    either. The erased memories' vectors leave the restored index too, as they leave the
    online one.
    """
    if not database.is_file():
        return {
            "replayed": False,
            "reason": "no database was restored",
            "namespaces": 0,
            "rows_removed": 0,
        }

    discovered = False
    if journal_path is not None and not journal_path.is_file():
        return {
            "replayed": False,
            "reason": (
                f"the erasure journal {journal_path} does not exist; if this store has never "
                f"served an erasure, create the file empty to say so"
            ),
            "namespaces": 0,
            "rows_removed": 0,
        }
    if journal_path is None:
        source = Path(manifest.get("source", ""))
        if not source.parent.is_dir():
            return {
                "replayed": False,
                "reason": (
                    f"no erasure journal was given and the original store directory "
                    f"({source.parent}) is not reachable from here, so whether an erasure "
                    f"must be replayed cannot be determined; pass --erasures "
                    f"<dir>/{source.stem}{JOURNAL_SUFFIX}"
                ),
                "namespaces": 0,
                "rows_removed": 0,
            }
        candidate = journal_path_for(source)
        if not candidate.is_file():
            return {
                "replayed": True,
                "journal": None,
                "reason": f"no erasure has been served: {candidate} does not exist",
                "namespaces": 0,
                "rows_removed": 0,
            }
        journal_path = candidate
        discovered = True

    journal = ErasureJournal(journal_path)
    cutoff = _copy_started(manifest)
    namespaces = journal.namespaces(served_after=cutoff)
    already_reflected = set(journal.namespaces()) - set(namespaces)
    removed = {"memories": 0, "sessions": 0, "turns": 0}
    erased_ids: list[str] = []
    if namespaces:
        # Opened the way the service opens a store: the schema script turns foreign keys on,
        # so deleting a memory cascades to its entity links and evidence exactly as it did
        # online, and an older backup gains any table the erasure touches.
        store = SQLiteMemoryStore(database)
        try:
            store.initialize()
            for user_id in namespaces:
                gone = store.hard_delete_user(user_id)
                erased_ids.extend(gone.pop("ids", []))
                for table in removed:
                    removed[table] += int(gone.get(table, 0))
        finally:
            store.close()
        # Opening the copy the service's way switched it to WAL. It is finished back in DELETE
        # mode, as the backup itself is, so the restored database stays one closed file and
        # the read-only checks that follow never depend on creating a WAL sidecar.
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("PRAGMA journal_mode=DELETE").fetchone()
    return {
        "replayed": True,
        "journal": str(journal_path),
        "found_beside_the_store": discovered,
        "cutoff": cutoff.isoformat() if cutoff else None,
        "namespaces": len(namespaces),
        "skipped_before_backup": len(already_reflected),
        "rows_removed": sum(removed.values()),
        "removed": removed,
        "vectors_removed": _drop_vectors(folder, database, erased_ids),
    }


def _copy_started(manifest: dict) -> datetime | None:
    """When the backup's copy began, or None when its manifest cannot say.

    Only schema 2 stamps `taken_at_utc` before copying. A version-1 stamp marks the end of
    the copy, so an erasure committed mid-copy can be older than the stamp and still be
    inside; with no usable cutoff every erasure is replayed, which can only err towards
    deleting a namespace's post-erasure writes, never towards serving erased data.
    """
    if int(manifest.get("schema_version") or 1) < 2:
        return None
    try:
        moment = datetime.fromisoformat(manifest.get("taken_at_utc") or "")
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _drop_vectors(folder: Path, database: Path, memory_ids: list[str]) -> int:
    """Take erased memories' vectors out of the restored index, as the online erasure does.

    A vector is derived from the erased text; an index still holding them for a namespace
    with no rows is not an erasure. Done through the index's own remove and save, so the
    files are the ones the service would have written. The dimension is read from the file:
    an index emptied by the removal must still accept the next write's vectors.
    """
    prefix = folder / f"{database.with_suffix('').name}-index"
    vectors_file = folder / f"{prefix.name}.npy"
    if not memory_ids or not vectors_file.is_file():
        return 0
    import numpy as np

    dim = int(np.load(vectors_file, allow_pickle=False).shape[-1])
    index = NumpyFlatIndex(prefix, dim=dim)
    removed = index.remove(memory_ids)
    if removed:
        index.save()
    return removed


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
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as connection:
        stored = {row[0] for row in connection.execute("SELECT id FROM memories")}
    if not (ids_file.is_file() and vectors_file.is_file()):
        detail = {"present": False, "consistent": False, "detail": "index files are missing"}
    else:
        import numpy as np

        indexed = json.loads(ids_file.read_text(encoding="utf-8"))
        # Read rather than memory-map: Windows cannot overwrite a mapped index.
        vectors = np.load(vectors_file, allow_pickle=False)
        missing = stored - set(indexed)
        orphans = set(indexed) - stored
        detail = {
            "present": True,
            "indexed": len(indexed),
            "in_database": len(stored),
            "rows_never_indexed": len(missing),
            "vectors_without_a_row": len(orphans),
            "duplicate_ids": len(indexed) - len(set(indexed)),
            "shape_matches_ids": vectors.ndim == 2 and vectors.shape[0] == len(indexed),
            "finite_vectors": bool(np.isfinite(vectors).all()),
        }
        detail["consistent"] = (
            not missing
            and not orphans
            and not detail["duplicate_ids"]
            and detail["shape_matches_ids"]
            and detail["finite_vectors"]
        )
    if rebuild and not detail["consistent"]:
        _rebuild(folder, database, stored)
        report["index_rebuilt"] = True
        checked = _check_index(folder, database, rebuild=False, report=report)
        checked["rebuilt"] = True
        return checked
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

    sweep = sub.add_parser("prune", help="delete all but the newest N backups in a folder")
    sweep.add_argument("--folder", type=Path, required=True)
    sweep.add_argument(
        "--keep",
        type=int,
        required=True,
        help="how many backups to retain; every one kept is another copy an erased "
        "namespace could be restored from",
    )

    drill = sub.add_parser("verify", help="restore into an empty folder and check it")
    drill.add_argument("--backup", type=Path, required=True)
    drill.add_argument("--into", type=Path, required=True)
    drill.add_argument(
        "--rebuild-index",
        action="store_true",
        help="re-encode the index from restored rows when it disagrees with them",
    )
    drill.add_argument(
        "--erasures",
        type=Path,
        help=(
            f"the store's <store>{JOURNAL_SUFFIX} journal, when it is not beside the "
            "original store. Erasures served after this backup are replayed; a path that "
            "does not exist fails the drill"
        ),
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

    if args.command == "prune":
        result = prune(args.folder, args.keep)
        print(f"kept {len(result['kept'])}, removed {len(result['removed'])}")
        for name in result["removed"]:
            print(f"  removed  {name}")
        print(f"  oldest kept backup was taken {result['oldest_kept']}")
        return 0

    report = restore(
        args.backup,
        args.into,
        rebuild_index=args.rebuild_index,
        erasures=args.erasures,
    )
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
    erasures = report["erasures"]
    if not erasures["replayed"]:
        print(f"  ERASURES       NOT replayed: {erasures['reason']}")
    elif erasures.get("journal") is None:
        print(f"  erasures       nothing to replay: {erasures['reason']}")
    else:
        print(
            f"  erasures       {erasures['namespaces']} replayed "
            f"({erasures['rows_removed']} rows, {erasures['vectors_removed']} vectors removed), "
            f"{erasures['skipped_before_backup']} already reflected in the backup"
            + ("  (journal found beside the store)" if erasures["found_beside_the_store"] else "")
        )
        if erasures["cutoff"] is None:
            print("                 this backup cannot date its copy: every erasure was replayed")
    window = report["data_window"]
    print(f"  data window    backup taken {window['backup_taken_at']}", end="")
    if window["behind_seconds"] is not None:
        print(f" ({window['behind_seconds'] / 3600:.2f}h of writes not in this copy)")
    else:
        print()
    print(f"  recovery time  {report['rto_seconds']}s measured for this drill")
    for item in report["damaged"]:
        print(f"  DAMAGED        {item['path']}: {item['problem']}")
    print(f"\n{'OK' if report['ok'] else 'FAILED'}: restore drill")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
