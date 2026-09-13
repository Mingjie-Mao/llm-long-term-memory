"""A backup nobody has restored is a belief, not a capability.

So the tests are drills, not assertions about the copy step. Each one damages the backup
in a way a real failure would and checks that the restore *reports* it rather than
returning a clean-looking directory.
"""

from __future__ import annotations

import errno
import json
import sqlite3
import sys
import weakref
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

br = pytest.importorskip("backup_restore")

from llm_long_term_memory.store import (  # noqa: E402
    Memory,
    NumpyFlatIndex,
    Session,
    SQLiteMemoryStore,
    Turn,
)


@pytest.fixture
def live(tmp_path):
    """A small store with a matching index, the way the service leaves one."""
    store = SQLiteMemoryStore(tmp_path / "live.db")
    store.initialize()
    store.add_session(
        Session(
            id="s1",
            user_id="alice",
            started_at=datetime(2026, 1, 1),
            source="test",
            turns=[
                Turn(
                    id="s1:0",
                    session_id="s1",
                    turn_index=0,
                    role="user",
                    content="I live in Canberra",
                    ts=datetime(2026, 1, 1),
                )
            ],
        )
    )
    memories = [
        Memory(
            f"m{i}",
            "alice",
            "semantic",
            f"fact {i}",
            4,
            subject="user",
            predicate="lives_in",
            object=f"place {i}",
            ingested_at=datetime(2026, 1, 1),
            source_session_id="s1",
        )
        for i in range(3)
    ]
    store.add_memories(memories)
    index = NumpyFlatIndex(tmp_path / "live-index", dim=2)
    index.add([m.id for m in memories], np.ones((3, 2), dtype=np.float32))
    index.save()
    store.close()
    return tmp_path / "live.db"


def test_a_backup_restores_into_an_empty_directory_intact(live, tmp_path):
    br.snapshot(live, tmp_path / "bk")
    report = br.restore(tmp_path / "bk", tmp_path / "out")

    assert report["ok"]
    assert report["integrity"] == "ok"
    assert report["counts_after"] == {"memories": 3, "sessions": 1, "turns": 1}
    assert report["index"]["consistent"]


def test_a_restore_refuses_a_directory_that_is_not_empty(live, tmp_path):
    """A drill that restored over existing data would be indistinguishable from one that
    worked, and could destroy the thing it was meant to protect."""
    br.snapshot(live, tmp_path / "bk")
    occupied = tmp_path / "out"
    occupied.mkdir()
    (occupied / "something.txt").write_text("in the way", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not empty"):
        br.restore(tmp_path / "bk", occupied)


def test_a_corrupted_backup_file_is_reported_not_restored(live, tmp_path):
    """Bit rot and truncated uploads are what checksums are for."""
    backup = tmp_path / "bk"
    br.snapshot(live, backup)
    (backup / "live.db").write_bytes(b"not a database any more")

    report = br.restore(backup, tmp_path / "out")

    assert not report["ok"]
    assert report["damaged"][0]["problem"] == "checksum changed since backup"
    # And the damaged file was not copied into the restore.
    assert not (tmp_path / "out" / "live.db").exists()


def test_a_missing_backup_file_is_reported(live, tmp_path):
    backup = tmp_path / "bk"
    br.snapshot(live, backup)
    (backup / "live-index.npy").unlink()

    report = br.restore(backup, tmp_path / "out")

    assert not report["ok"]
    assert any(d["problem"] == "missing from the backup" for d in report["damaged"])


# ------------------------------------------------------------ the asymmetry that matters


def test_an_index_missing_rows_is_reported_rather_than_hidden(live, tmp_path):
    """The database and the index are separate files and nothing makes them atomic. A
    backup taken mid-write can hold memories the index never saw, and a search would then
    silently miss them."""
    backup = tmp_path / "bk"
    manifest = br.snapshot(live, backup)
    # One memory drops out of the index, as a mid-write snapshot would leave it.
    ids_file = backup / "live-index.ids.json"
    kept = json.loads(ids_file.read_text(encoding="utf-8"))[:-1]
    ids_file.write_text(json.dumps(kept), encoding="utf-8")
    for item in manifest["files"]:
        if item["path"] == ids_file.name:
            item["sha256"] = br.sha256(ids_file)
    (backup / "backup.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = br.restore(backup, tmp_path / "out")

    assert not report["ok"]
    assert report["index"]["rows_never_indexed"] == 1
    assert not report["index"]["consistent"]


@pytest.fixture
def backup_with_an_empty_index(live, tmp_path, monkeypatch):
    """A backup whose index lost every row, resealed so only that asymmetry is wrong.

    The encoder is a stub, not the real one. What is under test is the rebuild logic, and CI
    deliberately omits the `embed` extra — a test that needed 2GB of torch to check a
    file-shuffling path would simply skip everywhere it matters.
    """
    backup = tmp_path / "bk"
    manifest = br.snapshot(live, backup)
    (backup / "live-index.ids.json").write_text(json.dumps([]), encoding="utf-8")
    np.save(backup / "live-index.npy", np.zeros((0, 2), dtype=np.float32))
    for item in manifest["files"]:
        if item["path"].startswith("live-index"):
            item["sha256"] = br.sha256(backup / item["path"])
    (backup / "backup.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    import llm_long_term_memory.embed as embed

    class _Stub:
        def encode(self, texts, show_progress=False):
            return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(embed, "Encoder", lambda *a, **k: _Stub())
    return backup


def test_rebuilding_the_index_repairs_that_asymmetry(backup_with_an_empty_index, tmp_path):
    """`--rebuild-index` re-encodes from the restored rows. Local encoder only, so a
    recovery drill never needs a provider."""
    report = br.restore(backup_with_an_empty_index, tmp_path / "out", rebuild_index=True)

    assert report["index_rebuilt"]
    assert report["ok"]
    rebuilt = json.loads((tmp_path / "out" / "live-index.ids.json").read_text(encoding="utf-8"))
    assert sorted(rebuilt) == ["m0", "m1", "m2"]


def test_the_rebuild_never_writes_a_file_that_is_still_mapped(
    backup_with_an_empty_index, tmp_path, monkeypatch
):
    """Windows' rule for mapped files, enforced on every platform.

    The index check used to open the vectors with `mmap_mode="r"` and was still holding the
    map when the rebuild wrote that same file. POSIX allows it; Windows refuses with EINVAL.
    So `--rebuild-index` failed on the real recovery path while the suite passed where it
    was written, and the test above caught it only on the Windows runner, after a push.
    Here a write to a file whose `np.load` map is still alive raises that error anywhere.
    """
    real_load, real_save = np.load, np.save
    mapped = []

    def load(file, *args, **kwargs):
        array = real_load(file, *args, **kwargs)
        if isinstance(array, np.memmap):
            mapped.append((Path(array.filename).resolve(), weakref.ref(array)))
        return array

    def save(file, arr, *args, **kwargs):
        target = Path(file).resolve()
        if any(path == target and alive() is not None for path, alive in mapped):
            raise OSError(errno.EINVAL, "written while still memory-mapped", str(file))
        return real_save(file, arr, *args, **kwargs)

    monkeypatch.setattr(np, "load", load)
    monkeypatch.setattr(np, "save", save)

    report = br.restore(backup_with_an_empty_index, tmp_path / "out", rebuild_index=True)

    assert report["index_rebuilt"]
    assert report["ok"]


# ------------------------------------------------------------ the copy itself


def test_the_backup_is_one_closed_file_with_no_wal_sidecar(live, tmp_path):
    """A file copy of a live WAL database can land mid-transaction. The copy goes through
    SQLite's own backup API and is finished in DELETE mode, so the hash describes the
    whole database rather than part of one."""
    backup = tmp_path / "bk"
    br.snapshot(live, backup)

    assert (backup / "live.db").is_file()
    assert not (backup / "live.db-wal").exists()
    assert not (backup / "live.db-shm").exists()
    with sqlite3.connect(f"file:{backup / 'live.db'}?mode=ro", uri=True) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"


def test_the_manifest_records_what_was_copied(live, tmp_path):
    manifest = br.snapshot(live, tmp_path / "bk")

    assert manifest["counts"] == {"memories": 3, "sessions": 1, "turns": 1}
    assert {item["path"] for item in manifest["files"]} == {
        "live.db",
        "live-index.npy",
        "live-index.ids.json",
    }
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])


def test_backing_up_a_missing_store_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        br.snapshot(tmp_path / "nope.db", tmp_path / "bk")
