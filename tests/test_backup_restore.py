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
    ErasureJournal,
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

    # Version 2 stamps the start of the copy, which is what the erasure replay cuts at.
    assert manifest["schema_version"] == 2
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


def test_database_only_backup_does_not_claim_search_is_restored(live, tmp_path):
    live.with_name("live-index.npy").unlink()
    live.with_name("live-index.ids.json").unlink()
    br.snapshot(live, tmp_path / "bk")
    report = br.restore(tmp_path / "bk", tmp_path / "out")
    assert report["counts_match"]
    assert not report["index"]["present"]
    assert not report["ok"], "restored rows without an index are not a searchable restore"


def test_missing_index_can_be_rebuilt_and_is_checked_again(live, tmp_path, monkeypatch):
    import llm_long_term_memory.embed as embed

    live.with_name("live-index.npy").unlink()
    live.with_name("live-index.ids.json").unlink()
    br.snapshot(live, tmp_path / "bk")

    class Encoder:
        def encode(self, texts):
            return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(embed, "Encoder", Encoder)
    report = br.restore(tmp_path / "bk", tmp_path / "out", rebuild_index=True)
    assert report["index_rebuilt"]
    assert report["index"]["indexed"] == 3
    assert report["ok"]


def test_duplicate_ids_are_not_a_consistent_index(live, tmp_path):
    live.with_name("live-index.ids.json").write_text(
        json.dumps(["m0", "m1", "m2", "m2"]), encoding="utf-8"
    )
    np.save(live.with_name("live-index.npy"), np.ones((4, 2), dtype=np.float32))
    br.snapshot(live, tmp_path / "bk")
    report = br.restore(tmp_path / "bk", tmp_path / "out")
    assert not report["ok"], "set equality must not hide duplicate vector identities"


def test_rebuild_success_is_not_assumed(backup_with_an_empty_index, tmp_path, monkeypatch):
    monkeypatch.setattr(br, "_rebuild", lambda *args: None)
    report = br.restore(backup_with_an_empty_index, tmp_path / "out", rebuild_index=True)
    assert not report["ok"], "a rebuild that did not repair the index cannot pass"


# ------------------------------------------------- erasure survives a restore

# An erasure that only deletes the online rows is incomplete, and the gap is invisible:
# every backup taken before it still holds the data. Restore one and the erased namespace
# is back, searchable, with nothing saying it should not be. These drills are that
# scenario, in the order it really happens — write, back up, erase, restore.


@pytest.fixture
def erased_after_backup(live, tmp_path):
    """A backup taken before an erasure, and the journal that records the erasure."""
    br.snapshot(live, tmp_path / "bk")

    store = SQLiteMemoryStore(live)
    store.initialize()
    removed = store.hard_delete_user("alice")
    store.close()

    journal = ErasureJournal.beside(live)
    journal.record(
        "alice",
        memories=removed["memories"],
        sessions=removed["sessions"],
        turns=removed["turns"],
    )
    return journal


def rows_for(database: Path, user_id: str) -> int:
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        return connection.execute(
            "SELECT count(*) FROM memories WHERE user_id = ?", (user_id,)
        ).fetchone()[0]


def test_a_restore_without_the_journal_brings_erased_data_back(erased_after_backup, tmp_path):
    """The failure this journal exists for. The restore itself is perfectly intact — that
    is the point: nothing about row counts or integrity can detect it."""
    journal = erased_after_backup
    journal.path.unlink()  # the operator does not have it, and the store dir has none

    report = br.restore(tmp_path / "bk", tmp_path / "out")

    assert report["integrity"] == "ok"
    assert report["counts_match"], "the copy is intact; that is exactly why this is unsafe"
    assert rows_for(tmp_path / "out" / "live.db", "alice") == 3, "erased data came back"


def test_replaying_the_journal_keeps_erased_data_erased(erased_after_backup, tmp_path):
    report = br.restore(tmp_path / "bk", tmp_path / "out", erasures=erased_after_backup.path)

    assert report["erasures"]["replayed"]
    assert report["erasures"]["namespaces"] == 1
    assert report["erasures"]["rows_removed"] == 5  # 3 memories, 1 session, 1 turn
    assert rows_for(tmp_path / "out" / "live.db", "alice") == 0


def test_the_replay_removes_the_turns_and_keys_too_not_only_the_memories(
    erased_after_backup, tmp_path
):
    """A restored turn is the original text the memory was extracted from, so leaving it
    behind would put back the very content the erasure removed."""
    br.restore(tmp_path / "bk", tmp_path / "out", erasures=erased_after_backup.path)

    with sqlite3.connect(tmp_path / "out" / "live.db") as connection:
        assert connection.execute("SELECT count(*) FROM turns").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0


def test_the_journal_is_found_beside_the_store_when_it_is_reachable(erased_after_backup, tmp_path):
    """The common case: restoring on the machine the store lives on. Nobody should have to
    remember a flag for the erasure to hold."""
    report = br.restore(tmp_path / "bk", tmp_path / "out")

    assert report["erasures"]["replayed"]
    assert report["erasures"]["found_beside_the_store"]
    assert rows_for(tmp_path / "out" / "live.db", "alice") == 0


def test_an_unreachable_store_directory_is_not_read_as_nothing_to_replay(live, tmp_path):
    """Restoring onto a new machine is when a restore actually happens, and there the
    absence of a journal proves nothing. Reporting "nothing to replay" would be a guess
    dressed as a check."""
    br.snapshot(live, tmp_path / "bk")
    manifest_path = tmp_path / "bk" / "backup.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"] = "/somewhere/that/does/not/exist/live.db"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = br.restore(tmp_path / "bk", tmp_path / "out")

    assert not report["erasures"]["replayed"]
    assert not report["ok"]
    assert "not reachable" in report["erasures"]["reason"]


def test_a_store_that_has_never_served_an_erasure_restores_ok(live, tmp_path):
    """The other side: requiring a journal that was never needed would make every clean
    restore report a failure, and a check that always fails is a check nobody reads."""
    br.snapshot(live, tmp_path / "bk")

    report = br.restore(tmp_path / "bk", tmp_path / "out")

    assert report["ok"]
    assert report["erasures"]["replayed"]
    assert report["erasures"]["namespaces"] == 0


def test_the_replay_does_not_touch_another_namespace(live, tmp_path):
    """An erasure is for one namespace. A replay that over-deleted would turn a safety
    mechanism into data loss."""
    store = SQLiteMemoryStore(live)
    store.initialize()
    store.add_memories(
        [Memory("b1", "bob", "semantic", "bob fact", 2, ingested_at=datetime(2026, 1, 1))]
    )
    store.close()
    br.snapshot(live, tmp_path / "bk")
    store = SQLiteMemoryStore(live)
    store.initialize()
    store.hard_delete_user("alice")
    store.close()
    ErasureJournal.beside(live).record("alice")

    br.restore(tmp_path / "bk", tmp_path / "out")

    assert rows_for(tmp_path / "out" / "live.db", "alice") == 0
    assert rows_for(tmp_path / "out" / "live.db", "bob") == 1


def test_a_corrupt_line_does_not_stop_the_other_erasures(live, tmp_path):
    """Refusing to read the journal because one line is malformed would leave every other
    erasure unapplied — the exact failure the file exists to prevent."""
    journal = ErasureJournal.beside(live)
    journal.path.write_text(
        '{not json at all\n{"user_id": "alice", "erased_at": "2026-01-01T00:00:00+00:00"}\n',
        encoding="utf-8",
    )

    assert journal.namespaces() == ["alice"]


@pytest.mark.parametrize(
    "damaged",
    [b'{"user_id": "bob",', b"[]", b'{"user_id": 12, "erased_at": "bad"}', b"\xff"],
)
def test_a_damaged_journal_replays_valid_deletions_but_fails_restore(live, tmp_path, damaged):
    """Losing a deletion record cannot be reported as having replayed every deletion."""
    br.snapshot(live, tmp_path / "bk")
    journal = ErasureJournal.beside(live)
    journal.record("alice")
    with journal.path.open("ab") as handle:
        handle.write(damaged + b"\n")

    report = br.restore(tmp_path / "bk", tmp_path / "out", erasures=journal.path)

    assert not report["ok"]
    assert not report["erasures"]["replayed"]
    assert report["erasures"]["invalid_lines"] == [2]
    assert "complete replay cannot be verified" in report["erasures"]["reason"]
    assert report["erasures"]["namespaces"] == 1
    assert rows_for(tmp_path / "out" / "live.db", "alice") == 0
    assert report["index"]["consistent"]


def test_a_wholly_damaged_journal_does_not_pass_as_empty(live, tmp_path):
    br.snapshot(live, tmp_path / "bk")
    journal = ErasureJournal.beside(live)
    journal.path.write_text('{"user_id": "alice",', encoding="utf-8")

    report = br.restore(tmp_path / "bk", tmp_path / "out", erasures=journal.path)

    assert not report["ok"]
    assert report["erasures"]["invalid_lines"] == [1]
    assert rows_for(tmp_path / "out" / "live.db", "alice") == 3


def test_a_namespace_used_again_after_its_erasure_keeps_what_it_wrote_since(live, tmp_path):
    """Only erasures served after the backup are replayed. One served before it is already
    reflected in the copy, and whatever that namespace holds there was written afterwards —
    a user who deleted their data and carried on. Replaying it anyway deleted that new data
    and reported the loss as erased rows correctly removed."""
    store = SQLiteMemoryStore(live)
    store.initialize()
    store.hard_delete_user("alice")
    store.add_memories(
        [Memory("fresh", "alice", "semantic", "said after", 3, ingested_at=datetime(2026, 2, 1))]
    )
    store.close()
    # An explicit time well before the backup, so the ordering does not depend on how finely
    # the clock resolves two calls made in quick succession.
    ErasureJournal.beside(live).path.write_text(
        json.dumps({"user_id": "alice", "erased_at": "2026-01-15T00:00:00+00:00"}) + "\n",
        encoding="utf-8",
    )
    br.snapshot(live, tmp_path / "bk")

    report = br.restore(tmp_path / "bk", tmp_path / "out")

    assert rows_for(tmp_path / "out" / "live.db", "alice") == 1
    assert report["erasures"]["namespaces"] == 0
    assert report["erasures"]["skipped_before_backup"] == 1


def test_the_replay_is_the_online_erasure_not_a_subset_of_it(live, tmp_path):
    """Entities hold the names a user mentioned; entity links and evidence cascade from
    memories. A hand-written list of DELETEs missed the first and ran without foreign keys,
    so a copy that had "replayed" the erasure still held 'Dr. Jane Roe'."""
    store = SQLiteMemoryStore(live)
    store.initialize()
    with store._conn as connection:
        connection.execute(
            "INSERT INTO entities (id, user_id, canonical_name, surface_form) "
            "VALUES ('e1', 'alice', 'jane roe', 'Dr. Jane Roe')"
        )
        connection.execute("INSERT INTO memory_entities (memory_id, entity_id) VALUES ('m0', 'e1')")
        connection.execute("INSERT INTO evidence (memory_id, source_memory_id) VALUES ('m1', 'm0')")
    store.close()
    br.snapshot(live, tmp_path / "bk")
    store = SQLiteMemoryStore(live)
    store.initialize()
    store.hard_delete_user("alice")
    store.close()
    ErasureJournal.beside(live).record("alice")

    br.restore(tmp_path / "bk", tmp_path / "out")

    with sqlite3.connect(tmp_path / "out" / "live.db") as connection:
        left = {
            table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("entities", "memory_entities", "evidence", "memories", "turns")
        }
    assert left == dict.fromkeys(left, 0), left


def test_the_replay_takes_the_erased_vectors_out_of_the_restored_index(
    erased_after_backup, tmp_path
):
    """The online erasure removes a memory's vector and the replay has to as well: a vector
    is derived from the erased text, and an index still holding three for a namespace with
    no rows is not an erasure. The replay runs before the index check, so the check
    describes the copy that would be served."""
    report = br.restore(tmp_path / "bk", tmp_path / "out", erasures=erased_after_backup.path)

    assert report["erasures"]["vectors_removed"] == 3
    ids = json.loads((tmp_path / "out" / "live-index.ids.json").read_text(encoding="utf-8"))
    assert ids == []
    assert report["index"]["consistent"]
    assert report["ok"]


def test_a_journal_path_that_does_not_exist_fails_the_drill(live, tmp_path):
    """The flag is what keeps erased data erased, so a typo in it must not read as an empty
    journal and pass."""
    br.snapshot(live, tmp_path / "bk")

    report = br.restore(tmp_path / "bk", tmp_path / "out", erasures=tmp_path / "typo.jsonl")

    assert not report["erasures"]["replayed"]
    assert "does not exist" in report["erasures"]["reason"]
    assert not report["ok"]


def test_an_erasure_served_by_another_store_is_not_replayed_into_this_one(live, tmp_path):
    """Stores in one directory used to share one journal, so restoring any of them replayed
    every store's erasures into it — deleting a namespace from a store that was never asked
    to delete anything."""
    br.snapshot(live, tmp_path / "bk")
    other = tmp_path / "other.db"
    store = SQLiteMemoryStore(other)
    store.initialize()
    store.close()
    ErasureJournal.beside(other).record("alice")

    report = br.restore(tmp_path / "bk", tmp_path / "out")

    assert rows_for(tmp_path / "out" / "live.db", "alice") == 3
    assert report["ok"]


def test_a_backup_that_cannot_date_its_copy_replays_every_erasure(live, tmp_path):
    """A version-1 manifest stamped the end of the copy, so an erasure committed mid-copy
    can be older than the stamp and still be inside. Without a usable cutoff the replay errs
    towards deleting, never towards serving erased data."""
    br.snapshot(live, tmp_path / "bk")
    manifest_path = tmp_path / "bk" / "backup.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ErasureJournal.beside(live).path.write_text(
        json.dumps({"user_id": "alice", "erased_at": "2020-01-01T00:00:00+00:00"}) + "\n",
        encoding="utf-8",
    )

    report = br.restore(tmp_path / "bk", tmp_path / "out")

    assert report["erasures"]["cutoff"] is None
    assert report["erasures"]["namespaces"] == 1
    assert rows_for(tmp_path / "out" / "live.db", "alice") == 0


def test_a_replayed_copy_is_left_in_the_backup_s_journal_mode(erased_after_backup, tmp_path):
    """The replay opens the copy the way the service does, which switches it to WAL. It is
    finished back in DELETE mode, as the backup is, so the read-only checks after it read
    one closed file rather than depending on a sidecar being creatable."""
    br.restore(tmp_path / "bk", tmp_path / "out", erasures=erased_after_backup.path)

    connection = sqlite3.connect(tmp_path / "out" / "live.db")
    try:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        connection.close()
    assert mode == "delete"


def test_the_drill_reports_a_measured_recovery_time_and_data_window(live, tmp_path):
    """An operator quoting a recovery objective needs a number from a drill that ran, not
    an estimate. The window is what the backup's own timestamp says was lost."""
    br.snapshot(live, tmp_path / "bk")

    report = br.restore(tmp_path / "bk", tmp_path / "out")

    assert report["rto_seconds"] >= 0
    assert report["data_window"]["backup_taken_at"]
    assert report["data_window"]["behind_seconds"] >= 0


def test_the_journal_lives_outside_the_database_it_describes(live, tmp_path):
    """The whole mechanism. A tombstone inside the store is restored along with everything
    else — that is, back to the state before the erasure, where it does not exist."""
    journal = ErasureJournal.beside(live)

    assert journal.path.parent == live.parent
    assert journal.path.name == "live.erasures.jsonl", "named for its store, not shared"

    journal.record("alice")
    br.snapshot(live, tmp_path / "bk")
    copied = {
        item["path"]
        for item in json.loads((tmp_path / "bk" / "backup.json").read_text(encoding="utf-8"))[
            "files"
        ]
    }

    assert journal.path.name not in copied, (
        "the journal must not be restorable to a pre-erasure state"
    )


# ------------------------------------------------------------------- retention


def test_pruning_keeps_the_newest_backups_by_their_own_timestamp(live, tmp_path):
    """By `taken_at_utc`, not file mtime: a copied or rsynced backup carries whatever
    mtime the transport gave it, and pruning by that deletes by accident."""
    folder = tmp_path / "backups"
    for day in (3, 1, 2):
        br.snapshot(live, folder / f"bk{day}")
        manifest = folder / f"bk{day}" / "backup.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["taken_at_utc"] = f"2026-01-0{day}T00:00:00+00:00"
        manifest.write_text(json.dumps(data), encoding="utf-8")

    result = br.prune(folder, keep=2)

    assert result["kept"] == ["bk3", "bk2"]
    assert result["removed"] == ["bk1"]
    assert not (folder / "bk1").exists()


def test_pruning_reports_how_far_back_a_restore_could_reach(live, tmp_path):
    """The number that matters for an erasure: how long deleted data stays recoverable
    from a backup."""
    folder = tmp_path / "backups"
    br.snapshot(live, folder / "bk1")

    result = br.prune(folder, keep=5)

    assert result["oldest_kept"]
    assert result["removed"] == []


def test_pruning_to_zero_is_refused(live, tmp_path):
    """ "Keep none" is not a retention policy; it is deleting the backups."""
    folder = tmp_path / "backups"
    br.snapshot(live, folder / "bk1")

    with pytest.raises(ValueError, match="at least one"):
        br.prune(folder, keep=0)


def test_pruning_ignores_directories_that_are_not_backups(live, tmp_path):
    """A folder shared with anything else must not lose that thing to the sweep."""
    folder = tmp_path / "backups"
    br.snapshot(live, folder / "bk1")
    (folder / "notes").mkdir()
    (folder / "notes" / "readme.txt").write_text("keep me", encoding="utf-8")

    br.prune(folder, keep=1)

    assert (folder / "notes" / "readme.txt").is_file()
