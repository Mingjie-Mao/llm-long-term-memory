"""Recomputing `event_time` on a store written before it meant anything.

Without this, the signal the split introduced cannot be evaluated on any evidence the
project already holds: every row of every existing store carries the conversation's
date, so `event_time_is_stated` is True everywhere and discriminates nothing. The
inputs needed to fix that are already columns, so the fix costs no model calls — but it
rewrites data, so it is opt-in and reports before it writes.
"""

from __future__ import annotations

import importlib.util
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from llm_long_term_memory.store import Memory, SQLiteMemoryStore

REPO = Path(__file__).resolve().parent.parent
MARCH = datetime(2023, 3, 15)


def _module():
    path = REPO / "tools/backfill_event_time.py"
    spec = importlib.util.spec_from_file_location("backfill_event_time_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _store(tmp_path: Path, *contents: str) -> Path:
    """A store in the state the old extractor left: the session date on every row."""
    path = tmp_path / "old.db"
    store = SQLiteMemoryStore(path)
    store.initialize()
    store.add_memories(
        [
            Memory(
                id=f"m{index}",
                user_id="u",
                type="semantic",
                content=content,
                token_count=1,
                event_time=MARCH,
                observed_at=MARCH,
                valid_from=MARCH,
                ingested_at=MARCH,
            )
            for index, content in enumerate(contents)
        ]
    )
    store.close()
    return path


def _event_times(path: Path) -> dict[str, str | None]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        return {
            r["id"]: r["event_time"]
            for r in connection.execute("SELECT id, event_time FROM memories")
        }
    finally:
        connection.close()


def test_a_fact_with_no_stated_time_gives_its_assumed_date_back(tmp_path):
    path = _store(tmp_path, "The user owns a fern.")
    module = _module()

    report, pending = module.recompute(path)

    assert report["by_kind"] == {"assumed_date_withdrawn": 1}
    assert pending == [(None, "m0")]


def test_a_fact_that_states_a_date_is_corrected_to_it(tmp_path):
    path = _store(tmp_path, "The user replaced the plugs on February 14.")
    module = _module()

    report, pending = module.recompute(path)

    assert report["by_kind"] == {"corrected_to_the_stated_date": 1}
    assert pending == [("2023-02-14T00:00:00", "m0")]
    assert report["carrying_a_stated_time_after"] == 1


def test_nothing_is_written_until_it_is_asked_for(tmp_path):
    path = _store(tmp_path, "The user owns a fern.")
    module = _module()
    before = _event_times(path)

    module.recompute(path)

    assert _event_times(path) == before


def test_applying_it_writes_exactly_the_rows_that_move(tmp_path):
    path = _store(
        tmp_path,
        "The user owns a fern.",
        "The user replaced the plugs on February 14.",
    )
    module = _module()

    _, pending = module.recompute(path)
    written = module.apply(path, pending)

    assert written == 2
    assert _event_times(path) == {"m0": None, "m1": "2023-02-14T00:00:00"}


def test_the_signal_only_exists_after_the_backfill(tmp_path):
    """The whole reason the tool exists, as an assertion."""
    path = _store(
        tmp_path,
        "The user owns a fern.",
        "The user replaced the plugs on February 14.",
    )
    module = _module()

    store = SQLiteMemoryStore(path)
    store.initialize()
    assert all(m.event_time_is_stated for m in store.iter_all("u")), "no discrimination"
    store.close()

    _, pending = module.recompute(path)
    module.apply(path, pending)

    store = SQLiteMemoryStore(path)
    store.initialize()
    stated = {m.id: m.event_time_is_stated for m in store.iter_all("u")}
    ordering = {m.id: m.occurred_at for m in store.iter_all("u")}
    store.close()

    assert stated == {"m0": False, "m1": True}
    # And ordering still has a time for both, because `observed_at` was never touched.
    assert ordering == {"m0": MARCH, "m1": datetime(2023, 2, 14)}


def test_a_store_without_the_column_is_refused_rather_than_guessed_at(tmp_path):
    path = _store(tmp_path, "The user owns a fern.")
    connection = sqlite3.connect(path)
    connection.execute("ALTER TABLE memories DROP COLUMN observed_at")
    connection.commit()
    connection.close()

    with pytest.raises(SystemExit, match="observed_at"):
        _module().recompute(path)
