from __future__ import annotations

import os

import pytest

from llm_long_term_memory.locking import AlreadyRunning, exclusive


def test_exclusive_lock_refuses_a_second_live_owner(tmp_path):
    target = tmp_path / "artifact.json"
    lock = target.with_suffix(".json.lock")

    with exclusive(target, what="test job"):
        assert lock.read_text(encoding="utf-8") == str(os.getpid())
        with (
            pytest.raises(AlreadyRunning, match="already running"),
            exclusive(target, what="test job"),
        ):
            pass

    assert not lock.exists()


def test_exclusive_lock_reclaims_a_dead_pid(tmp_path):
    target = tmp_path / "artifact.json"
    lock = target.with_suffix(".json.lock")
    lock.write_text("999999", encoding="utf-8")

    with exclusive(target):
        assert lock.read_text(encoding="utf-8") == str(os.getpid())


def test_fresh_incomplete_lock_is_not_mistaken_for_stale(tmp_path):
    target = tmp_path / "artifact.json"
    lock = target.with_suffix(".json.lock")
    lock.touch()

    with pytest.raises(AlreadyRunning, match="acquiring"), exclusive(target):
        pass
