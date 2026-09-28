"""One writer process per store.

SQLite serialises its own writes, but every service process holds the vector index in
memory and saves it whole. Two writer processes on one store would each overwrite the
other's vectors, and neither would notice. The guard makes the second writer refuse to
start instead.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from llm_long_term_memory.api.service import MemoryService
from llm_long_term_memory.locking import AlreadyRunning


class _Encoder:
    dim = 2

    def encode_one(self, text):
        return np.ones(2, dtype=np.float32)

    def encode(self, texts, show_progress=False):
        return np.ones((len(texts), 2), dtype=np.float32)


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LLTM_STORE_DIR", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "")
    return tmp_path


def _service(**kw):
    return MemoryService(
        config_path="configs/baselines.yaml", store_name="s", encoder=_Encoder(), **kw
    )


def test_a_writer_holds_the_store_lock_until_it_closes(store_dir):
    svc = _service()
    assert (store_dir / "s.db.lock").exists()
    svc.close()
    assert not (store_dir / "s.db.lock").exists()


def test_two_services_in_one_process_share_the_lock(store_dir):
    """Tests and per-request apps open a store twice in one process."""
    first, second = _service(), _service()
    first.close()
    assert (store_dir / "s.db.lock").exists(), "released while the second still writes"
    second.close()
    assert not (store_dir / "s.db.lock").exists()


def test_a_read_only_service_takes_no_lock(store_dir):
    _service().close()  # create the store
    reader = _service(read_only=True)
    assert not (store_dir / "s.db.lock").exists()
    reader.close()


def test_a_second_writer_process_is_refused(store_dir):
    """The case that corrupts: another process already writing this store."""
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import sys, time
                from pathlib import Path
                from llm_long_term_memory.locking import hold
                hold(Path({str(store_dir / "s.db")!r}), what="other writer")
                print("held", flush=True)
                time.sleep(30)
                """
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(AlreadyRunning):
            _service()
        reader = _service(read_only=True) if (store_dir / "s.db").exists() else None
        if reader is not None:
            reader.close()
    finally:
        holder.kill()
        holder.wait()


def test_a_lock_left_by_a_dead_process_is_reclaimed(store_dir):
    (store_dir / "s.db.lock").write_text("999999999", encoding="utf-8")
    svc = _service()
    svc.close()
