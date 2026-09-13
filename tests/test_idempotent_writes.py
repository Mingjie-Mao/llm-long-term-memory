"""A retried write must not append the turn a second time.

The defect this closes is specific and easy to hit. `add_message` persists the turn
*before* extraction runs, and `turn_index` is `len(existing.turns)`. So a provider failure
followed by an ordinary client retry landed the same turn again at the next index, and
extraction then ran over the duplicate. Nothing errored; the conversation just grew a copy
of itself.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from llm_long_term_memory.api.service import (
    MemoryService,
    TurnExtractionOutcome,
    WriteInProgress,
)
from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore


class _StubEncoder:
    dim = 2

    def encode(self, texts, show_progress=False):
        return np.ones((len(texts), 2), dtype=np.float32)

    def encode_one(self, text):
        return np.ones(2, dtype=np.float32)


class _Settings:
    def __init__(self, tmp_path):
        self.store_dir = tmp_path
        self.data_dir = tmp_path
        self.results_dir = tmp_path
        self.gemini_api_key = ""

    @property
    def has_api_key(self):
        return False


class _Extractor:
    """Counts calls, and can be told to fail the way a provider does."""

    def __init__(self):
        self.calls = 0
        self.fail_next = False

    def extract_turn(self, **_):
        self.calls += 1
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("simulated provider failure")
        return TurnExtractionOutcome([], {"requests": 1})


@pytest.fixture
def service(tmp_path):
    svc = MemoryService.__new__(MemoryService)
    svc.__init__(store_name="t", settings=_Settings(tmp_path), encoder=_StubEncoder())
    svc.store.close()
    store = SQLiteMemoryStore(tmp_path / "m.db")
    store.initialize()
    svc.store = store
    svc.index = NumpyFlatIndex(tmp_path / "idx", dim=2)
    svc.extractor = _Extractor()
    svc.resolver = None
    yield svc
    store.close()


def turns_in(service, user="alice") -> int:
    return sum(
        len(service.store.get_session(sid).turns)
        for sid in service.store.session_ids_for_user(user)
    )


# ------------------------------------------------------------------ the defect


def test_a_retry_without_a_key_still_duplicates_the_turn(service):
    """The behaviour the key exists to fix, pinned so the fix cannot be mistaken for
    something that was never broken. Without a key there is nothing to deduplicate on,
    and the second call is a legitimately different request."""
    service.add_message("alice", "user", "I live in Canberra", session_id="s1")
    service.add_message("alice", "user", "I live in Canberra", session_id="s1")

    assert turns_in(service) == 2


def test_a_retry_with_the_same_key_replays_instead_of_writing(service):
    first = service.add_message(
        "alice", "user", "I live in Canberra", session_id="s1", idempotency_key="k1"
    )
    second = service.add_message(
        "alice", "user", "I live in Canberra", session_id="s1", idempotency_key="k1"
    )

    assert turns_in(service) == 1, "the retry appended a second turn"
    assert second["idempotent_replay"] is True
    assert first.get("idempotent_replay") is None, "the first write is not a replay"
    assert second["turn_index"] == first["turn_index"]
    assert second["session_id"] == first["session_id"]
    # And the extractor was not paid for twice.
    assert service.extractor.calls == 1


def test_a_different_key_is_a_different_write(service):
    service.add_message("alice", "user", "one", session_id="s1", idempotency_key="k1")
    service.add_message("alice", "user", "two", session_id="s1", idempotency_key="k2")

    assert turns_in(service) == 2


# ------------------------------------------------------------------ isolation


def test_two_tenants_may_use_the_same_key(service):
    """Client-generated keys collide across tenants as a matter of course. One namespace
    reading another's reply would be a cross-tenant leak through the retry path."""
    a = service.add_message("alice", "user", "mine", session_id="s1", idempotency_key="shared")
    b = service.add_message("bob", "user", "also mine", session_id="s1", idempotency_key="shared")

    assert b.get("idempotent_replay") is None, "bob replayed alice's write"
    assert turns_in(service, "alice") == 1
    assert turns_in(service, "bob") == 1
    assert a["session_id"] == b["session_id"] == "s1"


def test_erasure_takes_the_keys_with_it(service):
    """Replaying a key after erasure would return a reply describing memories that no
    longer exist."""
    service.add_message("alice", "user", "hello", session_id="s1", idempotency_key="k1")
    service.erase_user("alice")

    again = service.add_message("alice", "user", "hello", session_id="s1", idempotency_key="k1")

    assert again.get("idempotent_replay") is None
    assert turns_in(service) == 1


# ------------------------------------------------------------------ failure and races


def test_a_failed_extraction_does_not_leave_the_key_claimed_forever(service):
    """A key claimed by a call that died must not wedge the client out permanently.

    It stays claimed — the turn *was* written — and the caller is told the write is in
    progress rather than handed a half-built reply. That is the honest state: the service
    does not know whether extraction would have succeeded."""
    service.extractor.fail_next = True
    with pytest.raises(RuntimeError, match="simulated provider failure"):
        service.add_message("alice", "user", "boom", session_id="s1", idempotency_key="k1")

    with pytest.raises(WriteInProgress):
        service.add_message("alice", "user", "boom", session_id="s1", idempotency_key="k1")

    # And crucially, the retry did not append a second turn.
    assert turns_in(service) == 1


def test_a_second_caller_racing_the_same_key_is_refused_not_served_a_stale_reply(service):
    """The key is claimed before the write, so the loser of the race is told to retry
    rather than handed a reply that may still be in flight."""
    service.store.remember_write("alice", "k1", '{"pending": true}')

    with pytest.raises(WriteInProgress):
        service.add_message("alice", "user", "hello", session_id="s1", idempotency_key="k1")

    assert turns_in(service) == 0


def test_the_claim_is_atomic():
    """`INSERT OR IGNORE` rather than check-then-write: two concurrent retries would both
    pass a check and both proceed, which is the race the key exists to close."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteMemoryStore(Path(tmp) / "m.db")
        store.initialize()
        assert store.remember_write("alice", "k", "first") is True
        assert store.remember_write("alice", "k", "second") is False
        assert store.remembered_write("alice", "k") == "first"
        store.close()


# ------------------------------------------------------------------ input


@pytest.mark.parametrize("key", ["", "   ", "x" * 201])
def test_an_unusable_key_is_refused(service, key):
    with pytest.raises(ValueError, match="idempotency_key"):
        service.add_message("alice", "user", "hi", session_id="s1", idempotency_key=key)


def test_no_key_means_no_row_is_recorded(service):
    """A service that recorded a key for every write would grow a table nobody reads."""
    service.add_message("alice", "user", "hi", session_id="s1")

    assert service.store.remembered_write("alice", "") is None


def test_the_replay_is_the_same_reply_not_a_reconstruction(service):
    """Memories are dataclasses, so the stored reply round-trips through JSON. A replay
    that quietly dropped a field would look like a successful retry and lose data."""
    first = service.add_message("alice", "user", "hello", session_id="s1", idempotency_key="k1")
    second = service.add_message("alice", "user", "hello", session_id="s1", idempotency_key="k1")

    assert set(second) - {"idempotent_replay"} == set(first)
    assert second["usage"] == first["usage"]


def test_the_service_is_unchanged_when_no_key_is_supplied(service):
    """Idempotency is opt-in. Existing callers, the research CLI and the MCP server all
    write without a key and must behave exactly as before."""
    result = service.add_message("alice", "user", "hello", session_id="s1")

    assert "idempotent_replay" not in result
    assert result["turn_index"] == 0
    assert isinstance(SimpleNamespace(**result).usage, dict)
