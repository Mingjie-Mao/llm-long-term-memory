"""A retried write must not append the turn a second time.

The defect this closes is specific and easy to hit. `add_message` persists the turn
*before* extraction runs, and `turn_index` is `len(existing.turns)`. So a provider failure
followed by an ordinary client retry landed the same turn again at the next index, and
extraction then ran over the duplicate. Nothing errored; the conversation just grew a copy
of itself.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from llm_long_term_memory.api.budget import BudgetExceeded
from llm_long_term_memory.api.service import (
    PENDING_WRITE_TAKEOVER,
    IdempotencyConflict,
    MemoryService,
    TurnExtractionOutcome,
    WriteInProgress,
    _request_fingerprint,
)
from llm_long_term_memory.llm.usage import CallRecord, UsageTracker
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


def test_a_failed_write_can_be_retried_and_does_not_duplicate_the_turn(service):
    """The failure a key exists to make safe is a provider error followed by a retry, so
    the retry has to be possible.

    The earlier version of this path left the key `pending` for good: every later attempt
    was told a write was still in progress, and the one thing the key promised — retry a
    failed write safely — was the one thing it prevented. It stayed that way because the
    turn had already been written and a retry would have appended it again.

    Both halves are fixed here. The turn is rolled back when extraction raises, so the
    retry starts from the state before the failure, and the key is marked `failed` rather
    than left claimed, so the retry is allowed to take it over."""
    service.extractor.fail_next = True
    with pytest.raises(RuntimeError, match="simulated provider failure"):
        service.add_message("alice", "user", "boom", session_id="s1", idempotency_key="k1")

    # The failed attempt left nothing behind to duplicate.
    assert turns_in(service) == 0
    assert service.store.write_key("alice", "k1")["state"] == "failed"

    reply = service.add_message("alice", "user", "boom", session_id="s1", idempotency_key="k1")

    assert reply.get("idempotent_replay") is None, "a failed write must not replay as done"
    assert turns_in(service) == 1


def test_a_failed_key_is_kept_rather_than_deleted(service):
    """A deleted key is indistinguishable from one that was never claimed. The operator
    asking "did that write ever run?" needs the difference, and so does anyone reading a
    retry as a retry."""
    service.extractor.fail_next = True
    with pytest.raises(RuntimeError):
        service.add_message("alice", "user", "boom", session_id="s1", idempotency_key="k1")

    row = service.store.write_key("alice", "k1")
    assert row is not None
    assert row["state"] == "failed"
    assert "simulated provider failure" in json.loads(row["response"])["detail"]


# ------------------------------------------------------- one key, one request


def test_the_same_key_with_a_different_message_is_refused(service):
    """Replaying the first reply would tell the caller their second message was stored
    when nothing was written — a silent loss the caller cannot detect. The key promises
    "this request at most once", not "this key answers anything"."""
    service.add_message("alice", "user", "hello", session_id="s1", idempotency_key="k1")

    with pytest.raises(IdempotencyConflict, match="different request"):
        service.add_message("alice", "user", "goodbye", session_id="s1", idempotency_key="k1")

    assert turns_in(service) == 1


@pytest.mark.parametrize(
    "second",
    [
        {"role": "assistant", "content": "hello", "session_id": "s1"},
        {"role": "user", "content": "hello ", "session_id": "s1"},
        {"role": "user", "content": "hello", "session_id": "s2"},
    ],
)
def test_every_part_of_the_request_is_in_the_fingerprint(service, second):
    """Role, content and session all change what gets written, so a key claimed for one
    may not answer another. Whitespace counts: it is content."""
    service.add_message("alice", "user", "hello", session_id="s1", idempotency_key="k1")

    with pytest.raises(IdempotencyConflict):
        service.add_message("alice", **second, idempotency_key="k1")


def test_the_identical_request_still_replays(service):
    """The conflict check must not break the case the key exists for."""
    first = service.add_message("alice", "user", "hello", session_id="s1", idempotency_key="k1")
    again = service.add_message("alice", "user", "hello", session_id="s1", idempotency_key="k1")

    assert again["idempotent_replay"] is True
    assert again["turn_index"] == first["turn_index"]
    assert turns_in(service) == 1


def test_an_abandoned_claim_can_be_taken_over_once_it_is_old_enough(service, monkeypatch):
    """A crash cannot run the failure path, so a key held by a dead process would stay
    unusable forever. The window is long on purpose: taking over a claim that is still
    running would write twice, which is worse than making a caller wait."""
    stale = (datetime.now() - PENDING_WRITE_TAKEOVER - timedelta(minutes=1)).isoformat()
    # The abandoned claim is this same request: a takeover retries what was interrupted,
    # it does not let a key be reused for something else.
    fingerprint = _request_fingerprint("user", "hello", "s1")
    service.store.remember_write("alice", "k1", '{"pending": true}', fingerprint)
    with service.store._conn as conn:
        conn.execute(
            "UPDATE write_keys SET claimed_at = ? WHERE user_id = 'alice' AND key = 'k1'",
            (stale,),
        )

    reply = service.add_message("alice", "user", "hello", session_id="s1", idempotency_key="k1")

    assert reply.get("idempotent_replay") is None
    assert turns_in(service) == 1
    assert service.store.write_key("alice", "k1")["state"] == "done"


def test_a_recent_claim_is_not_taken_over(service):
    """The other side of the window: a write that is genuinely still running must not be
    duplicated by an impatient retry."""
    service.store.remember_write(
        "alice", "k1", '{"pending": true}', _request_fingerprint("user", "hello", "s1")
    )

    with pytest.raises(WriteInProgress, match="still in progress"):
        service.add_message("alice", "user", "hello", session_id="s1", idempotency_key="k1")

    assert turns_in(service) == 0


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


def test_two_retries_that_read_the_same_failed_key_cannot_both_take_it_over(service):
    """The takeover used to be conditioned on `state IN ('failed', 'pending')`, which did
    not close the race it was written for: the first taker sets the row back to 'pending',
    the condition still matches, and a second retry that had read the same failed row took
    it over too — both wrote. The swap is now on exactly the row each caller read."""
    fingerprint = _request_fingerprint("user", "hello", "s1")
    service.store.remember_write("alice", "k1", '{"pending": true}', fingerprint)
    service.store.record_write_failure("alice", "k1", "boom")
    read = service.store.write_key("alice", "k1")

    def take_over():
        return service.store.take_over_write(
            "alice",
            "k1",
            fingerprint,
            expected_state=read["state"],
            expected_claimed_at=read["claimed_at"],
        )

    assert (take_over(), take_over()) == (True, False)


def test_claims_are_stamped_in_utc(service):
    """The takeover window is measured from `claimed_at`. In local wall-clock time it is
    wrong for an hour twice a year: across the spring-forward gap a claim seconds old reads
    as an hour old and is taken over while it is still running."""
    service.store.remember_write("alice", "k1", '{"pending": true}', "fingerprint")

    claimed = datetime.fromisoformat(service.store.write_key("alice", "k1")["claimed_at"])

    assert claimed.utcoffset() == timedelta(0)


# ------------------------------------------------------ what a write costs the account


class _RecordingExtractor:
    """Records attempts in a `UsageTracker`, as the live extractor's provider client does."""

    def __init__(self, attempts, *, succeed):
        self.usage = UsageTracker()
        self.attempts = attempts
        self.succeed = succeed

    def extract_turn(self, **_):
        started = len(self.usage.records)
        for ok, input_tokens, output_tokens in self.attempts:
            self.usage.record(
                CallRecord(
                    role="extract",
                    model="extractor-model",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=1.0,
                    ok=ok,
                )
            )
        if not self.succeed:
            raise RuntimeError("provider gave up")
        summary = UsageTracker(records=list(self.usage.records[started:])).summary()
        return TurnExtractionOutcome([], summary)


def test_a_budget_refusal_releases_the_key_instead_of_holding_it(service):
    """A 429 left the claim 'pending'. The caller's retry — once the cap was lifted or the
    day rolled over — was then told a write was still in progress, for the whole takeover
    window, about a write that had never started."""
    service.budgets.set_budget("alice", daily_calls=0)
    with pytest.raises(BudgetExceeded):
        service.add_message("alice", "user", "hi", session_id="s1", idempotency_key="k1")

    assert service.store.write_key("alice", "k1")["state"] == "failed"
    assert service.budgets.spent_today("alice").calls == 0, "nothing ran, so nothing is charged"

    service.budgets.set_budget("alice")  # the operator lifts the cap
    reply = service.add_message("alice", "user", "hi", session_id="s1", idempotency_key="k1")

    assert reply.get("idempotent_replay") is None
    assert turns_in(service) == 1


def test_a_request_refused_before_any_provider_call_is_neither_charged_nor_claimed(service):
    """A session id scoped to someone else is the caller's 422. It was charged to the account
    as a failed provider call — a client retrying its own 422 spent the day's budget on
    requests that never left the process — and it left the key 'failed' as though an
    attempt had run."""
    with pytest.raises(ValueError, match="another user"):
        service.add_message(
            "alice", "user", "hi", session_id="scoped-session-v1:3:bobx", idempotency_key="k1"
        )

    assert service.extractor.calls == 0
    assert service.budgets.spent_today("alice").calls == 0
    assert service.store.write_key("alice", "k1") is None


def test_a_failed_write_is_charged_for_every_attempt_that_reached_the_provider(service):
    """The provider client retries inside one write. Charging the write as a single call
    with no tokens let a retry storm through at a fraction of its cost."""
    service.extractor = _RecordingExtractor(
        [(False, 100, 0), (False, 100, 0), (False, 120, 0)], succeed=False
    )

    with pytest.raises(RuntimeError, match="provider gave up"):
        service.add_message("alice", "user", "hi", session_id="s1")

    spend = service.budgets.spent_today("alice")
    assert (spend.calls, spend.failed_calls, spend.input_tokens) == (3, 3, 320)


def test_a_write_that_failed_before_sending_anything_is_not_charged(service):
    """With per-attempt records there is nothing to guess: none recorded, none sent."""
    service.extractor = _RecordingExtractor([], succeed=False)

    with pytest.raises(RuntimeError):
        service.add_message("alice", "user", "hi", session_id="s1")

    assert service.budgets.spent_today("alice").calls == 0


def test_one_retry_inside_a_successful_write_is_one_failure_not_all_of_them(service):
    """`failed_calls` is how an operator tells a retry storm from one retry. Marking every
    call of a role failed because one of them was reported a storm where there was none."""
    service.extractor = _RecordingExtractor([(False, 50, 0), (True, 50, 10)], succeed=True)

    service.add_message("alice", "user", "hi", session_id="s1")

    spend = service.budgets.spent_today("alice")
    assert (spend.calls, spend.failed_calls, spend.tokens) == (2, 1, 110)


def test_an_extractor_that_keeps_no_records_is_charged_one_failed_call(service):
    """It cannot say what it sent, so the charge errs the way a budget must: towards one
    call rather than none."""
    service.extractor.fail_next = True
    with pytest.raises(RuntimeError):
        service.add_message("alice", "user", "boom", session_id="s1")

    spend = service.budgets.spent_today("alice")
    assert (spend.calls, spend.failed_calls) == (1, 1)


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


def test_a_key_written_before_fingerprints_existed_still_replays():
    """The migration backfills old rows to `done` with a NULL fingerprint. A NULL must be
    read as "claimed before this column existed" and let through — rejecting it would turn
    every legitimate retry against an upgraded store into a conflict, which is a new
    failure introduced by the fix for an old one."""
    import sqlite3
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        path = _Path(tmp) / "old.db"
        legacy = sqlite3.connect(path)
        legacy.executescript(
            "CREATE TABLE write_keys (user_id TEXT NOT NULL, key TEXT NOT NULL, "
            "response TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY (user_id, key));"
            "INSERT INTO write_keys VALUES "
            "('alice', 'k1', '{\"turn_index\": 0}', '2026-01-01T00:00:00');"
        )
        legacy.commit()
        legacy.close()

        store = SQLiteMemoryStore(path)
        store.initialize()
        row = store.write_key("alice", "k1")

        assert row["state"] == "done", "an old row is a completed write, not a pending one"
        assert row["fingerprint"] is None
        store._conn.close()
