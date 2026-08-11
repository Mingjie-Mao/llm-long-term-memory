"""Ingestion driver resilience.

A full run is ~1,920 requests spread over more than a day of quota. Every one of
these tests is about the same property: an interruption must cost the remaining
work, never the work already done.
"""

from __future__ import annotations

import numpy as np
import pytest

from chronomem.evaluation.datasets.longmemeval import HaystackSession, HaystackTurn
from chronomem.ingest.pipeline import (
    IngestionPipeline,
    IngestProgress,
    batched,
    namespaced_sessions,
)
from chronomem.llm.client import DailyQuotaExhausted
from chronomem.llm.rate_limiter import Wait
from chronomem.store import Memory, NumpyFlatIndex, SQLiteMemoryStore


def session(sid: str) -> HaystackSession:
    return HaystackSession(
        session_id=sid,
        date="2026/01/05 (Mon) 09:00",
        turns=[HaystackTurn(role="user", content=f"content of {sid}")],
    )


class FakeExtractor:
    """Yields one memory per session, and can be told to fail on the Nth batch."""

    user_id = "u1"

    def __init__(self, fail_on_batch: int | None = None, exc: Exception | None = None):
        self.fail_on_batch = fail_on_batch
        self.exc = exc or RuntimeError("boom")
        self.calls = 0

    def extract(self, sessions):
        self.calls += 1
        if self.fail_on_batch and self.calls == self.fail_on_batch:
            raise self.exc

        from chronomem.ingest.extract import ExtractionOutcome

        return ExtractionOutcome(
            [
                Memory(
                    # Namespaced, mirroring the real extractor: the pipeline sets
                    # `user_id` per batch and every memory is stamped with it.
                    id=f"mem_{self.user_id}_{s.session_id}",
                    user_id=self.user_id,
                    type="semantic",
                    content=f"fact from {s.session_id}",
                    token_count=5,
                    source_session_id=s.session_id,
                )
                for s in sessions
            ],
            0,
        )


class PassthroughDedup:
    def process(self, candidates, vectors=None):
        from chronomem.ingest.dedup import DedupOutcome

        return DedupOutcome(kept=list(candidates))


class FakeEncoder:
    def encode(self, texts, show_progress=False):
        return np.ones((len(texts), 4), dtype=np.float32)

    def encode_one(self, text):
        return np.ones(4, dtype=np.float32)


@pytest.fixture
def build(tmp_path):
    def make(extractor, sessions_per_request=2, checkpoint_every=1):
        store = SQLiteMemoryStore(tmp_path / "s.db")
        store.initialize()
        index = NumpyFlatIndex(tmp_path / "idx", dim=4)
        pipeline = IngestionPipeline(
            extractor,
            PassthroughDedup(),
            store,
            index,
            FakeEncoder(),
            checkpoint_path=tmp_path / "ckpt.json",
            sessions_per_request=sessions_per_request,
            checkpoint_every=checkpoint_every,
        )
        return pipeline, store, index

    return make


def test_happy_path_writes_everything(build):
    pipeline, store, index = build(FakeExtractor())
    outcome = pipeline.run([("u1", session(f"s{i}")) for i in range(6)])

    assert outcome.completed
    assert outcome.progress.memories_written == 6
    assert store.count("u1") == 6
    assert len(index) == 6
    store.close()


def test_quota_exhaustion_keeps_completed_batches(build):
    quota = DailyQuotaExhausted("m", Wait(3600.0, "rpd"))
    pipeline, store, _ = build(FakeExtractor(fail_on_batch=3, exc=quota))
    outcome = pipeline.run([("u1", session(f"s{i}")) for i in range(8)])

    assert not outcome.completed
    assert "daily quota" in outcome.stopped_reason
    assert store.count("u1") == 4, "the two batches that succeeded are on disk"
    assert len(outcome.progress.done_sessions) == 4
    store.close()


def test_a_network_failure_is_a_stop_not_a_lost_run(build):
    """The failure actually hit in practice: a DNS blip mid-run. It must not escape
    as a traceback, and it must not discard completed batches."""
    pipeline, store, _ = build(
        FakeExtractor(fail_on_batch=3, exc=ConnectionError("nodename nor servname provided"))
    )
    outcome = pipeline.run([("u1", session(f"s{i}")) for i in range(8)])

    assert not outcome.completed
    assert "ConnectionError" in outcome.stopped_reason
    assert store.count("u1") == 4
    store.close()


def test_the_failed_batch_is_not_marked_done(build):
    pipeline, store, _ = build(FakeExtractor(fail_on_batch=2, exc=RuntimeError("x")))
    outcome = pipeline.run([("u1", session(f"s{i}")) for i in range(4)])

    assert outcome.progress.done_sessions == {"u1:s0", "u1:s1"}
    assert "u1:s2" not in outcome.progress.done_sessions
    store.close()


def test_resume_skips_done_sessions_and_finishes(build, tmp_path):
    sessions = [("u1", session(f"s{i}")) for i in range(8)]

    first = FakeExtractor(fail_on_batch=3, exc=DailyQuotaExhausted("m", Wait(1.0, "rpd")))
    pipeline, store, _ = build(first)
    pipeline.run(sessions)
    store.close()

    second = FakeExtractor()
    pipeline2, store2, _ = build(second)
    outcome = pipeline2.run(sessions)

    assert outcome.completed
    assert second.calls == 2, "only the 4 remaining sessions, at 2 per request"
    assert len(outcome.progress.done_sessions) == 8
    assert store2.count("u1") == 8
    store2.close()


def test_reingesting_the_same_session_does_not_duplicate(build):
    """Memory ids are deterministic, so a batch replayed after a crash replaces its
    rows instead of doubling them."""
    sessions = [("u1", session("s0")), ("u1", session("s1"))]
    pipeline, store, _ = build(FakeExtractor())
    pipeline.run(sessions)
    pipeline.run(sessions, resume=False)

    assert store.count("u1") == 2
    store.close()


def test_progress_checkpoint_roundtrips(tmp_path):
    p = IngestProgress(done_sessions={"a", "b"}, memories_written=7, duplicates_dropped=2)
    path = tmp_path / "ckpt.json"
    p.save(path)

    loaded = IngestProgress.load(path)
    assert loaded.done_sessions == {"a", "b"}
    assert loaded.memories_written == 7
    assert loaded.duplicates_dropped == 2


def test_missing_checkpoint_starts_clean(tmp_path):
    assert IngestProgress.load(tmp_path / "nope.json").done_sessions == set()


def test_a_shared_session_is_ingested_once_per_question():
    """Deliberately *not* deduplicated across questions. LongMemEval pads each
    question's haystack with distractors from unrelated conversations, so the same
    session in two haystacks belongs to two different simulated users. Merging them
    produced a `lives_in` chain across seven cities and let retrieval for one
    question return another's evidence."""
    from chronomem.evaluation.datasets.longmemeval import Instance

    shared = session("shared")
    a = Instance("q1", "t", "q", "a", "d", [shared, session("only_a")], [])
    b = Instance("q2", "t", "q", "a", "d", [shared, session("only_b")], [])

    pairs = namespaced_sessions([a, b])
    assert sorted((ns, s.session_id) for ns, s in pairs) == [
        ("q1", "only_a"),
        ("q1", "shared"),
        ("q2", "only_b"),
        ("q2", "shared"),
    ]


def test_namespaces_are_not_deduplicated_across_questions():
    """A session appearing in two haystacks belongs to two different simulated
    users. Deduplicating it by session_id merges the personas — which is what
    produced a store where one 'user' had lived in Tokyo, Seattle, Shanghai and
    San Diego, with the temporal resolver chaining those into a move history."""
    from chronomem.evaluation.datasets.longmemeval import Instance
    from chronomem.ingest.pipeline import group_by_namespace, namespaced_sessions

    shared = session("shared")
    a = Instance("q1", "t", "q", "a", "d", [shared, session("only_a")], [])
    b = Instance("q2", "t", "q", "a", "d", [shared], [])

    pairs = namespaced_sessions([a, b])
    assert ("q1", shared) in [(ns, s) for ns, s in pairs]
    assert ("q2", shared) in [(ns, s) for ns, s in pairs]
    assert len(pairs) == 3, "the shared session is ingested once per question"

    groups = dict(group_by_namespace(pairs))
    assert sorted(groups) == ["q1", "q2"]
    assert len(groups["q1"]) == 2


def test_checkpoint_keys_are_namespaced():
    """Otherwise finishing a session under q1 would mark it done for q2 as well."""
    from chronomem.ingest.pipeline import _key

    assert _key("q1", session("s")) != _key("q2", session("s"))


def test_a_batch_never_straddles_two_namespaces(build):
    """One extraction request must not mix two users' sessions — the model would
    attribute facts across personas, and `session_index` maps into one list."""
    from chronomem.ingest.pipeline import group_by_namespace

    pairs = [("q1", session("a")), ("q2", session("b")), ("q1", session("c"))]
    groups = group_by_namespace(pairs)

    for _, sessions in groups:
        assert len({s.session_id for s in sessions}) == len(sessions)
    assert {ns for ns, _ in groups} == {"q1", "q2"}


def test_memories_are_written_under_their_own_namespace(build):
    pipeline, store, _ = build(FakeExtractor(), sessions_per_request=2)
    pipeline.run([("q1", session("a")), ("q2", session("b"))])

    assert store.count("q1") == 1
    assert store.count("q2") == 1
    assert store.count("user") == 0, "nothing lands in a shared bucket"
    store.close()


def test_batched_splits_evenly_and_keeps_the_remainder():
    assert list(batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(batched([], 3)) == []
