"""The repair as the pipeline actually runs it, not as the pilot measured it.

The pilot that earned this mechanism its gate ran its own loop: extract a batch, compare
each session, call, accept, score. The pipeline does four things that loop never did —
it scopes session ids before the repair sees them, it hands repaired memories through
deduplication, it writes them to the store and the vector index together, and it counts
them. Every one of those can silently turn 44.8% -> 59.3% into no change at all, and
none of them involves a model call, so none of them needs quota to check.

The measurement itself is not repeated here. What is checked is that the path the
measurement would travel is intact.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_long_term_memory.conversation import ConversationSession, ConversationTurn
from llm_long_term_memory.ingest.dedup import DedupOutcome
from llm_long_term_memory.ingest.extract import ExtractionOutcome
from llm_long_term_memory.ingest.pipeline import IngestionPipeline
from llm_long_term_memory.store import Memory, NumpyFlatIndex, SQLiteMemoryStore, scoped_session_id


def _session(session_id: str) -> ConversationSession:
    return ConversationSession(
        session_id=session_id,
        date="2023/05/20",
        turns=[ConversationTurn(role="user", content=f"I walked 416 steps on {session_id}.")],
    )


class _Extractor:
    """Writes one memory per session, with the raw session id the real one uses."""

    user_id = "u"

    def extract(self, sessions):
        return ExtractionOutcome(
            [
                Memory(
                    id=f"base_{s.session_id}",
                    user_id=self.user_id,
                    type="semantic",
                    content=f"The user did something on {s.session_id}.",
                    token_count=5,
                    source_session_id=s.session_id,
                )
                for s in sessions
            ],
            0,
        )


class _Repair:
    """Records what it was asked and returns one recovered memory per session."""

    version = "repair-test"

    def __init__(self, recover: bool = True) -> None:
        self.seen: list[tuple[str, list[str]]] = []
        self.recover = recover

    def prompt_texts(self):
        return ("repair-prompt",)

    def repair(self, session, memories, user_id):
        from llm_long_term_memory.ingest.repair import RepairOutcome

        self.seen.append((session.session_id, [m.id for m in memories]))
        if not self.recover:
            return RepairOutcome([], called=False, missing=[])
        return RepairOutcome(
            [
                Memory(
                    id=f"fix_{session.session_id}",
                    user_id=user_id,
                    type="semantic",
                    content=f"The user walked 416 steps on {session.session_id}.",
                    token_count=5,
                    source_session_id=session.session_id,
                )
            ],
            called=True,
            missing=["416"],
        )


class _PassthroughDedup:
    def __init__(self) -> None:
        self.saw: list[str] = []

    def process(self, candidates, vectors=None):
        self.saw = [m.id for m in candidates]
        return DedupOutcome(kept=list(candidates))


class _Encoder:
    def encode(self, texts, show_progress=False):
        return np.ones((len(texts), 4), dtype=np.float32)

    def encode_one(self, text):
        return np.ones(4, dtype=np.float32)


@pytest.fixture
def build(tmp_path):
    def make(repair=None, dedup=None):
        store = SQLiteMemoryStore(tmp_path / "s.db")
        store.initialize()
        index = NumpyFlatIndex(tmp_path / "idx", dim=4)
        pipeline = IngestionPipeline(
            _Extractor(),
            dedup or _PassthroughDedup(),
            store,
            index,
            _Encoder(),
            checkpoint_path=tmp_path / "ckpt.json",
            sessions_per_request=2,
            checkpoint_every=1,
            repair=repair,
        )
        return pipeline, store, index

    return make


def _pairs():
    return [("q1", _session("s1")), ("q1", _session("s2"))]


def test_repaired_memories_reach_the_store_and_the_index(build):
    repair = _Repair()
    pipeline, store, index = build(repair)

    pipeline.run(_pairs(), resume=False)

    stored = {m.id for m in store.iter_all("q1")}
    assert stored == {"base_s1", "base_s2", "fix_s1", "fix_s2"}
    assert len(index) == 4, "a memory in the store but not the index is unretrievable"


def test_the_repair_sees_this_session_s_memories_not_an_empty_set(build):
    """`_ingest_batch` scopes ids before the repair runs.

    Matching on the raw id would hand every session an empty baseline, so every session
    would look like it had lost everything and every one would buy a call — the cost
    going from 42% of sessions to 100% while the measurement still looked fine.
    """
    repair = _Repair()
    pipeline, _, _ = build(repair)

    pipeline.run(_pairs(), resume=False)

    assert dict(repair.seen) == {"s1": ["base_s1"], "s2": ["base_s2"]}


def test_repaired_memories_go_through_deduplication(build):
    """Skipping it would write a second copy of a fact the store already holds."""
    dedup = _PassthroughDedup()
    pipeline, _, _ = build(_Repair(), dedup=dedup)

    pipeline.run(_pairs(), resume=False)

    assert set(dedup.saw) == {"base_s1", "base_s2", "fix_s1", "fix_s2"}


def test_a_repair_that_deduplication_drops_is_not_counted_as_written(build):
    """The failure mode that would quietly undo the mechanism."""

    class _DropRepairs:
        def process(self, candidates, vectors=None):
            kept = [m for m in candidates if not m.id.startswith("fix_")]
            return DedupOutcome(kept=kept, duplicates=len(candidates) - len(kept))

    pipeline, store, _ = build(_Repair(), dedup=_DropRepairs())

    progress = pipeline.run(_pairs(), resume=False).progress

    assert {m.id for m in store.iter_all("q1")} == {"base_s1", "base_s2"}
    assert progress.repair_memories == 2, "recovered"
    assert progress.memories_written == 2, "and then dropped — the two must not be conflated"


def test_the_counters_separate_what_was_called_for_from_what_came_back(build):
    pipeline, _, _ = build(_Repair())

    progress = pipeline.run(_pairs(), resume=False).progress

    assert progress.repair_requests == 2
    assert progress.repair_memories == 2


def test_a_session_that_lost_nothing_is_not_counted_as_a_request(build):
    pipeline, store, _ = build(_Repair(recover=False))

    progress = pipeline.run(_pairs(), resume=False).progress

    assert progress.repair_requests == 0
    assert progress.repair_memories == 0
    assert {m.id for m in store.iter_all("q1")} == {"base_s1", "base_s2"}


def test_the_counters_survive_a_checkpoint_round_trip(build, tmp_path):
    """A resumed ingest that forgot them would under-report the spend."""
    from llm_long_term_memory.ingest.pipeline import IngestProgress

    pipeline, _, _ = build(_Repair())
    progress = pipeline.run(_pairs(), resume=False).progress
    progress.save(tmp_path / "round.json")

    reloaded = IngestProgress.load(tmp_path / "round.json")

    assert reloaded.repair_requests == progress.repair_requests
    assert reloaded.repair_memories == progress.repair_memories


def test_repaired_memories_carry_the_scoped_session_id(build):
    """Otherwise the raw archive cannot be found from them and tenants are not isolated."""
    pipeline, store, _ = build(_Repair())

    pipeline.run(_pairs(), resume=False)

    fixed = next(m for m in store.iter_all("q1") if m.id == "fix_s1")
    assert fixed.source_session_id == scoped_session_id("q1", "s1")


def test_without_a_repair_the_pipeline_behaves_exactly_as_before(build):
    pipeline, store, _ = build(repair=None)

    progress = pipeline.run(_pairs(), resume=False).progress

    assert {m.id for m in store.iter_all("q1")} == {"base_s1", "base_s2"}
    assert progress.repair_requests == 0
    assert progress.repair_memories == 0


def test_a_session_shared_by_two_tenants_keeps_both_tenants_repairs(build):
    """The real repair, not the stub: its ids must not collide across namespaces."""
    from llm_long_term_memory.ingest.grounded import (
        Anchored,
        GroundedFact,
        GroundingReport,
    )
    from llm_long_term_memory.ingest.repair import SpecificityRepair

    spoken = "I walked 416 steps on s1."

    class _Grounded:
        @staticmethod
        def prompt_texts():
            return ("grounded",)

        def extract(self, turns, session_date):
            fact = GroundedFact(
                content="The user walked 416 steps.", verbatim_span=spoken, turn_index=0
            )
            return GroundingReport(anchored=[Anchored(fact, 0, 0, len(spoken))])

    pipeline, store, index = build(SpecificityRepair(_Grounded()))

    pipeline.run([("q1", _session("s1")), ("q2", _session("s1"))], resume=False)

    for tenant in ("q1", "q2"):
        repaired = [m for m in store.iter_all(tenant) if m.id.startswith("g_")]
        assert len(repaired) == 1, tenant
        assert repaired[0].user_id == tenant
        assert repaired[0].source_session_id == scoped_session_id(tenant, "s1")
    assert len(index) == 4
