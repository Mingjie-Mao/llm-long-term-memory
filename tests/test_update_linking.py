"""Stage D in the write path: `UpdateLinker` inside `IngestionPipeline`."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from llm_long_term_memory.evaluation.datasets.longmemeval import HaystackSession, HaystackTurn
from llm_long_term_memory.ingest import fingerprint
from llm_long_term_memory.ingest.dedup import DedupOutcome
from llm_long_term_memory.ingest.extract import ExtractionOutcome
from llm_long_term_memory.ingest.pipeline import IngestionPipeline
from llm_long_term_memory.store import Memory, NumpyFlatIndex, SQLiteMemoryStore
from llm_long_term_memory.temporal import TemporalResolver
from llm_long_term_memory.temporal.transition import PairVerdict, UpdateLinker

FACTS = {
    # session id: (day, predicate, object, content)
    "s1": (1, "team_size", "4", "The user leads a team of 4 engineers."),
    "s2": (9, "team_count", "5", "The user leads a team of 5 engineers."),
}


def _session(sid):
    day = FACTS[sid][0]
    return HaystackSession(
        session_id=sid,
        date=f"2026/01/{day:02d} (Mon) 09:00",
        turns=[HaystackTurn(role="user", content=FACTS[sid][3])],
    )


class Extractor:
    user_id = "u1"
    version = "fake"
    model = "fake"

    def extract(self, sessions):
        out = []
        for s in sessions:
            day, predicate, obj, content = FACTS[s.session_id]
            when = datetime(2026, 1, day)
            out.append(
                Memory(
                    id=f"{self.user_id}-{s.session_id}",
                    user_id=self.user_id,
                    type="semantic",
                    content=content,
                    token_count=8,
                    subject="user",
                    predicate=predicate,
                    object=obj,
                    valid_from=when,
                    observed_at=when,
                    ingested_at=when,
                    source_session_id=s.session_id,
                    source_turn_index=0,
                )
            )
        return ExtractionOutcome(out, 1)


class Dedup:
    threshold = None

    def process(self, candidates, vectors=None):
        return DedupOutcome(kept=list(candidates))


class Encoder:
    def encode(self, texts, show_progress=False):
        vectors = np.ones((len(texts), 4), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class Judge:
    def __init__(self, relation="replace"):
        self.relation = relation
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)

        class Completion:
            text = PairVerdict(reason="count changed", relation=self.relation).model_dump_json()

        return Completion()


def _pipeline(tmp_path, judge, *, resolver=True, linker=True):
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()
    index = NumpyFlatIndex(tmp_path / "idx", dim=4)
    pipeline = IngestionPipeline(
        Extractor(),
        Dedup(),
        store,
        index,
        Encoder(),
        checkpoint_path=tmp_path / "ckpt.json",
        sessions_per_request=1,
        checkpoint_every=1,
        resolver=TemporalResolver(store) if resolver else None,
        update_linker=UpdateLinker(judge, "m") if linker else None,
    )
    return pipeline, store


@pytest.mark.parametrize("order", [["s1", "s2"], ["s2", "s1"]])
def test_a_cross_key_update_is_linked_whichever_session_arrives_first(tmp_path, order):
    judge = Judge()
    pipeline, store = _pipeline(tmp_path, judge)
    pipeline.run([("u1", _session(sid)) for sid in order])
    old, new = store.get("u1-s1"), store.get("u1-s2")
    assert old.status == "superseded" and old.superseded_by == "u1-s2"
    assert new.status == "active" and new.predicate == "team_size"
    assert len(judge.calls) == 1  # one judgment for the pair, made when the second arrives
    prompt = judge.calls[0]["prompt"]
    assert prompt.index("4 engineers") < prompt.index("5 engineers")  # earlier first, always


def test_coexist_changes_nothing(tmp_path):
    pipeline, store = _pipeline(tmp_path, Judge("coexist"))
    pipeline.run([("u1", _session("s1")), ("u1", _session("s2"))])
    assert {store.get("u1-s1").status, store.get("u1-s2").status} == {"active"}
    assert store.get("u1-s2").predicate == "team_count"


def test_other_users_facts_are_never_judged_or_touched(tmp_path):
    judge = Judge()
    pipeline, store = _pipeline(tmp_path, judge)
    pipeline.run([("u1", _session("s1")), ("u2", _session("s2"))])
    assert judge.calls == []
    assert store.get("u1-s1").status == "active" and store.get("u2-s2").status == "active"


def test_linking_needs_the_resolver_and_is_fingerprinted_only_when_on(tmp_path):
    with pytest.raises(ValueError, match="resolver"):
        _pipeline(tmp_path, Judge(), resolver=False)
    off = fingerprint.from_runtime(Extractor(), sessions_per_request=1).as_dict()
    on = fingerprint.from_runtime(
        Extractor(), sessions_per_request=1, linker=UpdateLinker(Judge(), "m")
    ).as_dict()
    assert "update_linking" not in off
    assert on["update_linking"].startswith("transition-pair-v3")
