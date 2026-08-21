"""`MemoryRunner` with a session-coherent context instead of a flat ranking.

The module-level assembly is tested in `test_coherent.py`. What these pin is the
wiring: that the runner actually uses it, that the row says which arm produced it,
and that memories a session contributes without having been retrieved do not get
invented signals.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from llm_long_term_memory.evaluation.datasets.longmemeval import Instance
from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
from llm_long_term_memory.retrieve import SessionBudget
from llm_long_term_memory.store import Memory, NumpyFlatIndex, Session, SQLiteMemoryStore


class ScriptedClient:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, *, role, model, prompt, **kw):
        self.prompts.append(prompt)

        class C:
            text = "an answer"
            input_tokens = 100
            output_tokens = 5
            thinking_tokens = 0
            api_latency_ms = 4.0

        return C()


class StubEncoder:
    def encode(self, texts, show_progress=False):
        return np.ones((len(texts), 2), dtype=np.float32)

    def encode_one(self, text):
        return np.ones(2, dtype=np.float32)


def fact(mid, session, day, content) -> Memory:
    return Memory(
        id=mid,
        user_id="q1",
        type="semantic",
        content=content,
        token_count=6,
        subject="user",
        predicate="did",
        object=content.split()[-1],
        event_time=datetime(2023, 1, day),
        valid_from=datetime(2023, 1, day),
        status="active",
        ingested_at=datetime(2026, 1, 1),
        source_session_id=session,
    )


@pytest.fixture
def wired(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "m.db")
    store.initialize()
    for sid, day in (("s1", 1), ("s2", 5)):
        store.add_session(
            Session(id=sid, user_id="q1", started_at=datetime(2023, 1, day), source="test")
        )
    memories = [
        fact("a1", "s1", 1, "The user began the sculpture"),
        fact("a2", "s1", 2, "The user bought extra clay downtown"),
        fact("a3", "s1", 3, "The user spent 10-12 hours on the sculpture"),
        fact("b1", "s2", 5, "The user repaired a bike"),
    ]
    store.add_memories(memories)
    index = NumpyFlatIndex(tmp_path / "idx", dim=2)
    index.add([m.id for m in memories], np.ones((len(memories), 2), dtype=np.float32))

    def build(**kwargs):
        return MemoryRunner(
            ScriptedClient(),
            model="m",
            encoder=StubEncoder(),
            store=store,
            index=index,
            temporal=True,
            **kwargs,
        )

    yield build
    store.close()


def instance() -> Instance:
    return Instance(
        question_id="q1",
        question_type="temporal-reasoning",
        question="How long did I spend on the sculpture?",
        answer="10-12 hours",
        question_date="2026-01-01",
        sessions=[],
        answer_session_ids=["s1"],
    )


def test_a_coherent_run_says_so_in_its_row(wired):
    flat = wired().answer(instance())
    assert flat.notes["context_shape"] == "flat"
    assert flat.notes["coherent"] is None

    coherent = wired(session_budget=SessionBudget(max_sessions=1)).answer(instance())
    assert coherent.notes["context_shape"] == "coherent"
    assert coherent.notes["coherent"]["budget"]["max_sessions"] == 1


def test_the_session_arrives_whole_and_in_order_in_the_prompt(wired):
    runner = wired(session_budget=SessionBudget(max_sessions=1))
    runner.answer(instance())
    prompt = runner.client.prompts[0]
    starts = [prompt.index(text) for text in ("began the sculpture", "extra clay", "10-12 hours")]
    assert starts == sorted(starts), "session memories must appear oldest first"
    assert "repaired a bike" not in prompt, "a second session is outside the budget of 1"


def test_the_session_carries_in_memories_the_cut_dropped(wired):
    """The production mechanism, reproduced: `top_k` truncates a ranking and takes
    part of a conversation with it. Selecting the conversation instead restores the
    rest — including memories that carry no signals, because they were never ranked.

    A memory with no hit is reported with no signals rather than signals of zero;
    zero would read as "scored and lost", which is a different fact about ranking.
    """
    flat = wired(top_k=2).answer(instance())
    assert len(flat.retrieved_ids) == 2, "the cut is what creates the loss"

    runner = wired(top_k=2, session_budget=SessionBudget(max_sessions=1))
    answer = runner.answer(instance())
    assert set(answer.retrieved_ids) == {"a1", "a2", "a3"}, "all of s1, cut or not"
    assert answer.notes["coherent"]["unretrieved"] >= 1

    reported = {row["memory_id"] for row in answer.notes["retrieval"]}
    assert reported < set(answer.retrieved_ids), "unranked memories carry no signals"
    assert "extra clay" in runner.client.prompts[0]


def test_dropped_sessions_are_named_rather_than_vanishing(wired):
    runner = wired(session_budget=SessionBudget(max_sessions=1))
    answer = runner.answer(instance())
    coherent = answer.notes["coherent"]
    assert len(coherent["sessions"]) == 1
    assert coherent["dropped_sessions"], "the session left out has to be reported"


def test_packing_and_coherence_together_are_refused_rather_than_resolved(wired):
    with pytest.raises(ValueError, match="Choose one"):
        wired(session_budget=SessionBudget(), token_budget=500)


def test_the_variant_is_the_only_thing_that_turns_the_shape_on(tmp_path, monkeypatch):
    """`two_stage_coherent` and `two_stage_fallback` differ in the context's shape
    and in nothing else, which is what makes the pair an ablation rather than two
    unrelated runs. The budget comes from the config so `scripts/freeze.py` can see
    it; a command-line flag would be a variable the freeze record cannot record.
    """
    from llm_long_term_memory.config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml("configs/fallback.yaml")
    assert cfg.context.max_sessions >= 1
    assert cfg.context.aggregate in {"max", "sum", "mean", "sum_top3"}
    assert cfg.context.session_order in {"score", "chronological"}
