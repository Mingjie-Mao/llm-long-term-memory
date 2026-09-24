"""The variant that reads from the memory store.

The point of these tests is that `chronomem` and `chronomem_no_temporal` differ in
exactly one thing — whether superseded facts reach the prompt. If anything else
differed, the results table could not attribute the gap to temporal resolution.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest

from llm_long_term_memory.evaluation.datasets.longmemeval import Instance
from llm_long_term_memory.evaluation.runners.memory import MemoryRunner, render_memory
from llm_long_term_memory.store import Memory, NumpyFlatIndex, Session, SQLiteMemoryStore, Turn


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


def fact(mid, content, status="active", valid_to=None, session="s1", user="q1") -> Memory:
    return Memory(
        id=mid,
        user_id=user,
        type="semantic",
        content=content,
        token_count=6,
        subject="user",
        predicate="uses_framework",
        object=content.split()[-1],
        event_time=datetime(2023, 1, 1),
        valid_from=datetime(2023, 1, 1),
        valid_to=valid_to,
        status=status,
        ingested_at=datetime(2026, 1, 1),
        source_session_id=session,
    )


@pytest.fixture
def wired(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "m.db")
    store.initialize()
    # The source session has to exist first: `source_session_id` is a real foreign
    # key, so a memory cannot claim provenance it does not have.
    store.add_session(
        Session(id="s1", user_id="q1", started_at=datetime(2023, 1, 1), source="test")
    )
    memories = [
        fact("old", "The user uses TensorFlow", "superseded", datetime(2023, 8, 1)),
        fact("new", "The user uses PyTorch"),
    ]
    store.add_memories(memories)

    index = NumpyFlatIndex(tmp_path / "idx", dim=2)
    index.add([m.id for m in memories], np.ones((2, 2), dtype=np.float32))

    def build(temporal: bool):
        return MemoryRunner(
            ScriptedClient(),
            model="m",
            encoder=StubEncoder(),
            store=store,
            index=index,
            temporal=temporal,
        )

    yield build
    store.close()


def instance() -> Instance:
    return Instance(
        question_id="q1",
        question_type="temporal-reasoning",
        question="What framework do I use?",
        answer="PyTorch",
        question_date="2026-01-01",
        sessions=[],
        answer_session_ids=["s1"],
    )


def test_without_temporal_the_superseded_fact_reaches_the_model(wired):
    """The failure the whole phase exists to fix: both answers offered, model picks."""
    runner = wired(temporal=False)
    answer = runner.answer(instance())

    prompt = runner.client.prompts[0]
    assert "TensorFlow" in prompt
    assert "PyTorch" in prompt
    assert answer.notes["superseded_shown"] == 1


def test_with_temporal_only_the_current_fact_is_shown(wired):
    runner = wired(temporal=True)
    answer = runner.answer(instance())

    prompt = runner.client.prompts[0]
    assert "TensorFlow" not in prompt
    assert "PyTorch" in prompt
    assert answer.notes["superseded_shown"] == 0


def test_the_two_variants_differ_only_in_filtering(wired):
    """Same store, same index, same encoder, same top_k. If ranking differed too,
    the table could not attribute the gap to the timeline."""
    off, on = wired(temporal=False), wired(temporal=True)
    assert off.top_k == on.top_k
    assert off.store is on.store
    assert off.index is on.index

    off.answer(instance())
    on.answer(instance())
    assert "What framework do I use?" in off.client.prompts[0]
    assert "What framework do I use?" in on.client.prompts[0]


def test_temporal_prompt_explains_the_window_notation(wired):
    runner = wired(temporal=True)
    runner.answer(instance())
    prompt = runner.client.prompts[0]
    assert "since" in prompt
    assert "no longer current" in prompt, "the note must define the closed-window form"


def test_render_shows_an_open_window_for_current_facts():
    line = render_memory(fact("a", "The user uses PyTorch"), temporal=True)
    assert "since 2023-01-01" in line
    assert "no longer current" not in line


def test_render_marks_a_closed_window_as_stale():
    stale = fact("a", "The user uses TensorFlow", "superseded", datetime(2023, 8, 1))
    line = render_memory(stale, temporal=True)
    assert "2023-01-01 to 2023-08-01" in line
    assert "no longer current" in line


def test_render_admits_when_a_date_is_unknown():
    undated = fact("a", "The user uses Keras")
    undated.valid_from = None
    # All three, not just `event_time`. A memory with no stated time still has the
    # date of the conversation it came from, and `occurred_at` finds it — so nulling
    # only `event_time` would leave this asserting something the renderer no longer
    # decides on, and it would pass for the wrong reason.
    undated.event_time = None
    undated.observed_at = None
    assert undated.occurred_at is None
    assert "date unknown" in render_memory(undated, temporal=True)


def test_render_without_temporal_carries_no_dates():
    line = render_memory(fact("a", "The user uses PyTorch"), temporal=False)
    assert line == "- The user uses PyTorch"


def test_evidence_recall_is_reported_from_source_sessions(wired):
    answer = wired(temporal=True).answer(instance())
    assert answer.notes["evidence_recalled"] is True
    assert answer.notes["source_session_recalled"] is True


def test_hydration_adds_local_verbatim_evidence_to_the_prompt(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "m.db")
    store.initialize()
    raw = "I started collecting vintage cameras three months ago. I now own 17 cameras."
    store.add_session(
        Session(
            id="s1",
            user_id="q1",
            started_at=datetime(2023, 1, 1),
            turns=[Turn("s1:0", "s1", 0, "user", raw, datetime(2023, 1, 1))],
        )
    )
    start = raw.index("I now own")
    memory = Memory(
        id="m1",
        user_id="q1",
        type="semantic",
        content="The user owns 17 vintage cameras.",
        token_count=8,
        source_session_id="s1",
        source_turn_index=0,
        source_char_start=start,
        source_char_end=len(raw),
    )
    store.add_memories([memory])
    index = NumpyFlatIndex(tmp_path / "idx", dim=2)
    index.add([memory.id], np.ones((1, 2), dtype=np.float32))
    client = ScriptedClient()
    runner = MemoryRunner(
        client,
        model="m",
        encoder=StubEncoder(),
        store=store,
        index=index,
        evidence_hydration=True,
    )

    answer = runner.answer(instance())

    assert "Verbatim source evidence" in client.prompts[0]
    assert "three months ago" in client.prompts[0]
    assert answer.notes["hydrated_memory_ids"] == ["m1"]
    store.close()


def test_prepare_is_a_no_op_because_the_store_is_prebuilt(wired):
    """Unlike the baselines, which index every question's haystack at query time."""
    runner = wired(temporal=True)
    runner.prepare(instance())
    assert runner.client.prompts == []


def test_runner_names_match_the_config_variants(wired):
    assert wired(temporal=True).name == "chronomem"
    assert wired(temporal=False).name == "chronomem_no_temporal"


def test_answer_is_json_serialisable_for_the_harness(wired):
    answer = wired(temporal=True).answer(instance())
    json.dumps(answer.notes)  # must not raise
    assert answer.notes["retrieval_latency_ms"] >= 0
    assert answer.notes["assembly_latency_ms"] >= 0
    assert answer.notes["answerer_api_latency_ms"] == 4.0


def test_another_questions_memories_are_never_retrieved(tmp_path):
    """LongMemEval questions are independent users with no shared history. A global
    index without a namespace filter answered one question from another's evidence
    — and merged their timelines, producing a `lives_in` chain across seven cities.
    """
    store = SQLiteMemoryStore(tmp_path / "x.db")
    store.initialize()
    for ns in ("q1", "q2"):
        store.add_session(
            Session(id=f"s_{ns}", user_id=ns, started_at=datetime(2023, 1, 1), source="t")
        )
    mine = fact("mine", "The user uses PyTorch", session="s_q1", user="q1")
    theirs = fact("theirs", "The user uses Fortran", session="s_q2", user="q2")
    store.add_memories([mine, theirs])

    index = NumpyFlatIndex(tmp_path / "i", dim=2)
    index.add(["mine", "theirs"], np.ones((2, 2), dtype=np.float32))

    runner = MemoryRunner(
        ScriptedClient(), model="m", encoder=StubEncoder(), store=store, index=index, temporal=True
    )
    runner.answer(instance())

    prompt = runner.client.prompts[0]
    assert "PyTorch" in prompt
    assert "Fortran" not in prompt, "another question's evidence must not leak in"
    store.close()
