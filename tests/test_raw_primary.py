"""Raw-primary arms: verbatim turns found in the whole archive by the question.

Registered in `results/prereg-raw-primary-heldout100-v1.md`. These pin what the arms
put in front of the answerer — and what they must never put there: another user's
turns.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from llm_long_term_memory.evaluation.datasets.longmemeval import Instance
from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
from llm_long_term_memory.retrieve.excerpts import archive_excerpts
from llm_long_term_memory.store import Memory, NumpyFlatIndex, Session, SQLiteMemoryStore, Turn


class _Client:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, *, role, model, prompt, **kw):
        self.prompts.append(prompt)

        class C:
            text = "an answer"
            input_tokens = 100
            output_tokens = 5
            thinking_tokens = 0
            api_latency_ms = 1.0

        return C()


class _Encoder:
    def encode(self, texts, show_progress=False):
        return np.ones((len(texts), 2), dtype=np.float32)

    def encode_one(self, text):
        return np.ones(2, dtype=np.float32)


def _session(sid, user, day, *contents):
    started = datetime(2023, 5, day)
    return Session(
        id=sid,
        user_id=user,
        started_at=started,
        turns=[
            Turn(f"{sid}-{i}", sid, i, "user" if i % 2 == 0 else "assistant", c, started)
            for i, c in enumerate(contents)
        ],
    )


@pytest.fixture
def store(tmp_path):
    s = SQLiteMemoryStore(tmp_path / "r.db")
    s.initialize()
    s.add_session(_session("late", "q1", 20, "I finally bought the kayak for $640.", "Nice."))
    s.add_session(_session("early", "q1", 2, "I am saving for a kayak.", "Good plan."))
    s.add_session(_session("other", "q2", 3, "My kayak cost $9,999.", "Wow."))
    yield s
    s.close()


def test_only_this_users_turns_are_found(store):
    found = archive_excerpts(store, "q1", "kayak", 4000)
    assert {t.session_id for t in found.turns} == {"late", "early"}
    assert "9,999" not in found.render()


def test_turns_are_shown_oldest_conversation_first_with_their_dates(store):
    text = archive_excerpts(store, "q1", "kayak", 4000).render()
    assert text.index("2023-05-02") < text.index("2023-05-20")
    assert text.index("saving for a kayak") < text.index("bought the kayak")


def test_the_budget_is_respected_and_an_overflowing_turn_is_skipped(store):
    found = archive_excerpts(store, "q1", "kayak", 8)
    assert found.tokens <= 8
    assert found.skipped_for_budget >= 1


def test_no_budget_means_no_excerpts(store):
    assert archive_excerpts(store, "q1", "kayak", 0).render() == ""


def _runner(store, tmp_path, **kw):
    memory = Memory(
        id="m1",
        user_id="q1",
        type="semantic",
        content="The user bought a kayak.",
        token_count=5,
        source_session_id="late",
        ingested_at=datetime(2026, 1, 1),
    )
    store.add_memories([memory])
    index = NumpyFlatIndex(tmp_path / "idx", dim=2)
    index.add(["m1"], np.ones((1, 2), dtype=np.float32))
    return MemoryRunner(_Client(), model="m", encoder=_Encoder(), store=store, index=index, **kw)


def _question():
    return Instance(
        question_id="q1",
        question_type="single-session-user",
        question="How much did my kayak cost?",
        answer="$640",
        question_date="2023-06-01",
        sessions=[],
        answer_session_ids=["late"],
    )


def test_raw_primary_adds_the_excerpts_to_the_memory_context(store, tmp_path):
    runner = _runner(store, tmp_path, raw_primary_tokens=4000)
    answer = runner.answer(_question())
    prompt = runner.client.prompts[0]
    assert "The user bought a kayak." in prompt
    assert "$640" in prompt
    assert "9,999" not in prompt
    assert answer.notes["raw_primary_turns"] >= 1
    assert answer.notes["raw_primary_only"] is False


def test_raw_only_drops_the_memory_context(store, tmp_path):
    runner = _runner(store, tmp_path, raw_primary_tokens=4000, raw_primary_only=True)
    runner.answer(_question())
    prompt = runner.client.prompts[0]
    assert "The user bought a kayak." not in prompt
    assert "$640" in prompt


def test_raw_only_without_a_budget_is_refused(store, tmp_path):
    with pytest.raises(ValueError):
        _runner(store, tmp_path, raw_primary_only=True)


def test_the_arms_differ_from_the_control_only_where_registered():
    """Same answer policy and hydration as `two_stage_hydrated`; raw-only has no fallback."""
    import inspect
    import re

    from llm_long_term_memory import cli

    source = inspect.getsource(cli)
    hydrated = re.search(
        r'evidence_hydration="_hydrated" in variant\s+or variant\s+in \{([^}]*)\}', source
    )
    assert hydrated is not None
    assert set(re.findall(r'"(\w+)"', hydrated.group(1))) == {
        "two_stage_raw_primary",
        "two_stage_raw_primary_t1",
        "two_stage_raw_primary_t2",
        "two_stage_raw_primary_t3",
        "two_stage_raw_primary_t4",
        "two_stage_raw_primary_v2",
        "two_stage_raw_primary_v4",
    }
    assert cli.RAW_PRIMARY_TOKENS == 4000
