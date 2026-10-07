"""The product answerer's opt-in raw-primary excerpts (`ServiceConfig.raw_primary_tokens`).

Off by default, so the served prompt is unchanged unless a deployment asks for it; when
on, only the asking tenant's turns may reach the prompt.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from llm_long_term_memory.config import ServiceConfig
from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.runtime import AnswerEngine
from llm_long_term_memory.store import Memory, NumpyFlatIndex, Session, SQLiteMemoryStore, Turn


class Encoder:
    def encode_one(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)


class Client:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, *, prompt, **kwargs):
        self.prompts.append(prompt)

        class Completion:
            text = "an answer"
            input_tokens = 10
            output_tokens = 3
            api_latency_ms = 1.0

        return Completion()


@pytest.fixture
def store(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    store.initialize()
    for sid, user, content in (
        ("a", "alice", "I bought the kayak a month ago for $640."),
        ("b", "bob", "My kayak cost $9,999 yesterday."),
    ):
        started = datetime(2025, 5, 24)
        store.add_session(
            Session(sid, user, started, turns=[Turn(f"{sid}-0", sid, 0, "user", content, started)])
        )
    store.add_memories(
        [
            Memory(
                id="m1",
                user_id="alice",
                type="semantic",
                content="The user owns a kayak.",
                token_count=5,
                status="active",
                valid_from=datetime(2025, 5, 24),
                ingested_at=datetime(2025, 5, 24),
            )
        ]
    )
    yield store
    store.close()


def _engine(store, tmp_path, **kw):
    index = NumpyFlatIndex(tmp_path / "memory-index", dim=2)
    index.add(["m1"], np.array([[1.0, 0.0]], dtype=np.float32))
    return AnswerEngine(
        Client(), model="answerer", encoder=Encoder(), store=store, index=index, **kw
    )


def _ask(user):
    return AnswerRequest(
        question="How much did my kayak cost?", asked_on="2025-06-14", user_id=user
    )


def test_off_by_default_in_the_engine_and_the_service_config(store, tmp_path):
    assert ServiceConfig().raw_primary_tokens == 0
    assert ServiceConfig().raw_primary_time_notes is False
    engine = _engine(store, tmp_path)
    answer = engine.answer_request(_ask("alice"))
    assert "$640" not in engine.client.prompts[0]
    assert answer.notes["raw_primary_turns"] == 0


def test_on_adds_only_the_asking_tenants_turns(store, tmp_path):
    engine = _engine(store, tmp_path, raw_primary_tokens=4000)
    answer = engine.answer_request(_ask("alice"))
    prompt = engine.client.prompts[0]
    assert "The user owns a kayak." in prompt
    assert "$640" in prompt
    assert "9,999" not in prompt, "another tenant's turn must never be selected"
    assert answer.notes["raw_primary_turns"] == 1


def test_time_notes_date_the_excerpts(store, tmp_path):
    engine = _engine(store, tmp_path, raw_primary_tokens=4000, raw_primary_time_notes=True)
    engine.answer_request(_ask("alice"))
    prompt = engine.client.prompts[0]
    assert "a month ago [≈ 2025-04-24]" in prompt
    assert "21 days, about 3.0 weeks, before the question" in prompt


def test_time_notes_without_excerpts_is_refused(store, tmp_path):
    with pytest.raises(ValueError):
        _engine(store, tmp_path, raw_primary_time_notes=True)
