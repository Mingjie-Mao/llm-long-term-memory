from __future__ import annotations

from datetime import datetime

import numpy as np

from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.runtime import AnswerEngine
from llm_long_term_memory.runtime.answer_engine import TEMPORAL_NOTE, render_context
from llm_long_term_memory.store import Memory, NumpyFlatIndex, SQLiteMemoryStore


class Encoder:
    def encode_one(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)

    def generate(self, **kwargs):
        text = next(self.replies)

        class Completion:
            input_tokens = 10
            output_tokens = 3
            api_latency_ms = 1.0

        completion = Completion()
        completion.text = text
        return completion


def test_product_answer_engine_needs_no_benchmark_instance(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    store.initialize()
    store.add_memories(
        [
            Memory(
                id="m1",
                user_id="alice",
                type="profile",
                content="The user lives in Lisbon.",
                token_count=5,
                status="active",
                valid_from=datetime(2025, 1, 1),
                ingested_at=datetime(2025, 1, 1),
            )
        ]
    )
    index = NumpyFlatIndex(tmp_path / "memory-index", dim=2)
    index.add(["m1"], np.array([[1.0, 0.0]], dtype=np.float32))
    engine = AnswerEngine(
        Client(["Lisbon"]),
        model="answerer",
        encoder=Encoder(),
        store=store,
        index=index,
        temporal=True,
        raw_fallback=False,
    )

    answer = engine.answer_request(
        AnswerRequest(question="Where do I live?", asked_on="2026-01-01", user_id="alice")
    )

    assert answer.text == "Lisbon"
    assert answer.retrieved_ids == ["m1"]
    store.close()


def test_live_service_no_longer_imports_the_benchmark_memory_runner():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "src/llm_long_term_memory/api/service.py"
    ).read_text(encoding="utf-8")

    assert "evaluation.runners.memory import MemoryRunner" not in source


def test_product_context_matches_the_established_default_runner_rendering():
    """Extraction must not simplify the product prompt that the API used before."""
    from llm_long_term_memory.evaluation.runners.memory import render_grouped

    memories = [
        Memory(
            id="old",
            user_id="alice",
            type="profile",
            scope="profile",
            subject="user",
            predicate="home_city",
            content="The user lived in Perth.",
            object="Perth",
            token_count=5,
            status="superseded",
            valid_from=datetime(2024, 1, 1),
            valid_to=datetime(2025, 1, 1),
            ingested_at=datetime(2024, 1, 1),
        ),
        Memory(
            id="current",
            user_id="alice",
            type="profile",
            scope="profile",
            subject="user",
            predicate="home_city",
            content="The user lives in Lisbon.",
            object="Lisbon",
            token_count=5,
            status="active",
            valid_from=datetime(2025, 1, 1),
            ingested_at=datetime(2025, 1, 1),
        ),
    ]

    expected = f"{TEMPORAL_NOTE}\n\n{render_grouped(memories, True)}"

    assert render_context(memories, True) == expected
