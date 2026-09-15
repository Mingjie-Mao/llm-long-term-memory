"""Answering from the conversation's store, not from a store named after the question.

A corpus can ask many questions of one conversation, and ingestion writes that conversation
once, under the conversation's name. A runner that still looked a question up by its own
id would find an empty store for every one of them, answer "I do not know" twenty times,
and every row would read as a memory failure rather than a wiring one.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from llm_long_term_memory.evaluation.datasets.longmemeval import Instance
from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
from llm_long_term_memory.store import Memory, NumpyFlatIndex, Session, SQLiteMemoryStore

CONVERSATION = "conv-1"


class ScriptedClient:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, *, role, model, prompt, **kw):
        self.prompts.append(prompt)

        class Completion:
            text = "an answer"
            input_tokens = 100
            output_tokens = 5
            thinking_tokens = 0
            api_latency_ms = 4.0

        return Completion()


class StubEncoder:
    def encode(self, texts, show_progress=False):
        return np.ones((len(texts), 2), dtype=np.float32)

    def encode_one(self, text):
        return np.ones(2, dtype=np.float32)


def runner_over_one_conversation(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "m.db")
    store.initialize()
    store.add_session(
        Session(
            id="s1",
            user_id=CONVERSATION,
            started_at=datetime(2024, 3, 10),
            source="corpus:s00c000",
        )
    )
    store.add_memories(
        [
            Memory(
                id="m1",
                user_id=CONVERSATION,
                type="semantic",
                content="The user moved to Lisbon",
                token_count=5,
                subject="user",
                predicate="lives_in",
                object="Lisbon",
                event_time=datetime(2024, 3, 10),
                valid_from=datetime(2024, 3, 10),
                valid_to=None,
                status="active",
                ingested_at=datetime(2026, 1, 1),
                source_session_id="s1",
            )
        ]
    )
    index = NumpyFlatIndex(tmp_path / "idx", dim=2)
    index.add(["m1"], np.ones((1, 2), dtype=np.float32))
    runner = MemoryRunner(
        ScriptedClient(),
        model="m",
        encoder=StubEncoder(),
        store=store,
        index=index,
        temporal=True,
    )
    return store, runner


def question(question_id, namespace):
    return Instance(
        question_id=question_id,
        question_type="knowledge_update",
        question="Where do I live now?",
        answer="Lisbon",
        question_date="2024/03/10 (Sun) 00:00",
        sessions=[],
        answer_session_ids=[],
        namespace=namespace,
    )


def test_every_question_of_a_conversation_reads_the_conversations_store(tmp_path):
    store, runner = runner_over_one_conversation(tmp_path)
    try:
        for question_id in (f"{CONVERSATION}-knowledge_update-0", f"{CONVERSATION}-abstention-1"):
            runner.answer(question(question_id, CONVERSATION))
        assert len(runner.client.prompts) == 2
        assert all("Lisbon" in prompt for prompt in runner.client.prompts)
    finally:
        store.close()


def test_without_the_namespace_a_question_looks_for_a_store_of_its_own(tmp_path):
    store, runner = runner_over_one_conversation(tmp_path)
    try:
        runner.answer(question(f"{CONVERSATION}-knowledge_update-0", None))
        assert "Lisbon" not in runner.client.prompts[0]
    finally:
        store.close()
