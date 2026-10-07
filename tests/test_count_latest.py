"""Answer policy `v2_cl`: latest value wins; counts and totals answered item by item.

Raw-primary t2 (step 2 after the dev100 error taxonomy). Every question gets the
latest-value rule; only a count or total switches to item notes and a larger output
allowance, so a lookup keeps v2's verdict and limit.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest

from llm_long_term_memory.answering import (
    ANSWER_SYSTEM,
    COUNT_LATEST_PROMPT_VERSION,
    AnswerVerdict,
    NotedAnswerVerdict,
    asks_for_aggregate,
)
from llm_long_term_memory.evaluation.datasets.longmemeval import Instance
from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
from llm_long_term_memory.store import Memory, NumpyFlatIndex, Session, SQLiteMemoryStore, Turn


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many fitness classes do I attend in a typical week?", True),
        ("How much money did I raise for charity in total?", True),
        ("How many hours in total did I spend driving to my three destinations combined?", True),
        ("What is the total number of model kits I bought?", True),
        ("How many days ago did I read the March 15th issue of The New Yorker?", False),
        ("How many weeks ago did I attend the festival?", False),
        ("What time do I usually go to the gym?", False),
        ("Which show did I start watching first?", False),
    ],
)
def test_count_and_total_questions_are_recognised(question, expected):
    assert asks_for_aggregate(question) is expected


class _Client:
    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, *, role, model, prompt, **kw):
        self.calls.append({"prompt": prompt, **kw})

        class C:
            text = json.dumps({"notes": [], "status": "answer", "answer": "x", "reason": ""})
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


@pytest.fixture
def runner(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "cl.db")
    store.initialize()
    started = datetime(2023, 5, 2)
    store.add_session(
        Session(
            "s",
            "q1",
            started,
            turns=[Turn("s-0", "s", 0, "user", "I go to yoga twice a week.", started)],
        )
    )
    store.add_memories(
        [
            Memory(
                id="m1",
                user_id="q1",
                type="semantic",
                content="The user goes to yoga.",
                token_count=5,
                source_session_id="s",
                ingested_at=datetime(2026, 1, 1),
            )
        ]
    )
    index = NumpyFlatIndex(tmp_path / "idx", dim=2)
    index.add(["m1"], np.ones((1, 2), dtype=np.float32))
    runner = MemoryRunner(
        _Client(),
        model="m",
        encoder=_Encoder(),
        store=store,
        index=index,
        raw_primary_tokens=4000,
        answer_policy="v2_cl",
    )
    yield runner
    store.close()


def _ask(question):
    return Instance(
        question_id="q1",
        question_type="multi-session",
        question=question,
        answer="",
        question_date="2023-06-01",
        sessions=[],
        answer_session_ids=["s"],
    )


def test_a_count_gets_item_notes_and_room_for_them(runner):
    answer = runner.answer(_ask("How many yoga classes do I attend in total each week?"))
    call = runner.client.calls[-1]
    assert "Before deciding, write `notes`: one note per candidate item" in call["system"]
    assert "most recent conversation is the current one" in call["system"]
    assert call["max_output_tokens"] == 1024
    assert runner.verdict_schema is NotedAnswerVerdict
    assert answer.notes["answer_mode"] == "count_notes"
    assert runner.answer_prompt_version == COUNT_LATEST_PROMPT_VERSION


def test_a_lookup_keeps_v2_with_only_the_latest_value_rule(runner):
    runner.answer(_ask("How many yoga classes do I attend in total each week?"))
    answer = runner.answer(_ask("When do I usually go to yoga?"))
    call = runner.client.calls[-1]
    assert call["system"].startswith(ANSWER_SYSTEM)
    assert "most recent conversation is the current one" in call["system"]
    assert "one note per candidate item" not in call["system"]
    assert call["max_output_tokens"] == 512, "reset after the previous count question"
    assert runner.verdict_schema is AnswerVerdict
    assert answer.notes["answer_mode"] == "latest_only"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many days passed between the two trips?", True),
        ("Which device did I set up first, the thermostat or the router?", True),
        ("Who did I go with to the music event last Saturday?", True),
        ("How many weeks ago did I attend the festival?", True),
        ("What time do I usually go to the gym?", False),
        ("Can you suggest some accessories for my phone?", False),
    ],
)
def test_time_questions_are_recognised(question, expected):
    from llm_long_term_memory.answering import asks_about_time

    assert asks_about_time(question) is expected


def test_v2_clt_routes_time_questions_to_notes(tmp_path, runner):
    runner.answer_policy = "v2_clt"
    answer = runner.answer(_ask("Which yoga class did I go to first?"))
    call = runner.client.calls[-1]
    assert "This question depends on dates, durations or order" in call["system"]
    assert "one note per candidate item" not in call["system"]
    assert call["max_output_tokens"] == 1024
    assert runner.verdict_schema is NotedAnswerVerdict
    assert answer.notes["answer_mode"] == "time_notes"
    runner.answer(_ask("How many yoga classes did I take in total since May?"))
    assert runner.client.calls[-1]["system"].count("Before deciding, write `notes`") == 2
    lookup = runner.answer(_ask("What do I do for anxiety?"))
    assert lookup.notes["answer_mode"] == "latest_only"
    assert runner.client.calls[-1]["max_output_tokens"] == 512
