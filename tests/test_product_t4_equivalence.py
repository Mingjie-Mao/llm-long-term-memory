"""The served `t4` profile sends exactly the requests the evaluated t4 arm sent.

The product engine and the evaluation runner are separate classes, so nothing but this
test stops the arm that was measured (87.67% on exposed heldout100) and the path the API
answers with from drifting apart again. Every `generate()` call is compared whole —
prompt, system, schema and output limit — across a plain lookup, a count, a date
question and a fallback, and only the asking tenant's turns may reach either.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from llm_long_term_memory.answering import AnswerVerdict, NotedAnswerVerdict
from llm_long_term_memory.config import ServiceConfig
from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
from llm_long_term_memory.runtime import AnswerEngine
from llm_long_term_memory.runtime.answer_engine import ANSWER_PROFILES
from llm_long_term_memory.store import Memory, NumpyFlatIndex, Session, SQLiteMemoryStore, Turn


class Encoder:
    def encode_one(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)


class Client:
    """Records every call; answers the first call per question with `first`."""

    def __init__(self, first: str):
        self.first = first
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        text = (
            self.first if "schema" in kwargs else "The kayak cost $640."  # fallback pass: prose
        )

        class Completion:
            input_tokens = 10
            output_tokens = 3
            api_latency_ms = 1.0

        completion = Completion()
        completion.text = text
        return completion


@pytest.fixture
def store(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    store.initialize()
    rows = (
        ("a", "alice", datetime(2025, 5, 24), "I bought a kayak last week for $640."),
        ("c", "alice", datetime(2025, 6, 2), "I took the kayak out on the lake yesterday."),
        ("b", "bob", datetime(2025, 5, 24), "My kayak cost $9,999 yesterday."),
        # Shares no word with "the music event last Saturday": only time finds it.
        ("s", "alice", datetime(2025, 6, 7), "Saw Billie Eilish live; my parents loved it."),
    )
    for sid, user, started, content in rows:
        store.add_session(
            Session(
                sid,
                user,
                started,
                turns=[
                    Turn(f"{sid}-0", sid, 0, "user", content, started),
                    Turn(f"{sid}-1", sid, 1, "assistant", "Sounds great!", started),
                ],
            )
        )
    # Enough kayak talk that the 4,000-token budget binds, so user-first packing and
    # the time window change which turns are kept rather than only their order.
    for day in range(300):
        sid, started = f"f{day}", datetime(2025, 1, 1) + timedelta(hours=10 * day)
        store.add_session(
            Session(
                sid,
                "alice",
                started,
                turns=[
                    Turn(
                        f"{sid}-0",
                        sid,
                        0,
                        "user",
                        f"Who should I go with to the kayak music event {day}? It matters a lot.",
                        started,
                    ),
                    Turn(
                        f"{sid}-1",
                        sid,
                        1,
                        "assistant",
                        "About kayaks: "
                        + "paddle strokes, the music event, hull shapes and kayak storage. " * 12,
                        started,
                    ),
                ],
            )
        )
    store.add_memories(
        [
            Memory(
                id="m1",
                user_id="alice",
                type="semantic",
                content="The user bought a kayak for $640.",
                token_count=8,
                status="active",
                scope="event",
                subject="user",
                predicate="owns_kayak",
                valid_from=datetime(2025, 5, 24),
                ingested_at=datetime(2025, 5, 24),
                source_session_id="a",
                source_turn_index=0,
                source_char_start=0,
                source_char_end=36,
            ),
            Memory(
                id="m2",
                user_id="alice",
                type="semantic",
                content="The user went kayaking on the lake.",
                token_count=7,
                status="active",
                scope="event",
                valid_from=datetime(2025, 6, 2),
                ingested_at=datetime(2025, 6, 2),
                source_session_id="c",
                source_turn_index=0,
                source_char_start=0,
                source_char_end=43,
            ),
        ]
    )
    yield store
    store.close()


def _index(tmp_path):
    index = NumpyFlatIndex(tmp_path / "memory-index", dim=2)
    index.add(["m1", "m2"], np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32))
    return index


def _evaluated_t4(client, store, index):
    """`two_stage_raw_primary_t4` exactly as `cli.py` builds it from configs/fallback.yaml."""
    return MemoryRunner(
        client,
        model="answerer",
        encoder=Encoder(),
        store=store,
        index=index,
        top_k=20,
        temporal=True,
        evidence_hydration=True,
        hydration_neighbouring_sentences=1,
        hydration_max_tokens=800,
        hydration_allocation="ranked",
        raw_fallback=True,
        raw_fallback_max_turns=3,
        raw_fallback_max_chars=2400,
        raw_primary_tokens=4000,
        raw_primary_dated=True,
        raw_primary_prefer_user=True,
        raw_primary_time_window=True,
        raw_primary_time_notes=True,
        max_output_tokens=512,
        answer_policy="v2_clt",
    )


def _served_t4(client, store, index):
    return AnswerEngine.from_profile(
        "t4",
        client,
        model="answerer",
        encoder=Encoder(),
        store=store,
        index=index,
        top_k=ServiceConfig().top_k,
        temporal=True,
    )


ANSWER = AnswerVerdict(status="answer", answer="$640").model_dump_json()
NOTED = NotedAnswerVerdict(
    notes=["2025-05-24: $640"], status="answer", answer="1"
).model_dump_json()
NEED_SOURCE = AnswerVerdict(
    status="need_source", reason="price missing", source_query="kayak price"
).model_dump_json()


@pytest.mark.parametrize(
    ("question", "first", "expected_calls"),
    [
        ("What kind of boat do I have?", ANSWER, 1),
        ("How many kayaks have I bought?", NOTED, 1),
        ("When did I last go kayaking?", NOTED, 1),
        ("What did I do with my kayak last week?", ANSWER, 1),
        ("Who did I go with to the music event last Saturday?", ANSWER, 1),
        ("What did my kayak cost?", NEED_SOURCE, 2),
    ],
)
def test_served_t4_sends_the_evaluated_requests(store, tmp_path, question, first, expected_calls):
    index = _index(tmp_path)
    evaluated, served = Client(first), Client(first)
    request = AnswerRequest(question=question, asked_on="2025-06-14", user_id="alice")

    a = _evaluated_t4(evaluated, store, index).answer_request(request)
    b = _served_t4(served, store, index).answer_request(request)

    assert len(evaluated.calls) == len(served.calls) == expected_calls
    assert evaluated.calls == served.calls
    assert a.text == b.text
    prompt = served.calls[0]["prompt"]
    assert "$640" in prompt and "9,999" not in prompt
    assert "Verbatim source evidence" in prompt  # hydration is part of t4
    if "last Saturday" in question:
        assert "Billie Eilish" in prompt  # reachable only through the time window


def test_service_defaults_to_the_t4_profile():
    assert ServiceConfig().answer_profile == "t4"
    assert ServiceConfig().top_k == 20
    assert set(ANSWER_PROFILES) == {"v2", "t4"}


def test_unknown_profile_is_refused(store, tmp_path):
    with pytest.raises(ValueError, match="unknown answer profile"):
        AnswerEngine.from_profile(
            "t9", Client(ANSWER), model="m", encoder=Encoder(), store=store, index=_index(tmp_path)
        )
