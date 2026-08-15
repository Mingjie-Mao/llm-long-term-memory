"""Memory first, source when needed — and silence when neither has it.

Four cases, one per branch the product has to get right:

1. memory suffices          -> answer, no fallback, one LLM call
2. memory-aware advice      -> answer from preferences, no fallback
3. detail dropped by extraction -> source-local fallback recovers the exact string
4. never discussed          -> archive search finds nothing, and it declines

The fourth is the one that keeps the other three honest. A fallback that answers
from whatever the keyword search dragged up would trade away 100% abstention for
confident guesses, which is the failure mode `full_context` already demonstrates at
50%.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest

from llm_long_term_memory.evaluation.datasets.longmemeval import Instance
from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
from llm_long_term_memory.retrieve.fallback import RawFallback
from llm_long_term_memory.store import Memory, NumpyFlatIndex, Session, SQLiteMemoryStore, Turn

NOW = datetime(2026, 8, 14)

MAYO_URL = "https://www.youtube.com/watch?v=UfOvNlX9Hh0"


class StubEncoder:
    dim = 2

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def encode(self, texts, **kw) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0] if "posture" in t.lower() or "video" in t.lower() else [0.0, 1.0]
                for t in texts
            ],
            dtype=np.float32,
        )


class ScriptedClient:
    """Returns queued payloads and records how many calls were made."""

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls: list[str] = []

    def generate(self, *, role, model, prompt, **kw):
        self.calls.append(prompt)
        payload = self.payloads.pop(0)

        class C:
            text = payload if isinstance(payload, str) else json.dumps(payload)
            input_tokens = 50
            output_tokens = 10
            thinking_tokens = 0
            api_latency_ms = 1.0

        return C()


def instance(question: str, qid: str = "u1") -> Instance:
    return Instance(
        question_id=qid,
        question_type="single-session-assistant",
        question=question,
        answer="gold",
        question_date="2026-08-14",
        sessions=[],
        answer_session_ids=[],
    )


@pytest.fixture
def wired(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()
    store.add_session(
        Session(
            id="s1",
            user_id="u1",
            started_at=NOW,
            turns=[
                Turn(
                    id="s1:0",
                    session_id="s1",
                    turn_index=0,
                    role="user",
                    content="any youtube video i can share about workplace posture?",
                    ts=NOW,
                ),
                Turn(
                    id="s1:1",
                    session_id="s1",
                    turn_index=1,
                    role="assistant",
                    content=(
                        "Certainly! The Mayo Clinic has one: 'How to Sit Properly at a "
                        f"Desk to Avoid Back Pain', {MAYO_URL}"
                    ),
                    ts=NOW,
                ),
            ],
        )
    )
    store.add_memories(
        [
            Memory(
                id="m_rec",
                user_id="u1",
                type="semantic",
                content="The assistant recommended a Mayo Clinic video about workplace posture.",
                token_count=10,
                ingested_at=NOW,
                valid_from=NOW,
                subject="assistant",
                predicate="recommendation",
                source_role="assistant",
                scope="recommendation",
                source_session_id="s1",
                source_turn_index=1,
            ),
            Memory(
                id="m_pref",
                user_id="u1",
                type="preference",
                content="The user prefers short instructional videos.",
                token_count=8,
                ingested_at=NOW,
                valid_from=NOW,
                subject="user",
                predicate="video_preference",
                scope="preference",
            ),
        ]
    )
    index = NumpyFlatIndex(tmp_path / "idx", dim=2)
    index.add(["m_rec", "m_pref"], np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32))
    yield store, index
    store.close()


def runner(store, index, client, **kw) -> MemoryRunner:
    return MemoryRunner(
        client,
        model="m",
        encoder=StubEncoder(),
        store=store,
        index=index,
        temporal=True,
        top_k=5,
        raw_fallback=True,
        **kw,
    )


def test_memory_alone_answers_without_touching_the_archive(wired):
    """The common case must stay a single LLM call: recovering raw text every time
    is what `two_stage_hydrated` did, at 3x the context for no measured gain."""
    store, index = wired
    client = ScriptedClient(
        {"status": "answer", "answer": "The assistant recommended a Mayo Clinic video."}
    )

    answer = runner(store, index, client).answer(instance("did you recommend a video?"))

    assert len(client.calls) == 1, "no second pass when memory sufficed"
    assert answer.notes["fallback_level"] == "none"
    assert "Mayo Clinic" in answer.text


def test_source_local_fallback_recovers_a_detail_extraction_dropped(wired):
    """The defining case. The memory is true, on topic, and cannot answer the
    question, because the URL never survived extraction."""
    store, index = wired
    client = ScriptedClient(
        {
            "status": "need_source",
            "reason": "The memory names a Mayo Clinic video but not its URL.",
            "source_query": "Mayo Clinic posture video URL",
        },
        f"The video is 'How to Sit Properly at a Desk to Avoid Back Pain': {MAYO_URL}",
    )

    answer = runner(store, index, client).answer(instance("what was the video link?"))

    assert len(client.calls) == 2, "exactly one extra call, only because it was needed"
    assert answer.notes["fallback_level"] == "source_local"
    assert answer.notes["fallback_turns"] == ["s1:1"]
    assert MAYO_URL in answer.text, "the exact string came back from the raw turn"
    # The second prompt must actually contain the verbatim turn.
    assert MAYO_URL in client.calls[1]


def test_a_question_never_discussed_is_declined_not_invented(wired):
    """Fallback must not become a licence to answer from whatever BM25 returns.
    Abstention is a measured strength; this is the test that protects it."""
    store, index = wired
    client = ScriptedClient(
        {
            "status": "no_evidence",
            "reason": "Nothing about a passport renewal was discussed.",
            "source_query": "passport renewal appointment",
        }
    )

    answer = runner(store, index, client).answer(instance("when is my passport appointment?"))

    assert len(client.calls) == 1, "no second call when the archive has nothing"
    assert answer.notes["fallback_level"] == "none"
    assert "do not know" in answer.text.lower()


def test_the_archive_is_searched_when_memory_found_nothing(wired):
    """Extraction can miss a fact entirely, so an empty memory result is not proof
    the conversation never contained it."""
    store, _ = wired
    fallback = RawFallback(store)

    evidence = fallback.recover("u1", "Mayo Clinic posture", memories=[])

    assert evidence.level == "archive_wide"
    assert any(MAYO_URL in t.content for t in evidence.turns)


def test_archive_search_never_crosses_a_namespace(wired):
    store, _ = wired
    assert RawFallback(store).recover("someone_else", "Mayo Clinic", memories=[]).turns == []


def test_a_store_indexed_before_turns_fts_existed_is_backfilled(tmp_path):
    """The index is created empty and its triggers only fire on later inserts, so a
    store ingested before it existed would search nothing and report 'no evidence' —
    indistinguishable from a genuine miss."""
    path = tmp_path / "old.db"
    store = SQLiteMemoryStore(path)
    store.initialize()
    store.add_session(
        Session(
            id="s9",
            user_id="u1",
            started_at=NOW,
            turns=[
                Turn(
                    id="s9:0",
                    session_id="s9",
                    turn_index=0,
                    role="assistant",
                    content="Use Mod Podge to seal the newspaper vase.",
                    ts=NOW,
                )
            ],
        )
    )
    # Simulate the pre-index state: empty the index and clear the built marker.
    store._conn.execute("INSERT INTO turns_fts(turns_fts) VALUES ('delete-all')")
    store._conn.execute("DELETE FROM meta WHERE key = 'turns_fts_built'")
    store._conn.commit()
    assert store.search_turns("u1", "Mod Podge") == []
    store.close()

    reopened = SQLiteMemoryStore(path)
    reopened.initialize()  # migration rebuilds
    assert [t.id for t in reopened.search_turns("u1", "Mod Podge")] == ["s9:0"]
    reopened.close()
