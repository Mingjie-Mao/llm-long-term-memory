"""The golden case: why memory-first with a raw-source fallback is necessary.

The Mayo Clinic question from LongMemEval — *"remind me of the video you
recommended"* — is one of four `single-session-assistant` questions that every
variant answered wrong, `full_context` and `naive_rag` excepted. Tracing it
(results/assistant-gap.md) showed why:

* the answer lives in an **assistant** turn, and extraction wrote user-profile facts
* so no structured memory holds the URL, and no amount of ranking can find it
* but the raw turn is in the store, intact

That makes it the one case that exercises the whole architecture end to end:
compression drops a detail, retrieval cannot recover it, and the archive can. If
this test ever passes trivially — because structured memory happens to hold the URL
— the fallback has stopped being the thing under test, and the first assertion says
so out loud.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from llm_long_term_memory.retrieve.fallback import RawFallback
from llm_long_term_memory.store import Memory, NumpyFlatIndex, Session, SQLiteMemoryStore, Turn

NOW = datetime(2026, 8, 14)
GOLD_URL = "https://www.youtube.com/watch?v=UfOvNlX9Hh0"

# Abridged from the real session `answer_sharegpt_81riySf_0`: the user asks, and the
# assistant supplies the substance. The user turn asserts no fact about the user,
# which is exactly why user-profile extraction produced nothing from this session.
USER_TURN = "any youtube video i can share with them about sitting posture?"
ASSISTANT_TURN = (
    "Certainly! There are plenty of helpful videos on YouTube. A good one is from the "
    "Mayo Clinic: 'How to Sit Properly at a Desk to Avoid Back Pain', "
    f"{GOLD_URL}. Another option is a short stretching routine, "
    "https://www.youtube.com/watch?v=LT_dFRnmdGs"
)


@pytest.fixture
def store(tmp_path):
    s = SQLiteMemoryStore(tmp_path / "golden.db")
    s.initialize()
    s.add_session(
        Session(
            id="s_posture",
            user_id="u",
            started_at=NOW,
            turns=[
                Turn(
                    id="t0",
                    session_id="s_posture",
                    turn_index=0,
                    role="user",
                    content=USER_TURN,
                    ts=NOW,
                ),
                Turn(
                    id="t1",
                    session_id="s_posture",
                    turn_index=1,
                    role="assistant",
                    content=ASSISTANT_TURN,
                    ts=NOW,
                ),
            ],
        )
    )
    # What extraction actually produced for this namespace: plausible, on-topic-ish,
    # and useless for the question. Reproduced rather than invented — the real store
    # holds "The user plans to start recording gameplay footage for their YouTube
    # channel" and nothing about Mayo Clinic.
    s.add_memories(
        [
            Memory(
                id="m1",
                user_id="u",
                type="semantic",
                token_count=9,
                content=(
                    "The user plans to start recording gameplay footage for their YouTube channel."
                ),
                ingested_at=NOW,
                valid_from=NOW,
                subject="user",
                predicate="plans",
            ),
            Memory(
                id="m2",
                user_id="u",
                type="semantic",
                token_count=9,
                content="The user decided to try OBS Studio for screen recording.",
                ingested_at=NOW,
                valid_from=NOW,
                subject="user",
                predicate="tools",
            ),
        ]
    )
    yield s
    s.close()


def test_structured_memory_cannot_answer_this(store):
    """The premise. If this ever fails, extraction improved and the case must be
    re-chosen — the test below would then be passing for the wrong reason."""
    memories = store.iter_active("u")

    assert memories, "the namespace is not empty; the failure is loss, not absence"
    assert not any(GOLD_URL in m.content for m in memories)
    assert not any("mayo" in m.content.lower() for m in memories)


def test_the_raw_archive_still_holds_the_answer(store):
    """Compression is lossy; the source is not. This is the asymmetry the whole
    fallback design rests on."""
    turns = store.turns_for_session("s_posture")

    assert any(GOLD_URL in t.content for t in turns)
    assert next(t for t in turns if GOLD_URL in t.content).role == "assistant"


def test_archive_wide_fallback_recovers_the_exact_url(store):
    """End to end: structured memory found nothing usable, so the archive is
    searched, and the answer comes back character for character."""
    evidence = RawFallback(store).recover("u", "Mayo Clinic posture video link", memories=[])

    assert evidence.level == "archive_wide"
    assert evidence.used
    # Substring, not token equality: real prose puts punctuation against a URL
    # ("...UfOvNlX9Hh0."), and tokenizing it back out is the caller's problem, not a
    # property of what the archive preserved.
    recovered = "\n".join(t.content for t in evidence.turns)
    assert GOLD_URL in recovered, "the gold URL verbatim, not a paraphrase of it"
    # And it is attributed, so an answer built on it can cite a source.
    assert evidence.turns[0].session_id == "s_posture"
    assert evidence.reason


def test_the_recovered_evidence_is_rendered_with_its_provenance(store):
    """An answer that cites nothing is indistinguishable from a hallucination, so
    the prompt the second pass receives has to carry the attribution."""
    evidence = RawFallback(store).recover("u", "Mayo Clinic video", memories=[])
    rendered = evidence.render()

    assert GOLD_URL in rendered
    assert "s_posture" in rendered
    assert "assistant" in rendered


def test_source_local_fallback_is_preferred_when_a_memory_points_at_the_turn(store):
    """Level 1 beats level 2 when it applies: a memory that names the right session
    is stronger evidence than a keyword match over everything."""
    store.add_memories(
        [
            Memory(
                id="m3",
                user_id="u",
                type="semantic",
                token_count=9,
                content="The assistant recommended a Mayo Clinic video about posture.",
                ingested_at=NOW,
                valid_from=NOW,
                subject="assistant",
                predicate="recommendation",
                source_role="assistant",
                scope="recommendation",
                source_session_id="s_posture",
                source_turn_index=1,
            ),
        ]
    )
    anchored = [m for m in store.iter_active("u") if m.id == "m3"]

    evidence = RawFallback(store).recover("u", "Mayo Clinic video", memories=anchored)

    assert evidence.level == "source_local"
    assert [t.turn_index for t in evidence.turns] == [1]
    assert GOLD_URL in evidence.turns[0].content


def test_a_question_the_archive_cannot_answer_stays_unanswered(store):
    """The guard rail. Fallback must not become a licence to answer from whatever
    the search returned — abstention is a measured strength worth protecting."""
    evidence = RawFallback(store).recover("u", "passport renewal appointment", memories=[])

    assert not evidence.used
    assert evidence.level == "none"


def test_the_archive_is_namespace_scoped(store):
    """`turns` has no user_id of its own; the join onto `sessions` is what stops one
    user's conversation answering another's question."""
    assert RawFallback(store).recover("someone_else", "Mayo Clinic", memories=[]).turns == []


def test_the_inspector_can_show_this_case_without_an_answerer(tmp_path, store):
    """The demo path: `/v1/raw/search` exposes the layer directly, so the case is
    presentable without spending answerer quota."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from llm_long_term_memory.api.app import app, set_service
    from llm_long_term_memory.api.service import MemoryService

    class StubEncoder:
        dim = 2

        def encode_one(self, text):
            return np.array([1.0, 0.0], dtype=np.float32)

        def encode(self, texts, **kw):
            return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)

    service = MemoryService.__new__(MemoryService)
    service.store = store
    service.store_name = "golden"
    service._encoder = StubEncoder()
    service.index = NumpyFlatIndex(tmp_path / "idx", dim=2)
    from llm_long_term_memory.config import ExperimentConfig

    service.config = ExperimentConfig()
    service.extractor = None
    service._answerer = None
    set_service(service)
    try:
        with TestClient(app) as client:
            body = client.post(
                "/v1/raw/search",
                json={"user_id": "u", "query": "Mayo Clinic posture video", "limit": 2},
            ).json()
    finally:
        set_service(None)

    assert any(GOLD_URL in t["content"] for t in body["turns"])
    assert any(t["role"] == "assistant" for t in body["turns"])
