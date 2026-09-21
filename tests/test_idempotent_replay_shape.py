"""A retried write must answer with the same body, not a 500.

`Memory` is a slots dataclass, so it has no `__dict__`. The replay serializer read one
through `vars()`, fell through to `str()`, and stored the memories as reprs; the response
model then tried to read `.id` off a string. The first write returned 200 and every retry
of it returned 500 — the one request an idempotency key exists to make safe. Found by the
simulated local trial, so the regression is pinned at the surface a client actually calls.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from llm_long_term_memory.api.app import app, set_service
from llm_long_term_memory.api.service import MemoryService, TurnExtractionOutcome
from llm_long_term_memory.store import Memory


class _Encoder:
    dim = 2

    def encode(self, texts, show_progress=False):
        return np.ones((len(texts), 2), dtype=np.float32)

    def encode_one(self, text):
        return np.ones(2, dtype=np.float32)


class _Extractor:
    def extract_turn(self, *, user_id, session_id, role, content, now, context_turns=()):
        return TurnExtractionOutcome(
            [
                Memory(
                    id="mem_1",
                    user_id=user_id,
                    type="episodic",
                    content=content,
                    token_count=4,
                    subject="user",
                    predicate="adopted",
                    object="a cat",
                    scope="event",
                    source_session_id=session_id,
                    source_turn_index=0,
                    valid_from=datetime(2026, 9, 14),
                )
            ],
            {"requests": 1},
        )


class _Settings:
    def __init__(self, tmp_path):
        self.store_dir = tmp_path
        self.data_dir = tmp_path
        self.results_dir = tmp_path
        self.gemini_api_key = ""

    @property
    def has_api_key(self):
        return False


@pytest.fixture
def client(tmp_path):
    service = MemoryService(
        store_name="replay",
        settings=_Settings(tmp_path),
        encoder=_Encoder(),
        extractor=_Extractor(),
    )
    set_service(service)
    yield TestClient(app, raise_server_exceptions=False)
    service.store.close()


BODY = {"user_id": "u1", "role": "user", "content": "I adopted a cat called Pepper."}
KEY = {"Idempotency-Key": "key-1"}


def test_a_retried_write_returns_the_first_response_again(client):
    first = client.post("/v1/messages", json=BODY, headers=KEY)
    retry = client.post("/v1/messages", json=BODY, headers=KEY)
    assert (first.status_code, retry.status_code) == (200, 200)
    assert retry.json() == first.json()


def test_the_replayed_memories_keep_their_fields_and_provenance(client):
    client.post("/v1/messages", json=BODY, headers=KEY)
    replayed = client.post("/v1/messages", json=BODY, headers=KEY).json()["memories"][0]
    assert replayed["id"] == "mem_1"
    assert replayed["predicate"] == "adopted"
    assert replayed["valid_from"].startswith("2026-09-14")
    # Provenance is the product's central claim; a replay that drops it is not a replay.
    assert replayed["source"]["turn_index"] == 0
    assert replayed["source"]["session_id"]
