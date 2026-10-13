import copy
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
from llm_long_term_memory.llm.client import Completion
from llm_long_term_memory.store import NumpyFlatIndex, Session, SQLiteMemoryStore, Turn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from t4_savings_repair import SavingsRepairRunner
from t4_scoped_dispatch import ScopedSavingsRepairRunner, applicable_repair


class Client:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        schema = kwargs.get("schema")
        self.calls.append({**kwargs, "schema": schema.model_json_schema() if schema else None})
        return Completion(
            '{"notes":"same","status":"answer","answer":"same"}', kwargs["model"], 0, 0, 0, 1, 0
        )


@pytest.fixture
def base(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()
    at = datetime(2026, 9, 1)
    for sid, user, text in (
        (
            "mine",
            "alice",
            "My kayak cost $640. I practise yoga three times a week. I collected 19 signed coins.",
        ),
        ("foreign", "bob", "I redeemed a coupon at Secret Place. I collected 200 signed posters."),
    ):
        store.add_session(
            Session(sid, user, at, "test", [Turn(sid + ":0", sid, 0, "user", text, at)])
        )

    class Encoder:
        def encode_one(self, text):
            return np.ones(2, dtype=np.float32)

    runner = MemoryRunner(
        Client(),
        "same-model",
        Encoder(),
        store,
        NumpyFlatIndex(tmp_path / "index", dim=2),
        answer_policy="v2_clt",
        raw_primary_tokens=4000,
        raw_primary_prefer_user=True,
        raw_primary_time_notes=True,
        raw_primary_time_window=True,
        raw_fallback=True,
    )
    yield runner
    store.close()


@pytest.mark.parametrize(
    "question",
    [
        "How much did my kayak cost?",
        "How often do I practise yoga?",
        "How many different doctors did I visit?",
    ],
)
def test_unmatched_repair_preserves_entire_baseline_request_and_result(base, question):
    request = AnswerRequest(question, "2026-10-01", "alice")
    assert applicable_repair(base, request) is None
    original = base.answer_request(request)
    calls = copy.deepcopy(base.client.calls)
    base.client.calls.clear()
    candidate = ScopedSavingsRepairRunner.from_t4(base)
    replay = candidate.answer_request(request)
    assert base.client.calls == calls
    assert replay.text == original.text and replay.context_tokens == original.context_tokens
    assert replay.notes["selection_repair"]["active"] is False
    assert all("Selection checks:" not in c["prompt"] for c in calls)


def test_active_collection_repair_preserves_previous_candidate_requests_and_override(base):
    request = AnswerRequest("How many signed posters are in my collection?", "2026-10-01", "alice")
    previous = SavingsRepairRunner.from_t4(base).answer_request(request)
    calls = copy.deepcopy(base.client.calls)
    base.client.calls.clear()
    current = ScopedSavingsRepairRunner.from_t4(base).answer_request(request)
    assert base.client.calls == calls and current.text == previous.text
    assert "don't have evidence" in current.text
    assert (
        current.notes["selection_repair"]["calculation"]["mechanism"]
        == "requested_collection_category_absent"
    )


def test_foreign_evidence_never_activates_repair_or_enters_baseline_context(base):
    request = AnswerRequest("Where did I redeem the coupon?", "2026-10-01", "alice")
    assert applicable_repair(base, request) is None
    ScopedSavingsRepairRunner.from_t4(base).answer_request(request)
    assert all(
        "Secret Place" not in c["prompt"] and "200 signed posters" not in c["prompt"]
        for c in base.client.calls
    )
