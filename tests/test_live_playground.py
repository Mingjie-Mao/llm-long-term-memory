"""Public LLM routes reuse real product logic; only the model transport is scripted."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from llm_long_term_memory.api.service import MemoryService
from llm_long_term_memory.config import Settings
from llm_long_term_memory.llm import CallRecord, Limits, QuotaManager, UsageTracker

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo-api"
sys.path.insert(0, str(DEMO))
from demo_live import CapacityExceeded, DemoClient  # noqa: E402


class Encoder:
    dim = 2

    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32).reshape(-1, 2)

    def encode_one(self, text):
        return self.encode([text])[0]


class Provider:
    def __init__(self):
        self.quota = QuotaManager(default=Limits(rpm=10_000, tpm=10**9, rpd=10_000))
        self.usage = UsageTracker()
        self.replies = []
        self.requests = []

    def generate(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        self.usage.record(
            CallRecord(
                kwargs["role"], kwargs["model"], 10, 5, 1, ok=not isinstance(reply, Exception)
            )
        )
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            text=json.dumps(reply) if isinstance(reply, dict) else reply,
            input_tokens=10,
            output_tokens=5,
            api_latency_ms=1,
            thinking_tokens=0,
        )


@pytest.fixture
def live(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("lltm_demo_test_app", DEMO / "app.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    service = MemoryService(
        config_path=ROOT / "configs/fallback.yaml",
        store_name="playground",
        settings=Settings(store_dir=tmp_path, gemini_api_key="test-only-no-network"),
        encoder=Encoder(),
    )
    provider = Provider()
    client = DemoClient(provider, service.store)
    service._live_client_instance = client
    service._live_usage = provider.usage
    service.store._conn.execute(
        "CREATE TABLE demo_sessions(namespace TEXT PRIMARY KEY, created_at TEXT NOT NULL)"
    )
    playground = module.Playground(
        service.store,
        service.index,
        service.encoder,
        service.retriever,
        service.resolver,
        {},
        service=service,
        live_client=client,
    )
    service._lock = playground.lock
    monkeypatch.setattr(module, "pg", playground)
    http = TestClient(module.app)
    header = {"x-demo-token": http.post("/demo/session").json()["token"]}
    yield module, playground, provider, http, header
    http.close()
    service.close()


def fact_replies(value):
    return [
        {
            "sessions": [
                {
                    "session_index": 0,
                    "facts": [
                        {
                            "content": f"The user lives in {value}.",
                            "source_role": "user",
                            "subject": "user",
                            "scope": "profile",
                        }
                    ],
                }
            ]
        },
        {
            "facts": [
                {"index": 0, "temporal_key": "lives_in", "object": value, "update_op": "replaces"}
            ]
        },
    ]


def write(http, header, content, session="conversation-1", request_id="r1"):
    return http.post(
        "/demo/messages",
        headers=header,
        json={"content": content, "session_id": session, "request_id": request_id},
    )


def test_natural_language_reaches_two_stages_and_t4_across_conversations(live):
    _, p, provider, http, header = live
    provider.replies.extend(fact_replies("Canberra"))
    response = write(http, header, "After years in Sydney, I have settled in Canberra.")
    assert response.status_code == 200, response.text
    assert len(provider.requests) == 2
    assert "settled in Canberra" in provider.requests[0]["prompt"]
    assert provider.requests[0]["schema"].__name__ == "FactsResult"
    assert provider.requests[1]["schema"].__name__ == "KeyingResult"
    memory = response.json()["memories"][0]
    assert memory["session_id"] == "conversation-1"
    provider.replies.append(
        {"status": "answer", "answer": "You live in Canberra.", "notes": "source"}
    )
    answer = http.post("/demo/answer", headers=header, json={"query": "Where do I live?"})
    assert answer.status_code == 200, answer.text
    payload = answer.json()
    assert payload["answer_profile"] == "t4" and payload["mode"] == "llm"
    assert "Canberra" in payload["answer"]
    assert "settled in Canberra" in payload["model_inputs"][0]["prompt"]
    assert payload["model_inputs"][0]["prompt"] == provider.requests[-1]["prompt"]
    assert payload["selected_sources"][0]["source"]["session_id"] == "conversation-1"
    assert p.live_client.inputs == [] and p.live_client.owner is None
    assert p.live_client.usage.records == []


def test_other_visitors_and_forged_namespace_cannot_reach_the_context(live):
    _, _, provider, http, header = live
    provider.replies.extend(fact_replies("SecretCity"))
    assert write(http, header, "I live in SecretCity.").status_code == 200
    other = {"x-demo-token": http.post("/demo/session").json()["token"]}
    provider.replies.append({"status": "no_evidence", "answer": "I do not know."})
    answer = http.post("/demo/answer", headers=other, json={"query": "Where do I live?"})
    assert answer.status_code == 200
    assert "SecretCity" not in json.dumps(answer.json())
    assert "SecretCity" not in provider.requests[-1]["prompt"]
    before = len(provider.requests)
    response = http.post(
        "/demo/answer",
        headers=other,
        json={"query": "Where?", "user_id": header["x-demo-token"].split(".")[0]},
    )
    assert response.status_code == 422
    assert len(provider.requests) == before
    assert http.post("/demo/answer", json={"query": "Where?"}).status_code == 401


def test_write_retry_replays_without_spending_more_capacity(live):
    _, p, provider, http, header = live
    provider.replies.extend(fact_replies("Canberra"))
    first = write(http, header, "I live in Canberra.")
    ns = header["x-demo-token"].split(".")[0]
    reserved = p.live_client.reserved(ns)
    second = write(http, header, "I live in Canberra.")
    assert first.status_code == second.status_code == 200
    assert len(provider.requests) == 2
    assert p.live_client.reserved(ns) == reserved
    assert len(list(p.store.iter_all(ns))) == 1


def test_provider_failure_is_sanitised_and_keeps_capacity_charge(live):
    _, p, provider, http, header = live
    provider.replies.append(RuntimeError("private-provider-body api-key-do-not-reflect"))
    response = write(http, header, "I live in Canberra.")
    assert response.status_code == 503
    assert "api-key" not in response.text and "private-provider" not in response.text
    detail = response.json()["detail"]
    assert detail["kind"] == "RuntimeError" and detail["provider_code"] is None
    ns = header["x-demo-token"].split(".")[0]
    assert p.live_client.reserved(ns)["calls"] == 2
    assert p.live_client.inputs == [] and p.live_client.usage.records == []
    assert list(p.store.iter_all(ns)) == []


def test_exhausted_capacity_refuses_before_transport_and_survives_delete(live):
    _, p, provider, http, header = live
    p.live_client.session_calls = 1
    response = write(http, header, "I live in Canberra.")
    assert response.status_code == 429
    assert provider.requests == []
    p.live_client.session_calls = 32
    provider.replies.extend(fact_replies("Canberra"))
    assert write(http, header, "I live in Canberra.", request_id="new").status_code == 200
    ns = header["x-demo-token"].split(".")[0]
    reserved = p.live_client.reserved(ns)
    p.live_client.global_calls = reserved["calls"]
    other = {"x-demo-token": http.post("/demo/session").json()["token"]}
    assert write(http, other, "I live in Berlin.").status_code == 429
    before = len(provider.requests)
    assert http.delete("/demo/session", headers=header).status_code == 200
    assert p.live_client.reserved(ns) == reserved
    assert len(provider.requests) == before
    assert not list(p.store.iter_all(ns)) and not p.index.ids


def test_expiry_deletes_real_live_memories_vectors_and_turns(live):
    module, p, provider, http, header = live
    provider.replies.extend(fact_replies("Canberra"))
    assert write(http, header, "I live in Canberra.").status_code == 200
    ns = header["x-demo-token"].split(".")[0]
    p.created[ns] = datetime.now() - timedelta(hours=2)
    assert module.sweep(p) == 1
    assert http.get("/demo/memories", headers=header).status_code == 410
    assert not p.index.ids and not list(p.store.iter_all(ns))
    assert p.store.search_turns(ns, "Canberra") == []


def test_disabled_llm_never_falls_back_to_rule_generation(live):
    _, p, provider, http, header = live
    p.service = None
    assert http.get("/demo/health").json()["llm_enabled"] is False
    assert write(http, header, "I live in Canberra.").status_code == 503
    assert http.post("/demo/answer", headers=header, json={"query": "Where?"}).status_code == 503
    assert provider.requests == []


def test_capacity_ledger_survives_adapter_restart_and_operation_deadline(live):
    _, p, provider, _, header = live
    ns = header["x-demo-token"].split(".")[0]
    with p.live_client.operation(ns):
        p.live_client._reserve(100)
    replacement = DemoClient(provider, p.store)
    assert replacement.reserved(ns) == {"calls": 2, "tokens": 100}
    with replacement.operation(ns):
        replacement.deadline = 0
        with pytest.raises(CapacityExceeded, match="timed out"):
            replacement.generate(prompt="hello", role="answerer", model="test")
    assert provider.requests == []


def test_update_uses_real_dedup_and_resolver_without_retiring_another_user(live):
    _, p, provider, http, header = live
    provider.replies.extend(fact_replies("Canberra"))
    assert write(http, header, "I live in Canberra.").status_code == 200
    provider.replies.extend(fact_replies("Melbourne"))
    provider.replies.append({"verdict": "UPDATE", "reason": "The user moved."})
    changed = write(
        http,
        header,
        "I moved from Canberra to Melbourne.",
        session="conversation-2",
        request_id="r2",
    )
    assert changed.status_code == 200, changed.text
    states = {m["object"]: m["status"] for m in changed.json()["memories"]}
    assert states == {"Canberra": "superseded", "Melbourne": "active"}
    chain = http.get("/demo/timeline", headers=header, params={"predicate": "lives_in"}).json()[
        "entries"
    ]
    assert any(row["replaced_by_next"] for row in chain)
    other = {"x-demo-token": http.post("/demo/session").json()["token"]}
    provider.replies.extend(fact_replies("Canberra"))
    before = len(provider.requests)
    assert write(http, other, "I live in Canberra.").status_code == 200
    assert len(provider.requests) - before == 2, "another user must not enter dedup"
    assert p.live_client.inputs == []


def test_fallback_inputs_are_the_actual_second_model_request(live):
    _, _, provider, http, header = live
    provider.replies.extend(fact_replies("Canberra"))
    assert (
        write(http, header, "I live in Canberra; my booking reference is VX-928173.").status_code
        == 200
    )
    provider.replies.extend(
        [
            {
                "status": "need_source",
                "source_query": "booking reference",
                "reason": "Need exact code",
            },
            "Your booking reference is VX-928173.",
        ]
    )
    response = http.post(
        "/demo/answer", headers=header, json={"query": "What is my booking reference?"}
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "VX-928173" in data["answer"]
    assert len(data["model_inputs"]) == 2
    assert data["model_inputs"][1]["prompt"] == provider.requests[-1]["prompt"]
    assert "VX-928173" in data["model_inputs"][1]["prompt"]
    assert data["fallback_level"] != "none"
    assert data["recovered_turns"] == [
        {
            "session_id": "conversation-1",
            "turn_index": 0,
            "text": "I live in Canberra; my booking reference is VX-928173.",
        }
    ]


def test_onnx_product_attach_configures_finite_sdk_transport_without_a_model_call(
    tmp_path, monkeypatch
):
    from demo_live import attach

    from llm_long_term_memory.store import NumpyFlatIndex, SQLiteMemoryStore

    monkeypatch.setenv("GEMINI_API_KEY", "test-only-not-a-real-key")
    store = SQLiteMemoryStore(tmp_path / "demo.db")
    store.initialize()
    playground = SimpleNamespace(
        store=store, encoder=Encoder(), lock=__import__("threading").RLock()
    )
    service, client = attach(playground, tmp_path / "demo")
    try:
        assert service.encoder is playground.encoder
        assert service.index is playground.index
        assert service.config.ingest.two_stage
        assert service.config.service.answer_profile == "t4"
        options = client.base._client._api_client._http_options
        assert options.timeout == 30000 and options.retry_options.attempts == 1
        assert client.base.max_retries == 2 and client.base.max_transport_retries == 1
        assert client.usage.records == []
        assert isinstance(service.index, NumpyFlatIndex)
    finally:
        client.base._client.close()
        service.close()


@pytest.mark.parametrize("recovers", [True, False])
def test_real_client_503_retry_is_bounded_accounted_and_preserves_write(
    live, monkeypatch, recovers
):
    from google.genai import errors

    from llm_long_term_memory.llm.client import GeminiClient

    _, p, provider, http, header = live
    base = GeminiClient(
        "test-only-no-network",
        provider.quota,
        provider.usage,
        max_retries=2,
        max_transport_retries=1,
    )
    p.live_client.base = base
    unavailable = errors.ServerError(
        503, {"error": {"code": 503, "message": "Service unavailable", "status": "UNAVAILABLE"}}
    )
    replies = [unavailable]
    replies.extend(fact_replies("Canberra") if recovers else [unavailable, "must not be called"])
    calls = []
    delays = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            text=json.dumps(reply),
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=5,
                thoughts_token_count=0,
            ),
        )

    monkeypatch.setattr(base._client.models, "generate_content", generate_content)
    monkeypatch.setattr("llm_long_term_memory.llm.client.time.sleep", delays.append)
    try:
        response = write(http, header, "I live in Canberra.")
        ns = header["x-demo-token"].split(".")[0]
        assert response.status_code == (200 if recovers else 503)
        assert len(calls) == (3 if recovers else 2)
        assert p.live_client.reserved(ns)["calls"] == (4 if recovers else 2)
        assert len(delays) == 1 and 2 <= delays[0] <= 3
        assert len(p.store.iter_all(ns)) == (1 if recovers else 0)
        assert p.store.iter_all("another-visitor") == []
        usage = p.store._conn.execute(
            "SELECT SUM(calls), SUM(failed_calls) FROM account_usage WHERE user_id=?", (ns,)
        ).fetchone()
        assert tuple(usage) == ((3, 1) if recovers else (2, 2))
        if recovers:
            replay = write(http, header, "I live in Canberra.")
            assert replay.status_code == 200 and len(calls) == 3
    finally:
        base._client.close()


def test_token_capacity_and_conflicting_retries_stop_before_new_calls(live):
    _, p, provider, http, header = live
    p.live_client.session_tokens = 1
    assert write(http, header, "I live in Canberra.").status_code == 429
    assert provider.requests == []
    p.live_client.session_tokens = 300_000
    provider.replies.extend(fact_replies("Canberra"))
    assert write(http, header, "I live in Canberra.", request_id="ok").status_code == 200
    before = len(provider.requests)
    response = write(http, header, "I live in Berlin.", request_id="ok")
    assert response.status_code == 409
    assert len(provider.requests) == before


def test_live_messages_reject_another_namespace_conversation_before_calls(live):
    from llm_long_term_memory.store import scoped_session_id

    _, _, provider, http, header = live
    response = write(
        http,
        header,
        "I live in Canberra.",
        session=scoped_session_id("another-user", "conversation-1"),
    )
    assert response.status_code == 422
    assert provider.requests == []


def test_precise_observation_does_not_override_stated_event_date(live):
    _, p, provider, http, header = live
    replies = fact_replies("Canberra")
    replies[0]["sessions"][0]["facts"][0]["content"] = "The user moved to Canberra on 2020-01-10."
    provider.replies.extend(replies)
    assert write(http, header, "I moved to Canberra on 2020-01-10.").status_code == 200
    ns = header["x-demo-token"].split(".")[0]
    memory = p.store.iter_all(ns)[0]
    assert memory.event_time.date().isoformat() == "2020-01-10"
    assert memory.observed_at.year >= 2026
    assert memory.event_time != memory.observed_at


def framework_replies(value, op="coexists", target=""):
    return [
        {
            "sessions": [
                {
                    "session_index": 0,
                    "facts": [
                        {
                            "content": f"The user uses {value} for personal projects.",
                            "source_role": "user",
                            "subject": "user",
                            "scope": "profile",
                        }
                    ],
                }
            ]
        },
        {
            "facts": [
                {
                    "index": 0,
                    "temporal_key": "uses_framework",
                    "object": value,
                    "update_op": op,
                    "target_object": target,
                }
            ]
        },
    ]


def dated_write(http, header, content, session, request_id, observed_on):
    return http.post(
        "/demo/messages",
        headers=header,
        json={
            "content": content,
            "session_id": session,
            "request_id": request_id,
            "observed_on": observed_on,
        },
    )


def timeline_shape(http, header):
    rows = http.get("/demo/memories", headers=header).json()
    memories = rows["active"] + rows["superseded"]
    names = {m["id"]: m["object"] for m in memories}
    return sorted(
        (
            m["object"],
            m["status"],
            names.get(m["superseded_by"]),
            (m["valid_to"] or "")[:10],
            m["observed_at"][:10],
        )
        for m in memories
    )


def test_dated_conversations_converge_out_of_order_and_answer_as_of(live):
    _, _, provider, http, header = live
    old = ("I use TensorFlow for my personal projects.", "conversation-1", "2024-03-10")
    new = ("I switched from TensorFlow to PyTorch.", "conversation-2", "2026-09-01")

    provider.replies.extend(framework_replies("TensorFlow"))
    assert dated_write(http, header, *old[:2], "a1", old[2]).status_code == 200
    provider.replies.extend(framework_replies("PyTorch", "replaces", "TensorFlow"))
    provider.replies.append({"verdict": "UPDATE", "reason": "The user switched."})
    assert dated_write(http, header, *new[:2], "a2", new[2]).status_code == 200

    reverse = {"x-demo-token": http.post("/demo/session").json()["token"]}
    provider.replies.extend(framework_replies("PyTorch", "replaces", "TensorFlow"))
    assert dated_write(http, reverse, *new[:2], "b1", new[2]).status_code == 200
    provider.replies.extend(framework_replies("TensorFlow"))
    provider.replies.append({"verdict": "DISTINCT", "reason": "An earlier framework."})
    assert dated_write(http, reverse, *old[:2], "b2", old[2]).status_code == 200

    chronological = timeline_shape(http, header)
    assert chronological == timeline_shape(http, reverse)
    assert chronological == [
        ("PyTorch", "active", None, "", "2026-09-01"),
        ("TensorFlow", "superseded", "PyTorch", "2026-09-01", "2024-03-10"),
    ]
    for token in (header, reverse):

        def held(day, token=token):
            response = http.get("/demo/as_of", headers=token, params={"at": day})
            assert response.status_code == 200, response.text
            return [m["object"] for m in response.json()["memories"]]

        assert held("2023-12-31") == []
        assert held("2025-01-01") == ["TensorFlow"]
        assert held("2026-10-01") == ["PyTorch"]


def test_conversation_dates_are_validated_before_calls_and_bound_to_the_write_key(live):
    _, _, provider, http, header = live
    future = (datetime.now() + timedelta(days=2)).date().isoformat()
    response = dated_write(http, header, "I live in Canberra.", "conversation-1", "k1", future)
    assert response.status_code == 422
    assert provider.requests == []
    provider.replies.extend(fact_replies("Canberra"))
    assert (
        dated_write(http, header, "I live in Canberra.", "conversation-1", "k1", "2025-05-01")
    ).status_code == 200
    # The same key with another date is another request, not a replay of the first.
    moved = dated_write(http, header, "I live in Canberra.", "conversation-1", "k1", "2025-06-01")
    assert moved.status_code == 409
    assert len(provider.requests) == 2


def test_as_of_and_answer_exclusions_stay_inside_the_namespace(live):
    _, _, provider, http, header = live
    provider.replies.extend(fact_replies("SecretCity"))
    assert (
        dated_write(http, header, "I live in SecretCity.", "conversation-1", "s1", "2025-01-01")
    ).status_code == 200
    other = {"x-demo-token": http.post("/demo/session").json()["token"]}
    response = http.get("/demo/as_of", headers=other, params={"at": "2026-01-01"})
    assert response.status_code == 200 and response.json()["memories"] == []
    assert http.get("/demo/as_of", params={"at": "2026-01-01"}).status_code == 401
    provider.replies.append({"status": "no_evidence", "answer": "I do not know."})
    answer = http.post("/demo/answer", headers=other, json={"query": "Where do I live?"})
    assert answer.status_code == 200
    assert answer.json()["not_selected"] == []
    assert "SecretCity" not in json.dumps(answer.json())


def test_answer_lists_the_memories_it_did_not_receive(live):
    _, _, provider, http, header = live
    provider.replies.extend(fact_replies("Canberra"))
    assert write(http, header, "I live in Canberra.").status_code == 200
    provider.replies.extend(fact_replies("Melbourne"))
    provider.replies.append({"verdict": "UPDATE", "reason": "The user moved."})
    assert (
        write(http, header, "I moved to Melbourne.", session="conversation-2", request_id="r2")
    ).status_code == 200
    provider.replies.append({"status": "answer", "answer": "Melbourne.", "notes": "source"})
    payload = http.post("/demo/answer", headers=header, json={"query": "Where do I live?"}).json()
    selected = {item["id"] for item in payload["selected"]}
    excluded = {row["object"]: row["reason"] for row in payload["not_selected"]}
    assert excluded == {"Canberra": "not_current"}
    assert not selected & {row["id"] for row in payload["not_selected"]}
