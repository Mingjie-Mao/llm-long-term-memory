"""API contract tests.

Everything here runs against a temporary store with a stub encoder: no API key, no
network, no torch. That is deliberate — the read path is the majority of the surface
and must stay testable in CI, which installs neither `embed` nor a credential.

The cases that matter most are the negative ones. Namespace isolation and the
`rejected` field are the two behaviours a memory service is judged on, and both fail
silently if untested: leakage looks like good recall, and a missing rejection looks
like a retrieval miss.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from llm_long_term_memory.api.app import app, set_service
from llm_long_term_memory.api.service import MemoryService
from llm_long_term_memory.store import Memory, Session, Turn, scoped_session_id

NOW = datetime(2026, 8, 14)
EARLIER = datetime(2026, 3, 2)


class StubEncoder:
    """Deterministic 2-d vectors keyed on a word, so ranking is predictable without
    loading a transformer."""

    dim = 2

    def encode_one(self, text: str) -> np.ndarray:
        return self._vec(text)

    def encode(self, texts: list[str], **kw) -> np.ndarray:
        return np.array([self._vec(t) for t in texts], dtype=np.float32)

    @staticmethod
    def _vec(text: str) -> np.ndarray:
        t = text.lower()
        if "sydney" in t or "moving" in t or "live" in t:
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)


def memory(mid: str, user: str, content: str, **kw) -> Memory:
    return Memory(
        id=mid,
        user_id=user,
        type=kw.pop("type", "semantic"),
        content=content,
        token_count=8,
        ingested_at=kw.pop("ingested_at", NOW),
        valid_from=kw.pop("valid_from", NOW),
        **kw,
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LLTM_STORE_DIR", str(tmp_path))
    # Keyless by construction: these tests assert the read path, and whether the
    # machine running them has a credential must not change the result.
    monkeypatch.setenv("GEMINI_API_KEY", "")
    service = MemoryService(
        config_path="configs/baselines.yaml",
        store_name="test-store",
        encoder=StubEncoder(),
    )
    # The session lands first: memories.source_session_id references it, and
    # that is the order ingestion writes in anyway.
    service.store.add_session(
        Session(
            id="s1",
            user_id="alice",
            started_at=NOW,
            turns=[
                Turn(
                    id="s1:0",
                    session_id="s1",
                    turn_index=0,
                    role="user",
                    content="The user is moving to Sydney next month. Booked the movers.",
                    ts=NOW,
                )
            ],
        )
    )

    # Two namespaces, so isolation is testable rather than assumed.
    service.store.add_memories(
        [
            memory(
                "m_old",
                "alice",
                "The user lives in Canberra.",
                subject="user",
                predicate="lives_in",
                object="Canberra",
                scope="profile",
                valid_from=EARLIER,
                ingested_at=EARLIER,
            ),
            memory(
                "m_new",
                "alice",
                "The user is moving to Sydney next month.",
                subject="user",
                predicate="lives_in",
                object="Sydney",
                scope="plan",
                source_session_id="s1",
                source_turn_index=0,
                source_char_start=0,
                source_char_end=38,
            ),
            memory(
                "m_rec",
                "alice",
                "The assistant recommended Mod Podge for the cork board.",
                subject="assistant",
                predicate="recommendation",
                source_role="assistant",
                scope="recommendation",
            ),
            memory("m_bob", "bob", "The user lives in Perth.", subject="user"),
        ]
    )
    # Establish the supersession through the production path rather than by
    # constructing the row: it is also what enforces the foreign key ordering.
    service.store.mark_superseded("m_old", "m_new", NOW)
    ids = ["m_old", "m_new", "m_rec", "m_bob"]
    service.index.add(
        ids,
        np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            dtype=np.float32,
        ),
    )
    service._answerer = None
    set_service(service)
    with TestClient(app) as c:
        c.app_store = service.store
        yield c
    set_service(None)
    service.close()


def test_healthz_reports_the_store_it_is_serving(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["memories"] == 4


def test_config_exposes_the_manifest_and_no_credentials(client):
    body = client.get("/v1/config").json()

    assert body["config"]["answerer"]
    assert "retrieval" in body["config"]
    assert body["capabilities"] == {"search": True, "write": False, "live_answer": False}
    serialised = str(body).lower()
    assert "api_key" not in serialised and "gemini_api_key" not in serialised


def test_service_fingerprint_reports_the_live_top_k(client):
    # Reach the running service through the same dependency the handlers use.
    from llm_long_term_memory.api.app import get_service

    running = get_service()
    assert running.top_k == 10
    assert "/top_k=10/" in running.fingerprint()
    assert running.config.retrieval.top_k == 20, "the benchmark value is deliberately different"


def test_search_returns_scored_memories_with_provenance(client):
    body = client.post(
        "/v1/memories/search", json={"user_id": "alice", "query": "where do I live?"}
    ).json()

    ids = [m["id"] for m in body["memories"]]
    assert "m_new" in ids
    hit = next(m for m in body["memories"] if m["id"] == "m_new")
    assert hit["source"]["session_id"] == "s1"
    assert hit["signals"]["semantic"] > 0
    assert hit["scope"] == "plan"


def test_superseded_memories_come_back_as_rejected_not_as_silence(client):
    """The differentiator: an absent memory is explained, not merely absent."""
    body = client.post(
        "/v1/memories/search", json={"user_id": "alice", "query": "where do I live?"}
    ).json()

    assert "m_old" not in [m["id"] for m in body["memories"]]
    rejected = {r["memory_id"]: r for r in body["rejected"]}
    assert rejected["m_old"]["reason"] == "superseded"
    assert rejected["m_old"]["superseded_by"] == "m_new"


def test_include_superseded_returns_the_old_fact(client):
    body = client.post(
        "/v1/memories/search",
        json={"user_id": "alice", "query": "where do I live?", "include_superseded": True},
    ).json()

    assert "m_old" in [m["id"] for m in body["memories"]]


def test_search_never_crosses_a_namespace(client):
    body = client.post(
        "/v1/memories/search", json={"user_id": "alice", "query": "where do I live?"}
    ).json()
    assert "m_bob" not in [m["id"] for m in body["memories"]]
    assert "m_bob" not in [r["memory_id"] for r in body["rejected"]]


def test_reading_another_namespaces_memory_is_404_not_403(client):
    """403 would confirm the id exists."""
    assert client.get("/v1/memories/m_bob", params={"user_id": "alice"}).status_code == 404


def test_memory_detail_includes_the_source_turn(client):
    body = client.get("/v1/memories/m_new", params={"user_id": "alice"}).json()

    assert body["memory"]["id"] == "m_new"
    assert body["evidence"]["session_id"] == "s1"
    assert "Sydney" in body["evidence"]["text"]


def test_listing_filters_by_scope_and_speaker(client):
    all_of_them = client.get("/v1/memories", params={"user_id": "alice"}).json()
    assert all_of_them["total"] == 3, "alice's three, and not bob's"

    assistant = client.get(
        "/v1/memories", params={"user_id": "alice", "source_role": "assistant"}
    ).json()
    assert [m["id"] for m in assistant["memories"]] == ["m_rec"]

    plans = client.get("/v1/memories", params={"user_id": "alice", "scope": "plan"}).json()
    assert [m["id"] for m in plans["memories"]] == ["m_new"]


def test_timeline_shows_the_supersession_chain_oldest_first(client):
    body = client.get(
        "/v1/timeline",
        params={"user_id": "alice", "subject": "user", "predicate": "lives_in"},
    ).json()

    assert [e["id"] for e in body["entries"]] == ["m_old", "m_new"]
    assert body["entries"][0]["valid_to"] is not None, "the old fact is closed"
    assert body["entries"][0]["superseded_by"] == "m_new"
    assert body["entries"][1]["valid_to"] is None, "the current one is still open"


def test_a_missing_user_id_is_rejected_before_any_store_access(client):
    assert client.post("/v1/memories/search", json={"query": "x"}).status_code == 422
    assert client.get("/v1/memories").status_code == 422


def test_the_write_path_reports_that_it_is_unconfigured(client):
    """No API key is a deployment state, not a malformed request."""
    response = client.post(
        "/v1/messages", json={"user_id": "alice", "role": "user", "content": "I moved."}
    )
    assert response.status_code == 503
    assert "extractor" in response.json()["detail"]


def test_forget_marks_evicted_rather_than_deleting(client):
    assert client.delete("/v1/memories/m_rec", params={"user_id": "alice"}).status_code == 204

    # Gone from search, but still on record: provenance survives a forget.
    listed = client.get("/v1/memories", params={"user_id": "alice"}).json()
    evicted = next(m for m in listed["memories"] if m["id"] == "m_rec")
    assert evicted["status"] == "evicted"


def test_forgetting_another_namespaces_memory_is_refused(client):
    assert client.delete("/v1/memories/m_bob", params={"user_id": "alice"}).status_code == 404


def test_every_request_is_logged_once_with_id_latency_and_status(client, capsys):
    """The log is the only record of a production request. It must carry enough to
    answer "why was this slow" and "which build served it" without a reproduction —
    and must never carry memory content or a credential."""
    import json as _json

    client.post("/v1/memories/search", json={"user_id": "alice", "query": "where do I live?"})
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{") and '"request"' in line
    ]

    assert len(lines) == 1, "one event per request, not per handler"
    event = _json.loads(lines[0])
    assert event["route"] == "/v1/memories/search"
    assert event["status"] == 200
    assert event["method"] == "POST"
    assert isinstance(event["latency_ms"], int | float)
    assert event["request_id"]

    # No user data, no secrets.
    body = lines[0].lower()
    assert "canberra" not in body and "sydney" not in body
    assert "api_key" not in body


def test_the_request_id_is_returned_to_the_caller(client):
    """So a user reporting a bad response can name the log line that produced it."""
    response = client.get("/healthz")
    assert response.headers.get("x-request-id")


def test_timeline_marks_which_entries_were_actually_replaced(client):
    """The inspector decides whether to draw a supersession chain from this field.

    Most predicates are multi-valued — owning one collectible does not replace
    another — so a UI that assumes consecutive entries supersede each other invents a
    history out of facts that merely coexist. `superseded_by` is what separates them.
    """
    body = client.get(
        "/v1/timeline",
        params={"user_id": "alice", "subject": "user", "predicate": "lives_in"},
    ).json()

    replaced = [e for e in body["entries"] if e["superseded_by"]]
    assert len(replaced) == 1, "exactly one entry was replaced"
    assert replaced[0]["id"] == "m_old"
    assert body["entries"][-1]["superseded_by"] is None, "the current fact replaces nothing"


def test_a_multi_valued_key_reports_no_supersession(client):
    """Two coexisting facts under one key must not look like a chain."""
    from datetime import timedelta

    client_store = client.app_store  # set by the fixture
    client_store.add_memories(
        [
            memory(
                "m_c1",
                "alice",
                "The user collects vintage stamps.",
                subject="user",
                predicate="collectibles",
                object="stamps",
            ),
            memory(
                "m_c2",
                "alice",
                "The user collects postcards.",
                subject="user",
                predicate="collectibles",
                object="postcards",
                valid_from=NOW + timedelta(days=1),
            ),
        ]
    )

    body = client.get(
        "/v1/timeline",
        params={"user_id": "alice", "subject": "user", "predicate": "collectibles"},
    ).json()

    assert len(body["entries"]) == 2
    assert all(e["superseded_by"] is None for e in body["entries"])
    assert all(e["valid_to"] is None for e in body["entries"])


def test_the_inspector_page_carries_its_state_in_the_url(client):
    """A view has to survive a reload and paste into someone else's browser.

    Not a JS test — it asserts the contract the page depends on: the served HTML
    reads `namespace`/`q` from the URL and rewrites it without navigating.

    The fixed `?demo=` routes this used to assert are deliberately gone; what is
    left is the general property, which is the one that matters. A link built by
    typing a namespace and a question still survives a reload and a paste.
    """
    page = client.get("/").text

    assert "URLSearchParams" in page
    assert "history.replaceState" in page, "must not reload; that would restart the search"
    assert 'params.get("namespace")' in page
    assert 'params.get("q")' in page

    # Shadowing `history` inside the search function turned replaceState into an
    # Array method lookup and blanked the panel. Cheap to assert, silent to break.
    assert "const history =" not in page


def test_the_page_does_not_cache_live_data_in_the_browser(client):
    """UI state is the browser's; memories are the store's. A copy in localStorage
    goes stale against the database and shows a state the backend no longer holds."""
    page = client.get("/").text

    assert "localStorage" in page
    for live in ("memories:", "rejected:", "turns:"):
        assert (
            f"localStorage.setItem(STORE_KEY, JSON.stringify({{...loadUiState(), {live}" not in page
        )


def test_a_recorded_run_is_served_without_calling_a_model(client, tmp_path, monkeypatch):
    """The demo page must cost nothing to open. Answering on every page load spends
    quota on every refresh; hard-coding an answer is a mock wearing a result's
    clothes. A recording is the third option, and it is labelled as one."""
    import json as _json

    from llm_long_term_memory.api import golden as golden_mod

    path = tmp_path / "golden.json"
    path.write_text(
        _json.dumps(
            {
                "runs": [
                    {
                        "name": "demo",
                        "namespace": "alice",
                        "query": "where do I live?",
                        "answer": "Sydney, from next month.",
                        "answer_status": "answer",
                        "fallback_level": "none",
                        "fallback_reason": None,
                        "evidence": [],
                        "judge_verdict": "PASS",
                        "recorded_at": "2026-08-14T00:00:00+10:00",
                        "runs": 3,
                        "fingerprint": "not-the-current-one",
                        "versions": {"answer_prompt": "memory-aware-v2"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(golden_mod, "GOLDEN_PATH", path)

    body = client.get("/v1/golden/demo").json()

    assert body["answer"] == "Sydney, from next month."
    assert body["runs"] == 3
    # The fingerprint deliberately does not match this process.
    assert body["is_current"] is False, "a recording from another build must say so"


def test_an_unknown_recorded_run_is_404(client):
    assert client.get("/v1/golden/does-not-exist").status_code == 404


def test_live_answering_reports_when_it_is_unconfigured(client):
    """Without a credential the button must fail honestly rather than appear broken."""
    response = client.post("/v1/answer", json={"user_id": "alice", "query": "where do I live?"})
    assert response.status_code == 503
    assert "answerer" in response.json()["detail"]


# --------------------------------------------------------- recorded golden runs


def test_a_recorded_run_is_labelled_and_not_executed(client, monkeypatch, tmp_path):
    """The demo shows a real answer without spending quota on every page load.

    Both halves matter: replaying is what makes the page free to leave running, and
    labelling is what stops a replay from passing as a live result.
    """
    import json as _json

    from llm_long_term_memory.api import golden

    path = tmp_path / "golden-runs.json"
    path.write_text(
        _json.dumps(
            {
                "runs": [
                    {
                        "name": "demo",
                        "namespace": "alice",
                        "query": "where do I live?",
                        "answer": "Sydney.",
                        "answer_status": "answer",
                        "fallback_level": "none",
                        "fallback_reason": None,
                        "evidence": [],
                        "memories_selected": 1,
                        "candidates_considered": 3,
                        "judge_verdict": "PASS",
                        "judge_reason": "ok",
                        "recorded_at": "2026-08-15T00:00:00+10:00",
                        "runs": 3,
                        "fingerprint": "deadbeefdeadbeef",
                        "versions": {"answer_prompt": "memory-aware-v2"},
                        "note": "",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(golden, "GOLDEN_PATH", path)

    body = client.get("/v1/golden/demo").json()

    assert body["answer"] == "Sydney."
    assert body["runs"] == 3, "three agreeing runs, not one anecdote"
    # The recording was made under a fingerprint this store cannot reproduce, so the
    # API must say so rather than presenting it as current behaviour.
    assert body["is_current"] is False


def test_an_unknown_golden_run_is_404(client):
    assert client.get("/v1/golden/nope").status_code == 404


def test_live_answer_is_refused_without_an_answerer(client):
    """`Run live answer` must fail loudly on a read-only deployment rather than
    silently falling back to the recording — that would make a replay masquerade as
    a fresh run, which is the thing this whole design avoids."""
    response = client.post("/v1/answer", json={"user_id": "alice", "query": "where do I live?"})

    assert response.status_code == 503
    assert "answerer" in response.json()["detail"].lower()


def test_the_demo_page_never_calls_a_model_on_load(client):
    """The property that makes a demo link shareable: opening it costs nothing.

    Asserted against the served page rather than by watching traffic, because the
    regression to prevent is someone adding an answer call to the render path.
    """
    page = client.get("/").text

    # The live call exists, but only inside a click handler.
    assert "/v1/answer" in page
    live_call_index = page.index("/v1/answer")
    handler_index = page.index('live.addEventListener("click"')
    assert handler_index < live_call_index, "the answer call must sit inside the click handler"
    assert 'api("/v1/answer"' not in page.split('live.addEventListener("click"')[0], (
        "no /v1/answer call before the click handler"
    )


def test_the_page_ships_no_benchmark_namespace_and_no_recorded_replay(client):
    """The inspector must not open onto evaluation data.

    It used to: four `?demo=` links hard-coded LongMemEval namespaces and the
    no-parameter case defaulted into one of them, so opening the page showed a
    synthetic persona's commute and battery habits with nothing saying these were
    fixtures. Two recorded answer runs were replayed on top of that.

    Asserted against the served page because the regression to prevent is someone
    re-adding a convenient default id for a screenshot.
    """
    page = client.get("/").text

    for namespace in ("41275add", "9a707b81", "01493427", "09d032c9"):
        assert namespace not in page, f"{namespace} is a benchmark namespace"
    assert "DEMOS" not in page
    assert "/v1/golden/" not in page, "recorded-run replay must not be in the render path"


def test_no_recorded_run_is_served(client):
    """The shipped recordings were of benchmark personas and are gone. The endpoint
    stays — a deployment may record its own — but must answer 404, not a leftover."""
    for name in ("mayo", "battery"):
        assert client.get(f"/v1/golden/{name}").status_code == 404


def test_the_query_box_cannot_inherit_browser_form_history(client):
    """Browser form history is keyed on a field's name, falling back to its id, and
    in Chrome that store is shared across sites. The query box was `id="q"` with no
    autocomplete, so it inherited suggestions from every search box on the web that
    also calls itself `q` — unrelated phrases from another site were observed
    appearing in it. A box that asks about a person's history is the last place that
    should re-offer what the visitor typed elsewhere."""
    page = client.get("/").text

    assert '<form id="ask" autocomplete="off">' in page
    assert page.count('autocomplete="off"') >= 3, "form and both inputs"
    assert 'id="q"' in page and 'name="lltm-question"' in page
    assert 'id="q" placeholder' not in page, "the unnamed, autocompleting field is back"


def test_health_reports_degraded_when_search_cannot_run(client, monkeypatch):
    """A health check that says ok while search returns 500 is worse than none: it
    tells an orchestrator to route traffic to a container that cannot answer it.

    Observed for real — the first Docker image omitted the `embed` extra, so
    `/healthz` returned ok and every search 500'd."""
    from llm_long_term_memory.api.service import MemoryService

    monkeypatch.setattr(MemoryService, "encoder_available", property(lambda self: False))

    body = client.get("/healthz").json()

    assert body["status"] == "degraded"
    assert body["search_available"] is False
    assert "embed" in body["detail"]
    # Readiness is non-200; liveness remains independently available.
    assert client.get("/healthz").status_code == 503
    assert client.get("/livez").status_code == 200


def test_a_missing_encoder_is_503_with_a_remedy_not_a_bare_500(client, monkeypatch):
    from llm_long_term_memory.api.service import EncoderUnavailable, MemoryService

    def boom(self):
        raise EncoderUnavailable("semantic retrieval needs the `embed` extra")

    monkeypatch.setattr(MemoryService, "encoder", property(boom))

    response = client.post(
        "/v1/memories/search", json={"user_id": "alice", "query": "where do I live?"}
    )

    assert response.status_code == 503
    assert "embed" in response.json()["detail"]


def test_live_answer_uses_the_requested_limit_without_mutating_the_shared_runner(client):
    from llm_long_term_memory.api.app import get_service

    class StubAnswerer:
        top_k = 10

        def answer(self, instance, *, limit=None):
            assert instance.question_id == "alice"
            assert instance.question == "where do I live?"
            assert limit == 3
            assert self.top_k == 10
            return SimpleNamespace(
                text="Sydney",
                context_tokens=42,
                notes={
                    "retrieval": [
                        {
                            "memory_id": "m_new",
                            "content": "The user moved to Sydney.",
                            "score": 0.9,
                            "scope": "profile",
                            "source_role": "user",
                        }
                    ],
                    "answer_status": "answer",
                    "fallback_level": "session",
                    "fallback_reason": "need_source",
                    "fallback_turns": ["The user moved to Sydney."],
                    "candidates_considered": 4,
                },
            )

    service = get_service()
    answerer = StubAnswerer()
    service._answerer = answerer

    response = client.post(
        "/v1/answer",
        json={"user_id": "alice", "query": "where do I live?", "limit": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Sydney"
    assert body["top_k"] == 3
    assert body["memories_selected"] == 1
    assert body["answerer_calls"] == 2
    assert answerer.top_k == 10


def test_write_path_persists_a_turn_and_indexes_extracted_memories(client):
    from llm_long_term_memory.api.app import get_service

    service = get_service()

    class StubExtractor:
        def extract_turn(self, *, user_id, session_id, role, content, now):
            assert user_id == "alice"
            assert role == "user"
            assert content == "I adopted a cat."
            return SimpleNamespace(
                memories=[
                    memory(
                        "m_cat",
                        "alice",
                        "The user adopted a cat.",
                        predicate="owns",
                        object="cat",
                        source_session_id=session_id,
                        source_turn_index=0,
                    )
                ],
                usage={"requests": 1},
            )

    service.extractor = StubExtractor()
    response = client.post(
        "/v1/messages",
        json={
            "user_id": "alice",
            "role": "user",
            "content": "I adopted a cat.",
            "session_id": "cat-chat",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "cat-chat"
    assert body["turn_index"] == 0
    assert [item["id"] for item in body["memories"]] == ["m_cat"]
    from llm_long_term_memory.store import external_session_id

    assert external_session_id(service.get("alice", "m_cat").source_session_id) == "cat-chat"


def test_api_key_auto_composes_the_real_batch_extractor_adapter(client, monkeypatch):
    """The default composition must not rely on the test-only `extract_turn` stub."""
    from llm_long_term_memory.api.app import get_service
    from llm_long_term_memory.api.service import LiveTurnExtractor
    from llm_long_term_memory.llm import UsageTracker

    class ScriptedClient:
        def __init__(self):
            self.payloads = [
                {
                    "sessions": [
                        {
                            "session_index": 0,
                            "facts": [{"content": "The user adopted a cat."}],
                        }
                    ]
                },
                {
                    "facts": [
                        {
                            "index": 0,
                            "temporal_key": "pets",
                            "update_op": "coexists",
                            "object": "cat",
                        }
                    ]
                },
                {"verdict": "DISTINCT", "reason": "Different from the recommendation."},
            ]

        def generate(self, **_kwargs):
            payload = self.payloads.pop(0)
            return SimpleNamespace(
                text=json.dumps(payload),
                input_tokens=10,
                output_tokens=5,
                thinking_tokens=0,
                api_latency_ms=1.0,
            )

    service = get_service()
    service.settings.gemini_api_key = "test-key"
    service._live_usage = UsageTracker()
    scripted = ScriptedClient()
    monkeypatch.setattr(service, "_get_live_client", lambda: scripted)

    response = client.post(
        "/v1/messages",
        json={
            "user_id": "alice",
            "role": "user",
            "content": "I adopted a cat.",
            "session_id": "real-adapter",
        },
    )

    assert response.status_code == 200, response.text
    assert isinstance(service.extractor, LiveTurnExtractor)
    assert response.json()["memories"][0]["source"]["session_id"] == "real-adapter"
    assert service.timeline("alice", "user", "pets")[0].object == "cat"


def test_shared_service_serializes_concurrent_writes(client):
    from llm_long_term_memory.api.app import get_service

    class ConcurrentExtractor:
        def __init__(self):
            self.guard = threading.Lock()
            self.active = 0
            self.max_active = 0

        def extract_turn(self, **_kwargs):
            with self.guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.01)
            with self.guard:
                self.active -= 1
            return SimpleNamespace(memories=[], usage={"total_requests": 0})

    service = get_service()
    extractor = ConcurrentExtractor()
    service.extractor = extractor
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(
            pool.map(
                lambda i: service.add_message(
                    "alice", "user", f"message {i}", session_id="concurrent-chat"
                ),
                range(8),
            )
        )

    assert extractor.max_active == 1
    assert sorted(result["turn_index"] for result in outcomes) == list(range(8))
    assert len(service.store.get_session(scoped_session_id("alice", "concurrent-chat")).turns) == 8


def test_evidence_is_none_when_a_memory_has_no_source_anchor(client):
    from llm_long_term_memory.api.app import get_service

    assert get_service().evidence("alice", "m_rec") is None
