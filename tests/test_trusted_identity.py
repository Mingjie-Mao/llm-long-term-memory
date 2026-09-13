"""A namespace the caller named is not a namespace the caller proved.

Before this, `user_id` arrived in the query string or the body, so every "tenant
isolation" check in the service compared a value against itself. These tests are written
as the attack rather than as the feature: the question is not whether a token works, it is
whether a caller holding tenant A's token can reach tenant B by any route the API exposes.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from llm_long_term_memory.api.identity import (
    AuthenticationConfigurationError,
    AuthenticationRequired,
    NamespaceForbidden,
    Principal,
    auth_enabled,
    configured_tokens,
    principal_from_token,
    resolve_namespace,
)

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from llm_long_term_memory.api import app as app_module  # noqa: E402
from llm_long_term_memory.api.service import MemoryService  # noqa: E402
from llm_long_term_memory.store import (  # noqa: E402
    Memory,
    NumpyFlatIndex,
    Session,
    SQLiteMemoryStore,
    Turn,
    scoped_session_id,
)

TOKENS = "tok-a:tenant-a,tok-b:tenant-b"


class _StubEncoder:
    dim = 2

    def encode(self, texts, show_progress=False):
        return np.ones((len(texts), 2), dtype=np.float32)

    def encode_one(self, text):
        return np.ones(2, dtype=np.float32)


def _memory(mid: str, user: str, content: str, session: str) -> Memory:
    return Memory(
        mid,
        user,
        "semantic",
        content,
        6,
        subject="user",
        predicate="likes",
        object=content.split()[-1],
        event_time=datetime(2026, 1, 1),
        valid_from=datetime(2026, 1, 1),
        ingested_at=datetime(2026, 1, 1),
        source_session_id=session,
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Two tenants with one memory each, and a token for each."""
    store = SQLiteMemoryStore(tmp_path / "m.db")
    store.initialize()
    index = NumpyFlatIndex(tmp_path / "idx", dim=2)

    memories = []
    for tenant, mid, text in (
        ("tenant-a", "mem-a", "the user likes apples"),
        ("tenant-b", "mem-b", "the user likes bananas"),
    ):
        session = scoped_session_id(tenant, "s1")
        store.add_session(
            Session(
                id=session,
                user_id=tenant,
                started_at=datetime(2026, 1, 1),
                source="test",
                turns=[
                    Turn(
                        id=f"{session}:0",
                        session_id=session,
                        turn_index=0,
                        role="user",
                        content=text,
                        ts=datetime(2026, 1, 1),
                    )
                ],
            )
        )
        memories.append(_memory(mid, tenant, text, session))
    store.add_memories(memories)
    index.add([m.id for m in memories], np.ones((2, 2), dtype=np.float32))

    service = MemoryService.__new__(MemoryService)
    service.__init__(store_name="t", settings=_Settings(tmp_path), encoder=_StubEncoder())
    service.store.close()
    service.store = store
    service.index = index
    app_module.set_service(service)

    monkeypatch.setenv("LLTM_API_TOKENS", TOKENS)
    yield TestClient(app_module.app)
    app_module.set_service(None)
    store.close()


class _Settings:
    def __init__(self, tmp_path):
        self.store_dir = tmp_path
        self.data_dir = tmp_path
        self.results_dir = tmp_path
        self.gemini_api_key = ""

    @property
    def has_api_key(self):
        return False


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ------------------------------------------------------------------ the attack


def test_a_token_holder_cannot_read_another_tenant_by_naming_it(client):
    """The whole point. Tenant A's token, tenant B's namespace in the query string."""
    response = client.get("/v1/memories", params={"user_id": "tenant-b"}, headers=auth("tok-a"))

    assert response.status_code == 403
    assert "does not match" in response.json()["detail"]


def test_a_token_holder_cannot_write_into_another_tenant(client):
    response = client.post(
        "/v1/messages",
        json={"user_id": "tenant-b", "role": "user", "content": "hello"},
        headers=auth("tok-a"),
    )

    assert response.status_code == 403


def test_a_token_holder_cannot_search_another_tenant_s_raw_conversations(client):
    response = client.post(
        "/v1/raw/search",
        json={"user_id": "tenant-b", "query": "bananas"},
        headers=auth("tok-a"),
    )

    assert response.status_code == 403


def test_a_token_holder_cannot_erase_another_tenant(client):
    response = client.request(
        "DELETE", "/v1/data", params={"user_id": "tenant-b", "confirm": True}, headers=auth("tok-a")
    )

    assert response.status_code == 403
    # And tenant B still has its data.
    assert client.get("/v1/memories", headers=auth("tok-b")).json()["total"] == 1


def test_omitting_user_id_falls_back_to_the_credential_not_to_everything(client):
    body = client.get("/v1/memories", headers=auth("tok-a")).json()

    assert body["total"] == 1
    assert body["memories"][0]["content"] == "the user likes apples"


def test_a_missing_or_wrong_token_is_refused_when_tokens_are_configured(client):
    assert client.get("/v1/memories", params={"user_id": "tenant-a"}).status_code == 401
    assert client.get("/v1/memories", headers=auth("nope")).status_code == 401
    assert client.get("/v1/memories", headers={"Authorization": "tok-a"}).status_code == 401


def test_a_memory_in_another_namespace_is_404_not_403(client):
    """Distinguishing 'not yours' from 'does not exist' would leak whether an id is in
    use. The namespace mismatch is 403 because the caller named it; guessing an id is
    404 because it must not confirm the guess."""
    response = client.get("/v1/memories/mem-b", headers=auth("tok-a"))

    assert response.status_code == 404


# ------------------------------------------------------------------ export and erasure


def test_export_returns_the_caller_s_own_data_including_raw_turns(client):
    body = client.get("/v1/export", headers=auth("tok-a")).json()

    assert body["user_id"] == "tenant-a"
    assert body["counts"] == {"memories": 1, "sessions": 1, "turns": 1}
    assert body["memories"][0]["content"] == "the user likes apples"
    assert body["sessions"][0]["turns"][0]["content"] == "the user likes apples"
    # The internal scoping prefix is an implementation detail of the store, not the
    # session id the caller ever used.
    assert body["sessions"][0]["session_id"] == "s1"
    assert "bananas" not in response_text(body)


def response_text(body) -> str:
    import json

    return json.dumps(body)


def test_erasure_requires_confirmation_and_then_actually_removes_everything(client):
    unconfirmed = client.request("DELETE", "/v1/data", headers=auth("tok-a"))
    assert unconfirmed.status_code == 400
    assert client.get("/v1/memories", headers=auth("tok-a")).json()["total"] == 1

    removed = client.request("DELETE", "/v1/data", params={"confirm": True}, headers=auth("tok-a"))
    assert removed.status_code == 200
    assert removed.json()["memories"] == 1
    assert removed.json()["turns"] == 1
    assert removed.json()["vectors"] == 1

    # Gone from every plane the API can reach, not merely filtered out of retrieval.
    assert client.get("/v1/memories", headers=auth("tok-a")).json()["total"] == 0
    assert client.get("/v1/export", headers=auth("tok-a")).json()["counts"]["memories"] == 0
    assert (
        client.post("/v1/raw/search", json={"query": "apples"}, headers=auth("tok-a")).json()[
            "turns"
        ]
        == []
    )
    # And the other tenant is untouched.
    assert client.get("/v1/memories", headers=auth("tok-b")).json()["total"] == 1


def test_erasure_is_a_different_operation_from_forget(client):
    """`forget` stays a soft delete: provenance is the product, and a removed row cannot
    explain why it is gone. Erasure exists because that reasoning does not survive a
    data-subject request, so the two must not be the same call."""
    client.request("DELETE", "/v1/memories/mem-a", headers=auth("tok-a"))

    listed = client.get("/v1/memories", headers=auth("tok-a")).json()
    assert listed["total"] == 1, "forget must not remove the row"
    assert listed["memories"][0]["status"] == "evicted"
    # An evicted memory is still the user's data and still exports.
    assert client.get("/v1/export", headers=auth("tok-a")).json()["counts"]["memories"] == 1


# ------------------------------------------------------------------ open mode


def test_open_mode_is_reported_rather_than_assumed(client, monkeypatch):
    assert client.get("/healthz").json()["authenticated_access"] is True

    monkeypatch.delenv("LLTM_API_TOKENS")
    assert client.get("/healthz").json()["authenticated_access"] is False


def test_open_mode_keeps_the_caller_supplied_namespace(client, monkeypatch):
    """The research CLI, the inspector and the offline suite all drive this without
    credentials. Open mode must keep working — what changed is that it is now a stated
    condition instead of an unexamined default."""
    monkeypatch.delenv("LLTM_API_TOKENS")

    body = client.get("/v1/memories", params={"user_id": "tenant-b"}).json()
    assert body["total"] == 1
    assert body["memories"][0]["content"] == "the user likes bananas"


# ------------------------------------------------------------------ the unit itself


@pytest.mark.parametrize("raw", ["orphan", "orphan:", ":tenant", "t:a,t:b", "t:a,"])
def test_invalid_token_configuration_cannot_enable_open_access(raw):
    with pytest.raises(AuthenticationConfigurationError):
        principal_from_token(None, {"LLTM_API_TOKENS": raw})


def test_valid_tokens_allow_surrounding_whitespace():
    assert configured_tokens({"LLTM_API_TOKENS": " t : tenant , u:other "}) == {
        "t": "tenant",
        "u": "other",
    }


def test_bad_server_configuration_refuses_requests_and_readiness(client, monkeypatch):
    monkeypatch.setenv("LLTM_API_TOKENS", "orphan")
    assert client.get("/v1/memories", params={"user_id": "tenant-b"}).status_code == 503
    assert client.get("/healthz").status_code == 503


def test_non_ascii_presented_token_is_rejected_without_a_server_error():
    with pytest.raises(AuthenticationRequired):
        principal_from_token("Bearer 错误", {"LLTM_API_TOKENS": "t:tenant"})


def test_no_tokens_means_open_mode():
    assert auth_enabled({"LLTM_API_TOKENS": ""}) is False
    assert auth_enabled({"LLTM_API_TOKENS": "t:tenant"}) is True


def test_an_untrusted_principal_never_claims_a_namespace():
    principal = principal_from_token(None, {"LLTM_API_TOKENS": ""})

    assert principal == Principal(user_id="", trusted=False)
    assert principal.is_anonymous
    assert resolve_namespace(principal, "whatever") == "whatever"


def test_a_trusted_principal_ignores_an_absent_request_and_refuses_a_conflicting_one():
    principal = principal_from_token("Bearer t", {"LLTM_API_TOKENS": "t:tenant"})

    assert principal == Principal(user_id="tenant", trusted=True)
    assert resolve_namespace(principal, None) == "tenant"
    assert resolve_namespace(principal, "tenant") == "tenant"
    with pytest.raises(NamespaceForbidden):
        resolve_namespace(principal, "other")


def test_a_bad_credential_raises_rather_than_falling_back_to_open_mode():
    """Falling back would make a typo in the token silently downgrade the deployment to
    no authentication at all."""
    env = {"LLTM_API_TOKENS": "t:tenant"}
    for header in (None, "", "Bearer", "Basic t", "Bearer wrong"):
        with pytest.raises(AuthenticationRequired):
            principal_from_token(header, env)
