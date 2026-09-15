"""`LLTM_REQUIRE_AUTH`: a deployment that would rather go red than go open.

Open mode is right for the research CLI and wrong for a host holding real user data, and
the difference between them is one environment variable somebody has to remember. Forgetting
`LLTM_API_TOKENS` produces a service that works perfectly and authorises everyone — the
failure that looks most like success, and the one no test catches because every response is
a 200.

So the switch refuses at three independent points, and each is tested on its own: one of
them will eventually be bypassed, and fail-closed means the other two still hold.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from llm_long_term_memory.api.app import app, set_service
from llm_long_term_memory.api.identity import (
    AuthenticationConfigurationError,
    AuthenticationRequired,
    auth_required,
    check_authentication_configuration,
    principal_from_token,
)
from llm_long_term_memory.api.service import MemoryService

REQUIRE = "LLTM_REQUIRE_AUTH"
TOKENS = "LLTM_API_TOKENS"


class StubEncoder:
    """Two dimensions, no transformer: this file tests the door, not the search."""

    dim = 2

    def encode_one(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def encode(self, texts: list[str], **kw) -> np.ndarray:
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


@pytest.fixture
def api_client(tmp_path, clean_env):
    """A live service, started while the environment is still clean, so that a test can
    flip the switch on a running process — which is one of the bypasses being guarded."""
    clean_env.setenv("LLTM_STORE_DIR", str(tmp_path))
    clean_env.setenv("GEMINI_API_KEY", "")
    service = MemoryService(
        config_path="configs/baselines.yaml", store_name="auth-store", encoder=StubEncoder()
    )
    service._answerer = None
    set_service(service)
    with TestClient(app) as client:
        yield client
    set_service(None)
    service.close()


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(REQUIRE, raising=False)
    monkeypatch.delenv(TOKENS, raising=False)
    return monkeypatch


def test_unset_keeps_open_mode_so_the_cli_and_the_suite_still_run(clean_env):
    assert auth_required() is False
    assert check_authentication_configuration() is False
    assert principal_from_token(None).trusted is False


@pytest.mark.parametrize("value", ("1", "true", "TRUE", "yes", "on"))
def test_the_switch_accepts_the_usual_spellings(clean_env, value):
    clean_env.setenv(REQUIRE, value)
    assert auth_required() is True


@pytest.mark.parametrize("value", ("0", "false", "no", "off", ""))
def test_and_the_usual_spellings_of_off(clean_env, value):
    clean_env.setenv(REQUIRE, value)
    assert auth_required() is False


def test_a_typo_is_an_error_rather_than_silently_off(clean_env):
    """`LLTM_REQUIRE_AUTH=ture` disabling the guard is the accident the guard prevents."""
    clean_env.setenv(REQUIRE, "ture")
    with pytest.raises(AuthenticationConfigurationError):
        auth_required()


def test_set_without_tokens_is_refused_rather_than_downgraded(clean_env):
    clean_env.setenv(REQUIRE, "1")
    with pytest.raises(AuthenticationConfigurationError, match="refuses to serve in open mode"):
        check_authentication_configuration()


def test_set_with_tokens_is_simply_enforced(clean_env):
    clean_env.setenv(REQUIRE, "1")
    clean_env.setenv(TOKENS, "secret:tenant-a")
    assert check_authentication_configuration() is True
    with pytest.raises(AuthenticationRequired):
        principal_from_token(None)
    assert principal_from_token("Bearer secret").user_id == "tenant-a"


def test_a_credential_check_refuses_even_if_startup_was_bypassed(clean_env):
    """An env edited on a running host, or an app injected by a test, must not leave the
    door open behind a process that validated once."""
    clean_env.setenv(REQUIRE, "1")
    with pytest.raises(AuthenticationConfigurationError):
        principal_from_token("Bearer anything")


def test_the_process_refuses_to_start(clean_env, tmp_path):
    clean_env.setenv("LLTM_STORE_DIR", str(tmp_path))
    clean_env.setenv(REQUIRE, "1")
    with pytest.raises(AuthenticationConfigurationError), TestClient(app):
        pass


def test_health_goes_red_rather_than_reporting_an_open_service(clean_env, api_client):
    """503, not "degraded but serving": an instance that cannot authenticate should be
    taken out of rotation."""
    clean_env.setenv(REQUIRE, "1")
    response = api_client.get("/healthz")
    assert response.status_code == 503
    body = response.json()
    assert body["authenticated_access"] is False
    assert body["authentication_required"] is True
    assert "open mode" in body["detail"]


def test_requests_are_refused_rather_than_served_from_a_named_namespace(clean_env, api_client):
    """503 rather than 401: no credential could make this instance serve, so inviting the
    caller to retry with one would be a lie, and taking the instance out of rotation is
    the point. A missing token where tokens *are* configured is still a 401 — that case is
    a credential problem and is tested in `test_trusted_identity.py`."""
    clean_env.setenv(REQUIRE, "1")
    response = api_client.post(
        "/v1/memories/search", json={"query": "anything", "user_id": "someone"}
    )
    assert response.status_code == 503
    assert "open mode" in response.json()["detail"]


def test_health_reports_an_open_deployment_as_open(clean_env, api_client):
    response = api_client.get("/healthz")
    assert response.status_code in (200, 503)
    body = response.json()
    assert body["authenticated_access"] is False
    assert body["authentication_required"] is False
