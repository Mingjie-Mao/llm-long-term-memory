"""The provider client must never block forever on a silent connection.

`HttpOptions.timeout` defaults to None in google-genai, which means no timeout at
all. That default cost a dev60 execution 31 minutes of wall clock and zero rows:
the first answerer request was sent, the peer neither answered nor closed, and the
read blocked indefinitely. The retry budgets in `generate` could not help, because
both of them are driven by caught exceptions and a blocking read raises nothing.

So the property under test is not "requests are fast". It is that a stalled socket
turns into an *exception*, which is the only thing the existing retry and resume
machinery can act on.
"""

from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
from google.genai import types

from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
from llm_long_term_memory.llm.client import GeminiClient


def _quota() -> QuotaManager:
    """Limits high enough that the rate limiter never becomes the thing being timed."""
    return QuotaManager(default=Limits(rpm=6000, rpd=100_000, tpm=10**9))


class SilentPeer:
    """A socket that accepts connections and then says nothing at all.

    This is the failure the timeout exists for, and it is deliberately not the same
    as a refused connection: refusal raises immediately on its own, so a client with
    no timeout still makes progress. A peer that accepts and stalls is what hangs.
    """

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._held: list[socket.socket] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            # Held open, never written to, never closed.
            self._held.append(conn)

    def close(self) -> None:
        self._stop.set()
        self._sock.close()
        for conn in self._held:
            conn.close()


@pytest.fixture
def silent_peer():
    peer = SilentPeer()
    yield peer
    peer.close()


def test_timeout_is_configured_on_the_sdk_client():
    """The value has to reach the SDK, not just be stored on our wrapper.

    Asserted through the SDK's own resolved options rather than our constructor
    argument, because the bug being pinned was a client built without the option
    at all — a wrapper attribute would have looked correct throughout.
    """
    client = GeminiClient(api_key="k", quota=_quota(), timeout_seconds=42.0)
    assert client._client._api_client._http_options.timeout == 42_000


def test_default_timeout_is_set_rather_than_unbounded():
    client = GeminiClient(api_key="k", quota=_quota())
    configured = client._client._api_client._http_options.timeout
    assert configured is not None
    # Comfortably above the slowest successful answerer call observed in tune3
    # (101s), so a slow provider does not become a missing row.
    assert configured >= 120_000


def test_a_stalled_connection_raises_instead_of_hanging(silent_peer):
    """The end-to-end property: a peer that never answers must not stall the run."""
    client = GeminiClient(
        api_key="k",
        quota=_quota(),
        max_transport_retries=1,
        timeout_seconds=0.25,
    )
    client._client = type(client._client)(
        api_key="k",
        http_options=types.HttpOptions(
            base_url=f"http://127.0.0.1:{silent_peer.port}",
            timeout=250,
        ),
    )

    started = time.monotonic()
    with pytest.raises(httpx.HTTPError):
        client.generate(role="answerer", model="gemini-3.5-flash-lite", prompt="hi")
    elapsed = time.monotonic() - started

    # The point is bounded time, not the exact bound: without a timeout this call
    # never returns at all.
    assert elapsed < 30.0


def test_read_timeouts_spend_the_transport_budget_not_the_api_budget(monkeypatch):
    """A timeout must be retried like a dropped connection, not like a bad request.

    `max_retries` guards against hammering a service that is answering; a request
    that timed out never got an answer, so it belongs to the far more generous
    transport budget. Getting this backwards would end a multi-day run on five
    unlucky minutes of network.
    """
    attempts = {"n": 0}

    class AlwaysTimesOut:
        def generate_content(self, *, model, contents, config):
            attempts["n"] += 1
            raise httpx.ReadTimeout("timed out")

    # Built field by field, as in test_client_fallbacks: the SDK exposes `models`
    # as a read-only property, so a real client cannot have it swapped out.
    client = GeminiClient.__new__(GeminiClient)
    client._client = type("C", (), {"models": AlwaysTimesOut()})()
    client.quota = _quota()
    client.usage = UsageTracker()
    client.max_retries = 2
    client.max_transport_retries = 4
    client._no_thinking_control = set()
    monkeypatch.setattr("time.sleep", lambda _: None)

    with pytest.raises(httpx.ReadTimeout):
        client.generate(role="answerer", model="gemini-3.5-flash-lite", prompt="hi")

    # Four transport attempts, not two API attempts.
    assert attempts["n"] == 4
