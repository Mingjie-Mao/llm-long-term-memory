"""Client behaviour around provider quirks.

These pin two failures found by running against the live API rather than by
reading docs, so they are the tests most likely to catch a regression when the
provider changes something.
"""

from __future__ import annotations

import httpx
import pytest
from google.genai import errors, types

from chronomem.llm import Limits, QuotaManager
from chronomem.llm.client import DailyQuotaExhausted, GeminiClient


class FakeAPIError(errors.APIError):
    """Must subclass the SDK's error type — the client catches that, and a plain
    Exception would sail straight past the handling under test."""

    def __init__(self, code: int, details=None, message: str = ""):
        Exception.__init__(self, message or f"{code}")
        self.code = code
        self.details = details


class FakeModels:
    """Records the config of every attempt so the test can assert on retries."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[types.GenerateContentConfig] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append(config)
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome

        class R:
            text = outcome

            class usage_metadata:
                prompt_token_count = 10
                candidates_token_count = 5
                thoughts_token_count = 0

        return R()


@pytest.fixture
def client(monkeypatch, tmp_path):
    def build(script):
        c = GeminiClient.__new__(GeminiClient)
        c._client = type("C", (), {"models": FakeModels(script)})()
        c.quota = QuotaManager(state_dir=tmp_path, default=Limits(rpm=1000, tpm=10**9, rpd=1000))
        from chronomem.llm import UsageTracker

        c.usage = UsageTracker()
        c.max_retries = 4
        c.max_transport_retries = 6
        c._no_thinking_control = set()
        return c

    return build


GENERIC_400 = FakeAPIError(
    400, message="400 INVALID_ARGUMENT. Request contains an invalid argument."
)


def test_generic_400_on_a_thinking_request_retries_without_thinking(client):
    """gemini-3.5-flash-lite rejects thinking_budget=0 with a message that never
    says 'thinking'. Matching on the text does not work; inferring from the config
    that was sent does."""
    c = client([GENERIC_400, "answer"])
    result = c.generate(role="answerer", model="gemini-3.5-flash-lite", prompt="hi", thinking=False)

    assert result.text == "answer"
    attempts = c._client.models.calls
    assert len(attempts) == 2
    assert attempts[0].thinking_config is not None
    assert attempts[1].thinking_config is None, "the retry must drop the rejected field"
    assert "gemini-3.5-flash-lite" in c._no_thinking_control


def test_the_quirk_is_remembered_for_later_calls(client):
    c = client([GENERIC_400, "one", "two"])
    c.generate(role="answerer", model="m", prompt="a", thinking=False)
    c.generate(role="answerer", model="m", prompt="b", thinking=False)

    assert c._client.models.calls[2].thinking_config is None, "should not re-learn every call"


def test_a_second_generic_400_is_not_retried_forever(client):
    """Once thinking has been ruled out, a 400 is a real bad request."""
    c = client([GENERIC_400, GENERIC_400])
    with pytest.raises(FakeAPIError):
        c.generate(role="answerer", model="m", prompt="hi", thinking=False)
    assert len(c._client.models.calls) == 2


def test_per_day_429_raises_immediately_instead_of_retrying(client):
    body = {
        "error": {
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [
                        {
                            "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                            "quotaValue": "20",
                        }
                    ],
                }
            ]
        }
    }
    c = client([FakeAPIError(429, details=body)])
    with pytest.raises(DailyQuotaExhausted):
        c.generate(role="answerer", model="gemini-3.5-flash", prompt="hi")

    assert len(c._client.models.calls) == 1, "retrying a daily cap only wastes time"
    # And the real limit is adopted for next time.
    assert c.quota.for_model("gemini-3.5-flash").limits.rpd == 20


def test_dropped_connection_is_retried(client, monkeypatch):
    """A multi-hour run will lose its connection at some point; that must not lose
    the run. Seen live as httpx.RemoteProtocolError mid-evaluation."""
    monkeypatch.setattr("time.sleep", lambda _: None)
    c = client([httpx.RemoteProtocolError("Server disconnected"), "recovered"])
    assert c.generate(role="answerer", model="m", prompt="hi").text == "recovered"
    assert len(c._client.models.calls) == 2


def test_persistent_transport_failure_eventually_raises(client, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    c = client([httpx.ConnectError("down")] * 6)
    with pytest.raises(httpx.ConnectError):
        c.generate(role="answerer", model="m", prompt="hi")
    assert len(c._client.models.calls) == 6  # max_transport_retries, not max_retries


def test_transport_retries_do_not_consume_the_api_error_budget(client, monkeypatch):
    """A dropped connection never reached the server, so it is no evidence that the
    request is bad. Counting it against max_retries meant a DNS blip could exhaust
    the budget and kill a multi-hour run — which is exactly what happened."""
    monkeypatch.setattr("time.sleep", lambda _: None)
    # More transport failures than max_retries (4), then success.
    c = client([httpx.ConnectError("down")] * 5 + ["answer"])

    result = c.generate(role="answerer", model="m", prompt="hi")

    assert result.text == "answer"
    assert len(c._client.models.calls) == 6


def test_failed_attempts_are_still_accounted_for(client, monkeypatch):
    """A failed call consumed quota; a run that burned its budget on retries must
    not look free in the usage report."""
    monkeypatch.setattr("time.sleep", lambda _: None)
    c = client([httpx.RemoteProtocolError("boom"), "ok"])
    c.generate(role="answerer", model="m", prompt="hi")
    assert c.usage.total_requests == 2
    assert c.usage.by_role()["answerer"].failures == 1


def test_server_refusal_outranks_the_local_counter(client):
    """Observed live: the server refused on a per-day quota while the limiter still
    counted 367 of a discovered 500. The local count undercounts — failed retries,
    token counting, and earlier processes all spend budget it never sees — so a
    refusal must pin the counter, or the operator is told the quota resets in 0.0h
    when it is 16 hours away."""
    body = {
        "error": {
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [
                        {
                            "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                            "quotaValue": "500",
                        }
                    ],
                }
            ]
        }
    }
    c = client([FakeAPIError(429, details=body)])
    limiter = c.quota.for_model("m")
    assert limiter.remaining_today > 0, "counter believes there is headroom"

    with pytest.raises(DailyQuotaExhausted) as caught:
        c.generate(role="answerer", model="m", prompt="hi")

    assert limiter.remaining_today == 0
    assert caught.value.wait.is_daily
    assert caught.value.wait.seconds > 60, "a real reset time, not a zero wait"


def test_404_is_not_retried(client):
    c = client([FakeAPIError(404, message="model not found")])
    with pytest.raises(FakeAPIError):
        c.generate(role="answerer", model="gemini-2.5-flash", prompt="hi")
    assert len(c._client.models.calls) == 1


def test_fenced_json_is_unwrapped_for_schema_calls(client):
    """gemma-4-31b-it wraps structured output in a markdown fence *intermittently*,
    despite response_mime_type=application/json. Intermittent is worse than always:
    the smoke test passed and the failure landed mid-evaluation."""
    from pydantic import BaseModel

    class Verdict(BaseModel):
        correct: bool

    c = client(['```json\n{"correct": true}\n```'])
    result = c.generate(role="judge", model="gemma-4-31b-it", prompt="hi", schema=Verdict)

    assert Verdict.model_validate_json(result.text).correct is True


def test_a_bare_fence_without_a_language_tag_is_handled(client):
    from pydantic import BaseModel

    class V(BaseModel):
        ok: bool

    c = client(['```\n{"ok": false}\n```'])
    assert (
        V.model_validate_json(c.generate(role="judge", model="m", prompt="x", schema=V).text).ok
        is False
    )


def test_unfenced_json_is_left_alone(client):
    from pydantic import BaseModel

    class V(BaseModel):
        ok: bool

    c = client(['{"ok": true}'])
    assert c.generate(role="judge", model="m", prompt="x", schema=V).text == '{"ok": true}'


def test_prose_answers_keep_their_fences(client):
    """No schema means the text is the deliverable — an answer may legitimately
    contain a code block, and stripping it would corrupt the response."""
    c = client(["Here is code:\n```python\nprint(1)\n```"])
    result = c.generate(role="answerer", model="m", prompt="x")
    assert "```python" in result.text
