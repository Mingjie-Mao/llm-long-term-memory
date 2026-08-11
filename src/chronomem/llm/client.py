"""Gemini client: quota-aware, retrying, and accounted for.

Every LLM call in ChronoMem goes through `GeminiClient.generate`, which

  * reserves a slot from that model's rate limiter before issuing the request,
  * retries 429/5xx with the server's own suggested delay when it offers one,
  * raises `DailyQuotaExhausted` instead of blocking when the daily budget is gone,
    so ingestion drivers can checkpoint and resume tomorrow, and
  * records tokens and latency for every attempt, successful or not.

Thinking is disabled by default. For extraction it buys nothing and costs both
tokens and latency; more importantly, thinking tokens vary between calls, which
would add noise to the token-cost column of the results table. Not every model
accepts `thinking_budget=0` — the ones that reject it are remembered after the
first failure and never sent the field again.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from google import genai
from google.genai import errors, types

from .rate_limiter import QuotaManager, Wait
from .usage import UsageTracker

# The 429 body carries "Please retry in 16.228217092s." — obeying it beats guessing.
_RETRY_HINT = re.compile(r"retry in ([0-9.]+)s", re.IGNORECASE)

# Some models wrap structured output in a markdown fence even when
# `response_mime_type="application/json"` was requested — gemma-4-31b-it does it
# intermittently, which is worse than doing it always: a smoke test passes and the
# failure surfaces mid-run. Stripping is done centrally rather than at each call
# site so that every schema caller is covered by one fix.
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    cleaned = _FENCE.sub("", text.strip())
    return cleaned.strip()


@dataclass(slots=True)
class QuotaViolation:
    quota_id: str
    value: int | None

    @property
    def is_tokens(self) -> bool:
        """Token quotas and request quotas share the `PerMinute` suffix but count
        completely different things — `GenerateContentInputTokensPerModelPerMinute`
        reports a value like 250,000. Reading that as a requests-per-minute limit
        removes rate limiting entirely, which is worse than having no discovery at
        all."""
        return "Token" in self.quota_id

    @property
    def is_per_day(self) -> bool:
        return "PerDay" in self.quota_id and not self.is_tokens

    @property
    def is_per_minute(self) -> bool:
        return "PerMinute" in self.quota_id and not self.is_tokens

    @property
    def is_tokens_per_minute(self) -> bool:
        return "PerMinute" in self.quota_id and self.is_tokens


def parse_quota_violations(exc: Exception) -> list[QuotaViolation]:
    """Pull the real limits out of a 429.

    Google does not publish per-model free-tier limits — the docs point at a
    dashboard — and they differ by more than an order of magnitude between models
    (gemini-3.6-flash allows 20 requests/day where flash-lite allows thousands).
    The one place the number is stated is the QuotaFailure detail of a 429, which
    carries both `quotaId` and `quotaValue`. Reading it there lets the limiter
    calibrate itself instead of trusting a constant that was going to be wrong.
    """
    details = getattr(exc, "details", None) or {}
    if isinstance(details, dict):
        details = details.get("error", {}).get("details", []) or []

    out: list[QuotaViolation] = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        if not detail.get("@type", "").endswith("QuotaFailure"):
            continue
        for v in detail.get("violations", []) or []:
            raw = v.get("quotaValue")
            out.append(
                QuotaViolation(
                    quota_id=v.get("quotaId", ""),
                    value=int(raw) if raw is not None and str(raw).isdigit() else None,
                )
            )
    return out


class DailyQuotaExhausted(RuntimeError):
    """The model's per-day free-tier budget is spent. Checkpoint and resume."""

    def __init__(self, model: str, wait: Wait) -> None:
        hours = wait.seconds / 3600
        super().__init__(f"daily quota exhausted for {model}; resets in {hours:.1f}h")
        self.model = model
        self.wait = wait


@dataclass(slots=True)
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    attempts: int
    api_latency_ms: float = 0.0
    """Time inside the API call only.

    Deliberately excludes time spent blocked on the rate limiter. Free-tier
    throttling can add tens of seconds of queueing, and folding that into the
    reported latency would make the p95 column a measurement of this project's
    quota tier rather than of the system being evaluated.
    """


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        quota: QuotaManager,
        usage: UsageTracker | None = None,
        max_retries: int = 5,
        max_transport_retries: int = 12,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self.quota = quota
        self.usage = usage or UsageTracker()
        self.max_retries = max_retries
        self.max_transport_retries = max_transport_retries
        # Models observed to reject thinking_budget=0; populated at runtime.
        self._no_thinking_control: set[str] = set()

    # ------------------------------------------------------------------ config

    def _build_config(
        self,
        *,
        system: str | None,
        temperature: float,
        max_output_tokens: int | None,
        schema: Any | None,
        thinking: bool,
        model: str,
    ) -> types.GenerateContentConfig:
        kwargs: dict[str, Any] = {"temperature": temperature}
        if system:
            kwargs["system_instruction"] = system
        if max_output_tokens:
            kwargs["max_output_tokens"] = max_output_tokens
        if schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = schema
        if not thinking and model not in self._no_thinking_control:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**kwargs)

    # ---------------------------------------------------------------- generate

    def generate(
        self,
        *,
        role: str,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        schema: Any | None = None,
        thinking: bool = False,
        est_input_tokens: int | None = None,
    ) -> Completion:
        limiter = self.quota.for_model(model)
        # A rough estimate is enough for TPM reservation; the exact count comes back
        # with the response and is what gets recorded.
        est = est_input_tokens if est_input_tokens is not None else len(prompt) // 4

        last_error: Exception | None = None
        transport_attempts = 0
        attempt = 0
        while attempt < self.max_retries:
            attempt += 1
            wait = limiter.acquire(tokens=est)
            if wait is not None and wait.is_daily:
                raise DailyQuotaExhausted(model, wait)

            config = self._build_config(
                system=system,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                schema=schema,
                thinking=thinking,
                model=model,
            )
            try:
                api_started = time.perf_counter()
                with self.usage.measure(role, model) as box:
                    resp = self._client.models.generate_content(
                        model=model, contents=prompt, config=config
                    )
                    u = resp.usage_metadata
                    box["input_tokens"] = u.prompt_token_count or 0
                    box["output_tokens"] = (u.candidates_token_count or 0) + (
                        u.thoughts_token_count or 0
                    )
                raw = resp.text or ""
                return Completion(
                    # Only when a schema was requested: leaving prose untouched
                    # matters, since an answer may legitimately contain a fence.
                    text=strip_code_fences(raw) if schema is not None else raw,
                    model=model,
                    input_tokens=u.prompt_token_count or 0,
                    output_tokens=u.candidates_token_count or 0,
                    thinking_tokens=u.thoughts_token_count or 0,
                    attempts=attempt,
                    api_latency_ms=(time.perf_counter() - api_started) * 1000,
                )

            except errors.APIError as exc:
                last_error = exc
                code = getattr(exc, "code", None)

                # Some models reject thinking_budget=0 — and gemini-3.5-flash-lite
                # rejects it with a bare "Request contains an invalid argument"
                # that never mentions thinking, so the message cannot be matched
                # on. Infer it instead: if this request carried a thinking config,
                # drop it and retry once before calling the request bad.
                if (
                    code == 400
                    and config.thinking_config is not None
                    and model not in self._no_thinking_control
                ):
                    self._no_thinking_control.add(model)
                    continue
                if code in (400, 401, 403, 404):
                    raise  # a bad request will not become good by repeating it

                if code == 429:
                    violations = parse_quota_violations(exc)
                    self.quota.learn(model, violations)
                    # A per-day violation will not clear by retrying. Surface it so
                    # the driver checkpoints, rather than burning the retry budget.
                    daily = next((v for v in violations if v.is_per_day), None)
                    if daily is not None:
                        # The server's refusal outranks the local counter, which
                        # undercounts (see RateLimiter.mark_exhausted). Without
                        # this the limiter still believes it has headroom, check()
                        # returns None, and the operator is told the quota resets
                        # in 0.0h when it is really most of a day away.
                        limiter.mark_exhausted()
                        raise DailyQuotaExhausted(
                            model, limiter.check() or Wait(0.0, "rpd")
                        ) from exc

                    # A tokens-per-minute refusal is "come back in a minute", not
                    # "this request is malformed" — it clears on its own and the
                    # server tells us exactly when. It therefore gets the patient
                    # budget rather than the API-error one, the same distinction
                    # already drawn for dropped connections.
                    #
                    # This matters because the local token window does not survive a
                    # process restart: a resumed run starts with an empty window and
                    # immediately fires two 110k-token requests while the server is
                    # still counting the previous minute. Charging those refusals to
                    # max_retries killed a 50-question run twice, at 20 and at 30.
                    if any(v.is_tokens_per_minute for v in violations):
                        limiter.absorb_tokens(est)
                        transport_attempts += 1
                        if transport_attempts >= self.max_transport_retries:
                            raise
                        attempt -= 1
                        time.sleep(self._backoff(exc, attempt))
                        continue

                if attempt == self.max_retries:
                    raise
                time.sleep(self._backoff(exc, attempt))

            except (httpx.HTTPError, TimeoutError, ConnectionError) as exc:
                # Transport-level failures — dropped connections, DNS blips, read
                # timeouts. A run spans hours and, on this quota, days; the network
                # will break at some point and that must not end the run. Seen in
                # practice as RemoteProtocolError mid-evaluation and as a DNS
                # ConnectError when the laptop's network dropped.
                #
                # These get their own, far more generous budget than API errors:
                # a failed connection never reached the server, so it consumed no
                # quota and waiting costs nothing but time. The API-error budget
                # exists to stop hammering a service that is answering; this one
                # only has to outlast a network outage.
                last_error = exc
                transport_attempts += 1
                if transport_attempts >= self.max_transport_retries:
                    raise
                # Does not count against the API-error budget: a request that never
                # reached the server says nothing about whether the request is valid.
                attempt -= 1
                # Capped exponential: ~2s, 4s, 8s … 60s, then 60s until the budget
                # runs out. Rides out several minutes of no network.
                time.sleep(min(2.0**transport_attempts, 60.0) + random.uniform(0, 1.0))

        raise RuntimeError(f"unreachable: {last_error}")

    def _backoff(self, exc: Exception, attempt: int) -> float:
        """Prefer the server's own retry hint; otherwise exponential with jitter."""
        hinted = _RETRY_HINT.search(str(exc))
        if hinted:
            return min(float(hinted.group(1)) + 0.5, 120.0)
        return min(2.0**attempt, 60.0) + random.uniform(0, 1.0)

    # ------------------------------------------------------------------ tokens

    def count_tokens(self, model: str, text: str) -> int:
        return self._client.models.count_tokens(model=model, contents=text).total_tokens or 0

    def list_models(self) -> list[str]:
        return [
            m.name.replace("models/", "")
            for m in self._client.models.list()
            if "generateContent" in (m.supported_actions or [])
        ]
