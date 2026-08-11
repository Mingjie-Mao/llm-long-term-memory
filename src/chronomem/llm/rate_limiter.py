"""Rate limiting for the Gemini free tier.

The free tier caps three dimensions independently — requests/minute, tokens/minute,
and requests/day — and exceeding any one returns 429 even when the other two have
headroom. Requests/day is the binding constraint for ChronoMem: ingesting
LongMemEval-S is tens of thousands of extraction calls, so the daily quota, not
cost, decides how long a run takes.

Two consequences shape this module:

  1. The daily counter must survive process restarts, because a full ingestion spans
     several days. It is persisted to disk and keyed by the Pacific-time date that
     Google resets on.
  2. Callers need to distinguish "wait 12 seconds" from "wait until tomorrow".
     `acquire()` returns a `Wait` describing which limit was hit so the ingestion
     driver can checkpoint and exit cleanly rather than sleeping for hours.

The clock and sleep function are injectable so the tests do not actually wait.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Google resets the daily quota at midnight Pacific.
QUOTA_TZ = ZoneInfo("America/Los_Angeles")


@dataclass(frozen=True, slots=True)
class Limits:
    """Free-tier defaults are deliberately conservative placeholders.

    The real numbers vary by model and change over time, so P0 measures them
    against the live API and writes the observed values into config rather than
    trusting any hard-coded constant.
    """

    rpm: int = 10
    tpm: int = 250_000
    rpd: int = 1_500


class RequestTooLarge(RuntimeError):
    """The request exceeds the model's entire per-minute token allowance.

    Not a rate-limit condition — no amount of waiting fixes it. The caller has to
    send less, which for ingestion means a smaller batch.
    """

    def __init__(self, tokens: int, tpm: int) -> None:
        super().__init__(
            f"request of ~{tokens:,} tokens exceeds the model's {tpm:,} tokens/minute "
            f"limit and can never be sent; reduce the batch size"
        )
        self.tokens = tokens
        self.tpm = tpm


@dataclass(frozen=True, slots=True)
class Wait:
    seconds: float
    reason: str  # 'rpm' | 'tpm' | 'rpd'

    @property
    def is_daily(self) -> bool:
        return self.reason == "rpd"


class RateLimiter:
    """Sliding-window limiter over RPM/TPM plus a persisted daily counter."""

    def __init__(
        self,
        limits: Limits,
        state_path: str | Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(QUOTA_TZ),
    ) -> None:
        self.limits = limits
        self.state_path = Path(state_path) if state_path else None
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()
        self._day: date = self._today()
        self._day_count = 0
        self._load()

    def _today(self) -> date:
        return self._wall_clock().date()

    def _load(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        data = json.loads(self.state_path.read_text())
        saved_day = date.fromisoformat(data["day"])
        if saved_day == self._today():
            self._day = saved_day
            self._day_count = data["count"]

    def _save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"day": self._day.isoformat(), "count": self._day_count})
        )

    def _roll_day(self) -> None:
        today = self._today()
        if today != self._day:
            self._day = today
            self._day_count = 0
            self._save()

    def _evict(self, now: float) -> None:
        cutoff = now - 60.0
        while self._requests and self._requests[0] <= cutoff:
            self._requests.popleft()
        while self._tokens and self._tokens[0][0] <= cutoff:
            self._tokens.popleft()

    def _seconds_until_reset(self) -> float:
        now = self._wall_clock()
        tomorrow = (
            datetime.combine(now.date(), datetime.min.time(), tzinfo=QUOTA_TZ).timestamp() + 86_400
        )
        return max(0.0, tomorrow - now.timestamp())

    def check(self, tokens: int = 0) -> Wait | None:
        """Return how long to wait, or None if the request may proceed now."""
        with self._lock:
            self._roll_day()
            now = self._clock()
            self._evict(now)

            # A request bigger than the whole per-minute token allowance can never
            # be satisfied, however long the caller waits. Saying so is the only
            # useful answer — waiting on it previously indexed an empty deque and
            # crashed with `IndexError: deque index out of range`, which told the
            # operator nothing about the actual problem (a 25.6k-token batch against
            # Gemma's 16k TPM).
            if tokens > self.limits.tpm:
                raise RequestTooLarge(tokens, self.limits.tpm)

            if self._day_count >= self.limits.rpd:
                return Wait(self._seconds_until_reset(), "rpd")
            if self._requests and len(self._requests) >= self.limits.rpm:
                return Wait(max(0.0, 60.0 - (now - self._requests[0])), "rpm")
            used = sum(t for _, t in self._tokens)
            if tokens and self._tokens and used + tokens > self.limits.tpm:
                return Wait(max(0.0, 60.0 - (now - self._tokens[0][0])), "tpm")
            return None

    def consume(self, tokens: int = 0) -> None:
        """Record a request that was actually issued."""
        with self._lock:
            self._roll_day()
            now = self._clock()
            self._requests.append(now)
            if tokens:
                self._tokens.append((now, tokens))
            self._day_count += 1
            self._save()

    def acquire(
        self,
        tokens: int = 0,
        sleep: Callable[[float], None] = time.sleep,
        max_wait: float = 300.0,
    ) -> Wait | None:
        """Block until a request may be issued.

        Returns None once the slot is reserved. If the wait exceeds `max_wait`
        — in practice this only happens when the daily quota is exhausted — the
        Wait is returned instead so the caller can checkpoint and stop rather
        than sleeping until tomorrow.
        """
        while True:
            wait = self.check(tokens)
            if wait is None:
                self.consume(tokens)
                return None
            if wait.seconds > max_wait:
                return wait
            sleep(wait.seconds + 0.01)

    @property
    def remaining_today(self) -> int:
        with self._lock:
            self._roll_day()
            return max(0, self.limits.rpd - self._day_count)

    def absorb_tokens(self, tokens: int) -> None:
        """Charge tokens the server counted but this window did not.

        The per-minute token window lives in memory, so it starts empty on every
        process. A resumed run therefore believes it has the full budget and fires
        immediately, while the server is still counting the previous minute's
        traffic. Recording the refused request's tokens makes the local window
        approximate the server's again, so the retry waits instead of colliding.
        """
        with self._lock:
            self._tokens.append((self._clock(), tokens))

    def mark_exhausted(self) -> None:
        """Record that the server refused on a per-day quota.

        The local counter is an estimate and runs low: retries that fail before a
        response, `count_tokens` calls, and requests issued by an earlier process
        all spend real budget without being counted here. So the server can (and
        does) refuse while this limiter still believes there is headroom — observed
        at 367 against a discovered limit of 500.

        When that happens the server is right. Pinning the counter to the limit
        makes `check()` report the true time to reset instead of falling back to a
        zero wait, and stops the rest of the process retrying against a budget that
        is already gone.
        """
        with self._lock:
            self._roll_day()
            self._day_count = max(self._day_count, self.limits.rpd)
            self._save()


class QuotaManager:
    """One RateLimiter per model.

    The free tier meters per model, not per project: the quota IDs returned in a
    429 body are `GenerateRequestsPerDayPerProjectPerModel-FreeTier` and
    `GenerateRequestsPerMinutePerProjectPerModel-FreeTier`. A single shared counter
    would therefore throttle the pipeline several times harder than the provider
    actually does.

    The practical consequence is that assigning the extractor, answerer, and judge
    to *different* models buys independent daily budgets — which is why D3's
    three-role split is a throughput decision as well as a methodological one.
    """

    def __init__(self, state_dir: str | Path | None = None, default: Limits | None = None) -> None:
        self.state_dir = Path(state_dir) if state_dir else None
        self.default = default or Limits()
        self._limits: dict[str, Limits] = {}
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = threading.Lock()

    def configure(self, model: str, limits: Limits) -> None:
        with self._lock:
            self._limits[model] = limits
            self._limiters.pop(model, None)  # rebuild with the new limits

    def for_model(self, model: str) -> RateLimiter:
        with self._lock:
            if model not in self._limiters:
                safe = model.replace("/", "_")
                self._limiters[model] = RateLimiter(
                    self._limits.get(model, self.default),
                    state_path=(self.state_dir / f"quota-{safe}.json") if self.state_dir else None,
                )
            return self._limiters[model]

    def remaining_today(self) -> dict[str, int]:
        with self._lock:
            return {m: rl.remaining_today for m, rl in self._limiters.items()}

    def learn(self, model: str, violations) -> None:
        """Adopt the real limits reported in a 429, and remember them.

        The observed limits are written next to the quota counters so the next run
        starts already calibrated instead of rediscovering them by getting
        rate-limited again.
        """
        current = self._limits.get(model, self.default)
        rpd, rpm, tpm = current.rpd, current.rpm, current.tpm
        for v in violations:
            if v.value is None:
                continue
            if v.is_tokens_per_minute:
                tpm = v.value
            elif v.is_per_day:
                rpd = v.value
            elif v.is_per_minute:
                rpm = v.value
        if (rpd, rpm, tpm) == (current.rpd, current.rpm, current.tpm):
            return

        learned = Limits(rpm=rpm, tpm=tpm, rpd=rpd)
        with self._lock:
            self._limits[model] = learned
            if model in self._limiters:
                self._limiters[model].limits = learned
        self._persist_learned(model, learned)

    def _persist_learned(self, model: str, limits: Limits) -> None:
        if not self.state_dir:
            return
        path = self.state_dir / "observed-limits.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(path.read_text()) if path.exists() else {}
        data[model] = {"rpm": limits.rpm, "tpm": limits.tpm, "rpd": limits.rpd}
        path.write_text(json.dumps(data, indent=2, sort_keys=True))

    def load_learned(self) -> dict[str, Limits]:
        if not self.state_dir:
            return {}
        path = self.state_dir / "observed-limits.json"
        if not path.exists():
            return {}
        out = {}
        for model, d in json.loads(path.read_text()).items():
            out[model] = Limits(rpm=d["rpm"], tpm=d["tpm"], rpd=d["rpd"])
            self.configure(model, out[model])
        return out
