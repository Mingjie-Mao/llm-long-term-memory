"""Rate limiter tests.

Time is injected, so these run instantly and deterministically — a limiter tested
with real sleeps is a test suite nobody runs.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from chronomem.llm import Limits, RateLimiter
from chronomem.llm.rate_limiter import QUOTA_TZ


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeWallClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


def make(limits: Limits, tmp_path=None, start_day="2026-01-01"):
    clock = FakeClock()
    wall = FakeWallClock(datetime.fromisoformat(f"{start_day}T09:00:00").replace(tzinfo=QUOTA_TZ))
    rl = RateLimiter(
        limits,
        state_path=(tmp_path / "quota.json") if tmp_path else None,
        clock=clock,
        wall_clock=wall,
    )
    return rl, clock, wall


def test_allows_up_to_rpm_then_blocks():
    rl, _, _ = make(Limits(rpm=3, tpm=10**9, rpd=10**9))
    for _ in range(3):
        assert rl.check() is None
        rl.consume()
    wait = rl.check()
    assert wait is not None and wait.reason == "rpm"
    assert wait.seconds == 60.0


def test_rpm_window_slides():
    rl, clock, _ = make(Limits(rpm=2, tpm=10**9, rpd=10**9))
    rl.consume()
    rl.consume()
    assert rl.check() is not None
    clock.advance(61)
    assert rl.check() is None


def test_tpm_is_enforced_independently_of_rpm():
    """Hitting any one of the three limits triggers 429 even with headroom on the
    others — that asymmetry is the whole reason this class exists."""
    rl, _, _ = make(Limits(rpm=100, tpm=1_000, rpd=10**9))
    rl.consume(tokens=900)
    wait = rl.check(tokens=200)
    assert wait is not None and wait.reason == "tpm"
    assert rl.check(tokens=50) is None  # still room under TPM


def test_rpd_reports_time_until_pacific_midnight():
    rl, _, _ = make(Limits(rpm=100, tpm=10**9, rpd=2))
    rl.consume()
    rl.consume()
    wait = rl.check()
    assert wait is not None and wait.reason == "rpd"
    assert wait.is_daily
    assert wait.seconds == 15 * 3600  # 09:00 -> midnight


def test_daily_counter_survives_restart(tmp_path):
    rl, _, _ = make(Limits(rpm=100, tpm=10**9, rpd=5), tmp_path)
    rl.consume()
    rl.consume()
    assert rl.remaining_today == 3

    revived, _, _ = make(Limits(rpm=100, tpm=10**9, rpd=5), tmp_path)
    assert revived.remaining_today == 3, "a multi-day ingestion must not lose its count"


def test_daily_counter_resets_on_new_pacific_day(tmp_path):
    rl, _, wall = make(Limits(rpm=100, tpm=10**9, rpd=2), tmp_path)
    rl.consume()
    rl.consume()
    assert rl.check() is not None

    wall.now = wall.now + timedelta(days=1)
    assert rl.check() is None
    assert rl.remaining_today == 2


def test_stale_state_file_from_yesterday_is_discarded(tmp_path):
    rl, _, _ = make(Limits(rpm=100, tpm=10**9, rpd=5), tmp_path, start_day="2026-01-01")
    rl.consume()
    rl.consume()

    later, _, _ = make(Limits(rpm=100, tpm=10**9, rpd=5), tmp_path, start_day="2026-01-02")
    assert later.remaining_today == 5


def test_acquire_sleeps_for_short_waits():
    rl, clock, _ = make(Limits(rpm=1, tpm=10**9, rpd=10**9))
    slept: list[float] = []

    def sleeper(s: float) -> None:
        slept.append(s)
        clock.advance(s)

    rl.consume()
    assert rl.acquire(sleep=sleeper) is None
    assert len(slept) == 1 and slept[0] > 59


def test_acquire_returns_instead_of_sleeping_until_tomorrow():
    """When the daily quota is gone the driver must checkpoint and exit, not block
    for 15 hours."""
    rl, _, _ = make(Limits(rpm=100, tpm=10**9, rpd=1))
    rl.consume()
    wait = rl.acquire(sleep=lambda _: None, max_wait=300.0)
    assert wait is not None and wait.is_daily


def test_a_request_larger_than_the_whole_tpm_budget_is_rejected_not_waited_on():
    """Waiting cannot help: the request exceeds the per-minute allowance outright.

    Found live — a 25.6k-token extraction batch against Gemma's 16k tokens/minute.
    The old code fell through to `self._tokens[0]` on an empty deque and died with
    `IndexError: deque index out of range`, which says nothing about the real
    problem or its fix.
    """
    from chronomem.llm.rate_limiter import RequestTooLarge

    rl, _, _ = make(Limits(rpm=10, tpm=16_000, rpd=1_500))
    with pytest.raises(RequestTooLarge, match="reduce the batch size"):
        rl.check(tokens=25_600)


def test_a_request_that_merely_does_not_fit_right_now_still_waits():
    rl, _, _ = make(Limits(rpm=100, tpm=16_000, rpd=1_500))
    rl.consume(tokens=15_000)
    wait = rl.check(tokens=5_000)
    assert wait is not None and wait.reason == "tpm"


def test_an_empty_window_never_indexes_out_of_range():
    """rpm=0 or an oversized first request used to reach `deque[0]` on an empty
    deque."""
    rl, _, _ = make(Limits(rpm=0, tpm=10**9, rpd=10**9))
    assert rl.check() is None


def test_absorb_tokens_makes_a_restarted_window_back_off():
    """The per-minute token window is in-memory, so a resumed run starts empty and
    bursts into a server that is still counting the previous minute. Absorbing the
    refused request's tokens restores the local view."""
    rl, _, _ = make(Limits(rpm=100, tpm=250_000, rpd=10**9))
    assert rl.check(tokens=110_000) is None, "a fresh window believes it is free"

    rl.absorb_tokens(110_000)
    rl.absorb_tokens(110_000)

    wait = rl.check(tokens=110_000)
    assert wait is not None and wait.reason == "tpm"
    assert wait.seconds > 0
