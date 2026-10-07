"""Bounded, durable recovery after the provider client's retries are exhausted."""

from __future__ import annotations

import json
import time
from pathlib import Path

from llm_long_term_memory.llm.client import DailyQuotaExhausted


class RecoveryExhausted(RuntimeError):
    pass


def transient(exc):
    if isinstance(exc, DailyQuotaExhausted):
        return False
    code = getattr(exc, "code", getattr(exc, "status_code", None))
    return code in {500, 502, 503, 504} or isinstance(exc, (ConnectionError, TimeoutError))


class RecoveringJudge:
    """Resume the same grading request; never invokes a reader or changes the judge.

    State is tied to an execution identity. Failed rounds and cooldown persist across
    restarts, preventing a restart from silently resetting the recovery budget.
    """

    def __init__(
        self,
        judge,
        usage,
        usage_path,
        state_path,
        identity,
        *,
        max_recoveries=3,
        max_wait_seconds=180,
        clock=time.time,
        sleep=time.sleep,
    ):
        if max_recoveries < 0 or max_wait_seconds < 0:
            raise ValueError("recovery budgets must be nonnegative")
        self.judge, self.usage = judge, usage
        self.usage_path, self.state_path = Path(usage_path), Path(state_path)
        self.identity = identity
        self.max_recoveries, self.max_wait_seconds = max_recoveries, max_wait_seconds
        self.clock, self.sleep = clock, sleep
        self.prompt_version = judge.prompt_version
        self.state = {
            "identity": identity,
            "failures": 0,
            "wait_seconds": 0,
            "cooldown_until": 0,
            "events": [],
            "max_recoveries": max_recoveries,
            "max_wait_seconds": max_wait_seconds,
        }
        if self.state_path.exists():
            saved = json.loads(self.state_path.read_text(encoding="utf-8"))
            for key in ("identity", "max_recoveries", "max_wait_seconds"):
                if saved[key] != self.state[key]:
                    raise ValueError("recovery execution identity/policy changed")
            self.state = saved

    def _save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.state_path)

    def grade(self, **kwargs):
        while True:
            if self.state["failures"] > self.max_recoveries:
                raise RecoveryExhausted("durable judge recovery budget exhausted")
            while self.clock() < self.state["cooldown_until"]:
                self.sleep(min(30, self.state["cooldown_until"] - self.clock()))
            try:
                return self.judge.grade(**kwargs)
            except Exception as exc:
                if not transient(exc):
                    raise
                self.state["failures"] += 1
                delay = min(60, 15 * 2 ** (self.state["failures"] - 1))
                exhausted = (
                    self.state["failures"] > self.max_recoveries
                    or self.state["wait_seconds"] + delay > self.max_wait_seconds
                )
                self.state["events"].append(
                    {
                        "at": self.clock(),
                        "error_type": type(exc).__name__,
                        "code": getattr(exc, "code", None),
                        "exhausted": exhausted,
                    }
                )
                if exhausted:
                    self.state["failures"] = self.max_recoveries + 1
                    self._save()
                    raise RecoveryExhausted(
                        "judge recovery stopped; saved readers remain intact"
                    ) from exc
                self.state["wait_seconds"] += delay
                self.state["cooldown_until"] = self.clock() + delay
                self._save()
            finally:
                # Includes permanent errors, quota stops and keyboard interrupts.
                self.usage.save(self.usage_path)
