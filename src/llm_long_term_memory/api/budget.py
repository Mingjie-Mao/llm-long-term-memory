"""Per-account provider budgets.

`llm/rate_limiter.py` already protects the *provider* quota — the requests-per-day the
whole process shares. This protects the *account*. They are not the same failure: with
only the provider limiter, one namespace can spend the day's entire allowance and every
other tenant sees an outage it did not cause and cannot explain.

Three properties this module exists to hold, each of which was easy to get wrong:

**A failed call still counts.** A call that reached the provider consumed quota and may
still be billed, so it is charged whether or not the write around it succeeded — every
attempt, the client's own retries included. A budget that excused failures is a budget any
retry loop walks straight through, and a retry loop is exactly what runs when things are
already going wrong. What never reached the provider is not charged: a request refused by
validation, by a missing extractor, or by this budget.

**Neither a restart nor an erasure resets it.** The ledger is a table in the same SQLite
store as the data, so a crash, a redeploy or a second process cannot hand a tenant a fresh
allowance, and a backup captures it. An erasure deliberately leaves it: the ledger holds
counts, not content, and clearing it would make "delete my data" a way to start the day
again.

**Both caps are checked before a write, and that write can overshoot them.** A call's token
count is not knowable until the provider replies, so the token cap is enforced on spend
already recorded: a caller at 99% of its token budget is allowed one more write, and is then
stopped. The call cap has the same shape for a different reason. Nothing is reserved in
advance, and one write can make several calls — extraction, a deduplication judgement, the
client's retries — so a caller one call under the cap may spend a whole write's worth.
Writes are serialised within one process; separate processes sharing a store can each pass
the check before either records. These are real limits, not oversights — stating them is
the only honest option, because the alternative is a budget that looks exact and is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

# The quota day is the provider's, not the server's. Reusing the limiter's zone keeps one
# definition of "today" across both budgets; a tenant whose day rolled at a different hour
# than the provider's would get a spend window that is briefly double.
from llm_long_term_memory.llm.rate_limiter import QUOTA_TZ

ANY = "*"


class BudgetExceeded(RuntimeError):
    """This account has spent its allowance for the day.

    Carries the dimension that was hit so the caller can say which, and the reset time so
    it can say when — "over budget" without either is an error a user cannot act on.
    """

    def __init__(self, user_id: str, dimension: str, spent: int, cap: int, resets_at: str) -> None:
        super().__init__(
            f"namespace {user_id!r} has used {spent:,} of its {cap:,} daily {dimension} "
            f"allowance; it resets at {resets_at}"
        )
        self.user_id = user_id
        self.dimension = dimension
        self.spent = spent
        self.cap = cap
        self.resets_at = resets_at


@dataclass(frozen=True, slots=True)
class Budget:
    """One account's caps. `None` is "no cap", which is not the same as zero."""

    daily_calls: int | None = None
    daily_tokens: int | None = None

    @property
    def unlimited(self) -> bool:
        return self.daily_calls is None and self.daily_tokens is None


@dataclass(frozen=True, slots=True)
class Spend:
    calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def quota_day(now: datetime | None = None) -> str:
    """Today in the provider's reset zone, as the ledger stores it."""
    moment = now or datetime.now(QUOTA_TZ)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=QUOTA_TZ)
    return moment.astimezone(QUOTA_TZ).date().isoformat()


def next_reset(now: datetime | None = None) -> str:
    """When the current quota day ends, for the message a stopped caller receives."""
    moment = (now or datetime.now(QUOTA_TZ)).astimezone(QUOTA_TZ)
    tomorrow = date.fromordinal(moment.date().toordinal() + 1)
    return datetime.combine(tomorrow, datetime.min.time(), tzinfo=QUOTA_TZ).isoformat()


class AccountBudgets:
    """Reads caps and writes the ledger. Owns no policy beyond "most specific wins"."""

    def __init__(self, store: Any) -> None:
        self.store = store

    # ----------------------------------------------------------------- caps

    def budget_for(self, user_id: str, model: str, operation: str) -> Budget:
        """The cap that applies, most specific first.

        Four rows can match — exact, per-model, per-operation, and the tenant-wide `*`/`*`
        — and picking the *first* match rather than the strictest is deliberate: an
        operator who writes a narrow row is overriding the broad one on purpose, and a
        silent `min()` across rows would make that override impossible to express.
        """
        for model_key, operation_key in (
            (model, operation),
            (model, ANY),
            (ANY, operation),
            (ANY, ANY),
        ):
            row = self.store.account_budget(user_id, model_key, operation_key)
            if row is not None:
                return Budget(row["daily_calls"], row["daily_tokens"])
        return Budget()

    def set_budget(
        self,
        user_id: str,
        *,
        daily_calls: int | None = None,
        daily_tokens: int | None = None,
        model: str = ANY,
        operation: str = ANY,
    ) -> None:
        if daily_calls is not None and daily_calls < 0:
            raise ValueError("daily_calls cannot be negative")
        if daily_tokens is not None and daily_tokens < 0:
            raise ValueError("daily_tokens cannot be negative")
        self.store.set_account_budget(user_id, model, operation, daily_calls, daily_tokens)

    # -------------------------------------------------------------- ledger

    def spent_today(self, user_id: str, day: str | None = None) -> Spend:
        """Everything this account has spent today, across models and operations.

        Summed across the whole account rather than per row: a per-model cap still has to
        stop a tenant that spreads the same spend over several models, and reading it any
        other way would make a cap avoidable by changing one field of the request.
        """
        return self.store.account_spend(user_id, day or quota_day())

    def check(self, user_id: str, model: str, operation: str) -> None:
        """Raise if this account may not make another call right now.

        Called before the request. The token check reads spend already recorded, so it
        stops the call *after* the one that crossed the line — see the module docstring.
        """
        budget = self.budget_for(user_id, model, operation)
        if budget.unlimited:
            return
        spend = self.spent_today(user_id)
        if budget.daily_calls is not None and spend.calls >= budget.daily_calls:
            raise BudgetExceeded(user_id, "call", spend.calls, budget.daily_calls, next_reset())
        if budget.daily_tokens is not None and spend.tokens >= budget.daily_tokens:
            raise BudgetExceeded(user_id, "token", spend.tokens, budget.daily_tokens, next_reset())

    def record(
        self,
        user_id: str,
        model: str,
        operation: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        ok: bool = True,
        calls: int = 1,
        failed_calls: int | None = None,
    ) -> None:
        """Add calls to the ledger, successful or not.

        A failed call still increments `calls`. Failures are also recorded separately so an
        operator can see a retry storm, but they are not excused: the request was sent.
        `failed_calls` says how many of `calls` failed when only some did; without it, `ok`
        says all or none.
        """
        if failed_calls is None:
            failed_calls = 0 if ok else calls
        self.store.record_account_usage(
            user_id,
            quota_day(),
            model,
            operation,
            calls=calls,
            failed_calls=min(max(0, failed_calls), calls),
            input_tokens=max(0, input_tokens),
            output_tokens=max(0, output_tokens),
        )
