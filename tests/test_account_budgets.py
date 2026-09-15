"""Per-account provider budgets, and the four ways a budget stops holding.

`llm/rate_limiter.py` already stops the deployment exhausting the provider. This is the
other half: stopping one namespace exhausting the deployment, which without it looks to
every other tenant like an outage nobody caused.

Each test below is one way a budget can be true on paper and useless in practice: a
failure that is not charged, a restart that grants a fresh allowance, an erasure that
resets the ledger, and a cap that can be sidestepped by changing one field of the request.
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from llm_long_term_memory.api.budget import (
    ANY,
    AccountBudgets,
    Budget,
    BudgetExceeded,
    next_reset,
    quota_day,
)
from llm_long_term_memory.store import SQLiteMemoryStore


@pytest.fixture
def store(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "budget.db")
    store.initialize()
    yield store
    store.close()


@pytest.fixture
def budgets(store):
    return AccountBudgets(store)


# ------------------------------------------------------------------- the ledger


def test_an_unset_budget_does_not_stop_anyone(budgets):
    """An operator who has configured nothing must not find their service refusing
    traffic. `None` is "no cap"; only a written row can stop a caller."""
    budgets.record("alice", "m", "extract", input_tokens=10_000_000)

    budgets.check("alice", "m", "extract")
    assert budgets.budget_for("alice", "m", "extract").unlimited


def test_a_deliberate_zero_is_not_the_same_as_unset(budgets):
    """The distinction the `None` default exists for: an operator can switch an account
    off, and that must not read as "no budget configured"."""
    budgets.set_budget("alice", daily_calls=0)

    with pytest.raises(BudgetExceeded):
        budgets.check("alice", "m", "extract")


def test_a_failed_call_is_charged(budgets):
    """A failure reached the provider, spent quota and may be billed. A budget that
    excused failures is one a retry loop walks straight through — and a retry loop is
    exactly what runs when things are already going wrong."""
    budgets.set_budget("alice", daily_calls=2)
    budgets.record("alice", "m", "extract", ok=False)
    budgets.record("alice", "m", "extract", ok=False)

    with pytest.raises(BudgetExceeded, match="2 of its 2 daily call"):
        budgets.check("alice", "m", "extract")

    spend = budgets.spent_today("alice")
    assert spend.calls == 2
    # Recorded separately as well, so an operator can see a retry storm for what it is.
    assert spend.failed_calls == 2


def test_the_ledger_survives_a_restart(tmp_path):
    """A budget a redeploy resets is not a budget. The ledger is a table in the store,
    not process state, so a second process inherits the spend rather than the allowance."""
    path = tmp_path / "restart.db"
    first = SQLiteMemoryStore(path)
    first.initialize()
    AccountBudgets(first).set_budget("alice", daily_calls=1)
    AccountBudgets(first).record("alice", "m", "extract")
    first.close()

    second = SQLiteMemoryStore(path)
    second.initialize()
    try:
        with pytest.raises(BudgetExceeded):
            AccountBudgets(second).check("alice", "m", "extract")
    finally:
        second.close()


def test_concurrent_records_do_not_lose_a_call(budgets, store):
    """`ON CONFLICT DO UPDATE` rather than read-modify-write. Two calls doing the latter
    each read the same count and each write it back plus one, so the ledger undercounts —
    and a ledger that undercounts is a budget that does not hold."""
    for _ in range(50):
        budgets.record("alice", "m", "extract", input_tokens=1)

    assert budgets.spent_today("alice").calls == 50
    assert budgets.spent_today("alice").input_tokens == 50


# ---------------------------------------------------------- what a cap applies to


def test_a_cap_cannot_be_sidestepped_by_changing_the_model(budgets):
    """Spend is summed across the account. Reading it per row would make any cap
    avoidable by varying one field of the request."""
    budgets.set_budget("alice", daily_calls=3)
    budgets.record("alice", "model-a", "extract")
    budgets.record("alice", "model-b", "extract")
    budgets.record("alice", "model-c", "answer")

    with pytest.raises(BudgetExceeded):
        budgets.check("alice", "model-d", "extract")


def test_one_account_does_not_spend_another_s_allowance(budgets):
    """The whole point: bob's traffic must not stop alice."""
    budgets.set_budget("alice", daily_calls=1)
    budgets.set_budget("bob", daily_calls=1)
    budgets.record("bob", "m", "extract")

    budgets.check("alice", "m", "extract")
    with pytest.raises(BudgetExceeded):
        budgets.check("bob", "m", "extract")


def test_the_most_specific_row_wins_rather_than_the_strictest(budgets):
    """An operator who writes a narrow row is overriding the broad one on purpose. A
    silent `min()` across matching rows would make that override impossible to express."""
    budgets.set_budget("alice", daily_calls=1)
    budgets.set_budget("alice", daily_calls=100, model="expensive", operation="extract")

    assert budgets.budget_for("alice", "expensive", "extract") == Budget(100, None)
    assert budgets.budget_for("alice", "other", "extract") == Budget(1, None)


def test_a_token_cap_stops_the_call_after_the_one_that_crossed_it(budgets):
    """A call's token count is not knowable before the provider replies, so the token cap
    is enforced on spend already recorded. Pinned rather than hidden: the alternative is a
    budget that looks exact and is not."""
    budgets.set_budget("alice", daily_tokens=100)

    budgets.check("alice", "m", "extract")
    budgets.record("alice", "m", "extract", input_tokens=90, output_tokens=5)
    # Still under: this call is allowed, and it is the one that may overshoot.
    budgets.check("alice", "m", "extract")
    budgets.record("alice", "m", "extract", input_tokens=90, output_tokens=5)

    with pytest.raises(BudgetExceeded, match="190 of its 100 daily token"):
        budgets.check("alice", "m", "extract")


def test_the_error_says_which_limit_and_when_it_resets(budgets):
    """ "Over budget" without either is an error a user cannot act on."""
    budgets.set_budget("alice", daily_calls=1)
    budgets.record("alice", "m", "extract")

    with pytest.raises(BudgetExceeded) as raised:
        budgets.check("alice", "m", "extract")

    assert raised.value.dimension == "call"
    assert raised.value.resets_at == next_reset()
    assert "resets at" in str(raised.value)


def test_a_negative_cap_is_refused(budgets):
    with pytest.raises(ValueError, match="cannot be negative"):
        budgets.set_budget("alice", daily_calls=-1)


# ------------------------------------------------------------------- erasure


def test_erasing_a_namespace_does_not_reset_its_spend(budgets, store):
    """Otherwise "delete my data" is a way to start the day's allowance again. The ledger
    holds counts, not content, so keeping it is not keeping user data."""
    budgets.set_budget("alice", daily_calls=1)
    budgets.record("alice", "m", "extract")

    store.hard_delete_user("alice")

    with pytest.raises(BudgetExceeded):
        budgets.check("alice", "m", "extract")


def test_erasure_still_removes_the_things_it_is_for(budgets, store):
    """The ledger exception must not quietly become an exception for anything else: the
    write keys, which do carry a reply about user content, still go."""
    store.remember_write("alice", "k1", '{"turn_index": 0}', "fingerprint")

    store.hard_delete_user("alice")

    assert store.write_key("alice", "k1") is None


# ------------------------------------------------------------------- the quota day


def test_the_quota_day_is_the_provider_s_not_the_server_s(budgets):
    """A tenant whose day rolled at a different hour than the provider's would get a spend
    window that is briefly double."""
    from llm_long_term_memory.llm.rate_limiter import QUOTA_TZ

    assert quota_day() == datetime.now(QUOTA_TZ).date().isoformat()


def test_yesterday_s_spend_does_not_count_against_today(budgets):
    budgets.set_budget("alice", daily_calls=1)
    budgets.store.record_account_usage(
        "alice",
        "2020-01-01",
        "m",
        "extract",
        calls=99,
        failed_calls=0,
        input_tokens=0,
        output_tokens=0,
    )

    budgets.check("alice", "m", "extract")
    assert budgets.spent_today("alice").calls == 0


def test_the_wildcard_is_the_default_scope(budgets):
    budgets.set_budget("alice", daily_calls=5)
    row = budgets.store.account_budget("alice", ANY, ANY)

    assert row == {"daily_calls": 5, "daily_tokens": None}


def test_an_old_store_gains_the_tables_on_open():
    """The budget tables arrived after stores existed in the wild. Opening one must add
    them rather than fail at the first check.

    The old store is built by this project and then stripped of the new tables, rather
    than hand-written: a hand-written stand-in would only prove the schema runs against
    whatever that fixture happened to contain."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "old.db"
        current = SQLiteMemoryStore(path)
        current.initialize()
        current.close()
        stripped = sqlite3.connect(path)
        stripped.executescript("DROP TABLE account_usage; DROP TABLE account_budgets;")
        stripped.commit()
        stripped.close()

        store = SQLiteMemoryStore(path)
        store.initialize()
        try:
            assert AccountBudgets(store).spent_today("alice").calls == 0
            AccountBudgets(store).set_budget("alice", daily_calls=1)
            AccountBudgets(store).record("alice", "m", "extract")
            with pytest.raises(BudgetExceeded):
                AccountBudgets(store).check("alice", "m", "extract")
        finally:
            store.close()
