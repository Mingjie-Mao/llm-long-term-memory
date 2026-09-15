"""The scheduled run: backup, offsite journal, retention, budget pressure.

A backup policy nobody executes is a plan, not a backup — so the part under test is the
one a scheduler depends on: the order the jobs run in, what counts as needing attention,
and whether a reporting failure can be mistaken for a backup failure.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

operate = pytest.importorskip("operate")


@pytest.fixture
def live(tmp_path):
    """A real store, because `snapshot` reads its tables."""
    from llm_long_term_memory.store import SQLiteMemoryStore

    path = tmp_path / "live.db"
    store = SQLiteMemoryStore(path)
    store.initialize()
    store.close()
    return path


def arguments(live, tmp_path, **kw):
    return SimpleNamespace(
        store=live,
        backups=tmp_path / "backups",
        journal_copy=kw.pop("journal_copy", None),
        keep=kw.pop("keep", 7),
        alert_at=kw.pop("alert_at", 0.8),
        log=kw.pop("log", None),
        **kw,
    )


def test_a_missing_store_fails_rather_than_writing_an_empty_backup(tmp_path):
    code, record = operate.run(arguments(tmp_path / "gone.db", tmp_path))
    assert code == 2
    assert "does not exist" in record["error"]


def test_a_clean_run_reports_success(live, tmp_path):
    code, record = operate.run(arguments(live, tmp_path))
    assert code == 0
    assert record["problems"] == []
    assert record["backup"]["files"] >= 1


def test_the_snapshot_is_taken_before_the_prune(live, tmp_path):
    """Pruning first would delete the copy the new snapshot has not replaced yet."""
    source = (REPO / "tools/operate.py").read_text(encoding="utf-8")
    assert source.index("backup_restore.snapshot") < source.index("backup_restore.prune")
    code, record = operate.run(arguments(live, tmp_path, keep=1))
    assert code == 0
    assert len(record["prune"]["kept"]) == 1


def test_a_store_that_never_erased_anything_has_no_journal_and_says_so(live, tmp_path):
    """Silence would be indistinguishable from a copy that failed."""
    offsite = tmp_path / "offsite"
    code, record = operate.run(arguments(live, tmp_path, journal_copy=offsite))
    assert code == 0
    assert record["journal"]["copied"] is False
    assert "no erasure journal" in record["journal"]["reason"]


def test_the_journal_is_copied_off_the_store_s_own_disk(live, tmp_path):
    """A deletion record lost with the store makes a restore replay nothing and report
    that the erased namespaces were correctly removed."""
    from llm_long_term_memory.store.erasure import journal_path_for

    journal_path_for(live).write_text('{"namespace":"a"}\n', encoding="utf-8")
    offsite = tmp_path / "offsite"
    code, record = operate.run(arguments(live, tmp_path, journal_copy=offsite))
    assert code == 0
    assert record["journal"]["copied"] is True
    assert record["journal"]["erasures"] == 1
    assert (offsite / journal_path_for(live).name).is_file()


def test_an_account_near_its_cap_needs_attention_before_it_is_refused(live, tmp_path):
    """`AccountBudgets` refuses at the cap, which the caller learns as a 429 at the worst
    moment. This is the part that speaks while there is still time to act."""
    from llm_long_term_memory.api.budget import ANY, AccountBudgets
    from llm_long_term_memory.store import SQLiteMemoryStore

    store = SQLiteMemoryStore(live)
    store.initialize()
    budgets = AccountBudgets(store)
    budgets.set_budget("tenant-a", daily_calls=10, model=ANY, operation=ANY)
    for _ in range(9):
        budgets.record("tenant-a", "m", "answer", input_tokens=1)
    store.close()

    code, record = operate.run(arguments(live, tmp_path, alert_at=0.8))
    assert code == 1
    approaching = record["budgets"]["approaching"]
    assert [row["user_id"] for row in approaching] == ["tenant-a"]
    assert approaching[0]["dimension"] == "calls"
    assert approaching[0]["share"] == pytest.approx(0.9)


def test_an_uncapped_account_is_not_reported_as_approaching_zero(live, tmp_path):
    """`None` is "no cap", which is not a cap of zero — treating it as one would page
    somebody about every account that has ever been used."""
    _, record = operate.run(arguments(live, tmp_path))
    assert record["budgets"]["approaching"] == []
    assert record["budgets"]["capped_dimensions_checked"] == 0


def test_every_run_leaves_one_line_a_scheduler_can_read(live, tmp_path, monkeypatch):
    log = tmp_path / "ops.log"
    monkeypatch.setattr(
        sys, "argv",
        ["operate.py", "--store", str(live), "--backups", str(tmp_path / "b"),
         "--log", str(log)],
    )  # fmt: skip
    assert operate.main() == 0
    line = json.loads(log.read_text(encoding="utf-8").strip())
    assert line["exit_code"] == 0
    assert line["started_at"] and line["finished_at"]


def test_a_broken_budget_report_does_not_read_as_a_failed_backup(live, tmp_path, monkeypatch):
    """The backup is the job; the budget report is commentary on it. Conflating them
    would have a scheduler retrying a backup that already succeeded."""
    monkeypatch.setattr(
        operate, "budget_pressure", lambda *a, **k: (_ for _ in ()).throw(sqlite3.Error("boom"))
    )
    code, record = operate.run(arguments(live, tmp_path))
    assert code == 1
    assert record["backup"]["files"] >= 1
    assert any("budget check failed" in p for p in record["problems"])
