import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import t4_nonnegative_wait_resume as adapter

from llm_long_term_memory.evaluation.judge import JudgeResult
from llm_long_term_memory.evaluation.recovery import RecoveringJudge, RecoveryExhausted
from llm_long_term_memory.llm.usage import UsageTracker


def make(tmp_path, clock, sleep, failures=0):
    judge = SimpleNamespace(
        prompt_version="same", grade=lambda **kw: JudgeResult(False, "same", 0, 0)
    )
    r = RecoveringJudge(
        judge,
        UsageTracker(),
        tmp_path / "usage.json",
        tmp_path / "state.json",
        "same",
        clock=clock,
        sleep=sleep,
    )
    r.state.update(cooldown_until=100, failures=failures)
    return r


def test_the_shared_recovery_no_longer_sleeps_a_negative_interval(tmp_path):
    # The boundary this adapter was registered for is now fixed in RecoveringJudge
    # itself (tests/test_prerequisite_recovery.py); the adapter stays for the run it
    # was bound to.
    now = iter([99.99, 100.01, 100.02])
    sleeps = []
    r = make(tmp_path, lambda: next(now), sleeps.append)
    assert r.grade(hypothesis="saved answer").correct is False
    assert all(s > 0 for s in sleeps)
    assert r.state["failures"] == 0


def test_positive_waits_are_unchanged_and_expired_waits_are_skipped():
    seen = []
    safe = adapter.safe_sleep(seen.append)
    for n in (-0.1, 0, 0.01, 30):
        safe(n)
    assert seen == [0.01, 30]


def test_adapter_does_not_reset_exhausted_budget(tmp_path):
    with adapter.runtime_adapter():
        r = make(tmp_path, lambda: 101, lambda n: None, failures=4)
        with pytest.raises(RecoveryExhausted):
            r.grade(hypothesis="same")
    assert r.state["failures"] == 4


def test_only_registered_child_drivers_are_injected(monkeypatch):
    seen = []
    monkeypatch.setattr(adapter.subprocess, "call", lambda cmd, **kw: seen.append(cmd) or 0)
    reader = ["python", str(adapter.REPO / "tools/test100_v2_worker.py"), "--execute"]
    driver = ["python", str(adapter.REPO / "tools/t4_savings_regression.py"), "--execute"]
    foreign = ["python", "/other/t4_savings_regression.py", "--execute"]
    with adapter.runtime_adapter():
        adapter.subprocess.call(reader)
        adapter.subprocess.call(driver)
        adapter.subprocess.call(foreign)
    assert seen[0] == reader and seen[2] == foreign
    assert seen[1][-3:] == ["--driver", "t4_savings_regression.py", "--execute"]


def test_confirmed_timer_bootstrap_is_durable_and_never_adds_budget(tmp_path, monkeypatch):
    path = tmp_path / "bootstrap.json"
    monkeypatch.setattr(adapter, "LEDGER", path)
    entry = {"wait_boundary_bootstrap": True, "identity": "same"}
    _, charge = adapter.bootstrap_adapters({"bootstrap_receipt": entry})
    assert charge(entry, ready=False) is False and not path.exists()
    assert charge(entry, ready=True)
    assert json.loads(path.read_text(encoding="utf-8"))["identity"] == "same"
    with pytest.raises(ValueError, match="consumed"):
        charge(entry, ready=True)


def test_nontransport_failure_cannot_be_misclassified_as_new_timer_bootstrap(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "LEDGER", tmp_path / "consumed")
    entry = {"wait_boundary_bootstrap": True, "identity": "registered"}
    monkeypatch.setattr(
        adapter.chain, "receipt", lambda p: (_ for _ in ()).throw(ValueError("permanent"))
    )
    monkeypatch.setattr(adapter, "bootstrap_receipt", lambda p: {**entry, "identity": "different"})
    receipt, _ = adapter.bootstrap_adapters({"bootstrap_receipt": entry})
    with pytest.raises(ValueError, match="interruption changed"):
        receipt({})
