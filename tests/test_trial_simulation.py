"""The simulated trial must actually fail when the product breaks.

A gate that always passes is worse than none, so this runs the whole simulation on a
temporary store and then breaks one guarantee — the idempotency key — to show the run
turns red rather than reporting a clean sheet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

pytest.importorskip("fastapi")
trial = pytest.importorskip("trial_simulation")


def test_the_simulated_trial_passes_every_check_and_calls_no_provider(tmp_path):
    report = trial.run(tmp_path / "run", accounts=2)
    assert report["provider_calls"] == 0
    assert report["failed"] == 0, [c for c in report["checks"] if not c["passed"]]
    assert report["turns_written"] == 2 * len(trial.SCRIPT)
    assert report["not_covered"]


def test_the_trial_reports_a_failure_when_a_guarantee_stops_holding(tmp_path, monkeypatch):
    original = trial.MemoryService.add_message

    def ignore_the_key(self, user_id, role, content, session_id=None, idempotency_key=None):
        return original(self, user_id, role, content, session_id)

    monkeypatch.setattr(trial.MemoryService, "add_message", ignore_the_key)
    report = trial.run(tmp_path / "broken", accounts=2)
    failed = {c["check"] for c in report["checks"] if not c["passed"]}
    assert "a retried write replays instead of appending" in failed
