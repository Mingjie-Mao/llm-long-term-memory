import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_original_test100_after_dev as chain


def test_incomplete_active_dev_only_waits_and_never_starts_test100():
    assert chain.prerequisite({"status": "RUNNING"}, None, True) == "WAIT"
    assert chain.prerequisite({"status": "RECOVERY_EXHAUSTED"}, None, True) == "WAIT"
    assert chain.prerequisite({"status": "RUNNING"}, None, False) == "STOPPED"


@pytest.mark.parametrize("gate", [None, {"pass": False}, {"pass": "true"}])
def test_missing_failed_or_untyped_gate_cannot_start_test100(gate):
    assert chain.prerequisite({"status": "COMPLETE", "pass": True}, gate, False) == "FAILED"


def test_quota_and_failed_parent_stop_without_automatic_restart():
    assert chain.prerequisite({"status": "QUOTA_WAIT"}, None, True) == "QUOTA_WAIT"
    assert (
        chain.prerequisite({"status": "COMPLETE", "pass": False}, {"pass": True}, True) == "FAILED"
    )


def test_only_complete_matching_pass_flags_allow_launch():
    assert (
        chain.prerequisite({"status": "COMPLETE", "pass": True}, {"pass": True}, False) == "READY"
    )


def test_source_change_during_wait_is_refused(tmp_path, monkeypatch):
    (tmp_path / "source.py").write_text("changed", encoding="utf-8")
    monkeypatch.setattr(chain, "REPO", tmp_path)
    with pytest.raises(ValueError, match="chain source changed"):
        chain.verify({"files": {"source.py": "original digest"}})
