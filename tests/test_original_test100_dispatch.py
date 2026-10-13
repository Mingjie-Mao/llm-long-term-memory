import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_original_test100_dispatch as profile


def test_new_profile_restores_old_protocol_after_use():
    original = {
        k: getattr(profile.experiment, k)
        for k in ("STEM", "STATE", "freeze", "SavingsRepairRunner")
    }
    with profile.configured():
        assert profile.experiment.STEM == profile.STEM
        assert profile.experiment.freeze is profile.freeze
        assert profile.experiment.SavingsRepairRunner is profile.ScopedSavingsRepairRunner
    assert {k: getattr(profile.experiment, k) for k in original} == original


def test_namespace_worker_changes_only_launcher_and_preserves_flags():
    old = ["python", str(profile.REPO / "tools/test100_v2_worker.py"), "--execute", "--rep", "3"]
    new = profile.namespace_worker_command(old)
    assert new[0] == old[0] and new[2:] == old[2:]
    assert Path(new[1]).name == "test100_v2_dispatch_worker.py"
    foreign = ["python", "/another/test100_v2_worker.py", "--execute"]
    assert profile.namespace_worker_command(foreign) == foreign


@pytest.mark.parametrize("mutate", ["false", "untyped", "missing", "binding", "fresh", "decision"])
def test_failed_or_incomplete_replay_cannot_start_new_original_test100(
    tmp_path, monkeypatch, mutate
):
    gate = {
        "pass": True,
        "candidate_prompt_version": profile.VERSION,
        "inventory_sha256": "bound",
        "fresh_reader_rows": 0,
        "fresh_grade_rows": 0,
        "dev100_replayed_rows": 300,
        "train21_replayed_rows": 21,
        "dev": [],
        "train": [],
        "decision": {"pass": True},
    }
    if mutate == "false":
        gate["pass"] = False
    elif mutate == "untyped":
        gate["pass"] = "true"
    elif mutate == "missing":
        gate["dev100_replayed_rows"] = 299
    elif mutate == "binding":
        gate["inventory_sha256"] = "changed"
    elif mutate == "fresh":
        gate["fresh_reader_rows"] = 300
    else:
        gate["decision"]["pass"] = False
    gp, ip = tmp_path / "gate.json", tmp_path / "inventory.json"
    gp.write_text(json.dumps(gate), encoding="utf-8")
    ip.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(profile, "PREREQ", gp)
    monkeypatch.setattr(profile.prerequisite, "INVENTORY", ip)
    monkeypatch.setattr(profile.prerequisite, "verify", lambda s: None)
    monkeypatch.setattr(profile, "sha256_file", lambda p: "bound")
    monkeypatch.setattr(profile.prerequisite, "decide", lambda d, t: {"pass": True})
    with pytest.raises(ValueError, match="acceptance required"):
        profile.passing_prerequisite()
