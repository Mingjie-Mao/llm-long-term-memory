import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


@pytest.fixture(
    params=[
        "t4_scoped_regression",
        "t4_event_count_regression",
        "t4_savings_regression",
    ]
)
def tool(request):
    return importlib.import_module(request.param)


def runs():
    return {
        arm: [
            {f"q{n}": {"correct": n < 85, "context_tokens": 6400} for n in range(100)}
            for _ in range(3)
        ]
        for arm in ("control", "candidate")
    }


@pytest.fixture
def manifest(monkeypatch, tool):
    monkeypatch.setattr(
        tool,
        "instances_for",
        lambda split: [
            SimpleNamespace(question_id=f"q{n}", question_type="user") for n in range(100)
        ],
    )


def test_regression_gate_separates_observed_equality_from_proof_of_no_loss(manifest, tool):
    result = tool.evaluate(runs())
    assert result["pass"] and result["accuracy"]["mean_net"] == 0
    assert not result["accuracy"]["statistical_no_loss_supported"]


def test_regression_rejects_negative_net_incomplete_runs_and_false_bool(manifest, tool):
    r = runs()
    r["candidate"][0]["q0"]["correct"] = False
    assert not tool.evaluate(r)["pass"]
    del r["candidate"][0]["q1"]
    with pytest.raises(ValueError, match="incomplete"):
        tool.evaluate(r)
    r = runs()
    r["candidate"][0]["q0"]["correct"] = "true"
    with pytest.raises(ValueError, match="incomplete"):
        tool.evaluate(r)


def test_failed_selection_prerequisite_forbids_regression_calls(tmp_path, monkeypatch, tool):
    path = tmp_path / "gate.json"
    path.write_text('{"pass": false}', encoding="utf-8")
    monkeypatch.setattr(tool, "PREREQ", path)
    with pytest.raises(ValueError, match="passing fixed21"):
        tool.acceptance_check()


def test_immutable_gate_cannot_be_replaced_after_it_is_saved(tmp_path, tool):
    path = tmp_path / "gate.json"
    tool.immutable_json(path, {"pass": False})
    with pytest.raises(ValueError, match="immutable"):
        tool.immutable_json(path, {"pass": True})


def test_stale_public_source_preflight_blocks_execution(tmp_path, monkeypatch, tool):
    import json

    monkeypatch.setattr(tool, "acceptance_check", lambda: None)
    monkeypatch.setattr(tool, "REPO", tmp_path)
    source = tmp_path / "store.db"
    source.write_bytes(b"changed source")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"inputs": {"store.db": "stale hash"}}), encoding="utf-8")
    monkeypatch.setattr(tool, "PREFLIGHT", preflight)
    with pytest.raises(ValueError, match="preflight input changed"):
        tool.freeze()
