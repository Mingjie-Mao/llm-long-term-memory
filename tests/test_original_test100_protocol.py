import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_original_test100 as tool


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "ANALYSIS", tmp_path)
    monkeypatch.setattr(
        tool,
        "instances",
        lambda: [SimpleNamespace(question_id=f"q{n}", question_type="user") for n in range(100)],
    )


def runs():
    return {
        arm: [
            {
                f"q{n}": {
                    "correct": n < 85,
                    "context_tokens": 6400,
                    "notes": {"end_to_end_wall_ms": 1000},
                }
                for n in range(100)
            }
            for _ in range(3)
        ]
        for arm in ("control", "candidate")
    }


def test_incomplete_or_string_verdict_cannot_publish_accuracy(protocol):
    r = runs()
    del r["candidate"][0]["q1"]
    with pytest.raises(ValueError, match="incomplete"):
        tool.report(r, {"control": [], "candidate": []}, "identity")
    r = runs()
    r["candidate"][0]["q1"]["correct"] = "false"
    with pytest.raises(ValueError, match="incomplete"):
        tool.report(r, {"control": [], "candidate": []}, "identity")
    assert not list(tool.ANALYSIS.glob("*.gate.*"))


def test_gate_keeps_historical_and_contemporaneous_scores_separate(protocol):
    result = tool.report(runs(), {"control": [], "candidate": []}, "identity")
    assert result["pass"]
    assert result["historical_v2_one_shot_accuracy"] == 72
    assert result["accuracy_percent"] == {"control": 85, "candidate": 85}
    assert not result["accuracy"]["statistical_no_loss_supported"]
    assert result["answer_end_to_end_latency"]["candidate"]["p95_ms"] == 1000
    assert "not unseen final" in result["limitations"]


def test_context_limit_and_negative_net_are_enforced(protocol):
    r = runs()
    for repeat in r["candidate"]:
        for row in repeat.values():
            row["context_tokens"] = 6501
    assert not tool.report(r, {"control": [], "candidate": []}, "identity")["pass"]


def test_negative_mean_cannot_pass_even_when_context_is_within_budget(protocol):
    r = runs()
    for repeat in r["candidate"]:
        repeat["q0"]["correct"] = False
    result = tool.report(r, {"control": [], "candidate": []}, "identity")
    assert result["accuracy"]["mean_net"] == -1
    assert not result["pass"]


def test_failed_dev_prerequisite_blocks_source_freeze(tmp_path, monkeypatch):
    path = tmp_path / "gate.json"
    path.write_text('{"pass": false}', encoding="utf-8")
    monkeypatch.setattr(tool, "PREREQ", path)
    with pytest.raises(ValueError, match="complete passing dev100"):
        tool.freeze()


def test_transport_adapter_preserves_false_and_only_wraps_transport():
    class Judge:
        prompt_version = "original"

        def __init__(self, value):
            self.value = value

        def grade(self, **kwargs):
            assert kwargs == {"question": "unchanged", "gold": "unchanged"}
            if isinstance(self.value, Exception):
                raise self.value
            return self.value

    kwargs = {"question": "unchanged", "gold": "unchanged"}
    assert tool.TransportAwareJudge(Judge(False)).grade(**kwargs) is False
    with pytest.raises(ConnectionError):
        tool.TransportAwareJudge(Judge(httpx.ConnectError("dns"))).grade(**kwargs)
    with pytest.raises(ValueError, match="invalid schema"):
        tool.TransportAwareJudge(Judge(ValueError("invalid schema"))).grade(**kwargs)


def test_resource_report_retains_failed_calls_and_arm_boundaries(protocol):
    from llm_long_term_memory.llm.usage import CallRecord

    workload = {
        "control": [CallRecord("answerer", "same-model", 100, 10, 50)],
        "candidate": [CallRecord("answerer", "same-model", 0, 0, 100, False, "timeout")],
    }
    result = tool.report(runs(), workload, "identity")
    assert result["resources_by_model"]["same-model"]["calls"] == 2
    assert result["resources_by_arm_and_model"]["candidate"]["same-model"]["failures"] == 1
    assert result["failed_call_tokens_may_be_unreported"]


def test_frozen_source_change_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("changed", encoding="utf-8")
    monkeypatch.setattr(tool, "REPO", tmp_path)
    with pytest.raises(ValueError, match="frozen original-test100 input changed"):
        tool.verify({"files": {"source.py": "original hash"}})
