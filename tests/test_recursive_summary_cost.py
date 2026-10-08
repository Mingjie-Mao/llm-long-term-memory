import copy
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from llm_long_term_memory.answering import Answer
from llm_long_term_memory.llm.usage import CallRecord, UsageTracker

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from continue_acceptance_job import next_reset, retryable_primary
from recursive_summary_report import amortization, model_workload, paired_accuracy, wilson
from run_recursive_summary_cost import GAPS, PILOT, TimedRunner, freeze, instances_for


def outcomes(value):
    return {str(i): {"correct": value} for i in range(100)}


def test_unchanged_100_questions_do_not_prove_zero_loss_by_repeating():
    rows = outcomes(True)
    result = paired_accuracy(
        {"control": [rows] * 3, "candidate": [rows] * 3}, {q: "temporal" for q in rows}
    )
    assert result["mean_net"] == 0
    assert result["majority_paired"]["questions"] == 100
    assert result["majority_paired"]["difference_ci95"][0] < 0
    assert result["majority_paired"]["difference_ci95"][1] > 0
    assert not result["statistical_no_loss_supported"]
    assert wilson(0, 100)[1] > 0


def test_majority_pair_direction_and_type_means():
    control = outcomes(False)
    candidate = copy.deepcopy(control)
    for i in range(25):
        candidate[str(i)]["correct"] = True
    result = paired_accuracy(
        {"control": [control] * 3, "candidate": [candidate] * 3}, {q: "temporal" for q in control}
    )
    assert result["mean_net"] == 25 and result["by_type_mean_net"]["temporal"] == 25
    assert result["majority_paired"]["wins"] == 25
    assert result["statistical_no_loss_supported"]


def test_pair_rejects_foreign_or_incomplete_outputs():
    rows = outcomes(True)
    with pytest.raises(ValueError, match="three fresh"):
        paired_accuracy({"control": [rows] * 2, "candidate": [rows] * 3}, {q: "x" for q in rows})
    bad = copy.deepcopy(rows)
    bad["foreign"] = {"correct": True}
    with pytest.raises(ValueError, match="incomplete"):
        paired_accuracy({"control": [rows] * 3, "candidate": [bad] * 3}, {q: "x" for q in rows})


def test_amortization_does_not_mix_histories_or_hide_update_cost():
    r = amortization(
        {"alice": 100, "bob": 1000},
        {"alice": 20, "bob": 200},
        {"alice": 50, "bob": 40},
        {"alice": 30, "bob": 60},
    )
    assert r["users"]["alice"]["break_even_queries"] == 6
    assert r["users"]["bob"]["break_even_queries"] is None
    assert r["users"]["alice"]["scenarios"]["1"]["summary_workload_tokens"] == 130
    assert r["users"]["alice"]["scenarios"]["1"]["summary_with_one_update_workload_tokens"] == 150
    assert r["users"]["bob"]["scenarios"]["1"]["summary_workload_tokens"] == 1060
    with pytest.raises(ValueError, match="scopes differ"):
        amortization({"alice": 10}, {}, {"bob": 1}, {"alice": 1})


def test_usage_retains_failed_calls_and_separates_models():
    records = [
        CallRecord("answerer", "answer-model", 100, 10, 1),
        CallRecord("answerer", "answer-model", 0, 0, 1, False, "500"),
        CallRecord("extractor", "summary-model", 300, 20, 1),
    ]
    r = model_workload(records)
    assert r["answer-model"]["calls"] == 2 and r["answer-model"]["failures"] == 1
    assert r["summary-model"]["input_tokens"] == 300
    assert set(model_workload(records, "answerer")) == {"answer-model"}


def test_timed_question_captures_all_calls_and_execution_identity():
    usage = UsageTracker()

    class Runner:
        answer_prompt_version = "memory-grounded-v18"
        model = "gemini-3.5-flash-lite"

        def answer_request(self, request):
            usage.records.extend(
                [
                    CallRecord("answerer", self.model, 100, 10, 1),
                    CallRecord("answerer", self.model, 200, 20, 1),
                ]
            )
            return Answer("test", 20, 300, 30, 2, {})

    r = TimedRunner(Runner(), "identity", "candidate", usage).answer_request(None)
    assert r.notes["end_to_end_wall_ms"] >= 0
    assert r.notes["whole_question_usage"]["input_tokens"] == 300
    assert r.notes["whole_question_usage"]["by_model"]["gemini-3.5-flash-lite"]["calls"] == 2
    assert r.notes["execution_identity"]["inventory_sha256"] == "identity"


def test_pilot_is_fixed_union_inside_exposed_train():
    assert len(PILOT) == len(set(PILOT)) == 21
    assert set(PILOT) >= GAPS and len(GAPS) == 3
    assert {i.question_id for i in instances_for("train150")} == set(PILOT)
    assert len(instances_for("dev100")) == 100


def test_summary_calls_require_completed_passing_primary(tmp_path, monkeypatch):
    import run_recursive_summary_cost as driver

    monkeypatch.setattr(driver, "ANALYSIS", tmp_path)
    with pytest.raises(ValueError, match="complete primary"):
        freeze()
    (tmp_path / f"{driver.PRIMARY}.gate.json").write_text('{"pass": false}', encoding="utf-8")
    with pytest.raises(ValueError, match="failed"):
        freeze()


def test_quota_reset_uses_pacific_date_across_sydney_dst():
    # Sydney changed daylight saving on Oct4; the provider reset is Pacific.
    r = next_reset(datetime(2026, 10, 3, 20, tzinfo=UTC))
    assert r == datetime(2026, 10, 4, 7, tzinfo=UTC)
    assert r.astimezone(__import__("zoneinfo").ZoneInfo("Australia/Sydney")).hour == 18


def test_continuation_never_retries_exhausted_or_permanent_failures():
    assert retryable_primary(2, {"status": "QUOTA_WAIT"})
    assert retryable_primary(2, {"status": "WAIT_FOR_READERS"})
    assert retryable_primary(
        2, {"status": "INCOMPLETE", "reason": "daily quota exhausted for model"}
    )
    assert not retryable_primary(
        2, {"status": "INCOMPLETE", "reason": "durable judge recovery budget exhausted"}
    )
    assert not retryable_primary(1, {"status": "INCOMPLETE"})
    assert not retryable_primary(2, {"status": "INCOMPLETE", "reason": "bad tenant"})


def test_summary_parent_call_count_at_group_boundaries():
    from recursive_summary_preflight import parent_count

    assert [parent_count(n) for n in (1, 2, 4, 5, 8, 16, 17, 32)] == [0, 1, 1, 2, 3, 5, 6, 11]


def configure_job(tmp_path, monkeypatch):
    import continue_acceptance_job as job
    import paged_pair_report

    monkeypatch.setattr(job, "ANALYSIS", tmp_path)
    monkeypatch.setattr(job, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(job, "LOG", tmp_path / "job.log")
    monkeypatch.setattr(job, "freeze", lambda: {})
    monkeypatch.setattr(job, "verify", lambda identity: None)
    monkeypatch.setattr(job, "wait_for_existing_writer", lambda *args: None)
    monkeypatch.setattr(paged_pair_report, "report", lambda: {})
    monkeypatch.setattr(sys, "argv", ["job", "--execute"])
    return job


def test_continuation_stops_at_primary_failed_gate_without_summary_calls(tmp_path, monkeypatch):
    job = configure_job(tmp_path, monkeypatch)
    job.save_json(tmp_path / f"{job.PRIMARY}.gate.json", {"pass": False})
    monkeypatch.setattr(job, "run", lambda *args: pytest.fail("no calls after failed gate"))
    assert job.main() == 1
    assert job.read(job.STATE)["status"] == "PRIMARY_FAILED"


def test_continuation_stops_on_exhausted_judge_without_quota_retry(tmp_path, monkeypatch):
    job = configure_job(tmp_path, monkeypatch)
    calls = []

    def fake_run(tool, stage, log):
        calls.append((tool, stage))
        progress = (
            {"status": "QUOTA_WAIT"}
            if stage == "readers"
            else {"status": "INCOMPLETE", "reason": "durable judge recovery budget exhausted"}
        )
        job.save_json(tmp_path / f"{job.PRIMARY}.progress.json", progress)
        return 2

    monkeypatch.setattr(job, "run", fake_run)
    monkeypatch.setattr(job, "quota_wait", lambda *args: pytest.fail("no exhausted-budget reset"))
    assert job.main() == 1
    assert job.read(job.STATE)["status"] == "STOPPED"
    assert [stage for _, stage in calls] == ["readers", "grades"]


def test_continuation_runs_summary_only_after_primary_pass(tmp_path, monkeypatch):
    job = configure_job(tmp_path, monkeypatch)
    calls = []

    def fake_run(tool, stage, log):
        calls.append((tool, stage))
        if tool == "run_paged_paired.py":
            job.save_json(tmp_path / f"{job.PRIMARY}.gate.json", {"pass": True})
        else:
            job.save_json(tmp_path / f"{job.SUMMARY}.progress.json", {"status": "COMPLETE"})
        return 0

    monkeypatch.setattr(job, "run", fake_run)
    assert job.main() == 0
    assert calls == [("run_paged_paired.py", "readers"), ("run_recursive_summary_cost.py", "all")]
    assert job.read(job.STATE)["status"] == "COMPLETE"
