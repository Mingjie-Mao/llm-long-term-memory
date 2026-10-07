import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_long_term_memory.evaluation.judge import JudgeResult
from llm_long_term_memory.evaluation.recovery import RecoveringJudge, RecoveryExhausted
from llm_long_term_memory.llm.client import DailyQuotaExhausted
from llm_long_term_memory.llm.usage import UsageTracker

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from grade_saved_recovering import grade_pending
from grounded_reader_diagnostic import grade_saved


class ProviderError(RuntimeError):
    def __init__(self, code):
        self.code = code


class Judge:
    prompt_version = "unchanged-judge"

    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def grade(self, **kwargs):
        self.calls.append(kwargs)
        reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        return JudgeResult(reply, "original verdict", 3, 2)


def build(tmp_path, judge, **kwargs):
    now = [100.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    tracker = UsageTracker()
    wrapped = RecoveringJudge(
        judge,
        tracker,
        tmp_path / "usage.json",
        tmp_path / "recovery.json",
        "identity",
        clock=lambda: now[0],
        sleep=sleep,
        **kwargs,
    )
    return wrapped, sleeps


def test_recover_exact_saved_answer_without_reader_and_preserve_false_grade(tmp_path):
    judge = Judge([ProviderError(500), ProviderError(503), False])
    wrapped, sleeps = build(tmp_path, judge)
    instance = SimpleNamespace(
        question_id="q",
        question="Where?",
        answer="gold",
        is_abstention=False,
        question_type="lookup",
    )
    reader = {"answer": {"text": "original wrong answer"}, "cohort": "regression"}
    sink = io.StringIO()
    result = grade_saved(wrapped, instance, reader, sink, "identity")
    assert not result["verdict"]["correct"]
    assert len(sink.getvalue().splitlines()) == 1
    assert len(judge.calls) == 3 and judge.calls[0] == judge.calls[1] == judge.calls[2]
    assert judge.calls[0]["hypothesis"] == "original wrong answer"
    assert sum(sleeps) == 45 and max(sleeps) <= 30
    assert json.loads((tmp_path / "recovery.json").read_text(encoding="utf-8"))["failures"] == 2
    assert (tmp_path / "usage.json").exists()


@pytest.mark.parametrize(
    "error",
    [
        ProviderError(400),
        ProviderError(401),
        ProviderError(429),
        ValueError("invalid schema"),
        KeyboardInterrupt(),
    ],
)
def test_permanent_quota_schema_and_interrupt_stop_without_retry(tmp_path, error):
    judge = Judge([error, True])
    wrapped, sleeps = build(tmp_path, judge)
    with pytest.raises(type(error)):
        wrapped.grade(hypothesis="same")
    assert len(judge.calls) == 1 and not sleeps
    assert (tmp_path / "usage.json").exists()


def test_daily_quota_is_not_transient(tmp_path):
    judge = Judge([DailyQuotaExhausted("same", SimpleNamespace(seconds=100)), True])
    wrapped, sleeps = build(tmp_path, judge)
    with pytest.raises(DailyQuotaExhausted):
        wrapped.grade(hypothesis="same")
    assert len(judge.calls) == 1 and not sleeps


def test_recovery_budget_survives_restart_and_foreign_identity_refused(tmp_path):
    judge = Judge([ProviderError(500)] * 4)
    wrapped, _ = build(tmp_path, judge, max_recoveries=1)
    with pytest.raises(RecoveryExhausted):
        wrapped.grade(hypothesis="same")
    resumed, _ = build(tmp_path, Judge([True]), max_recoveries=1)
    with pytest.raises(RecoveryExhausted):
        resumed.grade(hypothesis="same")
    assert not resumed.judge.calls
    with pytest.raises(ValueError, match="identity"):
        RecoveringJudge(
            judge,
            UsageTracker(),
            tmp_path / "usage.json",
            tmp_path / "recovery.json",
            "foreign",
            max_recoveries=1,
        )


def test_interrupt_consumption_is_recorded_by_usage_tracker():
    tracker = UsageTracker()
    with pytest.raises(KeyboardInterrupt), tracker.measure("judge", "same"):
        raise KeyboardInterrupt()
    assert tracker.total_requests == 1 and not tracker.records[0].ok


def test_resume_does_not_duplicate_grades_and_rejects_tampered_reader_before_call():
    instance = SimpleNamespace(
        question_id="q",
        question="Where?",
        answer="gold",
        is_abstention=False,
        question_type="lookup",
    )
    reader = {
        "question_id": "q",
        "inventory_sha256": "identity",
        "cohort": "regression",
        "answer": {"text": "original"},
    }
    sink, grades = io.StringIO(), {}
    judge = Judge([False])
    grade_pending(judge, {"q": instance}, {"q": reader}, grades, sink)
    grade_pending(judge, {"q": instance}, {"q": reader}, grades, sink)
    assert len(judge.calls) == 1 and len(sink.getvalue().splitlines()) == 1
    reader["answer"]["text"] = "tampered"
    with pytest.raises(ValueError, match="exact original"):
        grade_pending(judge, {"q": instance}, {"q": reader}, grades, sink)
    assert len(judge.calls) == 1
