from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_long_term_memory.answering import Answer
from llm_long_term_memory.evaluation.judge import JudgeResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from grounded_reader_diagnostic import grade_saved, save_reader


def instance():
    return SimpleNamespace(
        question_id="q",
        question="What did I buy?",
        question_date="2026-10-01",
        store_namespace="alice",
        answer="ring",
        is_abstention=False,
        question_type="single-session-user",
    )


def test_judge_failure_does_not_drop_saved_reader_answer_or_require_new_reader_call():
    requests = []

    def answer(request):
        requests.append(request)
        assert not hasattr(request, "answer")
        return Answer("ring", 50, notes={"evidence": "actual-reader-trace"})

    runner = SimpleNamespace(answer_request=answer, answer_prompt_version="memory-grounded-v6")
    sink = io.StringIO()
    row = save_reader(runner, instance(), sink, "identity", "originally-correct")

    class FailingJudge:
        def grade(self, **kwargs):
            raise RuntimeError("500 INTERNAL")

    with pytest.raises(RuntimeError, match="500"):
        grade_saved(FailingJudge(), instance(), row, io.StringIO(), "identity")
    saved = json.loads(sink.getvalue())
    assert saved["grading_status"] == "UNGRADED"
    assert saved["answer"]["notes"]["evidence"] == "actual-reader-trace"
    assert len(requests) == 1

    class WorkingJudge:
        prompt_version = "lme-type-aware-v2"

        def grade(self, **kwargs):
            assert kwargs["hypothesis"] == "ring"
            return JudgeResult(True, "same fact", 10, 5)

    grade_sink = io.StringIO()
    grade_saved(WorkingJudge(), instance(), saved, grade_sink, "identity")
    assert json.loads(grade_sink.getvalue())["verdict"]["correct"]
    assert len(requests) == 1
