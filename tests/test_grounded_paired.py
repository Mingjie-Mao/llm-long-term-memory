import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_long_term_memory.answering import Answer
from llm_long_term_memory.conversation import AnswerRequest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_grounded_paired import BoundRunner, combine


def fixture():
    instance = SimpleNamespace(
        question_id="q", question_type="multi-session", is_abstention=False, answer="ring"
    )
    identity = {
        "inventory_sha256": "frozen",
        "arm": "candidate",
        "answerer": "gemini-3.5-flash-lite",
    }
    reader = {
        "question_id": "q",
        "inventory_sha256": "frozen",
        "cohort": "candidate",
        "answer_prompt_version": "memory-grounded-v15",
        "answer": asdict(Answer("ring", 100, notes={"execution_identity": identity})),
    }
    grade = {
        "question_id": "q",
        "inventory_sha256": "frozen",
        "cohort": "candidate",
        "reader_row_sha256": hashlib.sha256(
            json.dumps(reader, sort_keys=True).encode()
        ).hexdigest(),
        "judge_prompt_version": "original-judge",
        "verdict": {"correct": True, "reason": "same fact"},
    }
    return instance, reader, grade


def test_grade_joins_exact_saved_reader_without_calling_any_model():
    instance, reader, grade = fixture()
    result = combine(instance, reader, grade, "frozen", "candidate")
    assert result.correct and result.hypothesis == "ring"
    assert result.notes == reader["answer"]["notes"]


@pytest.mark.parametrize(
    "change",
    [
        {"inventory_sha256": "other"},
        {"cohort": "baseline"},
        {"reader_row_sha256": "wrong"},
        {"question_id": "other"},
    ],
)
def test_foreign_grade_identity_source_or_arm_is_rejected(change):
    instance, reader, grade = fixture()
    grade.update(change)
    with pytest.raises(ValueError, match="identity"):
        combine(instance, reader, grade, "frozen", "candidate")


def test_correctness_and_answer_model_identity_are_checked():
    instance, reader, grade = fixture()
    grade["verdict"]["correct"] = 1
    with pytest.raises(ValueError, match="boolean"):
        combine(instance, reader, grade, "frozen", "candidate")
    grade["verdict"]["correct"] = True
    reader["answer"]["notes"]["execution_identity"]["answerer"] = "another-model"
    grade["reader_row_sha256"] = hashlib.sha256(
        json.dumps(reader, sort_keys=True).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="execution identity"):
        combine(instance, reader, grade, "frozen", "candidate")


def test_bound_runner_passes_only_answer_request_and_keeps_model():
    seen = []

    def answer(request):
        seen.append(request)
        assert not hasattr(request, "answer")
        return Answer("ring", 100)

    runner = SimpleNamespace(
        answer_request=answer,
        answer_prompt_version="memory-aware-v2",
        model="gemini-3.5-flash-lite",
    )
    request = AnswerRequest("What did I buy?", "2026-10-03", "alice")
    result = BoundRunner(runner, "frozen", "baseline").answer_request(request)
    assert seen == [request]
    assert result.notes["execution_identity"]["arm"] == "baseline"
