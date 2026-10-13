import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_long_term_memory.evaluation.datasets.longmemeval import Instance

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from t4_savings_replay import (
    ReplayMismatch,
    SavedReplyClient,
    cached_grade,
    grade_request_sha,
    row_sha,
)


def reader():
    return {
        "question_id": "q",
        "cohort": "old",
        "inventory_sha256": "old-id",
        "answer": {
            "text": "same answer",
            "notes": {
                "actual_answer_calls": [
                    {
                        "request": {
                            "role": "answerer",
                            "model": "fixed-model",
                            "prompt": "source",
                            "schema": None,
                        },
                        "response": "same answer",
                    }
                ]
            },
        },
    }


def test_replay_requires_exact_requests_and_all_responses_without_calling_provider():
    class Forbidden:
        def generate(self, **kwargs):
            raise AssertionError("provider must not be called")

    old = reader()
    client = SavedReplyClient(Forbidden(), old)
    response = client.generate(role="answerer", model="fixed-model", prompt="source")
    assert response.input_tokens == response.output_tokens == response.attempts == 0
    client.verify_output(SimpleNamespace(text="same answer"), old)
    with pytest.raises(ReplayMismatch):
        client.generate(role="answerer", model="fixed-model", prompt="source")
    with pytest.raises(ReplayMismatch):
        SavedReplyClient(Forbidden(), old).generate(
            role="answerer", model="fixed-model", prompt="changed"
        )
    with pytest.raises(ReplayMismatch):
        client.verify_output(SimpleNamespace(text="different"), old)


@pytest.mark.parametrize("changed", ["gold", "hypothesis", "binding", "prompt"])
def test_grade_replay_rejects_changed_grading_semantics_and_missing_source_binding(changed):
    old = reader()
    i = Instance("q", "multi-session", "question", "gold", "2023-01-01", [], [])
    original_grade = {
        "question_id": "q",
        "inventory_sha256": "old-id",
        "reader_row_sha256": row_sha(old),
        "judge_prompt_version": "same-judge",
        "verdict": {"correct": False},
    }
    current = {
        "cohort": "new",
        "answer": {
            "text": "same answer",
            "notes": {
                "exact_request_replay": {
                    "source_reader_sha256": row_sha(old),
                    "judge_request_sha256": {"original": grade_request_sha(i, "same answer")},
                    "judge_prompt_version": "same-judge",
                }
            },
        },
    }
    if changed == "gold":
        i.answer = "changed"
    elif changed == "hypothesis":
        current["answer"]["text"] = "changed"
    elif changed == "binding":
        original_grade["reader_row_sha256"] = "wrong"
    else:
        original_grade["judge_prompt_version"] = "changed"
    with pytest.raises(ValueError):
        cached_grade(i, current, old, original_grade, "new-id")


def test_replay_preserves_failed_grade_and_is_labeled_historical():
    old = reader()
    i = Instance("q", "multi-session", "question", "gold", "2023-01-01", [], [])
    original_grade = {
        "question_id": "q",
        "inventory_sha256": "old-id",
        "reader_row_sha256": row_sha(old),
        "judge_prompt_version": "same-judge",
        "verdict": {"correct": False},
    }
    current = {
        "cohort": "new",
        "answer": {
            "text": "same answer",
            "notes": {
                "exact_request_replay": {
                    "source_reader_sha256": row_sha(old),
                    "judge_request_sha256": {"original": grade_request_sha(i, "same answer")},
                    "judge_prompt_version": "same-judge",
                }
            },
        },
    }
    result = cached_grade(i, current, old, original_grade, "new-id")
    assert result["verdict"]["correct"] is False
    assert result["exact_grade_replay"] and result["inventory_sha256"] == "new-id"
