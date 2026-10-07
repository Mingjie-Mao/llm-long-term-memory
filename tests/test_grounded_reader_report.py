import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import grounded_reader_report as report


def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "REPO", tmp_path)
    inventory = tmp_path / "inventory.json"
    inventory.write_text("{}", encoding="utf-8")
    row = {
        "question_id": "q",
        "cohort": "reasoning-errors",
        "inventory_sha256": report.sha256_file(inventory),
        "answer": {"text": "5 sessions", "notes": {"grounded_calls": []}},
    }
    answers = tmp_path / "answers.jsonl"
    answers.write_text(json.dumps(row) + "\n", encoding="utf-8")
    usage = tmp_path / "usage.json"
    usage.write_text(json.dumps({"summary": {"total_requests": 2}}), encoding="utf-8")
    grades = tmp_path / "grades.jsonl"
    return row, answers, grades, inventory, usage


def test_ungraded_reader_report_preserves_unknown_correctness(tmp_path, monkeypatch):
    _, *paths = artifacts(tmp_path, monkeypatch)
    result = report.summarize(*paths)
    assert not result["grading_complete_for_saved_readers"]
    assert result["rows"][0]["correct"] is None
    assert result["cohorts"]["reasoning-errors"]["graded"] == 0


def test_report_rejects_grade_for_a_modified_saved_answer(tmp_path, monkeypatch):
    row, answers, grades, inventory, usage = artifacts(tmp_path, monkeypatch)
    grade = {
        "question_id": "q",
        "cohort": row["cohort"],
        "inventory_sha256": row["inventory_sha256"],
        "reader_row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
        "verdict": {"correct": True, "reason": "Same quantity"},
    }
    grades.write_text(json.dumps(grade) + "\n", encoding="utf-8")
    assert report.summarize(answers, grades, inventory, usage)["rows"][0]["correct"]
    row["answer"]["text"] = "6 sessions"
    answers.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="grade does not match"):
        report.summarize(answers, grades, inventory, usage)
