import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prerequisite_grade_continuation import MISSING, verify_rows
from prerequisite_reader_gate import QIDS


def rows():
    readers = {
        qid: {
            "question_id": qid,
            "cohort": "public",
            "inventory_sha256": "fixed",
            "answer": "saved",
        }
        for qid in QIDS
    }
    grades = {
        qid: {
            "question_id": qid,
            "cohort": "public",
            "inventory_sha256": "fixed",
            "reader_row_sha256": hashlib.sha256(
                json.dumps(reader, sort_keys=True).encode()
            ).hexdigest(),
            "verdict": {"correct": True},
        }
        for qid, reader in readers.items()
        if qid != MISSING
    }
    return readers, grades


def test_only_the_one_missing_grade_is_eligible():
    readers, grades = rows()
    verify_rows(readers, grades)
    grades.pop(QIDS[0])
    with pytest.raises(ValueError, match="one missing"):
        verify_rows(readers, grades)


def test_saved_answer_cannot_be_changed_before_supplement():
    readers, grades = rows()
    readers[QIDS[0]]["answer"] = "changed"
    with pytest.raises(ValueError, match="exact original reader"):
        verify_rows(readers, grades)


def test_foreign_grade_is_rejected():
    readers, grades = rows()
    grades["foreign"] = grades[QIDS[0]]
    with pytest.raises(ValueError, match="foreign"):
        verify_rows(readers, grades)


def test_completed_grade_cannot_be_requested_again():
    readers, grades = rows()
    grades[MISSING] = grades[QIDS[0]]
    with pytest.raises(ValueError, match="one missing"):
        verify_rows(readers, grades)
