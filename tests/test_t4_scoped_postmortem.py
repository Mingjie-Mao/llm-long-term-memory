import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from t4_scoped_postmortem import request_audit


def fixture():
    runs = {
        a: [{"q": {"correct": a == "control"}} for _ in range(3)] for a in ("control", "candidate")
    }
    readers = {
        a: [
            {"q": {"answer": {"notes": {"actual_answer_calls": [{"request": {"prompt": a}}]}}}}
            for _ in range(3)
        ]
        for a in runs
    }
    return runs, readers, {"q": "lookup"}


def test_audit_separates_stable_regression_from_changed_request_without_claiming_causation():
    runs, readers, types = fixture()
    row = request_audit(runs, readers, types)[0]
    assert row["majority_regression"] and row["mean_delta"] == -1
    assert row["unique_nonempty_first_requests"] == {"control": 1, "candidate": 1}
    assert row["same_first_request_across_arms"] == [False] * 3
    assert row["first_loss"].startswith("not assigned")


def test_audit_rejects_missing_repeats_ids_and_untyped_verdicts():
    for change in ("repeat", "id", "verdict", "capture"):
        runs, readers, types = fixture()
        if change == "repeat":
            runs["candidate"].pop()
        elif change == "id":
            readers["candidate"][0] = {}
        elif change == "verdict":
            runs["candidate"][0]["q"]["correct"] = "False"
        else:
            readers["candidate"][0]["q"]["answer"]["notes"] = {}
        with pytest.raises(ValueError):
            request_audit(runs, readers, types)


def test_empty_requests_are_unknown_not_proof_of_baseline_equivalence():
    runs, readers, types = fixture()
    readers = copy.deepcopy(readers)
    for arm in readers:
        readers[arm][0]["q"]["answer"]["notes"]["actual_answer_calls"] = []
    row = request_audit(runs, readers, types)[0]
    assert row["same_first_request_across_arms"][0] is None
