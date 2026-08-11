from __future__ import annotations

import csv

from chronomem.evaluation.agreement import score_worksheet, write_worksheet
from chronomem.evaluation.harness import QuestionResult


def result(qid: str, correct: bool) -> QuestionResult:
    return QuestionResult(
        question_id=qid,
        question_type="temporal-reasoning",
        is_abstention=False,
        correct=correct,
        hypothesis="PyTorch",
        gold="PyTorch",
        judge_reason="match",
        context_tokens=100,
        prompt_tokens=120,
        output_tokens=10,
        latency_ms=900.0,
    )


def test_worksheet_hides_the_judge_verdict(tmp_path):
    """Showing it would anchor the labeler, and the resulting agreement number
    would measure nothing."""
    path = write_worksheet([result("q1", True)], {"q1": "which framework?"}, tmp_path / "w.csv")
    rows = list(csv.DictReader(path.open()))

    assert rows[0]["human"] == ""
    assert "correct" not in rows[0]
    assert "judge_reason" not in rows[0]
    assert rows[0]["question"] == "which framework?"
    assert rows[0]["gold"] == "PyTorch"


def test_sampling_is_capped_and_deterministic(tmp_path):
    results = [result(f"q{i}", i % 2 == 0) for i in range(100)]
    a = write_worksheet(results, {}, tmp_path / "a.csv", n=10, seed=3)
    b = write_worksheet(results, {}, tmp_path / "b.csv", n=10, seed=3)

    ids_a = [r["question_id"] for r in csv.DictReader(a.open())]
    ids_b = [r["question_id"] for r in csv.DictReader(b.open())]
    assert len(ids_a) == 10
    assert ids_a == ids_b


def test_agreement_splits_lenient_from_strict(tmp_path):
    results = [result("q1", True), result("q2", True), result("q3", False), result("q4", False)]
    sheet = tmp_path / "w.csv"
    write_worksheet(results, {}, sheet)

    rows = list(csv.DictReader(sheet.open()))
    labels = {"q1": "1", "q2": "0", "q3": "1", "q4": "0"}
    with sheet.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            row["human"] = labels[row["question_id"]]
            writer.writerow(row)

    agreement = score_worksheet(sheet, results)
    assert agreement.n == 4
    assert agreement.agree == 2
    assert agreement.judge_lenient == 1, "q2: judge said correct, human said wrong"
    assert agreement.judge_strict == 1, "q3: judge said wrong, human said correct"
    assert agreement.rate == 0.5


def test_unlabeled_rows_are_skipped_not_counted_as_agreement(tmp_path):
    results = [result("q1", True), result("q2", True)]
    sheet = tmp_path / "w.csv"
    write_worksheet(results, {}, sheet)

    rows = list(csv.DictReader(sheet.open()))
    with sheet.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        rows[0]["human"] = "1"
        writer.writerows(rows)  # q2 left blank

    agreement = score_worksheet(sheet, results)
    assert agreement.n == 1, "a blank cell must not silently score as agreement"


def test_label_spellings_are_accepted(tmp_path):
    results = [result(f"q{i}", True) for i in range(4)]
    sheet = tmp_path / "w.csv"
    write_worksheet(results, {}, sheet)

    rows = list(csv.DictReader(sheet.open()))
    for row, label in zip(rows, ["yes", "TRUE", "n", "0"], strict=True):
        row["human"] = label
    with sheet.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    agreement = score_worksheet(sheet, results)
    assert agreement.n == 4
    assert agreement.agree == 2


def test_low_agreement_is_flagged_in_the_summary():
    from chronomem.evaluation.agreement import Agreement

    assert (
        "⚠️"
        in Agreement(n=50, agree=40, judge_lenient=8, judge_strict=2, disagreements=[]).summary()
    )
    assert (
        "⚠️"
        not in Agreement(
            n=50, agree=48, judge_lenient=1, judge_strict=1, disagreements=[]
        ).summary()
    )
