import csv

from chronomem.evaluation.failures import summarize_failure_worksheet, write_failure_worksheet
from chronomem.evaluation.harness import QuestionResult


def result(correct: bool = False) -> QuestionResult:
    return QuestionResult(
        question_id="q1",
        question_type="temporal-reasoning",
        is_abstention=False,
        correct=correct,
        hypothesis="17",
        gold="three months",
        judge_reason="wrong duration",
        context_tokens=10,
        prompt_tokens=20,
        output_tokens=2,
        latency_ms=100.0,
        source_session_recalled=True,
        notes={"retrieval": [{"memory_id": "m1"}]},
    )


def test_failure_worksheet_contains_only_judged_wrong_results(tmp_path):
    path = write_failure_worksheet(
        [result(False), result(True)], {"q1": "How long?"}, tmp_path / "audit.csv"
    )
    rows = list(csv.DictReader(path.open()))

    assert len(rows) == 1
    assert rows[0]["source_session_recalled"] == "True"
    assert rows[0]["primary_failure"] == ""


def test_failure_summary_counts_one_primary_cause_and_e1_detail(tmp_path):
    path = write_failure_worksheet([result()], {"q1": "How long?"}, tmp_path / "audit.csv")
    rows = list(csv.DictReader(path.open()))
    rows[0]["primary_failure"] = "E1"
    rows[0]["extraction_detail"] = "duration"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report = summarize_failure_worksheet(path)

    assert report.by_code == {"E1": 1}
    assert report.extraction_details == {"duration": 1}
    assert report.unclassified_ids == []


def test_invalid_or_missing_labels_remain_visible(tmp_path):
    path = write_failure_worksheet([result()], {}, tmp_path / "audit.csv")

    report = summarize_failure_worksheet(path)

    assert report.classified == 0
    assert report.unclassified_ids == ["q1"]
