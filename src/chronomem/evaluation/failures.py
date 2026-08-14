"""Manual, single-cause failure taxonomy for evaluation results."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from .harness import QuestionResult

FAILURE_CODES = {
    "E1": "extraction loss",
    "E2": "retrieval miss",
    "E3": "temporal resolution error",
    "E4": "context assembly loss",
    "E5": "answer reasoning failure",
    "E6": "judge error",
}

_FIELDS = [
    "question_id",
    "question_type",
    "question",
    "gold",
    "hypothesis",
    "judge_reason",
    "source_session_recalled",
    "retrieval_diagnostics",
    "primary_failure",
    "extraction_detail",
    "review_notes",
]


@dataclass(slots=True)
class FailureReport:
    total: int = 0
    classified: int = 0
    by_code: dict[str, int] = field(default_factory=dict)
    extraction_details: dict[str, int] = field(default_factory=dict)
    unclassified_ids: list[str] = field(default_factory=list)


def write_failure_worksheet(
    results: list[QuestionResult], questions: dict[str, str], path: str | Path
) -> Path:
    """Write every judged-wrong result for one primary-cause review."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        for result in results:
            if result.correct:
                continue
            source_recalled = (
                result.source_session_recalled
                if result.source_session_recalled is not None
                else result.evidence_recalled
            )
            diagnostics = {
                key: result.notes.get(key)
                for key in (
                    "retrieval",
                    "hydrated_memory_ids",
                    "hydrated_tokens",
                    "hydration_missing_anchors",
                )
                if key in result.notes
            }
            writer.writerow(
                {
                    "question_id": result.question_id,
                    "question_type": result.question_type,
                    "question": questions.get(result.question_id, ""),
                    "gold": result.gold,
                    "hypothesis": result.hypothesis,
                    "judge_reason": result.judge_reason,
                    "source_session_recalled": source_recalled,
                    "retrieval_diagnostics": json.dumps(diagnostics, ensure_ascii=False),
                    "primary_failure": "",
                    "extraction_detail": "",
                    "review_notes": "",
                }
            )
    return destination


def summarize_failure_worksheet(path: str | Path) -> FailureReport:
    """Validate and count completed E1-E6 failure labels."""
    report = FailureReport()
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            report.total += 1
            code = row.get("primary_failure", "").strip().upper()
            if code not in FAILURE_CODES:
                report.unclassified_ids.append(row.get("question_id", ""))
                continue
            report.classified += 1
            report.by_code[code] = report.by_code.get(code, 0) + 1
            detail = row.get("extraction_detail", "").strip().lower()
            if code == "E1" and detail:
                report.extraction_details[detail] = report.extraction_details.get(detail, 0) + 1
    return report
