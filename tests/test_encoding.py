"""Non-ASCII must survive every artifact round-trip, on every platform.

The CI matrix runs the Windows job with ``PYTHONUTF8=0``, so there the locale
codec is cp1252 rather than UTF-8. Without an explicit ``encoding=`` at each file
I/O site these writes raise ``UnicodeEncodeError`` or silently mojibake — and the
rest of the suite would not notice, because its fixtures are pure ASCII.

The corpus is not ASCII: LongMemEval questions and answers contain accented names,
curly quotes, and CJK. An evaluation artifact that cannot round-trip them is a
results-integrity bug, not a cosmetic one, so this is a gate in the same sense the
coverage and fidelity gates are.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict

from llm_long_term_memory.evaluation.agreement import score_worksheet, write_worksheet
from llm_long_term_memory.evaluation.failures import (
    summarize_failure_worksheet,
    write_failure_worksheet,
)
from llm_long_term_memory.evaluation.harness import QuestionResult
from llm_long_term_memory.evaluation.report import load_report
from llm_long_term_memory.llm.usage import CallRecord, UsageTracker

# An accented name, a CJK title, a curly apostrophe, and an em dash: all four
# appear in the benchmark corpus and all four are unrepresentable in cp1252 or
# corrupted by it.
UNICODE_QUESTION = "Où ai-je acheté 《红楼梦》 — l'édition de 1978?"
UNICODE_GOLD = "à Paris, chez Gibert Jeune"
UNICODE_HYPOTHESIS = "我不知道 — the memory says “bookshop” without naming it"


def result(question_id: str = "q1", correct: bool = False) -> QuestionResult:
    return QuestionResult(
        question_id=question_id,
        question_type="single-session-user",
        is_abstention=False,
        correct=correct,
        hypothesis=UNICODE_HYPOTHESIS,
        gold=UNICODE_GOLD,
        judge_reason="réponse incomplète",
        context_tokens=10,
        prompt_tokens=20,
        output_tokens=2,
        latency_ms=100.0,
        source_session_recalled=True,
        notes={"retrieval": [{"memory_id": "m1", "content": UNICODE_GOLD}]},
    )


def test_failure_worksheet_round_trips_non_ascii(tmp_path):
    """`retrieval_diagnostics` is dumped with ensure_ascii=False, so this column
    carries raw non-ASCII into the CSV even when every other JSON write escapes."""
    path = write_failure_worksheet([result()], {"q1": UNICODE_QUESTION}, tmp_path / "audit.csv")

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))

    assert rows[0]["question"] == UNICODE_QUESTION
    assert rows[0]["gold"] == UNICODE_GOLD
    assert rows[0]["hypothesis"] == UNICODE_HYPOTHESIS
    assert UNICODE_GOLD in rows[0]["retrieval_diagnostics"]

    # And the reader half must agree with the writer half.
    assert summarize_failure_worksheet(path).total == 1


def test_agreement_worksheet_round_trips_non_ascii(tmp_path):
    path = write_worksheet([result()], {"q1": UNICODE_QUESTION}, tmp_path / "labels.csv")

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert rows[0]["question"] == UNICODE_QUESTION
    assert rows[0]["hypothesis"] == UNICODE_HYPOTHESIS

    rows[0]["human"] = "n"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    agreement = score_worksheet(path, [result()])
    assert agreement.n == 1


def test_results_jsonl_round_trips_non_ascii(tmp_path):
    """The JSONL is the evidence behind every published number; `eval report`
    regenerates the table from it rather than from console output."""
    path = tmp_path / "raw.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in (result("q1"), result("q2", correct=True)):
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    report = load_report(path, variant="two_stage_hydrated")

    assert len(report.results) == 2
    assert report.results[0].gold == UNICODE_GOLD
    assert report.results[0].hypothesis == UNICODE_HYPOTHESIS


def test_usage_artifact_round_trips_non_ascii(tmp_path):
    """A provider error message is the one free-text field in the usage artifact,
    and provider errors are not guaranteed to be ASCII."""
    path = tmp_path / "usage.json"
    UsageTracker(
        records=[
            CallRecord(
                role="extractor",
                model="gemini-3.1-flash-lite",
                input_tokens=10,
                output_tokens=5,
                latency_ms=1.0,
                ok=False,
                error=f"429 — quota épuisée sur 《{UNICODE_QUESTION}》",
            )
        ]
    ).save(path)

    reloaded = UsageTracker(records=UsageTracker._load_records(path))
    assert reloaded.records[0].error is not None
    assert UNICODE_QUESTION in reloaded.records[0].error
