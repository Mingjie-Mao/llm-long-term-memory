"""Every result row records which prompts and extractor produced it.

A score is only interpretable alongside the code that generated it. This project
has already changed the answerer prompt once in a way that invalidates comparison
with earlier runs; without a stamp on each row, "51.6%" becomes unattributable
within weeks.
"""

from __future__ import annotations

import json

from test_harness import StubJudge, StubRunner, instance

from llm_long_term_memory.evaluation.harness import QuestionResult, run_eval
from llm_long_term_memory.evaluation.judge import JUDGE_PROMPT_VERSION
from llm_long_term_memory.evaluation.runners.base import ANSWER_PROMPT_VERSION
from llm_long_term_memory.store import SQLiteMemoryStore


def test_versions_are_written_to_every_result_row(tmp_path):
    out = tmp_path / "r.jsonl"
    run_eval(StubRunner(), StubJudge(), [instance("q1"), instance("q2")], out)

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["answer_prompt_version"] == ANSWER_PROMPT_VERSION
        assert row["judge_prompt_version"] == JUDGE_PROMPT_VERSION


def test_the_extractor_version_comes_from_the_runner_not_the_code(tmp_path):
    """It describes the store being evaluated. A store built by last month's
    extractor must not be relabelled by today's checkout."""
    runner = StubRunner()
    runner.extractor_version = "two-stage-v1"
    out = tmp_path / "r.jsonl"

    run_eval(runner, StubJudge(), [instance("q1")], out)

    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["extractor_version"] == "two-stage-v1"


def test_a_runner_without_a_store_reports_no_extractor_version(tmp_path):
    """`full_context` and `naive_rag` read raw sessions; there is no extractor to
    name, and inventing one would be a lie."""
    out = tmp_path / "r.jsonl"
    run_eval(StubRunner(), StubJudge(), [instance("q1")], out)

    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["extractor_version"] is None


def test_rows_written_before_versioning_still_load(tmp_path):
    """The frozen v1 artifacts have none of these fields. They must keep loading, and
    must report None rather than being silently backfilled with today's version."""
    out = tmp_path / "old.jsonl"
    out.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question_type": "temporal-reasoning",
                "is_abstention": False,
                "correct": True,
                "hypothesis": "Sydney",
                "gold": "Sydney",
                "judge_reason": "ok",
                "context_tokens": 10,
                "prompt_tokens": 12,
                "output_tokens": 2,
                "latency_ms": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    from llm_long_term_memory.evaluation.report import load_report

    report = load_report(out, variant="chronomem")
    assert report.n == 1
    assert report.results[0].answer_prompt_version is None
    assert report.results[0].extractor_version is None


def test_ingestion_stamps_the_store_with_its_extractor(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()

    assert store.get_meta("extractor_version") is None
    store.set_meta("extractor_version", "two-stage-p10-v2")
    assert store.get_meta("extractor_version") == "two-stage-p10-v2"

    # Re-ingesting under a new extractor relabels rather than accumulating.
    store.set_meta("extractor_version", "two-stage-p11-v3")
    assert store.get_meta("extractor_version") == "two-stage-p11-v3"
    store.close()


def test_question_result_defaults_are_none_not_a_guess():
    result = QuestionResult(
        question_id="q",
        question_type="t",
        is_abstention=False,
        correct=True,
        hypothesis="h",
        gold="g",
        judge_reason="r",
        context_tokens=1,
        prompt_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
    )
    assert result.answer_prompt_version is None
    assert result.judge_prompt_version is None
    assert result.extractor_version is None
