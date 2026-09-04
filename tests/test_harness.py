"""Evaluation driver: resume semantics and artifact integrity.

The published table is regenerated from the JSONL, so the file is the record and the
console summary is a convenience. These tests exist because the two silently
diverged: a run printed "50 questions, 56%" against a file holding 30 results.
"""

from __future__ import annotations

import pytest

from llm_long_term_memory.evaluation.datasets.longmemeval import Instance
from llm_long_term_memory.evaluation.harness import (
    ArtifactMismatch,
    StoreChanged,
    UsageArtifactMismatch,
    run_eval,
)
from llm_long_term_memory.evaluation.runners.base import Answer
from llm_long_term_memory.llm.usage import CallRecord, UsageTracker


def instance(qid: str, qtype: str = "single-session-user") -> Instance:
    return Instance(
        question_id=qid,
        question_type=qtype,
        question="q?",
        answer="Sydney",
        question_date="2026-01-01",
        sessions=[],
        answer_session_ids=[],
    )


class StubRunner:
    name = "stub"

    def __init__(self):
        self.prepared = []

    def prepare(self, inst):
        self.prepared.append(inst.question_id)

    def answer(self, inst):
        return Answer(text="Sydney", context_tokens=10, prompt_tokens=12, output_tokens=2)


class VersionedStubRunner(StubRunner):
    answer_prompt_version = "memory-reasoned-v3"


class StubJudge:
    def __init__(self, correct=True):
        self.correct = correct
        self.calls = 0
        self.graded_types: list[str | None] = []

    def grade(self, question, gold, hypothesis, is_abstention=False, question_type=None):
        from llm_long_term_memory.evaluation.judge import JudgeResult

        self.calls += 1
        self.graded_types.append(question_type)
        return JudgeResult(self.correct, "ok", 5, 2)


def test_results_are_written_as_they_complete(tmp_path):
    out = tmp_path / "r.jsonl"
    report = run_eval(StubRunner(), StubJudge(), [instance(f"q{i}") for i in range(5)], out)

    assert report.n == 5
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 5


def test_result_records_the_runners_actual_answer_prompt_version(tmp_path):
    report = run_eval(
        VersionedStubRunner(), StubJudge(), [instance("q1")], tmp_path / "reasoned.jsonl"
    )

    assert report.results[0].answer_prompt_version == "memory-reasoned-v3"


def test_the_question_type_reaches_the_judge(tmp_path):
    """The judge routes on it: preference questions are graded against a rubric
    rather than a reference answer. If the harness drops the type, that routing
    silently never fires."""
    judge = StubJudge()
    run_eval(
        StubRunner(),
        judge,
        [instance("q1", "single-session-preference"), instance("q2", "temporal-reasoning")],
        tmp_path / "r.jsonl",
    )

    assert judge.graded_types == ["single-session-preference", "temporal-reasoning"]


def test_resume_skips_completed_questions(tmp_path):
    out = tmp_path / "r.jsonl"
    instances = [instance(f"q{i}") for i in range(5)]
    run_eval(StubRunner(), StubJudge(), instances[:3], out)

    judge = StubJudge()
    report = run_eval(StubRunner(), judge, instances, out)

    assert judge.calls == 2, "only the two unfinished questions"
    assert report.n == 5


def test_usage_resume_loads_prior_calls_without_double_counting(tmp_path):
    out = tmp_path / "r.jsonl"

    class AccountingRunner(StubRunner):
        def __init__(self, usage):
            super().__init__()
            self.usage = usage

        def answer(self, inst):
            self.usage.record(CallRecord("answerer", "m", 10, 2, 1.0))
            return super().answer(inst)

    class AccountingJudge(StubJudge):
        def __init__(self, usage):
            super().__init__()
            self.usage = usage

        def grade(self, *args, **kwargs):
            self.usage.record(CallRecord("judge", "m", 10, 2, 1.0))
            return super().grade(*args, **kwargs)

    instances = [instance(f"q{i}") for i in range(3)]
    first = UsageTracker()
    run_eval(AccountingRunner(first), AccountingJudge(first), instances[:2], out, usage=first)
    resumed = UsageTracker()
    run_eval(AccountingRunner(resumed), AccountingJudge(resumed), instances, out, usage=resumed)

    saved = UsageTracker._load_records(out.with_suffix(".usage.json"))
    assert len(saved) == 6
    assert [record.role for record in saved].count("answerer") == 3
    assert [record.role for record in saved].count("judge") == 3


def test_rows_without_usage_are_refused_when_cost_accounting_is_requested(tmp_path):
    out = tmp_path / "r.jsonl"
    run_eval(StubRunner(), StubJudge(), [instance("q1")], out)

    with pytest.raises(UsageArtifactMismatch, match="undercount cost"):
        run_eval(
            StubRunner(),
            StubJudge(),
            [instance("q1"), instance("q2")],
            out,
            usage=UsageTracker(),
        )


def test_usage_is_checkpointed_when_a_metered_question_raises(tmp_path):
    out = tmp_path / "r.jsonl"
    usage = UsageTracker()

    class MeteredFailure(StubRunner):
        def answer(self, inst):
            usage.record(CallRecord("answerer", "m", 10, 0, 1.0, ok=False, error="boom"))
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_eval(MeteredFailure(), StubJudge(), [instance("q1")], out, usage=usage)

    saved = UsageTracker._load_records(out.with_suffix(".usage.json"))
    assert len(saved) == 1
    assert not saved[0].ok


class StoreRunner(StubRunner):
    """A runner that reads a prebuilt store, and so has a store to pin."""

    def __init__(self, fingerprint: str):
        super().__init__()
        self.store_fingerprint = fingerprint


def test_resume_refuses_answers_from_a_different_store(tmp_path):
    """The pilot/formal merge. A 31-question pilot ran against a store at 63% ingest;
    the formal 50-question run uses the finished store. Resuming would reuse the
    thirty-one — answered from evidence that was not yet there — and report them
    beside nineteen answered from the complete store as a single accuracy."""
    out = tmp_path / "r.jsonl"
    instances = [instance(f"q{i}") for i in range(5)]
    run_eval(StoreRunner("v4@1521"), StubJudge(), instances[:3], out)

    with pytest.raises(StoreChanged, match="different stores"):
        run_eval(StoreRunner("v4@2400"), StubJudge(), instances, out)

    # The same store still resumes: this must not break the case resume exists for,
    # a run that stopped on quota and continues tomorrow.
    report = run_eval(StoreRunner("v4@1521"), StubJudge(), instances, out)
    assert report.n == 5


def test_resume_refuses_rows_that_predate_store_stamping(tmp_path):
    """Unstamped rows are of unknown provenance, not of matching provenance — and
    every row of the real pilot artifact is unstamped, so treating None as "probably
    the same" would let through the one merge this guard was written for."""
    out = tmp_path / "r.jsonl"
    instances = [instance(f"q{i}") for i in range(3)]
    run_eval(StubRunner(), StubJudge(), instances[:2], out)  # no fingerprint written

    with pytest.raises(StoreChanged, match="unknown store"):
        run_eval(StoreRunner("v4@2400"), StubJudge(), instances, out)

    # --fresh remains the escape hatch, and it truncates.
    report = run_eval(StoreRunner("v4@2400"), StubJudge(), instances, out, resume=False)
    assert report.n == 3


def test_fresh_truncates_rather_than_appending(tmp_path):
    """Appending would silently merge two runs into one report — and the reason to
    re-run is usually that the judge or a prompt changed, so the merged rows are
    graded under different rules. Observed as n=56 and n=62 on a 50-question set."""
    out = tmp_path / "r.jsonl"
    instances = [instance(f"q{i}") for i in range(4)]

    run_eval(StubRunner(), StubJudge(correct=True), instances, out)
    report = run_eval(StubRunner(), StubJudge(correct=False), instances, out, resume=False)

    assert report.n == 4, "not 8"
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 4
    assert report.accuracy == 0.0, "the second run's grades, not a blend of both"


def test_a_diverged_artifact_raises_instead_of_reporting_a_number(tmp_path):
    """A concurrent writer, a truncation, or a lost write must not surface as a
    plausible-looking accuracy."""
    out = tmp_path / "r.jsonl"
    instances = [instance(f"q{i}") for i in range(3)]

    class SabotagingJudge(StubJudge):
        def grade(self, *a, **kw):
            result = super().grade(*a, **kw)
            if self.calls == 3:
                out.write_text(
                    "", encoding="utf-8"
                )  # stand-in for another process rewriting the file
            return result

    with pytest.raises(ArtifactMismatch, match=r"holds \d+ results but the run reported 3"):
        run_eval(StubRunner(), SabotagingJudge(), instances, out)


def test_report_regenerates_from_the_file_alone(tmp_path):
    """`eval report` reads only the artifacts, so a table can be rebuilt without
    re-spending quota."""
    from llm_long_term_memory.evaluation.report import load_report

    out = tmp_path / "naive_rag.jsonl"
    run_eval(StubRunner(), StubJudge(correct=True), [instance(f"q{i}") for i in range(6)], out)

    loaded = load_report(out, variant="naive_rag")
    assert loaded.n == 6
    assert loaded.accuracy == 1.0


def test_default_report_excludes_diagnostic_jsonl_artifacts(tmp_path):
    from llm_long_term_memory.evaluation.report import default_report_variants

    for name in (
        "chronomem",
        "full_context",
        "naive_rag.rep1",
        "chronomem_relevance_pack_250",
        "influence-chronomem",
    ):
        (tmp_path / f"{name}.jsonl").touch()

    assert default_report_variants(tmp_path) == ["full_context", "chronomem"]


def test_partial_final_line_from_a_killed_process_is_skipped(tmp_path):
    from llm_long_term_memory.evaluation.report import load_report

    out = tmp_path / "r.jsonl"
    run_eval(StubRunner(), StubJudge(), [instance(f"q{i}") for i in range(3)], out)
    with out.open("a", encoding="utf-8") as fh:
        fh.write('{"question_id": "q9", "corr')  # killed mid-write

    assert load_report(out).n == 3


def test_a_second_run_on_the_same_file_is_refused(tmp_path):
    """Two evaluations sharing an output path interleave appends, and if either used
    --fresh one truncates the other's finished work. Seen live: a 50-question file
    dropped to 34 and a second file was deleted mid-run."""
    import os

    from llm_long_term_memory.evaluation.harness import RunAlreadyInProgress

    out = tmp_path / "r.jsonl"
    lock = out.with_suffix(out.suffix + ".lock")
    lock.write_text(str(os.getpid()), encoding="utf-8")  # a live owner: this process

    with pytest.raises(RunAlreadyInProgress, match="already running this evaluation"):
        run_eval(StubRunner(), StubJudge(), [instance("q1")], out)


def test_a_lock_left_by_a_dead_process_is_reclaimed(tmp_path):
    """A crash during a quota-limited run is the normal case, so a lock outliving
    its owner must not block the resume path forever."""
    out = tmp_path / "r.jsonl"
    lock = out.with_suffix(out.suffix + ".lock")
    lock.write_text("999999", encoding="utf-8")  # a pid that does not exist

    report = run_eval(StubRunner(), StubJudge(), [instance("q1")], out)
    assert report.n == 1
    assert not lock.exists(), "released on the way out"


def test_liveness_check_does_not_signal_the_process_it_asks_about():
    """`os.kill(pid, 0)` reads as an existence check and is not one on Windows, where
    signal 0 is CTRL_C_EVENT: it delivers a real interrupt to the console group. The
    lock check would then have sent Ctrl+C to the very evaluation it exists to leave
    running. Asking about our own pid is the sharpest version of the question — if
    the check signals anything, this test takes the hit."""
    import os

    from llm_long_term_memory.locking import process_alive as _process_alive

    assert _process_alive(os.getpid()) is True
    assert _process_alive(999999) is False


def test_the_lock_is_released_even_when_the_run_raises(tmp_path):
    out = tmp_path / "r.jsonl"

    class Exploding(StubRunner):
        def answer(self, inst):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_eval(Exploding(), StubJudge(), [instance("q1")], out)

    assert not out.with_suffix(out.suffix + ".lock").exists()
