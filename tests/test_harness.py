"""Evaluation driver: resume semantics and artifact integrity.

The published table is regenerated from the JSONL, so the file is the record and the
console summary is a convenience. These tests exist because the two silently
diverged: a run printed "50 questions, 56%" against a file holding 30 results.
"""

from __future__ import annotations

import pytest

from chronomem.evaluation.datasets.longmemeval import Instance
from chronomem.evaluation.harness import ArtifactMismatch, run_eval
from chronomem.evaluation.runners.base import Answer


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


class StubJudge:
    def __init__(self, correct=True):
        self.correct = correct
        self.calls = 0

    def grade(self, question, gold, hypothesis, is_abstention=False):
        from chronomem.evaluation.judge import JudgeResult

        self.calls += 1
        return JudgeResult(self.correct, "ok", 5, 2)


def test_results_are_written_as_they_complete(tmp_path):
    out = tmp_path / "r.jsonl"
    report = run_eval(StubRunner(), StubJudge(), [instance(f"q{i}") for i in range(5)], out)

    assert report.n == 5
    assert len(out.read_text().strip().splitlines()) == 5


def test_resume_skips_completed_questions(tmp_path):
    out = tmp_path / "r.jsonl"
    instances = [instance(f"q{i}") for i in range(5)]
    run_eval(StubRunner(), StubJudge(), instances[:3], out)

    judge = StubJudge()
    report = run_eval(StubRunner(), judge, instances, out)

    assert judge.calls == 2, "only the two unfinished questions"
    assert report.n == 5


def test_fresh_truncates_rather_than_appending(tmp_path):
    """Appending would silently merge two runs into one report — and the reason to
    re-run is usually that the judge or a prompt changed, so the merged rows are
    graded under different rules. Observed as n=56 and n=62 on a 50-question set."""
    out = tmp_path / "r.jsonl"
    instances = [instance(f"q{i}") for i in range(4)]

    run_eval(StubRunner(), StubJudge(correct=True), instances, out)
    report = run_eval(StubRunner(), StubJudge(correct=False), instances, out, resume=False)

    assert report.n == 4, "not 8"
    assert len(out.read_text().strip().splitlines()) == 4
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
                out.write_text("")  # stand-in for another process rewriting the file
            return result

    with pytest.raises(ArtifactMismatch, match=r"holds \d+ results but the run reported 3"):
        run_eval(StubRunner(), SabotagingJudge(), instances, out)


def test_report_regenerates_from_the_file_alone(tmp_path):
    """`eval report` reads only the artifacts, so a table can be rebuilt without
    re-spending quota."""
    from chronomem.evaluation.report import load_report

    out = tmp_path / "naive_rag.jsonl"
    run_eval(StubRunner(), StubJudge(correct=True), [instance(f"q{i}") for i in range(6)], out)

    loaded = load_report(out, variant="naive_rag")
    assert loaded.n == 6
    assert loaded.accuracy == 1.0


def test_partial_final_line_from_a_killed_process_is_skipped(tmp_path):
    from chronomem.evaluation.report import load_report

    out = tmp_path / "r.jsonl"
    run_eval(StubRunner(), StubJudge(), [instance(f"q{i}") for i in range(3)], out)
    with out.open("a") as fh:
        fh.write('{"question_id": "q9", "corr')  # killed mid-write

    assert load_report(out).n == 3


def test_a_second_run_on_the_same_file_is_refused(tmp_path):
    """Two evaluations sharing an output path interleave appends, and if either used
    --fresh one truncates the other's finished work. Seen live: a 50-question file
    dropped to 34 and a second file was deleted mid-run."""
    import os

    from chronomem.evaluation.harness import RunAlreadyInProgress

    out = tmp_path / "r.jsonl"
    lock = out.with_suffix(out.suffix + ".lock")
    lock.write_text(str(os.getpid()))  # a live owner: this process

    with pytest.raises(RunAlreadyInProgress, match="already writing"):
        run_eval(StubRunner(), StubJudge(), [instance("q1")], out)


def test_a_lock_left_by_a_dead_process_is_reclaimed(tmp_path):
    """A crash during a quota-limited run is the normal case, so a lock outliving
    its owner must not block the resume path forever."""
    out = tmp_path / "r.jsonl"
    lock = out.with_suffix(out.suffix + ".lock")
    lock.write_text("999999")  # a pid that does not exist

    report = run_eval(StubRunner(), StubJudge(), [instance("q1")], out)
    assert report.n == 1
    assert not lock.exists(), "released on the way out"


def test_the_lock_is_released_even_when_the_run_raises(tmp_path):
    out = tmp_path / "r.jsonl"

    class Exploding(StubRunner):
        def answer(self, inst):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_eval(Exploding(), StubJudge(), [instance("q1")], out)

    assert not out.with_suffix(out.suffix + ".lock").exists()
