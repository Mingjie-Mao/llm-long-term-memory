"""The BEAM rubric judge, against the ways a score can move without anyone judging.

A missing grade scored as zero, or a duplicate averaged in, changes the number while
looking like a judgement; a score off BEAM's scale is not on it; an answer compared with
the reference text gets graded on wording; and a harness that hands rubric questions to
the reference-answer judge grades every row against the wrong thing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_long_term_memory.evaluation.beam_judge import (
    BEAM_JUDGE_PROMPT_VERSION,
    BeamRubricJudge,
    InvalidJudgement,
    RubricVerdict,
    kendall_tau_b,
    rescore,
)
from llm_long_term_memory.evaluation.datasets.longmemeval import Instance
from llm_long_term_memory.evaluation.harness import run_eval
from llm_long_term_memory.evaluation.judge import Judge
from llm_long_term_memory.evaluation.runners.base import Answer


class FakeClient:
    def __init__(self, grades):
        self.grades = grades
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            text=json.dumps({"grades": self.grades}),
            input_tokens=40,
            output_tokens=8,
            thinking_tokens=2,
            api_latency_ms=3.0,
        )


def grade(item, score, position=None):
    return {"item": item, "score": score, "reason": "r", "position": position}


def test_every_item_is_graded_in_one_call_and_averaged():
    client = FakeClient([grade(1, 1.0), grade(2, 0.5), grade(3, 0.0)])
    result = BeamRubricJudge(client, "judge").grade(
        "When did it happen?",
        "REFERENCE-TEXT",
        "an answer",
        question_type="information_extraction",
        rubric=("first", "second", "third"),
    )
    (call,) = client.calls
    assert call["schema"] is RubricVerdict
    assert "1. first\n2. second\n3. third" in call["prompt"]
    assert "When did it happen?" in call["prompt"]
    assert "REFERENCE-TEXT" not in call["prompt"]
    assert "position" not in call["prompt"]
    assert (result.score, result.correct, result.reason) == (0.5, True, "1.5 of 3 rubric points")
    assert result.output_tokens == 10


def test_saved_grades_recompute_the_verdict_without_a_call():
    client = FakeClient([grade(1, 1.0), grade(2, 0.0)])
    result = BeamRubricJudge(client, "judge").grade("q", "", "a", rubric=("x", "y"))
    assert rescore(result.details) == {"score": 0.5, "correct": True}
    assert rescore(result.details, correct_at=0.75)["correct"] is False
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "grades",
    [
        [grade(1, 1.0)],
        [grade(1, 1.0), grade(1, 0.0)],
        [grade(1, 1.0), grade(3, 1.0)],
    ],
)
def test_grades_that_miss_or_repeat_an_item_are_refused(grades):
    with pytest.raises(InvalidJudgement):
        BeamRubricJudge(FakeClient(grades), "judge").grade("q", "", "a", rubric=("x", "y"))


def test_a_score_off_beams_scale_is_refused():
    with pytest.raises(InvalidJudgement, match="not one of"):
        BeamRubricJudge(FakeClient([grade(1, 0.7)]), "judge").grade("q", "", "a", rubric=("x",))


def test_an_empty_answer_scores_zero_without_a_call():
    client = FakeClient([])
    result = BeamRubricJudge(client, "judge").grade("q", "", "   ", rubric=("x", "y"))
    assert (client.calls, result.score, result.correct) == ([], 0.0, False)


def test_a_question_without_a_rubric_is_refused():
    with pytest.raises(ValueError, match="rubric"):
        BeamRubricJudge(FakeClient([]), "judge").grade("q", "", "a")


@pytest.mark.parametrize(
    ("positions", "expected"),
    [((1, 2, 3), 1.0), ((3, 2, 1), 0.0), ((None, None, None), 0.0)],
)
def test_event_order_is_scored_from_the_positions_the_judge_reports(positions, expected):
    client = FakeClient([grade(n, 1.0, p) for n, p in enumerate(positions, start=1)])
    result = BeamRubricJudge(client, "judge").grade(
        "q", "", "a", question_type="event_ordering", rubric=("a", "b", "c")
    )
    assert "position" in client.calls[0]["prompt"]
    assert result.details["order_tau_norm"] == pytest.approx(expected)


def test_kendall_tau_b_matches_known_values():
    assert kendall_tau_b([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert kendall_tau_b([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert kendall_tau_b([1, 2, 3], [1, 1, 2]) == pytest.approx(2 / 6**0.5)
    assert kendall_tau_b([1, 2, 3], [4, 4, 4]) is None


def rubric_instance():
    return Instance(
        question_id="beam-100K-1-abstention-0",
        question_type="abstention",
        question="q?",
        answer="declines",
        question_date="2024/03/10 (Sun) 00:00",
        sessions=[],
        answer_session_ids=[],
        namespace="beam-100K-1",
        rubric=("declines",),
    )


class StubRunner:
    name = "stub"

    def __init__(self):
        self.prepared = []

    def prepare(self, inst):
        self.prepared.append(inst.question_id)

    def answer(self, inst):
        return Answer(text="I do not know.", context_tokens=1)


def test_the_harness_refuses_rubric_questions_for_a_reference_answer_judge(tmp_path):
    runner, out = StubRunner(), tmp_path / "rows.jsonl"
    with pytest.raises(ValueError, match="graded by rubric"):
        run_eval(runner, Judge(FakeClient([]), model="judge"), [rubric_instance()], out)
    assert runner.prepared == []
    assert not out.exists()


def test_rubric_rows_carry_the_judges_version_and_every_grade(tmp_path):
    out = tmp_path / "rows.jsonl"
    judge = BeamRubricJudge(FakeClient([grade(1, 1.0)]), "judge")
    report = run_eval(StubRunner(), judge, [rubric_instance()], out)
    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["judge_prompt_version"] == BEAM_JUDGE_PROMPT_VERSION
    assert rescore(row["notes"]["judge"]) == {"score": 1.0, "correct": True}
    assert report.results[0].correct is True
