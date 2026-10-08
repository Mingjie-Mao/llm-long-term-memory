import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from judge_request_cache import ExactRequestJudge, grade_kwargs, row_hash

from llm_long_term_memory.evaluation.judge import Judge, JudgeResult


def fixture(tmp_path, correct=True, name="a"):
    instance = SimpleNamespace(
        question_id="q",
        question="What?",
        answer="gold",
        is_abstention=False,
        question_type="single-session-user",
    )
    reader = {
        "question_id": "q",
        "cohort": "public",
        "inventory_sha256": "frozen",
        "answer": {"text": "saved"},
    }
    grade = {
        "question_id": "q",
        "cohort": "public",
        "inventory_sha256": "frozen",
        "reader_row_sha256": row_hash(reader),
        "judge_prompt_version": Judge.prompt_version,
        "verdict": asdict(JudgeResult(correct, "original", 10, 3)),
    }
    rp, gp = tmp_path / f"{name}.readers.jsonl", tmp_path / f"{name}.grades.jsonl"
    rp.write_text(json.dumps(reader) + "\n", encoding="utf-8")
    gp.write_text(json.dumps(grade) + "\n", encoding="utf-8")
    return instance, reader, grade, rp, gp


def wrapper():
    judge = Judge(None, "fixed-model")
    calls = []
    judge.grade = lambda **kw: calls.append(kw) or JudgeResult(False, "fresh", 5, 1)
    return ExactRequestJudge(judge), calls


@pytest.mark.parametrize("correct", [True, False])
def test_exact_cache_reuses_false_as_well_as_true(tmp_path, correct):
    args = fixture(tmp_path, correct)
    cache, calls = wrapper()
    cache.add(*args)
    result = cache.grade(**grade_kwargs(args[0], args[1]))
    assert result.correct is correct and calls == []
    assert result.input_tokens == result.output_tokens == 0
    assert result.details["judge_cache"]["fresh_provider_call"] is False


@pytest.mark.parametrize(
    "field", ["gold", "hypothesis", "question", "is_abstention", "question_type"]
)
def test_different_request_does_not_hit(tmp_path, field):
    args = fixture(tmp_path)
    cache, calls = wrapper()
    cache.add(*args)
    kwargs = grade_kwargs(args[0], args[1])
    kwargs[field] = (
        True
        if field == "is_abstention"
        else "single-session-preference"
        if field == "question_type"
        else "different"
    )
    cache.grade(**kwargs)
    assert len(calls) == 1 and cache.hits == 0


@pytest.mark.parametrize("field", ["model", "thinking"])
def test_model_or_thinking_change_does_not_hit(tmp_path, field):
    args = fixture(tmp_path)
    cache, calls = wrapper()
    cache.add(*args)
    setattr(cache.judge, field, False if field == "thinking" else "other-model")
    cache.grade(**grade_kwargs(args[0], args[1]))
    assert len(calls) == 1


def test_disagreeing_historical_grades_are_not_cherry_picked(tmp_path):
    first, second = fixture(tmp_path, True, "a"), fixture(tmp_path, False, "b")
    cache, calls = wrapper()
    cache.add(*first)
    cache.add(*second)
    cache.grade(**grade_kwargs(first[0], first[1]))
    assert len(calls) == 1 and cache.hits == 0


def test_changed_saved_grade_stops_cache(tmp_path):
    args = fixture(tmp_path)
    cache, calls = wrapper()
    cache.add(*args)
    args[2]["verdict"]["correct"] = False
    args[4].write_text(json.dumps(args[2]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence changed"):
        cache.grade(**grade_kwargs(args[0], args[1]))
    assert calls == []


def test_foreign_reader_cannot_seed_cache(tmp_path):
    args = fixture(tmp_path)
    cache, _ = wrapper()
    args[1]["question_id"] = "foreign"
    with pytest.raises(ValueError, match="question mismatch"):
        cache.add(*args)
