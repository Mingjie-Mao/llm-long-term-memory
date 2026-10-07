import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_long_term_memory.answering import Answer
from llm_long_term_memory.conversation import AnswerRequest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from paired_manifest_scope import select_manifest
from run_paged_paired import CostBoundRunner, ScopedRecoveringCache


def test_scope_is_applied_before_page_or_cost_processing():
    all_rows = [SimpleNamespace(question_id=str(i)) for i in range(500)]
    selected = select_manifest(all_rows, [str(i) for i in range(100)])
    assert len(selected) == 100 and [r.question_id for r in selected] == [
        str(i) for i in range(100)
    ]


@pytest.mark.parametrize("ids", [["x", "x"], ["missing"]])
def test_duplicate_or_missing_manifest_ids_are_rejected(ids):
    with pytest.raises(ValueError):
        select_manifest([SimpleNamespace(question_id="x")], ids)


def test_whole_question_cost_includes_failed_page_and_reader_calls():
    usage = SimpleNamespace(records=[SimpleNamespace(ok=True, input_tokens=999, output_tokens=99)])

    def answer(request):
        usage.records.extend(
            [
                SimpleNamespace(ok=False, input_tokens=0, output_tokens=0),
                SimpleNamespace(ok=True, input_tokens=7000, output_tokens=200),
                SimpleNamespace(ok=True, input_tokens=6000, output_tokens=100),
            ]
        )
        return Answer("saved", 5800)

    runner = SimpleNamespace(
        answer_request=answer,
        answer_prompt_version="memory-grounded-v18",
        model="gemini-3.5-flash-lite",
    )
    result = CostBoundRunner(runner, "frozen", "candidate", usage).answer_request(
        AnswerRequest("count?", "2026-10-04", "alice")
    )
    assert result.notes["whole_question_usage"] == {
        "requests": 3,
        "failures": 1,
        "input_tokens": 13000,
        "output_tokens": 300,
    }
    assert result.notes["execution_identity"]["inventory_sha256"] == "frozen"


def test_cached_grade_does_not_reset_exhausted_provider_recovery(tmp_path):
    from judge_request_cache import request_key

    from llm_long_term_memory.evaluation.judge import Judge, JudgeResult

    judge = Judge(None, "fixed")
    kwargs = {"question": "q", "gold": "a", "hypothesis": "a"}
    key = request_key(judge, kwargs)
    cache = SimpleNamespace(
        judge=judge,
        prompt_version=judge.prompt_version,
        entries={key: [{"result": {"correct": False}}]},
        grade=lambda **kw: JudgeResult(False, "cached negative", 0, 0),
    )
    wrapper = ScopedRecoveringCache(
        cache, None, tmp_path / "usage", tmp_path / "recovery", "unchanged"
    )
    assert wrapper.grade(**kwargs).correct is False
    assert not (tmp_path / "recovery").exists()
