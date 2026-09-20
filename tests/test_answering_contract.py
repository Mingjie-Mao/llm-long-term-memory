"""The answering contract belongs to the product, and must not move by accident.

`ANSWER_SYSTEM` and `ANSWER_PROMPT_VERSION` are stamped on every recorded result row,
because an accuracy figure is meaningless six months later if nobody can tell which
prompt produced it. They used to live in `evaluation/runners/base.py`, so the running
product's answering behaviour could only be read inside the harness that measures it.
"""

from __future__ import annotations

from llm_long_term_memory.answering import (
    ANSWER_PROMPT_VERSION,
    ANSWER_SYSTEM,
    Answer,
    AnswerVerdict,
)


def test_the_harness_still_sees_the_contract_at_its_old_import_path():
    from llm_long_term_memory.evaluation.runners import base

    assert base.ANSWER_SYSTEM is ANSWER_SYSTEM
    assert base.ANSWER_PROMPT_VERSION is ANSWER_PROMPT_VERSION
    assert base.Answer is Answer
    assert base.AnswerVerdict is AnswerVerdict


def test_the_version_stamped_on_every_row_is_the_one_this_prompt_was_registered_as():
    # Bumping the prompt without bumping this makes two different prompts report the
    # same identity, which is how a results table stops being comparable with itself.
    assert ANSWER_PROMPT_VERSION == "memory-aware-v2"
    assert "do not refuse merely because no memory" in ANSWER_SYSTEM
    assert "say you do not know rather than guessing" in ANSWER_SYSTEM


def test_the_verdict_offers_exactly_the_three_dispositions_the_fallback_routes_on():
    statuses = AnswerVerdict.model_fields["status"].annotation
    assert set(statuses.__args__) == {"answer", "need_source", "no_evidence"}


def test_the_product_answering_module_does_not_import_the_benchmark():
    import inspect

    from llm_long_term_memory import answering

    source = inspect.getsource(answering)
    assert "longmemeval" not in source
    assert "evaluation" not in source.split('"""', 2)[2]
