"""Object/fence leakage is guarded at every fallback return boundary and recorded.

This is an output-format check, not a proof that a prose answer is grounded.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.evaluation.runners.synthesis import (
    SYNTHESIS_ANSWER_SYSTEM,
    SynthesisVerdict,
)


class _Fake:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def generate(self, **kwargs):
        class _C:
            text = self.payload
            usage = None

        return _C()


def _runner(payload: str):
    from llm_long_term_memory.evaluation.runners.memory import MemoryRunner

    runner = MemoryRunner.__new__(MemoryRunner)
    runner.client = _Fake(payload)
    runner.model = "m"
    runner.answer_policy = "synthesis_v4"
    runner.max_output_tokens = 256
    runner.chars_per_token = 4.0
    runner.answer_system = SYNTHESIS_ANSWER_SYSTEM
    runner.verdict_schema = SynthesisVerdict
    runner.fallback = None
    runner._computation = None
    runner._answer_was_raw_structure = False
    return runner


def _request() -> AnswerRequest:
    return AnswerRequest(question="what is currently true?", asked_on="2026/01/01", user_id="q")


def test_an_empty_answer_falling_back_to_json_is_flagged():
    """The exact shape seen on the run: operation filled, answer empty."""
    payload = json.dumps({"status": "answer", "operation": "current_state", "answer": ""})
    runner = _runner(payload)
    _, text, _, _ = runner._answer_with_fallback(_request(), "ctx", [])
    assert runner._answer_was_raw_structure is True, "the defect must stay countable"
    # The structure itself no longer reaches the reader — see
    # test_raw_structure_never_reaches_the_reader — but the flag still records that the
    # model produced one.
    assert not text.lstrip().startswith("{")


def test_a_real_prose_answer_is_not_flagged():
    payload = json.dumps(
        {"status": "answer", "operation": "current_state", "answer": "San Francisco."}
    )
    runner = _runner(payload)
    _, text, _, _ = runner._answer_with_fallback(_request(), "ctx", [])
    assert runner._answer_was_raw_structure is False
    assert text == "San Francisco."


def test_json_inside_the_answer_field_is_also_caught():
    """The first version of this flag only looked at the empty-`answer` fallback, so a
    model that put JSON *into* `answer` was invisible to it — 10 of 18 real leaks on
    v4.0-flat arrived that way."""
    payload = json.dumps(
        {"status": "answer", "operation": "count", "answer": '```json\n{"items": []}\n```'}
    )
    runner = _runner(payload)
    _, _, _, _ = runner._answer_with_fallback(_request(), "ctx", [])
    assert runner._answer_was_raw_structure is True


def test_a_computed_answer_is_not_flagged_even_with_an_empty_answer_field():
    """count and duration have their reply rebuilt from operands, so an empty `answer`
    is survivable there. The old flag called these leaks anyway: it fired before
    `compute` ran, so a row reading "5 days" was reported as raw structure."""
    payload = json.dumps(
        {
            "status": "answer",
            "operation": "duration",
            "start_date": "2022-01-01",
            "end_date": "2022-01-19",
            "answer": "",
        }
    )
    runner = _runner(payload)
    _, text, _, _ = runner._answer_with_fallback(_request(), "ctx", [])
    assert text == "18 days"


def test_the_fallback_second_call_asks_for_prose_not_structure():
    """The second pass carries no schema, so a system prompt demanding a structured
    verdict has nothing to parse it back out. Under `synthesis_v4` the model emitted
    JSON and it became the answer on 9 of the 18 leaked rows of v4.0-flat."""
    import inspect

    from llm_long_term_memory.evaluation.runners.memory import MemoryRunner

    # Asserted as a property, not a spelling: a formatter reflowing the expression
    # broke the first version of this test without the behaviour changing at all.
    source = inspect.getsource(MemoryRunner._answer_with_fallback)
    second_call = source.split("second = self.client.generate", 1)[1]
    assert "ANSWER_SYSTEM" in second_call, "the fallback still demands a structured verdict"
    assert "synthesis_v4" in second_call, "the swap is not conditioned on the policy"


def test_raw_structure_never_reaches_the_reader():
    """A guarantee in code, not a request in a prompt.

    The prompt now asks for `answer` on every operation, and the previous two attempts to
    fix this by prompt alone each left rows behind — 55, then 18. Emitting the structure
    is never the right answer to anything, so when nothing usable survives, the honest
    rendering of "the model produced no reply" is an abstention.
    """
    payload = json.dumps({"status": "answer", "operation": "current_state", "answer": ""})
    runner = _runner(payload)
    _, text, _, _ = runner._answer_with_fallback(_request(), "ctx", [])
    assert not text.lstrip().startswith(("{", "```"))
    assert text == "I do not know."
    # Still flagged, so the prompt defect stays countable rather than being papered over.
    assert runner._answer_was_raw_structure is True


def test_a_computed_answer_is_preferred_over_the_abstention():
    """The guarantee must not swallow a perfectly good computed reply."""
    payload = json.dumps(
        {
            "status": "answer",
            "operation": "duration",
            "start_date": "2022-01-01",
            "end_date": "2022-01-19",
            "answer": "",
        }
    )
    runner = _runner(payload)
    _, text, _, _ = runner._answer_with_fallback(_request(), "ctx", [])
    assert text == "18 days"


def test_a_good_reply_survives_even_when_the_model_also_emitted_structure():
    """The regression this pins cost 11 correct answers.

    The flag is a diagnostic — true whenever the model produced a structure at any point.
    The repair is narrower — it may only fire when the text actually being returned is
    one. Driving the repair from the flag replaced sixteen replies with an abstention on
    v4.0-flat2, eleven of which had been correct the run before.
    """
    payload = json.dumps(
        {
            "status": "answer",
            "operation": "comparison",
            "start_date": "2023-05-22",
            "end_date": "2023-05-26",
            "answer": "The user taking painting classes came first.",
        }
    )
    runner = _runner(payload)
    _, text, _, _ = runner._answer_with_fallback(_request(), "ctx", [])
    assert text == "The user taking painting classes came first."
    assert "I do not know" not in text


@pytest.mark.parametrize("payload", ['{"status": "unexpected"}', '{"status":', "```json\n{}\n```"])
def test_invalid_verdict_structure_does_not_bypass_the_output_guard(payload):
    runner = _runner(payload)
    _, text, _, _ = runner._answer_with_fallback(_request(), "ctx", [])
    assert text == "I do not know."
    assert runner._answer_was_raw_structure is True


@pytest.mark.parametrize("status", ["need_source", "no_evidence"])
def test_no_source_does_not_return_structure_from_the_verdict_answer(status):
    runner = _runner(json.dumps({"status": status, "answer": '{"items": []}'}))
    runner.fallback = SimpleNamespace(recover=lambda *args: SimpleNamespace(used=False))
    _, text, _, _ = runner._answer_with_fallback(_request(), "ctx", [])
    assert text == "I do not know."
    assert runner._answer_was_raw_structure is True


@pytest.mark.parametrize(
    ("payload", "expected", "flagged"),
    [
        ('{"answer": "unparsed structure"}', "I do not know.", True),
        ('```json\n{"items": []}\n```', "I do not know.", True),
        ("The source says San Francisco.", "The source says San Francisco.", False),
    ],
)
def test_second_pass_is_checked_even_when_the_prompt_requests_prose(payload, expected, flagged):
    runner = _runner(json.dumps({"status": "need_source", "answer": ""}))
    first = runner.client.generate()
    replies = iter([first, SimpleNamespace(text=payload)])
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return next(replies)

    runner.client = SimpleNamespace(generate=generate)
    runner.raw_fallback_max_chars = 2400
    evidence = SimpleNamespace(used=True, render=lambda **kwargs: "The raw source text.")
    runner.fallback = SimpleNamespace(recover=lambda *args: evidence)
    _, text, _, _ = runner._answer_with_fallback(_request(), "ctx", [])
    assert text == expected
    assert runner._answer_was_raw_structure is flagged
    assert len(calls) == 2, "output validation must not spend another model call"
