"""The schema sent to the provider must be the schema the reply is parsed with.

Getting this wrong is silent. Every field a subclass adds carries a default, so parsing
a base-schema reply with the subclass succeeds and the extra fields simply read as
their defaults — indistinguishable from a model that chose to leave them empty.

That is not hypothetical. The whole v3 line ran this way: `answer_confidence` was
'medium' on 100% of rows in every phase because 'medium' is the default, and the
registered `no_new_confident_errors` dev60 gate compared a quantity that was zero by
construction. Eight of the nine gates tested something; that one did not.
"""

from __future__ import annotations

from llm_long_term_memory.evaluation.runners.base import AnswerVerdict
from llm_long_term_memory.evaluation.runners.reasoning import ReasonedAnswerVerdict
from llm_long_term_memory.evaluation.runners.synthesis import SynthesisVerdict


class _RecordingClient:
    """Captures the kwargs a runner hands the provider."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)

        class _Completion:
            text = '{"status": "answer", "answer": "x"}'
            usage = None

        return _Completion()


def _runner(policy: str):
    from llm_long_term_memory.evaluation.runners.memory import MemoryRunner

    runner = MemoryRunner.__new__(MemoryRunner)
    runner.client = _RecordingClient()
    runner.model = "test-model"
    runner.answer_policy = policy
    runner.max_output_tokens = 256
    runner.chars_per_token = 4.0
    from llm_long_term_memory.evaluation.runners.base import ANSWER_SYSTEM

    runner.answer_system = ANSWER_SYSTEM
    runner.verdict_schema = {
        "v2": AnswerVerdict,
        "reasoned_v3": ReasonedAnswerVerdict,
        "synthesis_v4": SynthesisVerdict,
    }[policy]
    return runner


class _Instance:
    question = "how many?"
    question_date = "2026/01/01"
    question_id = "q1"


def test_each_policy_sends_the_schema_it_parses_with():
    for policy, expected in (
        ("v2", AnswerVerdict),
        ("reasoned_v3", ReasonedAnswerVerdict),
        ("synthesis_v4", SynthesisVerdict),
    ):
        runner = _runner(policy)
        runner._complete(_Instance(), "some context", structured=True)
        sent = runner.client.calls[-1].get("schema")
        assert sent is expected, f"{policy} sent {sent}, parses with {expected}"


def test_an_unstructured_call_sends_no_schema():
    runner = _runner("v2")
    runner._complete(_Instance(), "some context")
    assert "schema" not in runner.client.calls[-1]


def test_the_subclasses_really_do_add_fields():
    """If they did not, the test above would pass for the wrong reason."""
    base = set(AnswerVerdict.model_fields)
    assert set(ReasonedAnswerVerdict.model_fields) - base
    assert set(SynthesisVerdict.model_fields) - base
