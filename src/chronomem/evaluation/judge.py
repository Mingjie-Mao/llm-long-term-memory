"""LLM-as-judge for LongMemEval.

Two prompts, because the two question kinds have opposite success conditions. A
normal question is correct when the hypothesis conveys the gold answer's facts. An
abstention question (`_abs`) has no answer in the history, and is correct only when
the system *declines* — a confident answer is the failure mode being tested.

The judge model is pinned for the life of the project and is deliberately a
different model from the answerer, so that a system is not grading its own prose.
Judge reliability is measured against hand labels rather than assumed; see
`chronomem eval judge-agreement` and D4 in docs/DECISIONS.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from chronomem.llm.client import GeminiClient

JUDGE_SYSTEM = (
    "You grade answers to questions about a long chat history. You are strict about "
    "factual equivalence and indifferent to wording, length, and style."
)

_QA_PROMPT = """\
Question: {question}

Reference answer (ground truth): {gold}

Candidate answer: {hypothesis}

Does the candidate answer convey the same factual content as the reference answer?

Rules:
- Ignore differences in phrasing, verbosity, formatting, and politeness.
- The candidate may add extra correct detail; that is still correct.
- If the candidate contradicts the reference on any fact the question asks about, \
it is incorrect.
- If the candidate hedges but still states the reference's fact, it is correct.
- If the candidate declines to answer or says it does not know, it is incorrect.
"""

_ABSTENTION_PROMPT = """\
Question: {question}

This question CANNOT be answered from the chat history — it presupposes something \
that was never discussed. The correct behavior is to decline, say the information \
is not available, or point out the false premise.

Candidate answer: {hypothesis}

Did the candidate correctly decline to answer?

Rules:
- Correct: it says it does not know, cannot find it, has no record, or challenges \
the premise.
- Incorrect: it supplies a specific answer as though the information existed, even \
a hedged one.
- Asking a clarifying question instead of asserting a fact counts as correct.
"""


class Verdict(BaseModel):
    correct: bool = Field(description="Whether the candidate answer is correct")
    reason: str = Field(description="One short sentence justifying the decision")


@dataclass(slots=True)
class JudgeResult:
    correct: bool
    reason: str
    input_tokens: int
    output_tokens: int
    api_latency_ms: float = 0.0


class Judge:
    def __init__(self, client: GeminiClient, model: str, thinking: bool = True) -> None:
        self.client = client
        self.model = model
        # Judging is a reasoning task at low volume, so thinking is left on here even
        # though it is off everywhere else.
        self.thinking = thinking

    def grade(
        self, question: str, gold: str, hypothesis: str, is_abstention: bool = False
    ) -> JudgeResult:
        if not hypothesis.strip():
            return JudgeResult(False, "empty answer", 0, 0)

        prompt = (
            _ABSTENTION_PROMPT.format(question=question, hypothesis=hypothesis)
            if is_abstention
            else _QA_PROMPT.format(question=question, gold=gold, hypothesis=hypothesis)
        )
        completion = self.client.generate(
            role="judge",
            model=self.model,
            prompt=prompt,
            system=JUDGE_SYSTEM,
            schema=Verdict,
            temperature=0.0,
            thinking=self.thinking,
        )
        verdict = Verdict.model_validate_json(completion.text)
        return JudgeResult(
            correct=verdict.correct,
            reason=verdict.reason,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens + completion.thinking_tokens,
            api_latency_ms=completion.api_latency_ms,
        )
