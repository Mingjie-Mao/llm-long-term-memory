"""LLM-as-judge for LongMemEval.

Three prompts, routed by question type, because the types have genuinely different
success conditions:

* **Normal QA** — correct when the hypothesis conveys the gold answer's facts.
* **Abstention** (`_abs`) — no answer exists in the history, so the correct behaviour
  is to *decline*. A confident answer is the failure mode being tested.
* **`single-session-preference`** — the "gold" is not an answer at all. It is a
  rubric describing what a well-personalized reply would contain, so grading it as a
  reference answer marks a correct reply wrong for not restating the rubric.

That third case was a real defect, not a hypothetical. Every system scored 0/3 on
this category, `full_context` included — and a hand audit found one of the three had
answered correctly (it recommended turbinado sugar, which is what the rubric asked
for) and was failed with the reason "the reference answer describes the user's
preferences, whereas the candidate answer provides actual suggestions". See
results/preference-audit.md.

The routing mirrors LongMemEval's own evaluation, which uses a separate
preference prompt instructing the judge that the reference is "a rubric for desired
personalized response", that the reply need not cover every point in it, and that
recalling and using the user's personal information correctly is what makes it
right.

The judge model is pinned for the life of the project and is deliberately a
different model from the answerer, so that a system is not grading its own prose.
Judge reliability is measured against hand labels rather than assumed; see
`lltm eval judge-agreement` and D4 in docs/DECISIONS.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from llm_long_term_memory.llm.client import GeminiClient

#   reference-v1     one reference-answer prompt plus an abstention prompt
#   lme-type-aware-v2  2026-08-14: routed by question type, so a preference gold is
#                      graded as the rubric it actually is
JUDGE_PROMPT_VERSION = "lme-type-aware-v2"

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


_PREFERENCE_PROMPT = """\
Question: {question}

Rubric describing a well-personalized response: {gold}

Candidate answer: {hypothesis}

Does the candidate satisfy the rubric?

Rules:
- The rubric is NOT a reference answer. It describes what a good personalized reply \
would draw on. A candidate that gives concrete advice reflecting the user's history \
is CORRECT; it must not restate the rubric.
- The candidate does NOT need to cover every point in the rubric. Recalling and \
using the user's relevant personal information correctly is enough.
- Correct: it acts on the preference or history the rubric names — recommending, \
advising, or tailoring its answer accordingly.
- Incorrect: it ignores the user's history and gives only generic advice, \
contradicts the stated preference, or declines to answer.
"""

# Question types whose "gold" is a rubric rather than an answer.
_RUBRIC_TYPES = frozenset({"single-session-preference"})


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
        self,
        question: str,
        gold: str,
        hypothesis: str,
        is_abstention: bool = False,
        question_type: str | None = None,
    ) -> JudgeResult:
        if not hypothesis.strip():
            return JudgeResult(False, "empty answer", 0, 0)

        prompt = self._prompt_for(question, gold, hypothesis, is_abstention, question_type)
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

    @staticmethod
    def _prompt_for(
        question: str,
        gold: str,
        hypothesis: str,
        is_abstention: bool,
        question_type: str | None,
    ) -> str:
        # Abstention wins over type: an `_abs` question has no answer to personalize
        # from, so declining is correct whatever category it belongs to.
        if is_abstention:
            return _ABSTENTION_PROMPT.format(question=question, hypothesis=hypothesis)
        if question_type in _RUBRIC_TYPES:
            return _PREFERENCE_PROMPT.format(question=question, gold=gold, hypothesis=hypothesis)
        return _QA_PROMPT.format(question=question, gold=gold, hypothesis=hypothesis)
