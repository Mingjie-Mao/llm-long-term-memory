"""Rubric judge for BEAM.

BEAM grades an answer against the rubric shipped with its question. Each item scores
1.0, 0.5 or 0.0 and the question scores their mean; event ordering is also read as
Kendall's tau-b over the order the events are given in, rescaled to [0, 1]. That is
BEAM's own protocol (github.com/mohammadtavakoli78/BEAM, `src/evaluation`), and the
scale here is the same.

Two departures, both made to fit a daily request quota:

* **One call grades every item of a question.** BEAM makes one call per item, which over
  this project's final half and four arms would be several thousand judge requests.
* **Event order comes from positions the judge reports** for each reference event. BEAM
  aligns newline-split answer lines to reference events with a pairwise LLM call per
  candidate pair. Answer lines that match no reference event are not ranked here.

So a score here compares this project's arms with each other. It is not a number to set
beside BEAM's published tables, which D2 rules out in any case.

Every item's grade is saved on the result row, and `rescore` recomputes the question
score, the binary verdict and the order metric from those grades. Changing how they are
aggregated never costs a judge call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from llm_long_term_memory.evaluation.judge import JudgeResult

if TYPE_CHECKING:
    from llm_long_term_memory.llm.client import GeminiClient

#   beam-rubric-v1  2026-09-14: every item of a question graded in one call on BEAM's
#                   1.0 / 0.5 / 0.0 scale; event order from positions the judge reports
BEAM_JUDGE_PROMPT_VERSION = "beam-rubric-v1"

CORRECT_AT = 0.5
"""A question counts as correct when its mean item score reaches this. The mean itself is
the measured quantity; the binary verdict exists so paired disagreement can be counted."""

SCALE = (0.0, 0.5, 1.0)

BEAM_JUDGE_SYSTEM = (
    "You grade an assistant's answer against a rubric. Judge meaning, not wording: accept "
    "paraphrases, and treat numbers, dates, durations and amounts as equal when their values "
    "are equal. Ignore tone, length and politeness unless an item explicitly requires a format."
)

_PROMPT = """\
Question the assistant was asked: {question}

The assistant's answer, between the markers:
<<<ANSWER
{hypothesis}
ANSWER>>>

Rubric items:
{items}

Grade every rubric item on its own, on this scale:
- 1.0: the answer fully meets the item. For an item saying what the answer must not do, \
the answer addresses the question and does not do it.
- 0.5: the answer meets the item only in part, or with a minor inaccuracy or omission.
- 0.0: the answer misses the item, contradicts it, or does not address the question.
An item that joins several requirements earns 1.0 only when all of them hold.
Return exactly one grade per item, numbered as above.
"""

_ORDER = """\
The items are events, listed in the order they happened. For each event the answer \
mentions, also give `position`: 1 for the event the answer places first, 2 for the one it \
places second, and so on, following the answer's own order. Leave `position` empty for an \
event the answer does not mention.
"""


class ItemGrade(BaseModel):
    item: int = Field(description="The rubric item's number, starting at 1")
    score: float = Field(description="1.0, 0.5 or 0.0")
    reason: str = Field(description="One short sentence justifying the score")
    position: int | None = Field(
        default=None,
        description="Event ordering only: where the answer places this event, from 1",
    )


class RubricVerdict(BaseModel):
    grades: list[ItemGrade]


class InvalidJudgement(ValueError):
    """The judge's grades do not cover the rubric exactly once on BEAM's scale."""


def render_prompt(question: str, rubric: Sequence[str], hypothesis: str, ordering: bool) -> str:
    items = "\n".join(f"{number}. {item}" for number, item in enumerate(rubric, start=1))
    prompt = _PROMPT.format(question=question, hypothesis=hypothesis, items=items)
    return f"{prompt}\n{_ORDER}" if ordering else prompt


def checked_grades(verdict: RubricVerdict, n_items: int) -> list[dict]:
    """Exactly one on-scale grade per item, in item order; anything else is refused.

    A missing item scored as zero, or a duplicate averaged in, would move the score while
    looking like a judgement, so neither is repaired.
    """
    numbers = sorted(grade.item for grade in verdict.grades)
    if numbers != list(range(1, n_items + 1)):
        raise InvalidJudgement(f"grades cover items {numbers}, expected 1..{n_items}")
    grades = []
    for grade in sorted(verdict.grades, key=lambda g: g.item):
        score = next((step for step in SCALE if abs(grade.score - step) < 1e-9), None)
        if score is None:
            raise InvalidJudgement(f"item {grade.item} scored {grade.score}, not one of {SCALE}")
        grades.append(
            {
                "item": grade.item,
                "score": score,
                "reason": grade.reason,
                "position": grade.position,
            }
        )
    return grades


def question_score(grades: Sequence[dict]) -> float:
    return sum(grade["score"] for grade in grades) / len(grades)


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Kendall's tau-b. None when either ranking is constant, where it is undefined."""
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0:
                ties_x += 1
            if dy == 0:
                ties_y += 1
            if dx and dy:
                if (dx > 0) == (dy > 0):
                    concordant += 1
                else:
                    discordant += 1
    pairs = len(x) * (len(x) - 1) // 2
    denominator = ((pairs - ties_x) * (pairs - ties_y)) ** 0.5
    return (concordant - discordant) / denominator if denominator else None


def order_score(grades: Sequence[dict]) -> float:
    """BEAM's `tau_norm` over the reference events: (tau-b + 1) / 2.

    The reference rank is the item number. An event the answer does not place takes one
    shared rank after every placed event, as in BEAM. Where tau is undefined — nothing
    placed, or a single event — the score is 0.0, because no order was shown.
    """
    unplaced = len(grades) + 1
    reference = [grade["item"] for grade in grades]
    answer = [grade["position"] or unplaced for grade in grades]
    tau = kendall_tau_b(reference, answer)
    return 0.0 if tau is None else (tau + 1) / 2


def rescore(
    judge_details: dict, question_type: str | None = None, correct_at: float = CORRECT_AT
) -> dict:
    """Recompute a row's score, verdict and order metric from its saved grades."""
    grades = judge_details["grades"]
    score = question_score(grades)
    out = {"score": score, "correct": score >= correct_at}
    if question_type == "event_ordering":
        out["order_tau_norm"] = order_score(grades)
    return out


class BeamRubricJudge:
    prompt_version = BEAM_JUDGE_PROMPT_VERSION
    accepts_rubric = True

    def __init__(self, client: GeminiClient, model: str, thinking: bool = True) -> None:
        self.client = client
        self.model = model
        self.thinking = thinking

    def grade(
        self,
        question: str,
        gold: str,
        hypothesis: str,
        is_abstention: bool = False,
        question_type: str | None = None,
        rubric: Sequence[str] = (),
    ) -> JudgeResult:
        """Grade one answer against its question's rubric.

        `gold` is written to the row by the harness and deliberately not shown to the
        judge. The rubric already states what the reference establishes, and a reference
        answer in the prompt invites grading by resemblance to its wording.
        """
        if not rubric:
            raise ValueError("the rubric judge needs the question's rubric")
        ordering = question_type == "event_ordering"
        if not hypothesis.strip():
            grades = [
                {"item": number, "score": 0.0, "reason": "empty answer", "position": None}
                for number in range(1, len(rubric) + 1)
            ]
            return self._result(grades, ordering, 0, 0, 0.0)
        completion = self.client.generate(
            role="judge",
            model=self.model,
            prompt=render_prompt(question, rubric, hypothesis, ordering),
            system=BEAM_JUDGE_SYSTEM,
            schema=RubricVerdict,
            temperature=0.0,
            thinking=self.thinking,
        )
        grades = checked_grades(RubricVerdict.model_validate_json(completion.text), len(rubric))
        return self._result(
            grades,
            ordering,
            completion.input_tokens,
            completion.output_tokens + completion.thinking_tokens,
            completion.api_latency_ms,
        )

    @staticmethod
    def _result(
        grades: list[dict], ordering: bool, input_tokens: int, output_tokens: int, latency: float
    ) -> JudgeResult:
        details: dict = {"version": BEAM_JUDGE_PROMPT_VERSION, "grades": grades}
        derived = rescore(details, "event_ordering" if ordering else None)
        details["score"] = derived["score"]
        if ordering:
            details["order_tau_norm"] = derived["order_tau_norm"]
        earned = sum(grade["score"] for grade in grades)
        return JudgeResult(
            correct=derived["correct"],
            reason=f"{earned:g} of {len(grades)} rubric points",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            api_latency_ms=latency,
            score=derived["score"],
            details=details,
        )
