"""Stage C of research/ROADMAP.md: let code do the arithmetic the reader gets wrong.

On the t4 baseline the evidence was in context and the reader still miscounted (lists
3+1+3+3, answers 8), applied a date offset twice ("3 weeks before July 15, which is 6
weeks ago") or reversed an order. Here, for three narrow question shapes only, the model
writes structured records — the items with their amounts, or the events with their
dates — and code computes the total, the interval or the order.

Two limits are deliberate. The records are the model's selection, not verified facts:
code can add 3+3+1+3 correctly and still be wrong if one item is out of scope, which is
why every computation is recorded. And an item that counts but has no stated amount
makes the total a known part, said as such ("at least ..."), never a complete total.

Any other question gets no new instruction, schema or limit: `compute_mode` returns None
and the caller sends exactly the request it sent before.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from llm_long_term_memory.answering import AnswerVerdict, asks_for_aggregate

COMPUTE_PROMPT_VERSION = "memory-aware-v2-clt-compute-v1"

ComputeMode = Literal["count", "date", "order"]

_UNIT = r"(?P<unit>days?|weeks?|months?|years?)"
_DATE_ARITHMETIC = re.compile(
    rf"\bhow\s+(?:many|much)\s+{_UNIT}\b(?=.*\b(?:ago|before|after|between|since|passed|"
    rf"until|from|when)\b)|\bhow\s+long\s+ago\b",
    re.I,
)
_TOTAL = re.compile(r"\b(?:in\s+total|total|combined|altogether|sum)\b", re.I)
# "How many more ...", "how much do I need", "how many years old": a difference, a rate or
# an age is not a sum of items, and summing its operands would be confidently wrong.
_NOT_A_SUM = re.compile(
    r"\b(?:more|less|fewer|than|left|remain\w*|need|difference|percent\w*|average|per|"
    r"each|old|age|rate|ratio|will)\b",
    re.I,
)
_ORDER = re.compile(r"\b(?:first|earlier|before|after|later|most\s+recent(?:ly)?)\b", re.I)
_ALTERNATIVES = re.compile(r"\S\s+or\s+\S", re.I)


def compute_mode(question: str) -> ComputeMode | None:
    """Which computation, if any, this question takes. Deliberately narrow."""
    if (
        _DATE_ARITHMETIC.search(question)
        and not _TOTAL.search(question)
        and not re.search(r"\b(?:will\s+i\s+be|old)\b", question, re.I)
    ):
        return "date"
    if asks_for_aggregate(question) and not _NOT_A_SUM.search(question):
        return "count"
    if _ORDER.search(question) and _ALTERNATIVES.search(question):
        return "order"
    return None


_HOW_MANY = re.compile(r"\bhow\s+many\b", re.I)
# "How many days did it take", "how many minutes did I exceed": a duration or a difference
# reached through a count's wording. "times" stays a count ("how many times did I ride").
_UNIT_COUNT = re.compile(
    r"\bhow\s+many\s+(?:\w+\s+)?(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.I,
)
_DERIVED = re.compile(r"\b(?:exceed\w*|faster|slower|longer|shorter|since)\b", re.I)


def count_mode_v2(question: str) -> ComputeMode | None:
    """t4c v2: plain "how many <things>" counts only, and the computation is recorded, never
    substituted. On t4c v1 no override ever turned a wrong answer right (75 overridden
    samples), date and order records lost samples, and "how much" questions were mostly
    differences or rates rather than sums (results/analysis/t4c-v1.md)."""
    if compute_mode(question) != "count" or not _HOW_MANY.search(question):
        return None
    if _UNIT_COUNT.search(question) or _DERIVED.search(question):
        return None
    return "count"


def question_unit(question: str) -> str:
    match = _DATE_ARITHMETIC.search(question)
    unit = (match.group("unit") if match and match.group("unit") else "days").lower()
    return unit if unit.endswith("s") else unit + "s"


class CountedItem(BaseModel):
    item: str = Field(description="What this note is about, in a few words")
    conversation_date: str = Field(description="Date of the conversation it comes from")
    user_words: str = Field(description="The user's own words about it")
    amount: float | None = Field(
        description=(
            "How much this item adds: 1 for one item; the stated number when the user "
            "gives one (times, hours, dollars); null if it counts but no amount is stated"
        )
    )
    verdict: Literal["COUNTS", "REPEAT", "OUTSIDE"]


class CountedAnswerVerdict(BaseModel):
    """`NotedAnswerVerdict` with each note's contribution as a number."""

    notes: list[CountedItem] = Field(default_factory=list)
    unit: str = Field(default="", description="Unit of the total; empty for a plain count")
    status: Literal["answer", "need_source", "no_evidence"] = Field(
        description=AnswerVerdict.model_fields["status"].description
    )
    answer: str = Field(default="", description="The reply, when status is 'answer'")
    reason: str = Field(default="", description="For need_source/no_evidence: what is missing")
    source_query: str = Field(default="", description="For need_source: search keywords")


class DatedEvent(BaseModel):
    event: str = Field(description="The event, in a few words")
    date: str = Field(
        description=(
            "When it happened as YYYY-MM-DD (or YYYY-MM if only the month is known), from "
            "the conversation date and any [date] note; empty if it cannot be dated"
        )
    )
    user_words: str = Field(description="The user's own words about it")


class DatedAnswerVerdict(BaseModel):
    events: list[DatedEvent] = Field(default_factory=list)
    status: Literal["answer", "need_source", "no_evidence"] = Field(
        description=AnswerVerdict.model_fields["status"].description
    )
    answer: str = Field(default="", description="The reply, when status is 'answer'")
    reason: str = Field(default="", description="For need_source/no_evidence: what is missing")
    source_query: str = Field(default="", description="For need_source: search keywords")


COMPUTE_RULES: dict[str, str] = {
    "count": (
        "Write each candidate item as a note with its `amount` (1 for one item, or the "
        "number the user states). The total is computed from the notes marked COUNTS, so "
        "every item you count must be its own note."
    ),
    "date": (
        "List in `events` the event or events the question is about, each dated as "
        "YYYY-MM-DD. The number of days, weeks, months or years is computed from those "
        "dates and the question's date, so give the dates, not only the interval."
    ),
    "order": (
        "List in `events` each event the question compares, each dated as YYYY-MM-DD (or "
        "YYYY-MM). Which came first or last is computed from those dates; leave a date "
        "empty when the conversations do not say when that event happened."
    ),
}

SCHEMAS = {"count": CountedAnswerVerdict, "date": DatedAnswerVerdict, "order": DatedAnswerVerdict}

_WORDS = {
    w: i
    for i, w in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ]
    )
}
_NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?|\b(?:" + "|".join(_WORDS) + r")\b", re.I)


def first_number(text: str) -> float | None:
    match = _NUMBER.search(text)
    if not match:
        return None
    token = match.group(0).lower().lstrip("$").replace(",", "")
    return float(_WORDS[token]) if token in _WORDS else float(token)


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}".rstrip("0")


def parse_date(text: str) -> tuple[date, str] | None:
    """A date and its precision ('day' or 'month'), or None."""
    text = (text or "").strip().replace("/", "-")
    for length, pattern, precision in ((10, "%Y-%m-%d", "day"), (7, "%Y-%m", "month")):
        try:
            return datetime.strptime(text[:length], pattern).date(), precision
        except ValueError:
            continue
    return None


def _interval(start: date, end: date, unit: str) -> int:
    days = abs((end - start).days)
    if unit == "days":
        return days
    if unit == "weeks":
        return round(days / 7)
    months = abs((end.year - start.year) * 12 + end.month - start.month)
    if unit == "months":
        return months
    return round(months / 12)


def apply(
    mode: str,
    verdict,
    model_answer: str,
    question: str,
    asked_on: datetime | None,
    *,
    override: bool = True,
) -> tuple[str, dict]:
    """The answer to return and a record of what was computed.

    With `override=False` the computation is recorded but the model's own answer is
    returned: t4c v1 showed the overrides cost more answers than they fixed.
    """
    if not override:
        _, detail = apply(mode, verdict, model_answer, question, asked_on)
        detail["would_override"] = detail.pop("overrode", False)
        detail["override_enabled"] = False
        return model_answer, detail
    detail: dict = {"mode": mode, "computed": False}
    if getattr(verdict, "status", "answer") != "answer":
        return model_answer, detail
    if mode == "count":
        counted = [n for n in verdict.notes if n.verdict == "COUNTS"]
        known = [n.amount for n in counted if n.amount is not None]
        total = sum(known)
        partial = len(known) < len(counted)
        stated = first_number(model_answer)
        detail.update(
            computed=True, total=total, partial=partial, items=len(counted), model_number=stated
        )
        if not counted or (stated == total and not partial):
            detail["overrode"] = False
            return model_answer, detail
        unit = f" {verdict.unit}" if verdict.unit else ""
        parts = "; ".join(
            f"{n.item} ({_fmt(n.amount) if n.amount is not None else 'amount not stated'})"
            for n in counted
        )
        head = (
            f"At least {_fmt(total)}{unit} (known part only)" if partial else f"{_fmt(total)}{unit}"
        )
        detail["overrode"] = True
        return f"{head}: {parts}.", detail
    dated = [(e, parse_date(e.date)) for e in verdict.events]
    if mode == "date":
        unit = question_unit(question)
        usable = [(e, d) for e, d in dated if d and d[1] == "day"]
        ago = re.search(r"\bago\b", question, re.I)
        if len(usable) >= 2 and not ago:
            (e1, (d1, _)), (e2, (d2, _)) = usable[:2]
            value = _interval(d1, d2, unit)
            text = f"{value} {unit} (between {e1.event} on {d1} and {e2.event} on {d2})."
        elif usable and ago and asked_on is not None:
            (e1, (d1, _)) = usable[0]
            value = _interval(d1, asked_on.date(), unit)
            text = f"{value} {unit} ago ({e1.event} on {d1})."
        else:
            return model_answer, detail
        stated = first_number(model_answer)
        detail.update(computed=True, value=value, unit=unit, model_number=stated)
        if stated == value:
            detail["overrode"] = False
            return model_answer, detail
        detail["overrode"] = True
        return text, detail
    # order: only when every listed event is dated, at a shared precision, without a tie.
    if len(dated) < 2 or any(d is None for _, d in dated):
        return model_answer, detail
    precision = "day" if all(d[1] == "day" for _, d in dated) else "month"
    key = (lambda d: d[0]) if precision == "day" else (lambda d: (d[0].year, d[0].month))
    ordered = sorted(dated, key=lambda pair: key(pair[1]))
    if key(ordered[0][1]) == key(ordered[1][1]):
        return model_answer, detail
    latest = re.search(r"\b(?:most\s+recent(?:ly)?|later|last)\b", question, re.I)
    chosen, other = (ordered[-1], ordered[-2]) if latest else (ordered[0], ordered[1])
    detail.update(computed=True, chosen=chosen[0].event, precision=precision)
    if chosen[0].event.lower() in model_answer.lower():
        detail["overrode"] = False
        return model_answer, detail
    detail["overrode"] = True
    word = "most recently" if latest else "first"
    return (
        f"{chosen[0].event} came {word} ({chosen[0].date}), compared with "
        f"{other[0].event} ({other[0].date}).",
        detail,
    )
