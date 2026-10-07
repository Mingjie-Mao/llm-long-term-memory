"""Relative dates in verbatim turns, resolved against the turn's conversation date.

The error taxonomy of `paged-paired-dev100-v18-v1` found that the raw-primary arm lost
nine of its 26 failed dev100 questions to temporal reasoning with every gold turn in
context: "a month ago" and "3 weeks ago" said on the same day, read the wrong way
round; "today" in a conversation three weeks before the question, answered as six.
The reader had the words and the conversation's date and did the arithmetic wrong.

This does the arithmetic first. Each relative expression keeps its words and gains a
bracketed date computed from the conversation it was said in — "a month ago
[≈ 2023-04-24]" — so the source wording and its provenance stay visible (AGENTS.md:
preserve raw relative-date expressions). `≈` marks an approximation; nothing is
annotated whose anchor is not the conversation date.
"""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import date, datetime, timedelta

from llm_long_term_memory.ingest.event_time import _is_title

_NUMBER = (
    r"\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"a\s+couple\s+of|a\s+few|several"
)
_COUNTS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "a couple of": 2,
    "a few": 3,
    "several": 3,
}
_VAGUE = {"a couple of", "a few", "several"}
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_DAY_WORDS = r"today|tonight|this\s+morning|this\s+afternoon|this\s+evening|yesterday|tomorrow"

_PATTERN = re.compile(
    rf"\b(?:"
    rf"(?P<day>{_DAY_WORDS})"
    rf"|(?:about\s+|around\s+|roughly\s+|almost\s+|nearly\s+|over\s+)?"
    rf"(?P<ago_n>{_NUMBER})\s+(?P<ago_unit>days?|weeks?|months?|years?)\s+ago"
    # "for about 4 years and 3 months now", "I've been working professionally for 9
    # years": a duration up to the conversation, so it has a start date. A bare "for
    # two weeks" ("I went to Japan for two weeks") does not, and is left alone.
    rf"|(?P<pp>(?:have|has|['\u2019]ve|['\u2019]s)\s+been\s+(?:[\w'\u2019-]+\s+){{0,6}}?)?"
    rf"for\s+(?:about\s+|around\s+|roughly\s+|almost\s+|nearly\s+|over\s+)?"
    rf"(?P<for_n>{_NUMBER})\s+(?P<for_unit>days?|weeks?|months?|years?)"
    rf"(?:\s+and\s+(?P<for_n2>{_NUMBER})\s+(?P<for_unit2>days?|weeks?|months?))?"
    rf"(?P<for_now>\s+now)?"
    rf"|(?P<rel>last|this\s+past|this|next)\s+(?P<rel_unit>week|weekend|month|year|"
    rf"{'|'.join(_WEEKDAYS)}|{'|'.join(_MONTHS)})"
    rf"|(?P<over>over\s+the\s+weekend)"
    rf")\b",
    re.I,
)


def _count(text: str) -> tuple[int, bool]:
    key = re.sub(r"\s+", " ", text.lower())
    if key.isdecimal():
        return int(key), False
    return _COUNTS[key], key in _VAGUE


def _shift_months(day: date, months: int) -> date:
    index = day.year * 12 + day.month - 1 + months
    year, month = divmod(index, 12)
    return date(year, month + 1, min(day.day, monthrange(year, month + 1)[1]))


def _weekday(day: date) -> str:
    return day.strftime("%a")


def _offset(day: date, n: int, unit: str, sign: int) -> date:
    unit = unit.lower().rstrip("s")
    if unit == "day":
        return day + timedelta(days=sign * n)
    if unit == "week":
        return day + timedelta(weeks=sign * n)
    return _shift_months(day, sign * n * (12 if unit == "year" else 1))


def resolve(match: re.Match, said: date) -> str | None:
    """The bracketed note for one expression, or None when it cannot be dated."""
    try:
        return _resolve(match, said)
    except (ValueError, OverflowError):
        return None  # "6,000 years ago" — no calendar date to write


def _resolve(match: re.Match, said: date) -> str | None:
    g = match.groupdict()
    if g["day"]:
        word = re.sub(r"\s+", " ", g["day"].lower())
        shift = {"yesterday": -1, "tomorrow": 1}.get(word, 0)
        when = said + timedelta(days=shift)
        return f"= {when.isoformat()} {_weekday(when)}"
    if g["ago_n"]:
        n, vague = _count(g["ago_n"])
        when = _offset(said, n, g["ago_unit"], -1)
        exact = g["ago_unit"].lower().startswith("day") and not vague and n <= 3
        return f"{'=' if exact else '≈'} {when.isoformat()}"
    if g["for_n"]:
        if not (g["for_now"] or g["pp"]):
            return None
        n, _ = _count(g["for_n"])
        start = _offset(said, n, g["for_unit"], -1)
        if g["for_n2"]:
            start = _offset(start, _count(g["for_n2"])[0], g["for_unit2"], -1)
        return f"since ≈ {start.isoformat()}"
    monday = said - timedelta(days=said.weekday())
    on_weekend = said.weekday() >= 5
    if g["over"]:
        # Said on a Saturday or Sunday, "over the weekend" is the one under way.
        saturday = monday + timedelta(days=5 if on_weekend else -2)
        return f"= {saturday.isoformat()} to {(saturday + timedelta(days=1)).isoformat()}"
    rel = re.sub(r"\s+", " ", g["rel"].lower())
    unit = g["rel_unit"].lower()
    if rel == "this past" and unit in {"week", "month", "year"}:
        return None  # "this past month" is a span back from today, not a calendar month.
    if unit == "week":
        start = monday + timedelta(weeks={"last": -1, "this past": -1, "this": 0, "next": 1}[rel])
        return f"= week of {start.isoformat()} to {(start + timedelta(days=6)).isoformat()}"
    if unit == "weekend":
        if rel == "this past" and on_weekend:
            return None  # the weekend under way, or the one before it
        saturday = monday + timedelta(
            days=5 + 7 * {"last": -1, "this past": -1, "this": 0, "next": 1}[rel]
        )
        mark = "≈" if rel == "next" else "="
        return f"{mark} {saturday.isoformat()} to {(saturday + timedelta(days=1)).isoformat()}"
    if unit == "month":
        shift = {"last": -1, "this past": -1, "this": 0, "next": 1}[rel]
        month = _shift_months(said.replace(day=1), shift)
        return f"= {month:%Y-%m} ({month:%B %Y})"
    if unit == "year":
        return f"= {said.year + {'last': -1, 'this past': -1, 'this': 0, 'next': 1}[rel]}"
    if unit in _WEEKDAYS:
        target = _WEEKDAYS.index(unit)
        if rel in {"last", "this past"}:
            back = (said.weekday() - target) % 7 or 7
            when = said - timedelta(days=back)
            return f"= {when.isoformat()} {_weekday(when)}"
        if rel == "next":
            ahead = (target - said.weekday()) % 7 or 7
            when = said + timedelta(days=ahead)
            return f"≈ {when.isoformat()} {_weekday(when)}"
        return None  # "this Friday" may be either side of the conversation.
    month = _MONTHS.index(unit) + 1
    if rel in {"last", "this past"}:
        year = said.year - (month >= said.month)
    elif rel == "next":
        year = said.year + (month <= said.month)
    else:
        return None  # "this May" may be past or coming.
    return f"= {year}-{month:02d}"


def annotate(text: str, said_on: datetime | date | None) -> tuple[str, int]:
    """`text` with a bracketed date after each resolvable expression, and the count."""
    if said_on is None:
        return text, 0
    said = said_on.date() if isinstance(said_on, datetime) else said_on
    pieces, last, count = [], 0, 0
    for match in _PATTERN.finditer(text):
        if match.group("day") and _is_title(text, match):
            continue
        note = resolve(match, said)
        if note is None:
            continue
        pieces += [text[last : match.end()], f" [{note}]"]
        last, count = match.end(), count + 1
    pieces.append(text[last:])
    return "".join(pieces), count
