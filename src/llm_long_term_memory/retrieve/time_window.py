"""The period a question's own relative time points at, for time-aware retrieval.

"Who did I go with to the music event last Saturday?" names a date the question's words
do not share with the turn that answers it ("I went to the Billie Eilish concert with my
parents"). Resolving "last Saturday" from the day the question is asked gives a window;
turns said in it, or whose own relative dates fall in it, can then be found by time —
LongMemEval's time-aware query expansion. Resolution reuses `time_notes`.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from llm_long_term_memory.ingest.event_time import _is_title
from llm_long_term_memory.retrieve.time_notes import _PATTERN, _shift_months, resolve

_ISO_DAY = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_ISO_MONTH = re.compile(r"^= (\d{4})-(\d{2}) \(")
_ISO_YEAR = re.compile(r"^= (\d{4})$")


def _span(note: str) -> tuple[date, date] | None:
    """The days a note covers, widened where it is approximate."""
    days = [date(*map(int, m.groups())) for m in _ISO_DAY.finditer(note)]
    if note.startswith("since"):
        return None  # a start date, not a time the question asks about
    if len(days) == 2:  # a week or a weekend
        return days[0] - timedelta(days=1), days[1] + timedelta(days=1)
    if len(days) == 1:
        slack = 1 if note.startswith("=") else 4
        return days[0] - timedelta(days=slack), days[0] + timedelta(days=slack)
    month = _ISO_MONTH.match(note)
    if month:
        first = date(int(month[1]), int(month[2]), 1)
        return first, _shift_months(first, 1) - timedelta(days=1)
    year = _ISO_YEAR.match(note)
    if year:
        return date(int(year[1]), 1, 1), date(int(year[1]), 12, 31)
    return None


def question_window(question: str, asked_on: datetime | date | None) -> tuple[date, date] | None:
    """The period a question's own relative time points at ("last Saturday", "a week
    ago"), resolved from the day it is asked, or None when it names none or several."""
    if asked_on is None:
        return None
    said = asked_on.date() if isinstance(asked_on, datetime) else asked_on
    spans = []
    for match in _PATTERN.finditer(question):
        if match.group("day") and _is_title(question, match):
            continue
        note = resolve(match, said)
        span = _span(note) if note else None
        if span:
            spans.append(span)
    return spans[0] if len(spans) == 1 else None


def turn_dates(text: str, said_on: datetime | date | None) -> list[tuple[date, date]]:
    """The periods a turn's own relative dates resolve to."""
    if said_on is None:
        return []
    said = said_on.date() if isinstance(said_on, datetime) else said_on
    spans = []
    for match in _PATTERN.finditer(text):
        if match.group("day") and _is_title(text, match):
            continue
        note = resolve(match, said)
        span = _span(note) if note else None
        if span:
            spans.append(span)
    return spans
