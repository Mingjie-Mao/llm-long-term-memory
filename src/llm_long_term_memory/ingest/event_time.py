"""Retain the time a fact states separately from the conversation's time.

Extraction produces a sentence and the conversation's date, and until now the date was
copied onto every memory as if it were the time the event happened. "I replaced the
plugs on February 14", restated in a March conversation, became a March event; a
duration between it and something else came out as zero.

This reads a date **out of the fact's own words** and returns None when there is none.
None is the useful answer, not a failure: `Memory.occurred_at` falls back to
`observed_at`, so ordering is unchanged, while `event_time` now means one thing —
the user said when.

**Deliberately narrow.** An unambiguous calendar day or short day-relative phrase
can become an exact event date. Coarser or ambiguous relative phrases retain their
original wording and an optional estimate, but never drive lifecycle order: a wrong
date would rank, sort, and supersede facts, while an unresolved one remains visible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_MONTH_NAMES = "|".join(sorted(_MONTHS, key=len, reverse=True))

# `2023-02-14`. Bare, so a version string or an id cannot match.
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

# `February 14, 2023` / `Feb 14 2023` / `February 14th, 2023`
_MONTH_DAY_YEAR = re.compile(
    rf"\b({_MONTH_NAMES})\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s+(\d{{4}})\b", re.I
)

# `14 February 2023`
_DAY_MONTH_YEAR = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_NAMES})\s+(\d{{4}})\b", re.I
)

# `February 14` with no year. The year comes from the conversation.
_MONTH_DAY = re.compile(rf"\b({_MONTH_NAMES})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.I)


# A date that something else is measured *from* is an anchor, not the event's time.
# "The party was about a month before May 21, 2023" happened in April, and reading the
# anchor as the answer was wrong on four of the first twenty dates this recovered from
# a real store. The extractor writes these constantly, because it resolves the user's
# "last month" by naming the conversation's date and keeping the offset in words.
_ANCHORED = re.compile(
    r"\b(?:before|after|prior\s+to|ahead\s+of|following|since|until|till|"
    r"up\s+to|leading\s+up\s+to|earlier\s+than|later\s+than)\s*$",
    re.I,
)

_RELATIVE_DAY = re.compile(r"\b(?:yesterday|today|tomorrow)\b", re.I)
_AGO = re.compile(
    r"\b(?P<count>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve)\s+(?P<unit>days?|weeks?|months?|years?)\s+ago\b",
    re.I,
)
_LAST_MONTH_NAME = re.compile(rf"\blast\s+({_MONTH_NAMES})\b", re.I)
_OTHER_RELATIVE = re.compile(
    r"\b(?:last|next|this)\s+(?:week|month|year|weekend|monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday)\b",
    re.I,
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
}


@dataclass(frozen=True, slots=True)
class TemporalEvidence:
    """One extracted expression, its session anchor and precision.

    `exact` alone may drive lifecycle order. `estimate` is descriptive when a phrase
    denotes a month or an approximate day; it must not silently close a validity
    interval. `observed_at` on the memory is the anchor used for either calculation.
    """

    exact: datetime | None = None
    estimate: datetime | None = None
    expression: str | None = None
    precision: str | None = None


def _shift_month(value: datetime, months: int) -> datetime:
    from calendar import monthrange

    count = value.year * 12 + value.month - 1 + months
    year, month_zero = divmod(count, 12)
    month = month_zero + 1
    return value.replace(year=year, month=month, day=min(value.day, monthrange(year, month)[1]))


def temporal_evidence(text: str, observed_at: datetime | None) -> TemporalEvidence:
    """Parse a narrow set of dates while retaining the expression that justified it.

    Day-precise expressions become `exact`. Month and approximate offsets are kept as
    estimates only; choosing one day for a whole month would fabricate an ordering.
    Multiple temporal expressions stay unresolved to avoid selecting the wrong event.
    """
    matches = sorted(
        [
            *[(m.start(), m) for m in _RELATIVE_DAY.finditer(text)],
            *[(m.start(), m) for m in _AGO.finditer(text)],
            *[(m.start(), m) for m in _LAST_MONTH_NAME.finditer(text)],
            *[(m.start(), m) for m in _OTHER_RELATIVE.finditer(text)],
        ],
        key=lambda item: item[0],
    )
    explicit_matches = [
        match
        for pattern in (_ISO, _MONTH_DAY_YEAR, _DAY_MONTH_YEAR, _MONTH_DAY)
        for match in pattern.finditer(text)
        if not _is_anchor(text, match.start())
    ]
    explicit = stated_event_time(text, observed_at)
    if explicit_matches and matches:
        # The relative phrase may refer to a different event, or use the named date
        # rather than the session as its anchor. Neither interpretation is safe.
        return TemporalEvidence(expression=text, precision="unresolved")
    if explicit is not None:
        return TemporalEvidence(explicit, explicit, explicit_matches[0].group(0), "day")
    if not matches:
        return TemporalEvidence()
    # A date plus a relative phrase, or two relative phrases, may describe different
    # events. Keep the words, but do not assign a date to the whole fact.
    expressions = list(dict.fromkeys(match.group(0) for _, match in matches))
    if len(expressions) != 1 or observed_at is None:
        return TemporalEvidence(expression="; ".join(expressions), precision="unresolved")
    phrase = expressions[0]
    day = _RELATIVE_DAY.fullmatch(phrase)
    if day:
        offset = {"yesterday": -1, "today": 0, "tomorrow": 1}[phrase.lower()]
        resolved = observed_at + timedelta(days=offset)
        return TemporalEvidence(resolved, resolved, phrase, "day")
    ago = _AGO.fullmatch(phrase)
    if ago:
        count_text = ago["count"].lower()
        count = int(count_text) if count_text.isdecimal() else _COUNTS[count_text]
        unit = ago["unit"].lower().rstrip("s")
        if unit == "day":
            resolved = observed_at - timedelta(days=count)
            return TemporalEvidence(resolved, resolved, phrase, "day")
        if unit == "week":
            estimate = observed_at - timedelta(weeks=count)
        else:
            estimate = _shift_month(observed_at, -count * (12 if unit == "year" else 1))
        return TemporalEvidence(estimate=estimate, expression=phrase, precision="approximate_day")
    named = _LAST_MONTH_NAME.fullmatch(phrase)
    if named:
        month = _MONTHS[named.group(1).lower()]
        year = observed_at.year - int(month >= observed_at.month)
        return TemporalEvidence(
            estimate=observed_at.replace(year=year, month=month, day=1),
            expression=phrase,
            precision="month",
        )
    return TemporalEvidence(expression=phrase, precision="unresolved")


def _is_anchor(text: str, start: int) -> bool:
    """Whether the date at `start` is what an offset is measured from."""
    return bool(_ANCHORED.search(text[max(0, start - 40) : start]))


def _valid(year: int, month: int, day: int) -> datetime | None:
    try:
        return datetime(year, month, day)
    except ValueError:
        return None


def stated_event_time(text: str, observed_at: datetime | None) -> datetime | None:
    """The date the fact states, or None.

    `observed_at` supplies the year for a date written without one, and only that. A
    month and day with no year almost always refers to the recent past relative to the
    conversation, so the most recent occurrence at or before the conversation's date is
    chosen — "February 14", said on 15 March 2023, is that year's; said on 3 January
    2023, it is the previous year's rather than a date six weeks in the future.

    A fact naming more than one date is left undated. Two dates usually means a span or
    a comparison, and picking one of them would assert something the sentence does not.
    """
    if not text:
        return None

    found: list[datetime] = []
    anchored = 0

    def record(when: datetime | None, start: int) -> None:
        nonlocal anchored
        if when is None:
            return
        if _is_anchor(text, start):
            anchored += 1
            return
        found.append(when)

    for match in _ISO.finditer(text):
        year, month, day = match.groups()
        record(_valid(int(year), int(month), int(day)), match.start())

    for match in _MONTH_DAY_YEAR.finditer(text):
        month, day, year = match.groups()
        record(_valid(int(year), _MONTHS[month.lower()], int(day)), match.start())

    for match in _DAY_MONTH_YEAR.finditer(text):
        day, month, year = match.groups()
        record(_valid(int(year), _MONTHS[month.lower()], int(day)), match.start())

    if not found and not anchored and observed_at is not None:
        for match in _MONTH_DAY.finditer(text):
            month, day = match.groups()
            number = _MONTHS[month.lower()]
            candidate = _valid(observed_at.year, number, int(day))
            if candidate is None:
                continue
            if candidate > observed_at:
                candidate = _valid(observed_at.year - 1, number, int(day))
            record(candidate, match.start())

    # A sentence that measures an offset from a date states a time it does not name.
    # Falling back to any other date in it would answer a different question.
    if anchored:
        return None

    unique = {when for when in found}
    if len(unique) != 1:
        return None
    return unique.pop()
