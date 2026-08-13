"""Features the utility predictor sees at serving time.

Every one is computable from the store and the query alone — no LLM call, no gold
answer. That constraint is the point: the predictor has to run inside the packer's
budget, so anything it cannot see in a few microseconds cannot be a feature.

The first feature is the retrieval score. Including it is what makes the experiment
answerable: if predicted utility never beats relevance, the model has learned
nothing beyond what ranking already knew, and the honest conclusion is that
downstream utility is not separable from retrieval score at this scale.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime

from chronomem.store import Memory

FEATURE_NAMES = (
    "semantic_score",
    "rank",
    "token_count",
    "importance",
    "confidence",
    "age_days",
    "is_superseded",
    "has_number",
    "has_date_words",
    "query_term_overlap",
    "is_profile",
    "is_episodic",
    "duplicate_content_rate",
)


@dataclass(slots=True)
class Features:
    semantic_score: float
    rank: float
    token_count: float
    importance: float
    confidence: float
    age_days: float
    is_superseded: float
    has_number: float
    has_date_words: float
    query_term_overlap: float
    is_profile: float
    is_episodic: float
    duplicate_content_rate: float
    """How much of this memory's wording is repeated by its neighbours in the same
    retrieval. The one feature aimed squarely at redundancy, which leave-one-out
    cannot see but a packer must."""

    def as_list(self) -> list[float]:
        return [getattr(self, f.name) for f in fields(self)]


_DATE_WORDS = frozenset(
    [
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
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "yesterday",
        "today",
        "tomorrow",
        "week",
        "month",
        "year",
        "ago",
        "since",
        "until",
        "before",
        "after",
    ]
)


def _tokens(text: str) -> set[str]:
    return {
        w for w in "".join(c if c.isalnum() else " " for c in text.lower()).split() if len(w) > 2
    }


def build(
    memory: Memory,
    *,
    query: str,
    semantic_score: float,
    rank: int,
    neighbours: list[Memory],
    now: datetime | None = None,
) -> Features:
    now = now or datetime.now()
    content = memory.content
    words = _tokens(content)

    when = memory.event_time or memory.ingested_at
    age = (now - when).days if when else 0.0

    others = [m for m in neighbours if m.id != memory.id]
    if others and words:
        shared = max(len(words & _tokens(o.content)) / len(words) for o in others)
    else:
        shared = 0.0

    return Features(
        semantic_score=float(semantic_score),
        rank=float(rank),
        token_count=float(memory.token_count),
        importance=float(memory.importance),
        confidence=float(memory.confidence),
        age_days=float(age),
        is_superseded=1.0 if memory.status != "active" else 0.0,
        has_number=1.0 if any(c.isdigit() for c in content) else 0.0,
        has_date_words=1.0 if words & _DATE_WORDS else 0.0,
        query_term_overlap=(len(words & _tokens(query)) / len(words)) if words else 0.0,
        is_profile=1.0 if memory.type == "profile" else 0.0,
        is_episodic=1.0 if memory.type == "episodic" else 0.0,
        duplicate_content_rate=shared,
    )
