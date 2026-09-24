"""Deterministic query-time repairs for the v2c hybrid-memory candidate.

v2b showed that more extracted facts are not enough. Its remaining failures were
visible in the already-retrieved evidence: an explicit replacement ranked behind an
older value, an exact-date question received only month-level structured text, and a
chronological answer omitted one of three dated events. These repairs operate on that
evidence without another model call and without mutating the store.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from llm_long_term_memory.evaluation.runners.base import ANSWER_SYSTEM
from llm_long_term_memory.store import Memory

V2C_ANSWER_PROMPT_VERSION = "memory-aware-v2c.8"
V2C_ANSWER_SYSTEM = (
    f"{ANSWER_SYSTEM}\n\n"
    "When resolving the exact name, title, URL, or link of something mentioned or "
    "recommended earlier, compare every qualifier against the verbatim source. If no "
    "single candidate satisfies them all, do not force a match. State the conflict "
    "briefly and distinguish the candidates: which item was actually recommended and "
    "which item matches the other requested property. Treat source evidence as more "
    "authoritative than the question's possibly mistaken description."
)

_WORD = re.compile(r"[A-Za-z0-9]+")
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "s",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
        "user",
    }
)
_PAST = re.compile(r"\b(?:before|previously|used to|at the time|back then|former)\b", re.I)
_ORDER = re.compile(r"\b(?:order|sequence|chronological|first|then|next|last)\b", re.I)
_DATE_DETAIL = re.compile(r"\b(?:when|what date|which date|what day|which day)\b", re.I)
_EXACT_REFERENCE = re.compile(r"\b(?:name|title|url|link)\b", re.I)
_PRIOR_RECOMMENDATION = re.compile(r"\b(?:recommend(?:ed|ation)?|suggest(?:ed|ion)?)\b", re.I)
_DAY_PRECISION = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|"
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?\b",
    re.I,
)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_SPORTS = frozenset(
    {
        "sport",
        "sports",
        "nba",
        "nfl",
        "basketball",
        "football",
        "baseball",
        "soccer",
        "hockey",
        "tennis",
        "playoffs",
        "championship",
        "game",
        "games",
        "match",
        "matches",
    }
)
_OBSERVED = frozenset({"watch", "watched", "watching", "attend", "attended", "saw", "viewed"})


def _tokens(text: str) -> set[str]:
    return {word.lower() for word in _WORD.findall(text) if word.lower() not in _STOP}


def _when(memory: Memory) -> datetime:
    match = _ISO_DATE.search(memory.content)
    if match:
        return datetime(*(int(part) for part in match.groups()))
    return memory.occurred_at or memory.valid_from or datetime.max


def _relevant(question_tokens: set[str], memory: Memory, minimum: int = 2) -> bool:
    return len(question_tokens & _tokens(f"{memory.content} {memory.object or ''}")) >= minimum


def _update_suppression(question: str, memories: list[Memory]) -> tuple[list[Memory], list[str]]:
    """Hide an older related value when a retrieved fact explicitly replaces it.

    Stage B can assign neighbouring predicates (``running_achievements`` versus
    ``running_personal_best``), so exact predicate equality is too strict. The repair
    requires the same subject, three shared content terms, and relevance of both facts
    to this query. It declines past-state questions.
    """
    if _PAST.search(question):
        return list(memories), []
    question_tokens = _tokens(question)
    suppressed: set[str] = set()
    replacers = [m for m in memories if m.update_op == "replaces" or m.replaces_previous]
    for newer in replacers:
        if not _relevant(question_tokens, newer):
            continue
        newer_time = newer.occurred_at or newer.valid_from
        newer_tokens = _tokens(f"{newer.content} {newer.object or ''}")
        for older in memories:
            if older.id == newer.id or older.subject != newer.subject:
                continue
            older_time = older.occurred_at or older.valid_from
            if newer_time and older_time and older_time >= newer_time:
                continue
            if not _relevant(question_tokens, older):
                continue
            shared = newer_tokens & _tokens(f"{older.content} {older.object or ''}")
            if len(shared) >= 3:
                suppressed.add(older.id)
    return [memory for memory in memories if memory.id not in suppressed], sorted(suppressed)


def _detail_gap(question: str, memories: list[Memory]) -> str | None:
    """Return a deterministic reason to hydrate source text before answering."""
    # A compressed recommendation can preserve a plausible sibling item while losing
    # which exact item matched a later qualifier. Asking the answerer whether it needs
    # source text is unsafe here: a plausible sibling makes it confidently say no.
    # Restrict this route to explicit exact-identifier questions about a prior
    # recommendation, then search only sessions already located by retrieval.
    if _EXACT_REFERENCE.search(question) and _PRIOR_RECOMMENDATION.search(question):
        return (
            "The question asks for the exact identity of a prior recommendation; "
            "verify it against the located source conversation."
        )
    if not _DATE_DETAIL.search(question):
        return None
    question_tokens = _tokens(question)
    scored = [
        (len(question_tokens & _tokens(f"{memory.content} {memory.object or ''}")), memory)
        for memory in memories
    ]
    best = max((score for score, _ in scored), default=0)
    relevant = [memory for score, memory in scored if score == best and score > 0]
    if relevant and not any(_DAY_PRECISION.search(memory.content) for memory in relevant):
        return "The question asks for an exact date, but structured memory has no day-level date."
    return None


def _timeline(question: str, memories: list[Memory]) -> tuple[str, list[str]]:
    if not _ORDER.search(question):
        return "", []

    query_tokens = _tokens(question)
    asks_sports = bool(query_tokens & _SPORTS)
    asks_observed = bool(query_tokens & _OBSERVED)
    candidates: list[Memory] = []
    for memory in memories:
        if memory.scope != "event":
            continue
        tokens = _tokens(f"{memory.content} {memory.predicate or ''}")
        if asks_sports and not (tokens & _SPORTS):
            continue
        if asks_observed and not (tokens & _OBSERVED):
            continue
        if not asks_sports and len(query_tokens & tokens) < 2:
            continue
        candidates.append(memory)

    if len(candidates) < 2:
        return "", []
    candidates.sort(key=lambda memory: (_when(memory), memory.id))
    lines = [
        "Question-specific chronological evidence (oldest first; include every matching event).",
        "The complete answer must include every event listed below in exactly this order; "
        "do not omit an event or demote it to a footnote.",
    ]
    if asks_observed:
        lines.append(
            "For an event-list question, attending an event in person is one way of watching "
            "it; do not drop an attended event unless the question explicitly excludes it."
        )
    for memory in candidates:
        when = _when(memory)
        stamp = when.strftime("%Y-%m-%d") if when != datetime.max else "date unknown"
        lines.append(f"- {stamp}: {memory.content}")
    return "\n".join(lines), [memory.id for memory in candidates]


@dataclass(slots=True)
class V2CPlan:
    memories: list[Memory]
    context_note: str = ""
    suppressed_update_ids: list[str] = field(default_factory=list)
    timeline_ids: list[str] = field(default_factory=list)
    detail_hydration_reason: str | None = None


def plan(question: str, memories: list[Memory]) -> V2CPlan:
    """Build the query-time v2c evidence plan without model calls or store writes."""
    resolved, suppressed = _update_suppression(question, memories)
    timeline, timeline_ids = _timeline(question, resolved)
    detail_reason = _detail_gap(question, resolved)
    notes = []
    if suppressed:
        notes.append(
            "Query-time update check: an older related value was omitted because a newer "
            "retrieved fact explicitly replaces it."
        )
    if timeline:
        notes.append(timeline)
    if detail_reason and "exact identity of a prior recommendation" in detail_reason:
        notes.append(
            "Exact-reference rule: resolve every qualifier in the question jointly against "
            "the verbatim source evidence. Return the one candidate that satisfies all of "
            "them. Do not lead with, or present as the answer, a sibling candidate that "
            "matches only some qualifiers. If no candidate satisfies every qualifier, do "
            "not force one: state that the description conflicts with the source and "
            "distinguish the candidate that was actually recommended from the candidate "
            "that matches the other requested property."
        )
    return V2CPlan(
        memories=resolved,
        context_note="\n\n".join(notes),
        suppressed_update_ids=suppressed,
        timeline_ids=timeline_ids,
        detail_hydration_reason=detail_reason,
    )
