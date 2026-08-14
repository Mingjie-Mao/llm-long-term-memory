"""Stage B: a fact string becomes a `Memory`.

Rules, not a model call. Everything this stage needs is either in the sentence or in
the session it came from, and spending a request per fact to recover it would cost
more than Stage A itself.

The split has a diagnostic payoff beyond cost. With one stage, a memory missing its
predicate and a memory that was never extracted look identical from outside — both
are simply absent from a query's results. Separating them makes "the fact was lost"
and "the fact was stored badly" different measurements.

Where rules are weakest is `predicate`, and that matters more than it looks: P4
matches supersessions on `(subject, predicate)`, so a predicate assigned by keyword
is the difference between a timeline and a pile. The heuristics below are
deliberately conservative — an unmatched fact gets `states`, which collides with
nothing and therefore supersedes nothing, rather than a guess that might wrongly
retire a true fact.
"""

from __future__ import annotations

import re
from datetime import datetime

from chronomem.store import Memory, MemoryType

from .extract import _parse_date, memory_id

# Ordered: the first match wins, so more specific patterns come first.
_PREDICATE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("lives_in", re.compile(r"\b(lives? in|moved to|relocated to|resides? in)\b", re.I)),
    (
        "works_as",
        re.compile(
            r"\b(works? as|is an? .{0,20}"
            r"(engineer|designer|teacher|nurse|lawyer|manager|developer|analyst))\b",
            re.I,
        ),
    ),
    ("works_at", re.compile(r"\bworks? (at|for)\b", re.I)),
    ("studies", re.compile(r"\b(studies|studied|is studying|majors? in)\b", re.I)),
    (
        "dislikes",
        re.compile(
            r"\b(dislikes?|hates?|no longer|stopped|gave up|"
            r"does not (like|enjoy)|doesn't (like|enjoy))\b",
            re.I,
        ),
    ),
    ("prefers", re.compile(r"\b(prefers?|likes?|enjoys?|favou?rite|switched to)\b", re.I)),
    ("owns", re.compile(r"\b(owns?|has|have|bought|purchased|acquired)\b", re.I)),
    ("plans_to", re.compile(r"\b(plans? to|intends? to|is going to|will)\b", re.I)),
    (
        "completed",
        re.compile(
            r"\b(completed|finished|attended|visited|went to|did|made|read|watched)\b", re.I
        ),
    ),
    ("spent", re.compile(r"\b(spent|paid|cost)\b", re.I)),
    ("scheduled", re.compile(r"\b(scheduled|books?|booked|appointment)\b", re.I)),
    (
        "has_relationship",
        re.compile(
            r"\b(sister|brother|mother|father|wife|husband|partner|friend|daughter|son)\b", re.I
        ),
    ),
    ("has_health_condition", re.compile(r"\b(diagnosed|allergic|condition|symptoms?)\b", re.I)),
    ("uses_tool", re.compile(r"\b(uses?|using)\b", re.I)),
)

_TYPE_RULES: tuple[tuple[MemoryType, re.Pattern[str]], ...] = (
    ("preference", re.compile(r"\b(prefers?|likes?|dislikes?|hates?|enjoys?|favou?rite)\b", re.I)),
    (
        "procedural",
        re.compile(r"\b(usually|always|every (morning|day|week)|routine|habitually)\b", re.I),
    ),
    (
        "episodic",
        re.compile(
            r"\b(attended|visited|went|bought|purchased|completed|finished|"
            r"last (week|month|year)|ago|yesterday)\b",
            re.I,
        ),
    ),
    (
        "profile",
        re.compile(
            r"\b(lives? in|works? (as|at|for)|is \d+ years old|name is|"
            r"sister|brother|mother|father|wife|husband)\b",
            re.I,
        ),
    ),
)

_REPLACES = re.compile(
    r"\b(no longer|stopped|switched (to|from)|moved (to|from)|used to|instead of|"
    r"gave up|quit|replaced)\b",
    re.I,
)

_ASSISTANT = re.compile(r"^the assistant\b", re.I)
_PROPER_NOUN = re.compile(r"\b(?:[A-Z][a-z']+\s+){0,3}[A-Z][a-z']{2,}\b")
_NUMBER = re.compile(r"\d")

_SENTENCE_LEAD = frozenset({"The", "A", "An", "In", "On", "At", "For", "It", "They", "User"})


def infer_predicate(text: str) -> str:
    for name, pattern in _PREDICATE_RULES:
        if pattern.search(text):
            return name
    # `states` is a deliberate dead end: it matches no other fact, so P4 will never
    # use it to retire something. A wrong guess here would close a true fact's
    # validity window, which is worse than leaving a timeline unbuilt.
    return "states"


def infer_type(text: str) -> MemoryType:
    for name, pattern in _TYPE_RULES:
        if pattern.search(text):
            return name
    return "semantic"


def infer_entities(text: str) -> list[str]:
    out = []
    for match in _PROPER_NOUN.finditer(text):
        value = match.group(0).strip()
        head = value.split()[0]
        if head in _SENTENCE_LEAD and len(value.split()) == 1:
            continue
        if value.lower().startswith("the user") or value.lower().startswith("the assistant"):
            continue
        out.append(value)
    return list(dict.fromkeys(out))


def infer_importance(text: str) -> float:
    """Facts carrying a specific are likelier to be asked about than general ones.

    A heuristic, and labelled as one: it is metadata for the downstream selector,
    not a gate. Nothing is dropped for scoring low.
    """
    score = 0.4
    if _NUMBER.search(text):
        score += 0.2
    if infer_entities(text):
        score += 0.15
    if _REPLACES.search(text):
        score += 0.15
    return min(1.0, score)


def infer_object(text: str, predicate: str) -> str:
    """The value side of the triple, taken as the span after the matched verb.

    Crude by design. It feeds supersede matching only when the predicate is
    single-valued, and those predicates have narrow, reliable patterns.
    """
    for name, pattern in _PREDICATE_RULES:
        if name != predicate:
            continue
        match = pattern.search(text)
        if match:
            tail = text[match.end() :].strip(" .,")
            return tail[:80]
    return ""


def structure(
    fact: str,
    *,
    session_id: str,
    session_date: str,
    user_id: str,
    chars_per_token: float = 4.6,
    now: datetime | None = None,
) -> Memory:
    text = fact.strip()
    subject = "assistant" if _ASSISTANT.match(text) else "user"
    predicate = infer_predicate(text)
    event_time = _parse_date(session_date)

    return Memory(
        id=memory_id(user_id, session_id, text),
        user_id=user_id,
        type=infer_type(text),
        content=text,
        token_count=max(1, int(len(text) / chars_per_token)),
        subject=subject,
        predicate=predicate,
        object=infer_object(text, predicate),
        importance=infer_importance(text),
        event_time=event_time,
        valid_from=event_time,
        valid_to=None,
        ingested_at=now or datetime.now(),
        replaces_previous=bool(_REPLACES.search(text)),
        entities=infer_entities(text),
        source_session_id=session_id,
    )
