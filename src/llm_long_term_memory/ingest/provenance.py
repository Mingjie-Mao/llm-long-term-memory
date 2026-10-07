"""Deterministically anchor a structured memory to verbatim source text."""

from __future__ import annotations

import re

from llm_long_term_memory.conversation import ConversationSession
from llm_long_term_memory.store import Memory

from .event_time import temporal_evidence

_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"[^.!?]+(?:[.!?]+|$)")
_NUMBER = re.compile(r"\d")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def source_span_for(statement: str, session: ConversationSession) -> tuple[int, int, int] | None:
    """Choose the most lexically supported verbatim sentence in a source session.

    The extractor does not supply character offsets, so this is an auditable anchor
    rather than a claim that the LLM emitted an exact quote. Hydration expands one
    neighbouring sentence at read time, which recovers local temporal or numeric
    context without restoring the complete session.
    """
    statement_tokens = _tokens(statement)
    if not statement_tokens:
        return None

    best: tuple[float, int, int, int] | None = None
    for turn_index, turn in enumerate(session.turns):
        for match in _SENTENCE.finditer(turn.content):
            raw = match.group(0)
            start = match.start() + len(raw) - len(raw.lstrip())
            end = match.end() - len(raw) + len(raw.rstrip())
            text = turn.content[start:end]
            tokens = _tokens(text)
            shared = statement_tokens & tokens
            if not shared:
                continue
            score = float(len(shared))
            score += sum(2.0 for token in shared if _NUMBER.search(token))
            candidate = (score, -turn_index, -start, end)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        return None
    _, negative_turn, negative_start, end = best
    return -negative_turn, -negative_start, end


_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "did",
        "do",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "user",
        "users",
        "assistant",
    ]
)


def source_span_for_v2(
    statement: str, session: ConversationSession, role: str | None = None
) -> tuple[int, int, int] | None:
    """`source_span_for`, weighting what is distinctive rather than what is shared.

    v1 counts shared tokens, so "the", "user" and "from" outvote the one name the fact is
    about, and ties go to the earlier turn. v2 weights each shared token by how rare it
    is among the session's sentences (stop-words dropped), and adds a large bonus for
    each of the fact's specifics — numbers, quantities, multi-word names, as the
    fidelity detector finds them — that the sentence actually contains. Registered in
    `results/prereg-anchor-v2-offline-v1.md`.
    """
    import math

    from .fidelity import _extract_facets

    statement_tokens = _tokens(statement) - _STOP
    if not statement_tokens:
        return None
    specifics = {
        value.rstrip(",.;:!?").strip()
        for values in _extract_facets(statement).values()
        for value in values
    } - {""}
    # v3: a fact is anchored among the turns of whoever said it, as Zep links a fact to
    # the episode it came from. An assistant's essay that repeats "UCLA" and "Computer
    # Science" is not where the user said they studied there. Falls back to every turn
    # when the speaker said nothing the fact shares.
    if role is not None and any(t.role == role for t in session.turns):
        own_turns = ConversationSession(
            session_id=session.session_id,
            date=session.date,
            turns=[
                t if t.role == role else type(t)(role=t.role, content="") for t in session.turns
            ],
        )
        found = source_span_for_v2(statement, own_turns)
        if found is not None:
            return found
    sentences: list[tuple[int, int, int, str]] = []
    for turn_index, turn in enumerate(session.turns):
        for match in _SENTENCE.finditer(turn.content):
            raw = match.group(0)
            start = match.start() + len(raw) - len(raw.lstrip())
            end = match.end() - len(raw) + len(raw.rstrip())
            if end > start:
                sentences.append((turn_index, start, end, turn.content[start:end]))
    if not sentences:
        return None
    frequency: dict[str, int] = {}
    token_sets = []
    for *_, text in sentences:
        tokens = _tokens(text) - _STOP
        token_sets.append(tokens)
        for token in tokens:
            frequency[token] = frequency.get(token, 0) + 1

    def idf(token: str) -> float:
        return math.log(1 + len(sentences) / frequency.get(token, 1))

    best: tuple[float, int, int, int] | None = None
    for (turn_index, start, end, text), tokens in zip(sentences, token_sets, strict=True):
        shared = statement_tokens & tokens
        lowered = text.lower()
        carried = [value for value in specifics if value in lowered]
        if not shared and not carried:
            continue
        score = sum(idf(token) for token in shared)
        score += sum(3.0 * max((idf(t) for t in _tokens(v)), default=1.0) for v in carried)
        candidate = (score, -turn_index, -start, end)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None
    _, negative_turn, negative_start, end = best
    return -negative_turn, -negative_start, end


# v1  shared-token count, +2 per shared number; ties to the earlier turn.
# v2  idf-weighted shared tokens plus a bonus for the fact's own specifics. On the
#     no-gold reference of results/prereg-anchor-v2-offline-v1.md: 87.8% -> 96.5% on
#     train150 and 87.8% -> 96.2% on dev100; idf alone, which the reference cannot
#     flatter, gives 93.6% / 93.4%.
# v3  v2 restricted to the memory's own speaker's turns (`role=`) was tried and failed
#     the registered keep-rule: 97.9% / 98.2% of v1's correct anchors kept, against 99%.
ANCHOR_VERSION = "provenance-v2"


def attach_source_span(memory: Memory, session: ConversationSession) -> Memory:
    """Mutate a new memory with its raw-session anchor and return it."""
    span = source_span_for_v2(memory.content, session)
    if span is not None:
        memory.source_turn_index, memory.source_char_start, memory.source_char_end = span
        turn_index, start, end = span
        source_text = session.turns[turn_index].content[start:end]
        source_time = temporal_evidence(source_text, memory.observed_at)
        memory.event_time_source_expression = source_time.expression
    return memory
