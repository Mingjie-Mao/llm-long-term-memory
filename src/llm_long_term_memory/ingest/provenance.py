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


def attach_source_span(memory: Memory, session: ConversationSession) -> Memory:
    """Mutate a new memory with its raw-session anchor and return it."""
    span = source_span_for(memory.content, session)
    if span is not None:
        memory.source_turn_index, memory.source_char_start, memory.source_char_end = span
        turn_index, start, end = span
        source_text = session.turns[turn_index].content[start:end]
        source_time = temporal_evidence(source_text, memory.observed_at)
        memory.event_time_source_expression = source_time.expression
    return memory
