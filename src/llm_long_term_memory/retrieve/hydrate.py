"""Rehydrate compact, verbatim source evidence after structured retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from chronomem.store import Memory, MemoryStore

_SENTENCE = re.compile(r"[^.!?]+(?:[.!?]+|$)")


@dataclass(frozen=True, slots=True)
class HydratedEvidence:
    memory_id: str
    session_id: str
    turn_index: int
    role: str
    text: str
    token_count: int


@dataclass(slots=True)
class HydrationResult:
    evidence: list[HydratedEvidence] = field(default_factory=list)
    tokens: int = 0
    missing_anchors: int = 0
    skipped_for_budget: int = 0


def _expand_to_sentences(text: str, start: int, end: int, neighbours: int) -> str:
    sentences = list(_SENTENCE.finditer(text))
    if not sentences:
        return text.strip()
    selected = next(
        (
            index
            for index, sentence in enumerate(sentences)
            if sentence.start() <= start and end <= sentence.end()
        ),
        None,
    )
    if selected is None:
        return text[max(0, start) : min(len(text), end)].strip()
    first = max(0, selected - neighbours)
    last = min(len(sentences) - 1, selected + neighbours)
    return text[sentences[first].start() : sentences[last].end()].strip()


class EvidenceHydrator:
    """Recover raw local context without making it a second retrieval index."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        neighbouring_sentences: int = 1,
        chars_per_token: float = 4.6,
    ) -> None:
        self.store = store
        self.neighbouring_sentences = max(0, neighbouring_sentences)
        self.chars_per_token = chars_per_token

    def hydrate(self, memories: list[Memory], max_tokens: int = 0) -> HydrationResult:
        result = HydrationResult()
        seen: set[tuple[str, int, int, int]] = set()
        sessions: dict[str, dict[int, object]] = {}

        for memory in memories:
            anchor = (
                memory.source_session_id,
                memory.source_turn_index,
                memory.source_char_start,
                memory.source_char_end,
            )
            if any(value is None for value in anchor):
                result.missing_anchors += 1
                continue
            session_id, turn_index, start, end = anchor
            key = (session_id, turn_index, start, end)
            if key in seen:
                continue
            seen.add(key)

            turns = sessions.setdefault(
                session_id,
                {turn.turn_index: turn for turn in self.store.turns_for_session(session_id)},
            )
            turn = turns.get(turn_index)
            if turn is None:
                result.missing_anchors += 1
                continue
            text = _expand_to_sentences(turn.content, start, end, self.neighbouring_sentences)
            token_count = max(1, int(len(text) / self.chars_per_token)) if text else 0
            if max_tokens and result.tokens + token_count > max_tokens:
                result.skipped_for_budget += 1
                continue
            result.evidence.append(
                HydratedEvidence(
                    memory_id=memory.id,
                    session_id=session_id,
                    turn_index=turn_index,
                    role=turn.role,
                    text=text,
                    token_count=token_count,
                )
            )
            result.tokens += token_count
        return result


def render_evidence(evidence: list[HydratedEvidence]) -> str:
    """Render source text as an explicitly quoted, inspectable prompt section."""
    return "\n".join(
        f"- [{item.session_id}:{item.turn_index} {item.role}] {item.text}" for item in evidence
    )
