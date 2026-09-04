"""Rehydrate compact, verbatim source evidence after structured retrieval."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from llm_long_term_memory.store import Memory, MemoryStore

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
    redundant_anchors: int = 0
    eligible_sessions: int = 0
    hydrated_sessions: int = 0


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
        allocation: Literal["ranked", "session_fair"] = "ranked",
    ) -> None:
        if allocation not in {"ranked", "session_fair"}:
            raise ValueError(f"unknown hydration allocation: {allocation!r}")
        self.store = store
        self.neighbouring_sentences = max(0, neighbouring_sentences)
        self.chars_per_token = chars_per_token
        self.allocation = allocation

    @staticmethod
    def _session_fair_order(evidence: list[HydratedEvidence]) -> list[HydratedEvidence]:
        """Keep rank within a session but give every session an equal first chance.

        Retrieval rank still decides which session appears first and which source span
        represents it. Round-robin only prevents several memories from that first
        session consuming the whole raw-evidence budget before another selected
        session contributes anything.
        """
        by_session: dict[str, list[HydratedEvidence]] = defaultdict(list)
        for item in evidence:
            by_session[item.session_id].append(item)
        ordered: list[HydratedEvidence] = []
        depth = 0
        while True:
            added = False
            for items in by_session.values():
                if depth < len(items):
                    ordered.append(items[depth])
                    added = True
            if not added:
                return ordered
            depth += 1

    def hydrate(self, memories: list[Memory], max_tokens: int = 0) -> HydrationResult:
        result = HydrationResult()
        seen: set[tuple[str, int, int, int]] = set()
        sessions: dict[str, dict[int, object]] = {}
        candidates: list[HydratedEvidence] = []

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
            candidates.append(
                HydratedEvidence(
                    memory_id=memory.id,
                    session_id=session_id,
                    turn_index=turn_index,
                    role=turn.role,
                    text=text,
                    token_count=token_count,
                )
            )

        if self.allocation == "session_fair":
            # Different extracted facts can point at different character spans in
            # the same source sentence. Sending that sentence twice costs tokens
            # without adding evidence, so compact mode keeps its first-ranked copy.
            unique: list[HydratedEvidence] = []
            seen_source: set[tuple[str, int, str]] = set()
            for item in candidates:
                source = (item.session_id, item.turn_index, item.text)
                if source in seen_source:
                    result.redundant_anchors += 1
                    continue
                seen_source.add(source)
                unique.append(item)
            candidates = self._session_fair_order(unique)

        result.eligible_sessions = len({item.session_id for item in candidates})
        for item in candidates:
            if max_tokens and result.tokens + item.token_count > max_tokens:
                result.skipped_for_budget += 1
                continue
            result.evidence.append(item)
            result.tokens += item.token_count
        result.hydrated_sessions = len({item.session_id for item in result.evidence})
        return result


def render_evidence(evidence: list[HydratedEvidence]) -> str:
    """Render source text as an explicitly quoted, inspectable prompt section."""
    return "\n".join(
        f"- [{item.session_id}:{item.turn_index} {item.role}] {item.text}" for item in evidence
    )
