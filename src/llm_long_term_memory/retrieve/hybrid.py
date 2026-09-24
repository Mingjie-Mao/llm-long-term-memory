"""Deterministic, explainable hybrid retrieval.

All five signals are normalized to ``[0, 1]`` before weights are applied. This is
not cosmetic: SQLite FTS5's BM25 values are negative and unbounded, whereas cosine
similarity is bounded. Adding their raw values would make the lexical term dominate
or vanish according to an implementation detail instead of a configured weight.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from llm_long_term_memory.store import Memory, MemoryStore, VectorIndex

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class RetrievalSignals:
    semantic: float
    bm25: float
    recency: float
    importance: float
    entity: float

    def weighted(self, weights: Mapping[str, float]) -> float:
        return sum(
            getattr(self, name) * float(weights.get(name, 0.0))
            for name in self.__dataclass_fields__
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "semantic": self.semantic,
            "bm25": self.bm25,
            "recency": self.recency,
            "importance": self.importance,
            "entity": self.entity,
        }


@dataclass(frozen=True, slots=True)
class RetrievedMemory:
    memory: Memory
    score: float
    """The weighted hybrid score. Retained even when reranking replaced the
    ordering, so that "ranked 7th by hybrid, 1st after reranking" stays visible."""
    signals: RetrievalSignals
    semantic_raw: float
    bm25_raw: float | None
    strength: float
    rerank_score: float | None = None
    """Cross-encoder score; None when reranking is disabled or the candidate was
    outside the rerank pool. Not comparable to `score` — different scale."""


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """What recall looked like *before* truncation and reranking.

    A single end-of-pipeline recall number cannot tell you whether a stage rescued
    the evidence or discarded it. Comparing recall at each stage can: if candidate
    recall is 100% and post-rerank recall is 90%, the reranker is deleting correct
    evidence, and that is a different problem from failing to find it.
    """

    candidate_ids: tuple[str, ...]
    """Every candidate that survived namespace and status filtering, in hybrid order."""
    candidate_session_ids: frozenset[str]
    reranked: bool


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN.findall(text.lower()) if len(token) > 2}


def _min_max(values: dict[str, float], *, lower_is_better: bool = False) -> dict[str, float]:
    """Normalize observed candidates; one unambiguous hit gets full credit."""
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if math.isclose(low, high):
        return {key: 1.0 for key in values}
    if lower_is_better:
        return {key: (high - value) / (high - low) for key, value in values.items()}
    return {key: (value - low) / (high - low) for key, value in values.items()}


class HybridRetriever:
    """Union semantic and lexical candidates, then score five independent signals."""

    def __init__(
        self,
        store: MemoryStore,
        index: VectorIndex,
        *,
        weights: Mapping[str, float],
        candidate_limit: int = 50,
        recency_halflife_days: float = 30.0,
        use_strength: bool = False,
        now: datetime | None = None,
        reranker=None,
    ) -> None:
        self.store = store
        self.index = index
        self.weights = {name: float(value) for name, value in weights.items()}
        self.candidate_limit = max(1, candidate_limit)
        self.recency_halflife_days = max(0.001, recency_halflife_days)
        self.use_strength = use_strength
        self.now = now
        # Injected rather than constructed here, so this module stays free of a
        # torch dependency and the reranker can be stubbed in tests.
        self.reranker = reranker

    def retrieve(
        self,
        query: np.ndarray,
        query_text: str,
        namespace: str,
        *,
        temporal: bool,
        limit: int,
        as_of: datetime | None = None,
    ) -> list[RetrievedMemory]:
        """Return ranked candidates without leaking namespaces or evicted records."""
        return self.retrieve_with_trace(
            query, query_text, namespace, temporal=temporal, limit=limit, as_of=as_of
        )[0]

    def retrieve_with_trace(
        self,
        query: np.ndarray,
        query_text: str,
        namespace: str,
        *,
        temporal: bool,
        limit: int,
        as_of: datetime | None = None,
    ) -> tuple[list[RetrievedMemory], RetrievalTrace]:
        """`retrieve`, plus the pre-truncation candidate set for recall attribution.

        `as_of` is the moment the question is being asked. Age was measured against
        `datetime.now()` and nothing ever passed a reference, so a memory's recency was
        its distance from **the wall clock of whoever ran the code**. Two consequences,
        and the second is the serious one: a question set in 2023 scored every 2023
        memory as ancient, and the same store ranked differently tomorrow than today —
        which quietly breaks the property every offline falsification in this project
        rests on, that retrieval is deterministic given the store.

        Masked until now because the shipped weight on recency is 0.0, so the signal is
        multiplied away. It stops being masked the moment anyone turns it on.
        """
        semantic_hits = self.index.search(query, limit=len(self.index))
        semantic_raw = dict(semantic_hits)
        semantic_memories = self.store.get_many([memory_id for memory_id, _ in semantic_hits])
        semantic_ids = [
            memory.id
            for memory in semantic_memories
            if memory.user_id == namespace
            and memory.status != "evicted"
            and (not temporal or memory.status == "active")
        ][: self.candidate_limit]

        lexical_hits = self.store.search_lexical(
            namespace,
            query_text,
            self.candidate_limit,
            include_superseded=not temporal,
        )
        bm25_raw = {hit.memory_id: hit.score for hit in lexical_hits}

        candidate_ids = list(dict.fromkeys([*semantic_ids, *bm25_raw]))
        memories = self.store.get_many(candidate_ids)
        memories = [
            memory
            for memory in memories
            if memory.user_id == namespace
            and memory.status != "evicted"
            and (not temporal or memory.status == "active")
        ]
        if not memories:
            return [], RetrievalTrace((), frozenset(), self.reranker is not None)

        bm25 = _min_max(
            {memory.id: bm25_raw[memory.id] for memory in memories if memory.id in bm25_raw},
            lower_is_better=True,
        )
        query_terms = _tokens(query_text)
        now = as_of or self.now or datetime.now()
        scored = []
        for memory in memories:
            raw = semantic_raw.get(memory.id, -1.0)
            when = memory.occurred_at or memory.ingested_at
            age_days = max(0.0, (now - when).total_seconds() / 86_400) if when else 0.0
            recency = math.exp(-math.log(2) * age_days / self.recency_halflife_days)
            entity = self._entity_overlap(memory, query_terms)
            signals = RetrievalSignals(
                semantic=_clip((raw + 1.0) / 2.0),
                bm25=_clip(bm25.get(memory.id, 0.0)),
                recency=_clip(recency),
                importance=_clip(memory.importance),
                entity=_clip(entity),
            )
            scored.append(
                RetrievedMemory(
                    memory=memory,
                    score=signals.weighted(self.weights)
                    * (memory.strength if self.use_strength else 1.0),
                    signals=signals,
                    semantic_raw=raw,
                    bm25_raw=bm25_raw.get(memory.id),
                    strength=memory.strength,
                )
            )

        scored.sort(key=lambda hit: (-hit.score, -hit.semantic_raw, hit.memory.id))
        trace = RetrievalTrace(
            candidate_ids=tuple(hit.memory.id for hit in scored),
            candidate_session_ids=frozenset(
                hit.memory.source_session_id for hit in scored if hit.memory.source_session_id
            ),
            reranked=self.reranker is not None,
        )
        # Reranking sits between recall and truncation on purpose: it can only
        # rescue a memory the hybrid stage ranked below `limit` if it runs before
        # the cut, and it can only be afforded at all because the cut happens after.
        if self.reranker is not None:
            return self.reranker.rerank(query_text, scored, limit), trace
        return scored[:limit], trace

    @staticmethod
    def _entity_overlap(memory: Memory, query_terms: set[str]) -> float:
        if not memory.entities or not query_terms:
            return 0.0
        overlaps = []
        for entity in memory.entities:
            entity_terms = _tokens(entity)
            if entity_terms:
                overlaps.append(len(entity_terms & query_terms) / len(entity_terms))
        return max(overlaps, default=0.0)
