"""Cluster related memories and synthesize a traceable semantic summary."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field

from chronomem.embed import Encoder
from chronomem.llm.client import GeminiClient
from chronomem.store import Memory, MemoryStore, VectorIndex

_SYSTEM = (
    "You consolidate related long-term memories. Preserve facts, qualifiers, and time "
    "ordering; never invent a fact that is not present in the source memories."
)
_PROMPT = """\
Synthesize the related memories below into one short, self-contained semantic memory.

Keep specific names, quantities, and dates whenever they distinguish the facts. Do
not erase a contradiction; state it as a time-qualified change. Return confidence as
a number from 0 to 1 based only on agreement among the source memories.

{memories}
"""


class Synthesis(BaseModel):
    content: str = Field(description="One self-contained consolidated memory")
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass(slots=True)
class ConsolidationReport:
    clusters_found: int = 0
    memories_created: list[str] = field(default_factory=list)
    source_ids: dict[str, list[str]] = field(default_factory=dict)


class Consolidator:
    """A bounded background job; it never deletes source memories."""

    def __init__(
        self,
        client: GeminiClient,
        model: str,
        encoder: Encoder,
        store: MemoryStore,
        index: VectorIndex,
        *,
        similarity_threshold: float = 0.84,
        min_cluster_size: int = 3,
        source_strength_multiplier: float = 0.5,
        chars_per_token: float = 4.6,
    ) -> None:
        self.client = client
        self.model = model
        self.encoder = encoder
        self.store = store
        self.index = index
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = max(2, min_cluster_size)
        self.source_strength_multiplier = min(1.0, max(0.0, source_strength_multiplier))
        self.chars_per_token = chars_per_token

    def consolidate(self, user_id: str) -> ConsolidationReport:
        report = ConsolidationReport()
        for cluster in self._clusters(user_id):
            report.clusters_found += 1
            summary = self._synthesize(user_id, cluster)
            self.store.add_memories([summary])
            self.store.add_evidence(summary.id, [memory.id for memory in cluster])
            self.store.set_strengths(
                {
                    memory.id: memory.strength * self.source_strength_multiplier
                    for memory in cluster
                },
                at=datetime.now(),
            )
            self.index.add([summary.id], self.encoder.encode([summary.content]))
            report.memories_created.append(summary.id)
            report.source_ids[summary.id] = [memory.id for memory in cluster]
        return report

    def _clusters(self, user_id: str) -> list[list[Memory]]:
        consumed_sources = self.store.evidence_source_ids()
        candidates = [
            memory
            for memory in self.store.iter_active(user_id)
            if memory.type != "profile"
            and memory.id not in consumed_sources
            and not self.store.evidence_for(memory.id)
        ]
        by_type: dict[str, list[Memory]] = {}
        for memory in candidates:
            by_type.setdefault(memory.type, []).append(memory)

        clusters: list[list[Memory]] = []
        for memories in by_type.values():
            if len(memories) < self.min_cluster_size:
                continue
            vectors = self.encoder.encode([memory.content for memory in memories])
            clusters.extend(self._connected_components(memories, vectors))
        return clusters

    def _connected_components(
        self, memories: list[Memory], vectors: np.ndarray
    ) -> list[list[Memory]]:
        neighbours = [set() for _ in memories]
        similarities = vectors @ vectors.T
        for left in range(len(memories)):
            for right in range(left + 1, len(memories)):
                if similarities[left, right] >= self.similarity_threshold:
                    neighbours[left].add(right)
                    neighbours[right].add(left)

        clusters: list[list[Memory]] = []
        seen: set[int] = set()
        for start in range(len(memories)):
            if start in seen:
                continue
            stack, component = [start], []
            seen.add(start)
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbour in neighbours[node]:
                    if neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
            if len(component) >= self.min_cluster_size:
                clusters.append([memories[index] for index in sorted(component)])
        return clusters

    def _synthesize(self, user_id: str, cluster: list[Memory]) -> Memory:
        rendered = "\n".join(f"- {memory.content}" for memory in cluster)
        completion = self.client.generate(
            role="extractor",
            model=self.model,
            prompt=_PROMPT.format(memories=rendered),
            system=_SYSTEM,
            schema=Synthesis,
            temperature=0.0,
            est_input_tokens=int(len(rendered) / self.chars_per_token),
        )
        synthesis = Synthesis.model_validate_json(completion.text)
        content = synthesis.content.strip()
        digest = hashlib.sha256(
            f"{user_id}|{'|'.join(sorted(memory.id for memory in cluster))}|{content}".encode()
        ).hexdigest()[:20]
        return Memory(
            id=f"consolidated:{digest}",
            user_id=user_id,
            type="semantic",
            content=content,
            token_count=max(1, int(len(content) / self.chars_per_token)),
            subject="user",
            predicate="consolidated_memory",
            object="",
            confidence=synthesis.confidence,
            importance=max(memory.importance for memory in cluster),
            ingested_at=datetime.now(),
            entities=list(
                dict.fromkeys(entity for memory in cluster for entity in memory.entities)
            ),
        )
