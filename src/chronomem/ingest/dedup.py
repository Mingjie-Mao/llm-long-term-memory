"""Deciding whether a candidate fact is already known.

Two stages, for a reason worth stating plainly: **a similarity threshold alone
cannot do this job.**

"The user likes Python" and "The user does not like Python" embed at roughly 0.95
cosine similarity — they share every content word and differ by one negation. Any
threshold high enough to catch real duplicates also catches that pair, and dropping
it means the memory store silently keeps whichever opinion happened to arrive first
and never learns the user changed their mind. Meanwhile a threshold low enough to
separate them lets genuine restatements through and the store fills with near-copies.

So embeddings are used only as a cheap *recall* filter — cheap enough to run against
every candidate — and the actual decision is made by an LLM on the handful of
neighbours that survive. That keeps the expensive call rare (it fires on a few
percent of candidates) while making the decision on semantics rather than on cosine
distance.

The three-way verdict matters too. A binary duplicate/not-duplicate would collapse
UPDATE into one of the other two: treat "user moved to Sydney" as a duplicate of
"user lives in Canberra" and the move is lost; treat it as distinct and the store
now asserts two contradictory locations with nothing marking which is current.
UPDATE is what P4 consumes to set `valid_to` on the old fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from chronomem.embed import Encoder
from chronomem.llm.client import GeminiClient
from chronomem.store import Memory, MemoryStore, NumpyFlatIndex

from .schemas import DedupDecision, is_single_valued

DEDUP_SYSTEM = (
    "You compare two statements about a user for a memory system and decide whether "
    "the new one is already covered by the old one."
)

_PROMPT = """\
Existing memory: {old}
New candidate:   {new}

Classify the relationship:

- DUPLICATE — the same fact. Storing both adds nothing.
- UPDATE — the same attribute of the user, but a different value. The new statement \
supersedes the old one.
- DISTINCT — different facts. They may be worded similarly or share a topic, but \
both are worth keeping.

Pay attention to negation, quantity, and time. "The user likes Python" and "The user \
does not like Python" are UPDATE, never DUPLICATE. "The user has a sister in Osaka" \
and "The user has a brother in Osaka" are DISTINCT.
"""


@dataclass(slots=True)
class DedupOutcome:
    kept: list[Memory] = field(default_factory=list)
    duplicates: int = 0
    updates: list[tuple[Memory, Memory]] = field(default_factory=list)
    """(new, superseded_existing) pairs, handed to P4's temporal resolver."""
    adjudications: int = 0
    """How many LLM calls the stage actually cost."""


class Deduplicator:
    def __init__(
        self,
        client: GeminiClient,
        model: str,
        encoder: Encoder,
        store: MemoryStore | None = None,
        index: NumpyFlatIndex | None = None,
        threshold: float = 0.92,
        max_neighbours: int = 3,
    ) -> None:
        self.client = client
        self.model = model
        self.encoder = encoder
        self.store = store
        self.index = index
        self.threshold = threshold
        self.max_neighbours = max_neighbours

    def adjudicate(self, old: Memory, new: Memory) -> DedupDecision:
        completion = self.client.generate(
            role="extractor",
            model=self.model,
            prompt=_PROMPT.format(old=old.content, new=new.content),
            system=DEDUP_SYSTEM,
            schema=DedupDecision,
            temperature=0.0,
        )
        return DedupDecision.model_validate_json(completion.text)

    def process(self, candidates: list[Memory], vectors: np.ndarray | None = None) -> DedupOutcome:
        """Filter a batch of candidates against each other and against the store."""
        outcome = DedupOutcome()
        if not candidates:
            return outcome

        if vectors is None:
            vectors = self.encoder.encode([m.content for m in candidates])

        # Accepted-so-far, so that duplicates *within* one batch are caught too —
        # ten sessions of the same user restate the same facts constantly.
        seen_vectors: list[np.ndarray] = []
        seen_memories: list[Memory] = []

        for candidate, vector in zip(candidates, vectors, strict=True):
            neighbours = self._neighbours(candidate, vector, seen_vectors, seen_memories)
            verdict = None

            for neighbour in neighbours[: self.max_neighbours]:
                decision = self.adjudicate(neighbour, candidate)
                outcome.adjudications += 1
                if decision.verdict == "DUPLICATE":
                    verdict = "DUPLICATE"
                    break
                if decision.verdict == "UPDATE":
                    verdict = "UPDATE"
                    outcome.updates.append((candidate, neighbour))
                    break

            if verdict == "DUPLICATE":
                outcome.duplicates += 1
                continue

            outcome.kept.append(candidate)
            seen_vectors.append(vector)
            seen_memories.append(candidate)

        return outcome

    def _neighbours(
        self,
        candidate: Memory,
        vector: np.ndarray,
        seen_vectors: list[np.ndarray],
        seen_memories: list[Memory],
    ) -> list[Memory]:
        """Cheap recall filter. Everything it returns still gets adjudicated."""
        found: list[tuple[float, Memory]] = []

        if seen_vectors:
            sims = np.asarray(seen_vectors) @ vector
            for sim, mem in zip(sims, seen_memories, strict=True):
                if sim >= self.threshold:
                    found.append((float(sim), mem))

        if self.index is not None and self.store is not None and len(self.index):
            for mid, sim in self.index.search(vector, limit=self.max_neighbours):
                if sim >= self.threshold:
                    existing = self.store.get(mid)
                    if existing is not None:
                        found.append((sim, existing))

        # A (subject, predicate) collision is worth adjudicating only when the
        # predicate is single-valued. "lives in Canberra" vs "relocated to Sydney"
        # are far apart in embedding space and are exactly the collision UPDATE
        # exists for. But `user/owns/peace lily` vs `user/owns/Fitbit` collide on
        # the same key and are simply two different possessions — firing on those
        # cost 72 LLM calls per 60 sessions for zero verdicts, twelve times the
        # extraction budget.
        if (
            self.store is not None
            and candidate.subject
            and candidate.predicate
            and is_single_valued(candidate.predicate)
        ):
            for existing in self.store.find_by_predicate(
                candidate.user_id, candidate.subject, candidate.predicate
            ):
                if existing.id != candidate.id:
                    found.append((1.0, existing))

        found.sort(key=lambda pair: -pair[0])
        deduped: dict[str, Memory] = {}
        for _, mem in found:
            deduped.setdefault(mem.id, mem)
        return list(deduped.values())
