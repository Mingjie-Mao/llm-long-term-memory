"""Exhaustive relation scan, used only where a router is confident enough to claim it.

Top-k similarity returns the *most relevant* memories, not *all* of them, so a question
that counts a set is structurally under-served: on the development count probes only
**57%** have every fact the correct answer needs, against 100% for every other operation.
Scanning one relation instead returned **100%** of a set's members on 68.5 median tokens,
against top-k's 58.3% on 439 (`results/analysis/exhaustive-scan.json`).

That measurement was taken with an oracle telling it which relation to scan. This module
is the same mechanism behind the real router, and it inherits the router's refusal: when
`Route.routed` is false, nothing here runs and the caller keeps its similarity ranking.

**Why a wrong scan is worse than no scan.** A scan returns a *complete* set. If the
relation is wrong, the answer is a confident, complete count of the wrong thing — and
completeness is exactly what makes such an answer look trustworthy. On 72 real questions
the confident routes were confidently wrong about what was being counted, which is why
the router abstains on 75% of them and why that is the correct behaviour rather than a
shortfall.

Union, not replacement: scanned memories are merged into the ranking rather than
substituted for it, so a scan that finds nothing cannot make retrieval worse than it was.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_long_term_memory.store import Memory, MemoryStore

from .hybrid import RetrievalSignals, RetrievedMemory
from .relation_router import RelationRouter, Route


@dataclass(frozen=True, slots=True)
class ScanOutcome:
    memories: list[RetrievedMemory]
    route: Route
    added: int


class RelationScanner:
    """Adds every member of a routed relation to what similarity already found."""

    def __init__(
        self,
        store: MemoryStore,
        router: RelationRouter,
        relation_of: dict[str, str],
        *,
        max_added: int = 40,
    ) -> None:
        self.store = store
        self.router = router
        # predicate_raw -> relation_type, from results/analysis/predicate-map.csv. Passed
        # in rather than read here so the mapping stays one file with one hash, which is
        # what the probe set and its held-out split are bound to.
        self.relation_of = relation_of
        # A namespace holds 100-180 memories and a set holds 3-13, so this bound is a
        # guard against a mis-mapped predicate dragging in a whole namespace, not a
        # ranking decision.
        self.max_added = max_added

    def expand(self, question: str, namespace: str, ranked: list[RetrievedMemory]) -> ScanOutcome:
        route = self.router.route(question)
        if not route.routed:
            return ScanOutcome(ranked, route, 0)

        wanted = set(route.relations)
        already = {hit.memory.id for hit in ranked}
        extra: list[Memory] = [
            memory
            for memory in self.store.iter_active(namespace)
            if memory.user_id == namespace
            and memory.subject == "user"
            and memory.id not in already
            and self.relation_of.get(memory.predicate or "") in wanted
        ]
        if len(extra) > self.max_added:
            return ScanOutcome(ranked, route, 0)

        # Scanned members carry no similarity score — they were not ranked, they were
        # enumerated. Giving them a fabricated one would put them in competition with
        # ranked results on a number that means nothing; they are appended instead, so
        # the existing order is untouched and the addition is visible as an addition.
        appended = [
            RetrievedMemory(
                memory=memory,
                score=0.0,
                signals=RetrievalSignals(0.0, 0.0, 0.0, memory.importance, 0.0),
                semantic_raw=-1.0,
                bm25_raw=None,
                strength=memory.strength,
            )
            for memory in extra
        ]
        return ScanOutcome([*ranked, *appended], route, len(appended))
