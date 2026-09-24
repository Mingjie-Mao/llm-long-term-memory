"""Add back the specifics batched extraction dropped, and nothing else.

Batching sessions into one request is what the free tier bought, and it is also what
makes the extractor summarise: at fifteen sessions a request it retained 37.3% of the
user's own numbers, durations, names and dates, at eight 64.9%, at one 78.4%. Batch
eight is the affordable point on that curve, and it still loses a third.

This recovers part of that third without re-extracting anything. After the batch has
been extracted, each session's user assertions are compared against what was written
using the same facet detector the fidelity ruler uses. A session that lost nothing costs
nothing. A session that lost something gets **one** grounded extraction call, and a fact
from it is accepted only when

* it is anchored to an exact span of a **user** turn — the grounded extractor refuses a
  fact whose quoted span is not in the turn it names, rather than storing it anyway; and
* at least one of that session's baseline-missing specifics appears in both the quoted
  span and the fact's own content.

So the repair is **additive**: it never edits or removes a baseline memory, which is why
its measured losses are zero by construction rather than by luck. Registered and
measured in `results/prereg-specificity-repair-pilot.md`: 44.8% -> 59.3% on holdout
sessions 61-120, 21 specifics gained and 0 lost, 25 of 60 sessions calling, 0.33 added
memories per session, 0 grounding failures.

**Off by default.** It costs up to one request per session, which is the same order as
the batched extraction it supplements, and `ingest.specificity_repair` has to be set for
it to run. Turning it on changes what gets written, so it is part of the ingest
fingerprint and a store built without it refuses to resume with it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from llm_long_term_memory.conversation import ConversationSession
from llm_long_term_memory.store import Memory

from .fidelity import _extract_facets, user_assertions
from .grounded import Anchored, GroundedExtractor, normalise

REPAIR_VERSION = "specificity-repair-v1"


def missing_specifics(session: ConversationSession, memories: list[Memory]) -> set[str]:
    """Specifics the user stated that no memory of this session carries.

    The same detector and the same haystack the fidelity ruler scores with, so what
    triggers a repair is exactly what the ruler would have counted as lost.
    """
    stated = _extract_facets(user_assertions(session))
    blob = " || ".join(f"{memory.content} {memory.object or ''}" for memory in memories).lower()
    return {value for values in stated.values() for value in values if value not in blob}


def grounded_memory(anchored: Anchored, session_id: str, user_id: str) -> Memory:
    """One anchored fact as a `Memory`, carrying the span it was taken from.

    The id is a digest of the tenant, the session, the quoted span and the content, so
    re-running a repair produces the same id in any process and the write stays
    idempotent. The tenant is in it because one haystack session can belong to several
    tenants, and the store writes with `INSERT OR REPLACE`: an id shared across tenants
    would let one tenant's repair overwrite another's.
    """
    fact = anchored.fact
    key = f"{user_id}|{session_id}|{fact.verbatim_span}|{fact.content}"
    return Memory(
        id=f"g_{hashlib.sha1(key.encode()).hexdigest()[:16]}",
        user_id=user_id,
        type="semantic",
        content=fact.content,
        token_count=max(1, len(fact.content.split())),
        subject=fact.subject,
        predicate=fact.attribute or None,
        object=(f"{fact.value} {fact.unit}".strip() or None) if fact.value else None,
        source_session_id=session_id,
        source_turn_index=anchored.turn_index,
        source_char_start=anchored.char_start,
        source_char_end=anchored.char_end,
    )


@dataclass(slots=True)
class RepairOutcome:
    memories: list[Memory]
    called: bool
    """Whether a request was spent. A session that lost nothing costs nothing."""

    missing: list[str]
    attempted: int = 0
    rejected: int = 0
    """Facts the grounded extractor refused because their span was not in the turn."""


class SpecificityRepair:
    """One conditional grounded call per session that lost a specific."""

    version = REPAIR_VERSION

    def __init__(self, extractor: GroundedExtractor) -> None:
        self.extractor = extractor

    def prompt_texts(self) -> tuple[str, ...]:
        """Part of the ingest fingerprint: the repair prompt decides what is written."""
        return (REPAIR_VERSION, *self.extractor.prompt_texts())

    def repair(
        self, session: ConversationSession, memories: list[Memory], user_id: str
    ) -> RepairOutcome:
        missing = sorted(missing_specifics(session, memories))
        if not missing:
            return RepairOutcome([], called=False, missing=[])

        report = self.extractor.extract(
            [(turn.role, turn.content) for turn in session.turns], session.date
        )
        accepted: list[Memory] = []
        for anchored in report.anchored:
            # Only the user's own statements. An assistant turn is a legitimate source
            # for other memories, but what was measured as lost is what the *user*
            # said, and repairing from elsewhere would be answering another question.
            if session.turns[anchored.turn_index].role != "user":
                continue
            content = normalise(anchored.fact.content)
            span = normalise(anchored.fact.verbatim_span)
            # In both, not either. A specific present only in the content is the model
            # asserting it, and present only in the span is the repair not carrying it.
            if any(value in content and value in span for value in missing):
                accepted.append(grounded_memory(anchored, session.session_id, user_id))
        return RepairOutcome(
            accepted,
            called=True,
            missing=missing,
            attempted=len(report.anchored) + len(report.rejected),
            rejected=len(report.rejected),
        )
