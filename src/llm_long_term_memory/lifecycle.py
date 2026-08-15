"""P5 retention policy: gradual decay and explicit, auditable eviction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from llm_long_term_memory.store import Memory, MemoryStore


@dataclass(slots=True)
class DecayReport:
    examined: int = 0
    updated: int = 0
    strengths: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class EvictionReport:
    user_id: str
    limit: int
    evicted_ids: list[str] = field(default_factory=list)


def decayed_strength(memory: Memory, *, now: datetime, halflife_days: float) -> float:
    """Apply one exponential half-life from the last reinforcing interaction.

    The calculation is pure so a request never double-decays the same interval. The
    persisted value changes only through :func:`apply_decay`, and retrieval calls it
    before selection when decay is enabled.
    """
    if halflife_days <= 0:
        raise ValueError("halflife_days must be positive")
    anchor = memory.strength_updated_at or memory.last_accessed_at or memory.ingested_at
    if anchor is None:
        return memory.strength
    elapsed_days = max(0.0, (now - anchor).total_seconds() / 86_400)
    return memory.strength * math.exp(-math.log(2) * elapsed_days / halflife_days)


def apply_decay(
    store: MemoryStore,
    user_id: str,
    *,
    now: datetime,
    halflife_days: float,
) -> DecayReport:
    """Persist decayed active-memory strengths for one user namespace."""
    report = DecayReport()
    updates: dict[str, float] = {}
    for memory in store.iter_active(user_id):
        report.examined += 1
        strength = decayed_strength(memory, now=now, halflife_days=halflife_days)
        if not math.isclose(strength, memory.strength):
            updates[memory.id] = strength
    store.set_strengths(updates, at=now)
    report.updated = len(updates)
    report.strengths = updates
    return report


def evict_to_limit(store: MemoryStore, user_id: str, *, limit: int) -> EvictionReport:
    """Evict the weakest active records while retaining their database provenance."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    active = store.iter_active(user_id)
    if len(active) <= limit:
        return EvictionReport(user_id=user_id, limit=limit)

    def eviction_key(memory: Memory) -> tuple[float, datetime, str]:
        # Lower strength x importance is less valuable. Stable ties ensure a re-run
        # cannot churn a fixed-capacity store merely because SQLite returned rows in
        # a different physical order.
        when = memory.last_accessed_at or memory.ingested_at or datetime.min
        return (memory.strength * memory.importance, when, memory.id)

    victims = sorted(active, key=eviction_key)[: len(active) - limit]
    ids = [memory.id for memory in victims]
    store.mark_evicted(ids)
    return EvictionReport(user_id=user_id, limit=limit, evicted_ids=ids)
