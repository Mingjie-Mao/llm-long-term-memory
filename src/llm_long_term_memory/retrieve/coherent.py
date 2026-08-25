"""Assemble retrieved memories into whole sessions instead of a flat ranking.

Production hands the answerer the top twenty memories in score order, which mixes
conversations: a memory from March's bike repair can sit between two from June's
sculpture class. `results/context-arms.md` found that presenting the same facts as
the gold sessions' memories in event order changed the behaviour, while trimming
`top_k` without changing the shape made it *worse* — so the variable is the shape,
not the amount.

That experiment used `answer_session_ids`, which the product does not have. This
builds the same shape from retrieval alone, which is plausible because the gold
session is already being found: `source_session_recalled` is 94% on `heldout100`
and source-session recall has been 93.5-94.0% on every store measured. Nothing
new has to be retrieved. The sessions simply have to be recognised as units.

Nothing here calls a model. Every knob below is a **development decision to be made
on `train150` and frozen before `dev100` is touched**
([the pre-registration](../../../results/prereg-context-shape.md)), so the defaults
are starting points and not findings. The one thing this module must not do is
choose them by looking at validation data.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from ..store import Memory, external_session_id
from .hybrid import RetrievedMemory

Aggregate = Literal["max", "sum", "mean", "sum_top3"]
SessionOrder = Literal["score", "chronological"]


@dataclass(frozen=True, slots=True)
class SessionBudget:
    """How much of the archive a coherent context is allowed to spend.

    `max_total_memories` exists because the product claim is 75x less context than
    the transcript, and a whole session can be thirty memories. Without a ceiling
    this becomes `naive_rag` with extra steps — which is the failure mode
    `two_stage_hydrated` already demonstrated at 3x the context for no gain.
    """

    max_sessions: int = 3
    window_radius: int | None = None
    """Memories to keep either side of the session's best-scoring hit, in event
    order. `None` keeps the whole session."""
    max_total_memories: int | None = 20
    """Hard ceiling across all sessions, applied session by session so a session is
    never half-included in the middle of its own timeline."""
    aggregate: Aggregate = "sum_top3"
    """How a session's memories combine into one session score. `max` rewards a
    single strong hit; `sum` rewards long sessions regardless of relevance;
    `sum_top3` is the compromise and is the default only because it has to be
    something."""
    session_order: SessionOrder = "chronological"
    """Order sessions are laid out in. `chronological` gives the model a timeline
    across sessions, which is the thing `temporal-reasoning` questions need and the
    flat ranking destroys; `score` puts the most relevant first."""
    include_superseded: bool = False
    """Production applies temporal resolution and shows only facts in force.
    Including superseded ones would change what a memory means, not just its
    position, so it is off by default and is its own experiment."""


@dataclass(frozen=True, slots=True)
class CoherentContext:
    memories: tuple[Memory, ...]
    sessions: tuple[str, ...]
    """Selected session ids, in the order they appear in `memories`."""
    session_scores: dict[str, float] = field(default_factory=dict)
    dropped_sessions: tuple[str, ...] = ()
    """Sessions retrieval surfaced that the budget excluded. Kept for the same
    reason search reports `below_rank`: an absence without a reason is
    indistinguishable from a bug."""
    truncated: bool = False
    """Whether `max_total_memories` cut a session that would otherwise be included."""


def _event_key(memory: Memory) -> tuple[str, int]:
    stamp = memory.event_time or memory.valid_from
    return (str(stamp) if stamp else "", memory.source_turn_index or 0)


def rank_sessions(
    hits: Sequence[RetrievedMemory], aggregate: Aggregate = "sum_top3"
) -> list[tuple[str, float]]:
    """Score each session by the retrieved memories that came from it.

    Memories with no `source_session_id` cannot belong to a session and are
    dropped here rather than silently grouped under a `None` key — a null session
    that outranks real ones is the kind of bug that reads as a ranking problem.
    """
    by_session: dict[str, list[float]] = {}
    for hit in hits:
        session_id = hit.memory.source_session_id
        if not session_id:
            continue
        by_session.setdefault(session_id, []).append(hit.score)

    scored: list[tuple[str, float]] = []
    for session_id, scores in by_session.items():
        ordered = sorted(scores, reverse=True)
        if aggregate == "max":
            value = ordered[0]
        elif aggregate == "sum":
            value = sum(ordered)
        elif aggregate == "mean":
            value = sum(ordered) / len(ordered)
        elif aggregate == "sum_top3":
            value = sum(ordered[:3])
        else:  # pragma: no cover - Literal keeps this unreachable
            raise ValueError(f"unknown aggregate {aggregate!r}")
        scored.append((session_id, value))

    # Ties broken by session id so the same store and query always produce the same
    # context. A context that varies between runs would be indistinguishable from
    # the answerer variance this is meant to reduce.
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored


def _window(memories: list[Memory], best_ids: set[str], radius: int | None) -> list[Memory]:
    """Keep a contiguous slice of a session's timeline around its best hits.

    Contiguous in *event order*, not in score order: the point of a window is that
    what surrounds a fact explains it, and a window assembled by score is just a
    smaller flat ranking.
    """
    if radius is None:
        return memories
    anchors = [i for i, memory in enumerate(memories) if memory.id in best_ids]
    if not anchors:
        return memories
    low = max(0, min(anchors) - radius)
    high = min(len(memories), max(anchors) + radius + 1)
    return memories[low:high]


def build_coherent_context(
    hits: Sequence[RetrievedMemory],
    session_memories: Callable[[str], Iterable[Memory]],
    budget: SessionBudget | None = None,
    *,
    forced_session_ids: Iterable[str] | None = None,
) -> CoherentContext:
    """Group retrieved memories into whole sessions, in event order.

    `session_memories` returns every memory the store holds for one session,
    injected rather than taken from a store so this can be tested without one.
    """
    budget = budget or SessionBudget()
    ranked = rank_sessions(hits, budget.aggregate)
    forced_mode = forced_session_ids is not None
    forced = list(dict.fromkeys(forced_session_ids or ()))
    if not ranked and not forced_mode:
        return CoherentContext(memories=(), sessions=())

    if forced_mode:
        # Evaluation ceiling only: the real system does not know these ids. Keep
        # every other assembly rule identical. A forced session with no retrieved
        # hit gets score 0 and, because it has no anchor, contributes its whole
        # active timeline (still subject to the same hard context cap).
        scores = dict(ranked)
        forced_ranked = [(session_id, scores.get(session_id, 0.0)) for session_id in forced]
        chosen = forced_ranked[: budget.max_sessions]
        chosen_ids = {session_id for session_id, _ in chosen}
        dropped = [session_id for session_id, _ in forced_ranked[budget.max_sessions :]]
        dropped.extend(session_id for session_id, _ in ranked if session_id not in chosen_ids)
        reported_scores = dict(forced_ranked)
    else:
        chosen = ranked[: budget.max_sessions]
        dropped = [session_id for session_id, _ in ranked[budget.max_sessions :]]
        reported_scores = dict(ranked)

    best_by_session: dict[str, set[str]] = {}
    for hit in hits:
        session_id = hit.memory.source_session_id
        if session_id:
            best_by_session.setdefault(session_id, set()).add(hit.memory.id)

    blocks: list[tuple[str, list[Memory]]] = []
    for session_id, _ in chosen:
        memories = [
            memory
            for memory in session_memories(session_id)
            if budget.include_superseded or memory.status == "active"
        ]
        memories.sort(key=_event_key)
        block = _window(memories, best_by_session.get(session_id, set()), budget.window_radius)
        if block:
            blocks.append((session_id, block))

    if budget.session_order == "chronological":
        blocks.sort(key=lambda pair: _event_key(pair[1][0]))

    selected: list[Memory] = []
    kept: list[str] = []
    truncated = False
    for session_id, block in blocks:
        if (
            budget.max_total_memories is not None
            and len(selected) + len(block) > budget.max_total_memories
        ):
            # Whole sessions or nothing. Half a timeline is the shape this is
            # trying to get away from, and a session cut mid-sequence would
            # reintroduce it under a name that claims otherwise.
            truncated = True
            continue
        selected.extend(block)
        kept.append(session_id)

    dropped.extend(session_id for session_id, _ in blocks if session_id not in kept)
    return CoherentContext(
        memories=tuple(selected),
        sessions=tuple(kept),
        session_scores=reported_scores,
        dropped_sessions=tuple(dropped),
        truncated=truncated,
    )


def session_recall(
    hits: Sequence[RetrievedMemory], gold_sessions: Iterable[str], top_m: int
) -> bool:
    """Does any gold session survive into the top `top_m` sessions?

    This is the gate the context experiment runs before spending answerer quota.
    If a coherent context cannot contain the gold session, measuring how it is
    ordered is measuring nothing — and this costs no API calls to find out.
    """
    gold = set(gold_sessions)
    if not gold:
        return False
    ranked = [external_session_id(session_id) for session_id, _ in rank_sessions(hits)][:top_m]
    return any(session_id in gold for session_id in ranked)
