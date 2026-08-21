"""Session-coherent context assembly, with no model and no store.

The properties pinned here are the ones a flat ranking violates by construction:
a session arrives whole, in the order it happened, or it does not arrive.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from llm_long_term_memory.retrieve import (
    SessionBudget,
    build_coherent_context,
    rank_sessions,
    session_recall,
)
from llm_long_term_memory.retrieve.hybrid import RetrievalSignals, RetrievedMemory
from llm_long_term_memory.store import Memory

NOW = datetime(2026, 1, 31)


def memory(memory_id: str, session: str, day: int, turn: int = 0, **kwargs) -> Memory:
    return Memory(
        id=memory_id,
        user_id="u",
        type="semantic",
        content=f"{memory_id} from {session}",
        token_count=8,
        ingested_at=NOW,
        source_session_id=session,
        source_turn_index=turn,
        event_time=datetime(2026, 1, day),
        **kwargs,
    )


def hit(mem: Memory, score: float) -> RetrievedMemory:
    return RetrievedMemory(
        memory=mem,
        score=score,
        signals=RetrievalSignals(semantic=score, bm25=0.0, recency=0.0, importance=0.0, entity=0.0),
        semantic_raw=score,
        bm25_raw=None,
        strength=1.0,
    )


def library(*memories: Memory):
    """A `session_memories` callable over an in-memory list."""
    by_session: dict[str, list[Memory]] = {}
    for mem in memories:
        by_session.setdefault(mem.source_session_id or "", []).append(mem)
    return lambda session_id: list(by_session.get(session_id, []))


# --------------------------------------------------------------- session ranking


def test_sessions_rank_by_their_retrieved_memories():
    a1, a2 = memory("a1", "sess_a", 10), memory("a2", "sess_a", 11)
    b1 = memory("b1", "sess_b", 12)
    ranked = rank_sessions([hit(a1, 0.3), hit(a2, 0.3), hit(b1, 0.5)], "sum")
    assert ranked[0][0] == "sess_a"  # 0.6 beats 0.5

    ranked_max = rank_sessions([hit(a1, 0.3), hit(a2, 0.3), hit(b1, 0.5)], "max")
    assert ranked_max[0][0] == "sess_b"  # one strong hit wins under max


def test_a_memory_with_no_session_is_dropped_rather_than_grouped_under_none():
    orphan = memory("x", "sess_a", 10)
    object.__setattr__(orphan, "source_session_id", None)
    ranked = rank_sessions([hit(orphan, 0.9), hit(memory("a1", "sess_a", 10), 0.1)])
    assert [session for session, _ in ranked] == ["sess_a"]


def test_ties_break_deterministically_so_the_context_does_not_vary_between_runs():
    a = memory("a1", "sess_a", 10)
    b = memory("b1", "sess_b", 10)
    first = rank_sessions([hit(a, 0.5), hit(b, 0.5)])
    second = rank_sessions([hit(b, 0.5), hit(a, 0.5)])
    assert first == second == [("sess_a", 0.5), ("sess_b", 0.5)]


# ------------------------------------------------------------------ assembly


def test_a_session_arrives_whole_and_in_event_order():
    late, early = memory("a2", "sess_a", 20), memory("a1", "sess_a", 10)
    unretrieved = memory("a3", "sess_a", 15)
    ctx = build_coherent_context(
        [hit(late, 0.9)], library(late, early, unretrieved), SessionBudget(max_sessions=1)
    )
    # every memory of the session, oldest first — including the one retrieval missed
    assert [m.id for m in ctx.memories] == ["a1", "a3", "a2"]


def test_max_total_memories_never_cuts_a_session_in_half():
    big = [memory(f"a{i}", "sess_a", 10, turn=i) for i in range(15)]
    small = [memory("b1", "sess_b", 20)]
    ctx = build_coherent_context(
        [hit(big[0], 0.9), hit(small[0], 0.8)],
        library(*big, *small),
        SessionBudget(max_sessions=2, max_total_memories=10, session_order="score"),
    )
    # sess_a is 15 memories and does not fit; it is skipped entirely, not sliced
    assert ctx.sessions == ("sess_b",)
    assert "sess_a" in ctx.dropped_sessions
    assert ctx.truncated


def test_sessions_can_be_laid_out_chronologically_rather_than_by_score():
    older, newer = memory("a1", "sess_a", 5), memory("b1", "sess_b", 25)
    hits = [hit(newer, 0.9), hit(older, 0.4)]

    by_score = build_coherent_context(
        hits, library(older, newer), SessionBudget(session_order="score")
    )
    assert by_score.sessions == ("sess_b", "sess_a")

    by_time = build_coherent_context(
        hits, library(older, newer), SessionBudget(session_order="chronological")
    )
    assert by_time.sessions == ("sess_a", "sess_b")


def test_a_window_is_contiguous_in_event_order_not_in_score_order():
    session = [memory(f"a{i}", "sess_a", 10 + i) for i in range(7)]
    # the best hit is the middle one; a score-assembled window would pick strays
    ctx = build_coherent_context(
        [hit(session[3], 0.9), hit(session[0], 0.8)],
        library(*session),
        SessionBudget(max_sessions=1, window_radius=1),
    )
    # anchors are positions 0 and 3, so the slice spans 0-1 through 3+1
    assert [m.id for m in ctx.memories] == ["a0", "a1", "a2", "a3", "a4"]


def test_superseded_memories_are_excluded_by_default_and_includable_on_request():
    live = memory("a1", "sess_a", 10)
    dead = memory("a2", "sess_a", 11, status="superseded")
    hits = [hit(live, 0.9)]

    default = build_coherent_context(hits, library(live, dead))
    assert [m.id for m in default.memories] == ["a1"]

    including = build_coherent_context(
        hits, library(live, dead), SessionBudget(include_superseded=True)
    )
    assert [m.id for m in including.memories] == ["a1", "a2"]


def test_sessions_beyond_the_budget_are_reported_rather_than_vanishing():
    mems = [memory(f"m{i}", f"sess_{i}", 10 + i) for i in range(5)]
    ctx = build_coherent_context(
        [hit(m, 0.9 - i * 0.1) for i, m in enumerate(mems)],
        library(*mems),
        SessionBudget(max_sessions=2),
    )
    assert len(ctx.sessions) == 2
    assert len(ctx.dropped_sessions) == 3
    assert set(ctx.session_scores) == {f"sess_{i}" for i in range(5)}


def test_no_hits_yields_an_empty_context_rather_than_raising():
    ctx = build_coherent_context([], library())
    assert ctx.memories == () and ctx.sessions == ()


# --------------------------------------------------------------------- the gate


@pytest.mark.parametrize(
    ("top_m", "expected"),
    [(1, False), (2, True), (3, True)],
)
def test_session_recall_is_the_gate_the_experiment_runs_before_spending_quota(top_m, expected):
    strong = memory("b1", "sess_b", 10)
    gold = memory("a1", "sess_a", 10)
    hits = [hit(strong, 0.9), hit(gold, 0.4)]
    assert session_recall(hits, ["sess_a"], top_m) is expected


def test_session_recall_with_no_gold_is_false_rather_than_vacuously_true():
    assert session_recall([hit(memory("a1", "sess_a", 10), 0.9)], [], 3) is False
