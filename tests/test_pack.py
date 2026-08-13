"""Budget packing."""

from __future__ import annotations

from chronomem.pack import pack
from chronomem.store import Memory


def mem(mid: str, content: str, tokens: int, mtype: str = "semantic") -> Memory:
    return Memory(id=mid, user_id="u", type=mtype, content=content, token_count=tokens)


def test_value_per_token_beats_value_alone():
    """A memory worth 0.9 at 60 tokens is a worse buy than three worth 0.4 at 12.
    Sorting by score prefers long memories, which is what a token budget least
    wants."""
    big = mem("big", "one long expensive fact about travel", 60)
    smalls = [mem(f"s{i}", f"short cheap fact number {i}", 12) for i in range(3)]

    result = pack([big, *smalls], [0.9, 0.4, 0.4, 0.4], budget=40)

    assert "big" not in {m.id for m in result.selected}
    assert len(result.selected) == 3


def test_a_harmful_memory_is_excluded_even_with_budget_to_spare():
    """Negative utility is why the label is signed rather than a ranking: a stale
    fact left in because there was room is exactly the v1 failure."""
    good = mem("good", "the user lives in Sydney", 10)
    stale = mem("stale", "the user lives in Canberra", 10)

    result = pack([good, stale], [0.9, -1.0], budget=1000)

    assert [m.id for m in result.selected] == ["good"]
    assert result.dropped_negative == 1
    assert result.tokens_used == 10, "the budget is not spent just because it exists"


def test_a_near_duplicate_is_penalised_not_paid_for_twice():
    a = mem("a", "the user lives in Canberra with two cats", 12)
    twin = mem("twin", "the user lives in Canberra and has two cats", 12)
    other = mem("other", "the user prefers oat milk in coffee", 12)

    result = pack([a, twin, other], [0.9, 0.9, 0.5], budget=24)
    chosen = {m.id for m in result.selected}

    assert "a" in chosen
    assert "other" in chosen, "the distinct fact beat the near-copy"
    assert "twin" not in chosen


def test_a_type_floor_reserves_room_a_ranking_would_have_taken():
    """Without a floor, high-scoring episodic memories crowd out the profile facts
    nearly every question needs some of."""
    episodics = [mem(f"e{i}", f"episode {i}", 10, "episodic") for i in range(5)]
    profile = mem("p", "the user is a graphic designer", 10, "profile")

    floored = pack(
        [*episodics, profile], [0.9] * 5 + [0.2], budget=30, type_floors={"profile": 0.34}
    )
    unfloored = pack([*episodics, profile], [0.9] * 5 + [0.2], budget=30)

    assert "p" in {m.id for m in floored.selected}
    assert "p" not in {m.id for m in unfloored.selected}


def test_nothing_exceeds_the_budget():
    memories = [mem(f"m{i}", f"fact {i}", 30) for i in range(10)]
    result = pack(memories, [0.5] * 10, budget=100)
    assert result.tokens_used <= 100
    assert sum(m.token_count for m in result.selected) == result.tokens_used


def test_a_memory_larger_than_the_whole_budget_is_skipped_not_crashed():
    result = pack([mem("huge", "x", 5000)], [1.0], budget=100)
    assert result.selected == []
    assert result.tokens_used == 0


def test_empty_inputs_are_handled():
    assert pack([], [], budget=100).selected == []
    assert pack([mem("m", "x", 5)], [1.0], budget=0).selected == []
