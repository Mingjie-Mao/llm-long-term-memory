"""A supersession chain must reach the answerer as a chain.

The store already decides which value of a key is current — it closes the old one's
validity window and sets its status. Flat rendering discards that and leaves the model
to rediscover it from two bullet points differing only in a parenthesised date. On
`train150` it did not: asked where a guitar was serviced, with both the plan and the
completion in context, v3.3 answered that the user *plans* to send it.

These tests pin the rendering, not the model. They are cheap and they fail loudly if
the grouping regresses to a flat list.
"""

from __future__ import annotations

from datetime import datetime

from llm_long_term_memory.evaluation.runners.memory import (
    render_grouped,
    render_timelines,
)
from llm_long_term_memory.store import Memory


def _memory(
    identifier: str,
    content: str,
    *,
    subject: str = "user",
    predicate: str = "guitar_repair",
    start: str | None = None,
    end: str | None = None,
    status: str = "active",
    scope: str | None = "plan",
) -> Memory:
    return Memory(
        id=identifier,
        user_id="u1",
        type="episodic",
        content=content,
        token_count=len(content) // 4,
        subject=subject,
        predicate=predicate,
        scope=scope,
        valid_from=datetime.fromisoformat(start) if start else None,
        valid_to=datetime.fromisoformat(end) if end else None,
        status=status,
    )


def test_a_chain_is_rendered_in_event_order_with_the_current_value_marked():
    chain = [
        _memory("m2", "sent the guitar to the shop on Main St", start="2026-07-05"),
        _memory(
            "m1",
            "plans to send the guitar for repair",
            start="2026-07-01",
            end="2026-07-05",
            status="superseded",
        ),
    ]
    rendered, remaining = render_timelines(chain, temporal=True)

    assert not remaining, "both values belong to the chain"
    assert rendered.index("2026-07-01") < rendered.index("2026-07-05"), "not in event order"
    # The distinction the flat renderer lost: which one holds now.
    current_line = next(line for line in rendered.splitlines() if "[CURRENT]" in line)
    assert "Main St" in current_line
    was_line = next(line for line in rendered.splitlines() if "[was]" in line)
    assert "plans to send" in was_line


def test_a_single_value_is_not_dressed_up_as_a_timeline():
    """One value nothing contradicted is not a history, and labelling it 'current'
    would imply something was superseded."""
    rendered, remaining = render_timelines(
        [_memory("m1", "owns a Fender Stratocaster", start="2026-07-01")], temporal=True
    )
    assert rendered == ""
    assert len(remaining) == 1


def test_nothing_is_grouped_without_dates():
    """`temporal=False` means no validity windows were retrieved, so any ordering
    shown would be invented."""
    chain = [
        _memory("m1", "plans to send the guitar for repair", start="2026-07-01"),
        _memory("m2", "sent the guitar to the shop", start="2026-07-05"),
    ]
    rendered, remaining = render_timelines(chain, temporal=False)
    assert rendered == ""
    assert len(remaining) == 2


def test_a_chained_memory_is_not_also_listed_flat():
    """Seeing a superseded value twice — once marked stale, once not — is worse than
    either rendering on its own."""
    chain = [
        _memory(
            "m1",
            "plans to send the guitar",
            start="2026-07-01",
            end="2026-07-05",
            status="superseded",
        ),
        _memory("m2", "sent the guitar to Main St", start="2026-07-05"),
        _memory(
            "m3", "prefers flat-wound strings", predicate="string_preference", start="2026-06-01"
        ),
    ]
    context = render_grouped(chain, temporal=True)

    assert context.count("plans to send the guitar") == 1
    assert context.count("sent the guitar to Main St") == 1
    # The unchained memory still reaches the model.
    assert "prefers flat-wound strings" in context


def test_undated_members_sort_last_rather_than_crashing():
    """A store can hold a fact with no date; it must not take the renderer down."""
    chain = [
        _memory("m2", "sent the guitar to Main St", start="2026-07-05"),
        _memory("m1", "mentioned the guitar", start=None),
    ]
    rendered, _ = render_timelines(chain, temporal=True)
    assert "date unknown" in rendered
    assert rendered.index("2026-07-05") < rendered.index("date unknown")
