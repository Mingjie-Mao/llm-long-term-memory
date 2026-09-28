"""Verbatim turns from the whole archive, found by the question, under a token budget.

The memory-as-index gate (`results/memory-as-index-offline-decision.md`) found that
extracted memories locate the evidence a question needs far less often than searching
the raw archive with the question itself: 50.0% against 78.1% all-gold coverage at
4,000 tokens on train150. Memories are a good state layer and a poor locator. This is
the locator the gate measured — `bm25_turns` — as an answer-time component: BM25 over
every turn in the namespace, whole turns in rank order until the budget is spent, then
shown oldest first with each conversation's date, so the reader can tell when
something was said.

Namespace-scoped by construction: `SQLiteMemoryStore.search_turns` joins turns onto
their session's user, and nothing here reads outside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from llm_long_term_memory.store import MemoryStore, Turn


@dataclass(slots=True)
class ArchiveExcerpts:
    turns: list[Turn] = field(default_factory=list)
    tokens: int = 0
    session_dates: dict[str, datetime | None] = field(default_factory=dict)
    skipped_for_budget: int = 0

    @property
    def sessions(self) -> list[str]:
        return sorted({turn.session_id for turn in self.turns})

    def render(self) -> str:
        """Oldest conversation first, turns in their original order, each dated."""
        if not self.turns:
            return ""

        def when(session_id: str) -> tuple:
            date = self.session_dates.get(session_id)
            return (date is None, date or datetime.min, session_id)

        by_session: dict[str, list[Turn]] = {}
        for turn in self.turns:
            by_session.setdefault(turn.session_id, []).append(turn)
        blocks = []
        for session_id in sorted(by_session, key=when):
            date = self.session_dates.get(session_id)
            header = f"Conversation on {date:%Y-%m-%d}:" if date else "Conversation (undated):"
            lines = [
                f"{turn.role}: {turn.content}"
                for turn in sorted(by_session[session_id], key=lambda t: t.turn_index)
            ]
            blocks.append("\n".join([header, *lines]))
        return (
            "Excerpts from the user's earlier conversations, found for this question "
            "(verbatim, oldest first):\n\n" + "\n\n".join(blocks)
        )


def archive_excerpts(
    store: MemoryStore,
    user_id: str,
    question: str,
    max_tokens: int,
    *,
    chars_per_token: float = 4.6,
    candidates: int = 500,
) -> ArchiveExcerpts:
    """Whole turns in BM25 rank order until `max_tokens` is spent.

    A turn that would overflow is skipped and filling continues, as the hydrator does
    and as the offline gate measured: a later, shorter turn may still fit.
    """
    result = ArchiveExcerpts()
    if max_tokens <= 0:
        return result
    for turn in store.search_turns(user_id, question, limit=candidates):
        cost = max(1, int(len(turn.content) / chars_per_token))
        if result.tokens + cost > max_tokens:
            result.skipped_for_budget += 1
            continue
        result.turns.append(turn)
        result.tokens += cost
    for session_id in result.sessions:
        session = store.get_session(session_id)
        result.session_dates[session_id] = session.started_at if session else None
    return result
