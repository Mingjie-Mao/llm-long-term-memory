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

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from llm_long_term_memory.store import MemoryStore, Turn

RRF_K = 60
_TOKEN = re.compile(r"[a-z0-9]+")


def bm25_rank(query: str, keys: dict[str, str], k1: float = 1.2, b: float = 0.75) -> list[str]:
    """Okapi BM25 over one namespace's turn keys; keys sharing no term are left out.

    Used when a turn's key is not its text alone (`fact_keys`): the store's full-text
    index holds the turns as spoken, so an expanded key has to be ranked here.
    """
    docs = {key: _TOKEN.findall(text.lower()) for key, text in keys.items()}
    n = len(docs)
    average = sum(len(d) for d in docs.values()) / max(1, n)
    frequency: dict[str, int] = defaultdict(int)
    for tokens in docs.values():
        for term in set(tokens):
            frequency[term] += 1
    terms = set(_TOKEN.findall(query.lower()))
    scores: dict[str, float] = {}
    for key, tokens in docs.items():
        counts: dict[str, int] = defaultdict(int)
        for token in tokens:
            counts[token] += 1
        score = 0.0
        for term in terms & counts.keys():
            idf = math.log(1 + (n - frequency[term] + 0.5) / (frequency[term] + 0.5))
            tf = counts[term]
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(tokens) / average))
        if score > 0:
            scores[key] = score
    return sorted(scores, key=lambda key: -scores[key])


def fact_keyed_texts(store: MemoryStore, user_id: str, turns: list[Turn]) -> dict[str, str]:
    """Each turn's key: its text, followed by the facts anchored to it.

    LongMemEval's fact-augmented key expansion — the facts help a turn be found, and the
    turn is still what is returned. Every memory counts, superseded ones included: a fact
    later replaced still says what its turn was about.
    """
    facts: dict[tuple[str, int], list[str]] = defaultdict(list)
    for memory in store.iter_all(user_id):
        if memory.source_session_id and memory.source_turn_index is not None:
            facts[(memory.source_session_id, memory.source_turn_index)].append(memory.content)
    return {
        turn.id: turn.content
        + (
            "\n" + " ".join(facts[(turn.session_id, turn.turn_index)])
            if facts.get((turn.session_id, turn.turn_index))
            else ""
        )
        for turn in turns
    }


def fuse(*rankings: list[str], k: int = RRF_K) -> list[str]:
    """Reciprocal-rank fusion of turn-id rankings; ties keep first appearance.

    `hybrid-turn-retrieval-offline-v1` measured this against BM25 alone at 4,000
    tokens: 78.1% -> 83.6% all-gold coverage on train150, 86.0% -> 88.2% on heldout100.
    """
    score: dict[str, float] = defaultdict(float)
    first: dict[str, int] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            score[key] += 1 / (k + rank)
            first.setdefault(key, len(first))
    return sorted(score, key=lambda key: (-score[key], first[key]))


@dataclass(slots=True)
class ArchiveExcerpts:
    turns: list[Turn] = field(default_factory=list)
    tokens: int = 0
    session_dates: dict[str, datetime | None] = field(default_factory=dict)
    skipped_for_budget: int = 0
    asked_on: datetime | None = None
    """When set, each conversation header also says how long before the question it
    was, so a gap between two dates is a subtraction the code has already done."""
    window_turns: int = 0
    """Turns packed because they fall in the period the question names (`time_window`)."""
    time_notes: bool = False
    """When set, relative dates inside each turn ("a month ago") gain the date they
    resolve to from that conversation's date (`time_notes.annotate`), and headers
    name the weekday."""

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
            if date and self.time_notes:
                header = f"Conversation on {date:%Y-%m-%d} ({date:%A}):"
            if date and self.asked_on and self.time_notes:
                days = (self.asked_on.date() - date.date()).days
                side = "before" if days >= 0 else "after"
                header = (
                    f"Conversation on {date:%Y-%m-%d} ({date:%A}; {abs(days)} days, about "
                    f"{abs(days) / 7:.1f} weeks, {side} the question):"
                )
            elif date and self.asked_on:
                days = (self.asked_on.date() - date.date()).days
                side = "before" if days >= 0 else "after"
                header = (
                    f"Conversation on {date:%Y-%m-%d} ({abs(days)} days, about "
                    f"{abs(days) / 7:.1f} weeks, {side} the question):"
                )
            lines = [
                f"{turn.role}: {self._text(turn, date)}"
                for turn in sorted(by_session[session_id], key=lambda t: t.turn_index)
            ]
            blocks.append("\n".join([header, *lines]))
        note = (
            " Dates in square brackets were added: each relative date resolved from its "
            "own conversation's date; ≈ marks an approximation."
            if self.time_notes
            else ""
        )
        return (
            "Excerpts from the user's earlier conversations, found for this question "
            f"(verbatim, oldest first):{note}\n\n" + "\n\n".join(blocks)
        )

    def _text(self, turn: Turn, date: datetime | None) -> str:
        # The user's own turns only: the assistant's "today" is small talk, and its
        # restatements of the user's dates repeat what the user's turn already says.
        if not self.time_notes or turn.role != "user":
            return turn.content
        from llm_long_term_memory.retrieve.time_notes import annotate

        return annotate(turn.content, date)[0]


def archive_excerpts(
    store: MemoryStore,
    user_id: str,
    question: str,
    max_tokens: int,
    *,
    chars_per_token: float = 4.6,
    candidates: int = 500,
    turn_index=None,
    query_vector: np.ndarray | None = None,
    asked_on: datetime | None = None,
    memory_anchors: list[tuple[str, int]] | None = None,
    fact_keys: bool = False,
    prefer_user: bool = False,
    user_fraction: float = 0.75,
    time_notes: bool = False,
    time_window: bool = False,
    window_fraction: float = 0.5,
) -> ArchiveExcerpts:
    """Whole turns in rank order until `max_tokens` is spent.

    BM25 alone by default. With `turn_index` (a `NumpyFlatIndex` over turn ids) and the
    question's `query_vector`, the BM25 ranking is fused with a dense ranking of this
    namespace's turns. The dense side searches the whole index and keeps only turns of
    this user's sessions, so another tenant's turn can never be selected however close
    it is.

    `memory_anchors` — `(session_id, turn_index)` of the retrieved memories, in memory
    rank — adds the turns the memories point at as a third ranking. The memory layer holds
    as a user's history grows while turn search does not: at 4x history, 74.7% -> 80.8%
    all-gold coverage (`results/prereg-growth-fusion-offline-v1.md`).

    A turn that would overflow is skipped and filling continues, as the hydrator does
    and as the offline gate measured: a later, shorter turn may still fit.
    """
    result = ArchiveExcerpts(asked_on=asked_on, time_notes=time_notes)
    if not 0 <= user_fraction <= 1:
        raise ValueError("user_fraction must be between zero and one")
    if max_tokens <= 0:
        return result
    ranked = store.search_turns(user_id, question, limit=candidates)
    dense_on = turn_index is not None and query_vector is not None and len(turn_index)
    if dense_on or memory_anchors or fact_keys:
        own = {
            turn.id: turn
            for session_id in store.session_ids_for_user(user_id)
            for turn in store.turns_for_session(session_id)
        }
        if fact_keys:
            # BM25 over turn + facts replaces the store's full-text ranking of the turn
            # alone. Measured: the lexical side carries the gain (train150 77.4% ->
            # 81.5%; expanding the dense side alone gave 77.4%).
            lexical = bm25_rank(question, fact_keyed_texts(store, user_id, list(own.values())))
            ranked = [own[tid] for tid in lexical[:candidates]]
        rankings = [[t.id for t in ranked]]
        if dense_on:
            rankings.append(
                [
                    tid
                    for tid, _ in turn_index.search(query_vector, limit=len(turn_index))
                    if tid in own
                ]
            )
        if memory_anchors:
            at = {(turn.session_id, turn.turn_index): turn.id for turn in own.values()}
            led = [at[a] for a in memory_anchors if a in at]
            rankings.append(list(dict.fromkeys(led)))
        own.update({turn.id: turn for turn in ranked})
        ranked = [own[tid] for tid in fuse(*rankings)]
    selected = set()
    window_turns: list[Turn] = []
    if time_window and asked_on is not None:
        # The question's own relative time ("last Saturday") resolved from the day it is
        # asked: the user's turns said in that period, or whose own relative dates fall
        # in it, are packed first, up to `window_fraction` of the budget, in rank order.
        # Offline (tools/time_window_offline.py): all-gold reach on the 30 questions of
        # train150, dev100 and heldout100 that name a period, 15 -> 25, none lost.
        from llm_long_term_memory.retrieve.time_window import question_window, turn_dates

        window = question_window(question, asked_on)
        if window is not None:
            position = {turn.id: rank for rank, turn in enumerate(ranked)}
            dates = {
                session_id: (session.started_at if session else None)
                for session_id in store.session_ids_for_user(user_id)
                for session in [store.get_session(session_id)]
            }

            def inside(turn: Turn) -> bool:
                said = dates.get(turn.session_id)
                if said and window[0] <= said.date() <= window[1]:
                    return True
                return any(
                    start <= window[1] and window[0] <= end
                    for start, end in turn_dates(turn.content, said)
                )

            window_turns = sorted(
                (
                    turn
                    for session_id in dates
                    for turn in store.turns_for_session(session_id)
                    if turn.role == "user" and inside(turn)
                ),
                key=lambda t: (position.get(t.id, 10**6), t.session_id, t.turn_index),
            )

    def pack(turns, limit):
        for turn in turns:
            if turn.id in selected:
                continue
            cost = max(1, int(len(turn.content) / chars_per_token))
            if result.tokens + cost > limit:
                continue
            result.turns.append(turn)
            selected.add(turn.id)
            result.tokens += cost

    if window_turns:
        pack(window_turns, int(max_tokens * window_fraction))
    if prefer_user:
        pack([t for t in ranked if t.role == "user"], int(max_tokens * user_fraction))
        pack([t for t in ranked if t.role != "user"], max_tokens)
        pack([t for t in ranked if t.role == "user"], max_tokens)
    else:
        pack(ranked, max_tokens)
    result.skipped_for_budget = sum(t.id not in selected for t in ranked)
    result.window_turns = sum(t.id in selected for t in window_turns)
    for session_id in result.sessions:
        session = store.get_session(session_id)
        result.session_dates[session_id] = session.started_at if session else None
    return result
