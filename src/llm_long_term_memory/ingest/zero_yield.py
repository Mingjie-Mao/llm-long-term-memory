"""Classify sessions that reached the archive but produced no memory.

An empty extraction is not automatically a model failure: some conversations
contain no durable personal fact.  This audit only makes claims the stored evidence
can support.  Reused LongMemEval sessions are especially useful: if identical
source text yields memories in one namespace and none in another, the miss is
demonstrably batch/model dependent without paying for another API call.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from llm_long_term_memory.conversation import ConversationSession, ConversationSource
from llm_long_term_memory.store import MemoryStore, external_session_id

from .extract import _parse_date
from .pipeline import IngestProgress, _key, group_by_namespace, namespaced_sessions


@dataclass(frozen=True, slots=True)
class ZeroYieldCase:
    question_id: str
    session_id: str
    category: str
    is_evidence: bool
    turn_count: int
    char_count: int
    occurrences: int
    positive_occurrences: int
    pending_occurrences: int
    source_variants: int
    source_text_variants: int
    source_date_variants: int
    batch_position: int
    format_issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ZeroYieldReport:
    total_sessions: int
    terminal_sessions: int
    min_turns: int
    batch_size: int
    position_totals: tuple[int, ...]
    all_zero_yield_sessions: int
    short_zero_yield_sessions: int
    all_cases: tuple[ZeroYieldCase, ...]

    @property
    def complete(self) -> bool:
        return self.terminal_sessions == self.total_sessions

    @property
    def cases(self) -> tuple[ZeroYieldCase, ...]:
        """Substantive cases retained for compatibility with earlier artifacts."""
        return tuple(case for case in self.all_cases if case.turn_count >= self.min_turns)

    @property
    def categories(self) -> dict[str, int]:
        return dict(sorted(Counter(case.category for case in self.cases).items()))

    @property
    def all_categories(self) -> dict[str, int]:
        return dict(sorted(Counter(case.category for case in self.all_cases).items()))

    @property
    def short_categories(self) -> dict[str, int]:
        return dict(
            sorted(
                Counter(
                    case.category for case in self.all_cases if case.turn_count < self.min_turns
                ).items()
            )
        )

    def to_dict(self) -> dict:
        position_zeros = Counter(case.batch_position for case in self.cases)

        def band(positions: range) -> dict[str, int | float]:
            total = sum(self.position_totals[position] for position in positions)
            zero = sum(position_zeros[position] for position in positions)
            return {"total": total, "zero": zero, "rate": zero / total if total else 0.0}

        front = band(range(0, min(4, self.batch_size)))
        later = band(range(min(4, self.batch_size), self.batch_size))
        expected_later_at_front_rate = later["total"] * front["rate"]
        excess_later = max(0.0, later["zero"] - expected_later_at_front_rate)
        return {
            "complete": self.complete,
            "total_sessions": self.total_sessions,
            "terminal_sessions": self.terminal_sessions,
            "pending_sessions": self.total_sessions - self.terminal_sessions,
            "min_turns": self.min_turns,
            "batch_size": self.batch_size,
            "all_zero_yield_sessions": self.all_zero_yield_sessions,
            "short_zero_yield_sessions_below_min_turns": self.short_zero_yield_sessions,
            "substantive_zero_yield_sessions": len(self.cases),
            # Compatibility with the partial artifact's field name.
            "zero_yield_sessions": len(self.cases),
            "categories": self.categories,
            "all_categories": self.all_categories,
            "short_categories": self.short_categories,
            "evidence_zero_yield": sum(case.is_evidence for case in self.all_cases),
            "substantive_evidence_zero_yield": sum(case.is_evidence for case in self.cases),
            "position_bands": {
                "front_0_3": front,
                "later_4_end": later,
            },
            "batch_position_effect": {
                "later_vs_front_risk_ratio": (
                    later["rate"] / front["rate"] if front["rate"] else None
                ),
                "expected_later_zeros_at_front_rate": expected_later_at_front_rate,
                "excess_later_zeros": excess_later,
                "interpretation": (
                    "population estimate only; do not label individual sessions from position. "
                    "The separate randomized batch-position pilot supplies the causal evidence."
                ),
            },
            "positions": [
                {
                    "position": position,
                    "total": total,
                    "zero": position_zeros[position],
                    "rate": position_zeros[position] / total if total else 0.0,
                }
                for position, total in enumerate(self.position_totals)
            ],
            "cases": [asdict(case) for case in self.cases],
            "all_cases": [asdict(case) for case in self.all_cases],
        }


def _format_issues(session: ConversationSession) -> tuple[str, ...]:
    issues: list[str] = []
    if _parse_date(session.date) is None:
        issues.append("invalid_date")
    if any(not turn.content.strip() for turn in session.turns):
        issues.append("blank_turn")
    if any(turn.role not in {"user", "assistant"} for turn in session.turns):
        issues.append("invalid_role")
    if not any(turn.role == "user" for turn in session.turns):
        issues.append("missing_user_turn")
    return tuple(issues)


def _source_fingerprint(session: ConversationSession) -> str:
    payload = {
        "date": session.date,
        "turns": [{"role": turn.role, "content": turn.content} for turn in session.turns],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _source_text_fingerprint(session: ConversationSession) -> str:
    payload = [{"role": turn.role, "content": turn.content} for turn in session.turns]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def memory_counts(store: MemoryStore, namespaces: set[str]) -> dict[tuple[str, str], int]:
    """Count memories by public source id, hiding scoped database keys."""
    counts: Counter[tuple[str, str]] = Counter()
    for namespace in namespaces:
        for memory in store.iter_all(namespace):
            if memory.source_session_id:
                counts[(namespace, external_session_id(memory.source_session_id))] += 1
    return dict(counts)


def audit_zero_yield(
    sources: list[ConversationSource],
    progress: IngestProgress,
    counts: dict[tuple[str, str], int],
    *,
    min_turns: int = 6,
    batch_size: int = 15,
) -> ZeroYieldReport:
    """Return evidence-backed categories for every terminal zero-yield session."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pairs = namespaced_sessions(sources)
    blocked = progress.blocked_sessions
    terminal = progress.done_sessions | blocked
    occurrences: dict[str, list[tuple[str, ConversationSession, str]]] = defaultdict(list)
    positions: dict[str, int] = {}
    for namespace, sessions in group_by_namespace(pairs):
        for ordinal, session in enumerate(sessions):
            positions[_key(namespace, session)] = ordinal % batch_size

    for namespace, session in pairs:
        key = _key(namespace, session)
        if key in blocked:
            state = "blocked"
        elif key in progress.done_sessions:
            state = "positive" if counts.get((namespace, session.session_id), 0) else "zero"
        else:
            state = "pending"
        occurrences[session.session_id].append((namespace, session, state))

    all_cases: list[ZeroYieldCase] = []
    for namespace, session in pairs:
        key = _key(namespace, session)
        if key not in terminal:
            continue
        if key not in blocked and counts.get((namespace, session.session_id), 0):
            continue

        peers = occurrences[session.session_id]
        states = [state for _, _, state in peers]
        source_variants = len({_source_fingerprint(peer) for _, peer, _ in peers})
        source_text_variants = len({_source_text_fingerprint(peer) for _, peer, _ in peers})
        source_date_variants = len({peer.date for _, peer, _ in peers})
        issues = _format_issues(session)
        if key in blocked:
            category = "content_policy"
        elif issues:
            category = "source_format"
        elif session.is_evidence:
            # `has_answer` is ground-truth annotation: an empty extraction here is
            # a real information loss, not merely a chat with nothing worth saving.
            category = "evidence_extraction_miss"
        elif source_text_variants > 1:
            # A shared public id is only repeat evidence when the source text is
            # byte-for-byte equivalent after parsing. Otherwise this is a dataset
            # identity collision, not model variance.
            category = "source_id_collision"
        elif source_date_variants > 1 and "positive" in states:
            # LongMemEval reuses identical conversation text under different
            # synthetic dates. The extractor sees the date, so these are near
            # repeats but not controlled repeats and cannot prove randomness.
            category = "date_variant_outcome_difference"
        elif source_date_variants > 1 and "pending" in states:
            category = "date_variant_comparison_pending"
        elif source_date_variants > 1:
            category = "date_variant_consistent_zero"
        elif "positive" in states:
            category = "inconsistent_repeat"
        elif "pending" in states:
            category = "comparison_pending"
        elif len(peers) > 1:
            category = "consistent_repeat_zero"
        elif len(session.turns) < min_turns:
            # Short chats often have no durable personal fact. This is a supported
            # candidate explanation, not proof that extraction succeeded.
            category = "short_no_durable_fact_candidate"
        else:
            # Needs a blinded source-content audit or an isolated repeat. Calling it
            # random now would turn absence of evidence into a conclusion.
            category = "unclassified_single_zero"

        all_cases.append(
            ZeroYieldCase(
                question_id=namespace,
                session_id=session.session_id,
                category=category,
                is_evidence=session.is_evidence,
                turn_count=len(session.turns),
                char_count=session.char_count,
                occurrences=len(peers),
                positive_occurrences=states.count("positive"),
                pending_occurrences=states.count("pending"),
                source_variants=source_variants,
                source_text_variants=source_text_variants,
                source_date_variants=source_date_variants,
                batch_position=positions[key],
                format_issues=issues,
            )
        )

    position_totals = Counter(
        positions[_key(namespace, session)]
        for namespace, session in pairs
        if _key(namespace, session) in terminal and len(session.turns) >= min_turns
    )
    return ZeroYieldReport(
        total_sessions=len(pairs),
        terminal_sessions=sum(_key(*pair) in terminal for pair in pairs),
        min_turns=min_turns,
        batch_size=batch_size,
        position_totals=tuple(position_totals[position] for position in range(batch_size)),
        all_zero_yield_sessions=len(all_cases),
        short_zero_yield_sessions=sum(case.turn_count < min_turns for case in all_cases),
        all_cases=tuple(all_cases),
    )
