"""Pre-declared selection rule for the train150 session-context grid."""

from __future__ import annotations

from dataclasses import dataclass


class NoEligibleContext(ValueError):
    """No train-only candidate clears the registered recall and context gates."""


@dataclass(frozen=True, slots=True)
class ContextCandidate:
    name: str
    aggregate: str
    window_radius: int | None
    max_total_memories: int
    rank_top3: float
    assembled_recall: float
    median_context_tokens: float
    flat_median_tokens: float
    truncated_questions: int

    @property
    def context_ratio(self) -> float:
        return (
            self.median_context_tokens / self.flat_median_tokens
            if self.flat_median_tokens
            else float("inf")
        )


def choose_context_candidate(
    candidates: list[ContextCandidate],
    *,
    min_top3: float = 0.80,
    min_assembled: float = 0.80,
    max_context_ratio: float = 1.50,
) -> ContextCandidate:
    """Choose by a rule written before the complete train150 results exist.

    The gate is Top-3 source-session recall.  Among candidates that pass it and
    actually preserve the source session in the assembled context, prefer higher
    assembled recall.  Exact ties prefer higher rank recall, then the smaller hard
    memory cap and window, then fewer median tokens and truncations.  This prevents
    a 40-memory ceiling winning over an equally accurate 30-memory ceiling merely
    because the larger arm truncated one fewer irrelevant session.
    """
    eligible = [
        candidate
        for candidate in candidates
        if candidate.rank_top3 >= min_top3
        and candidate.assembled_recall >= min_assembled
        and candidate.context_ratio <= max_context_ratio
    ]
    if not eligible:
        raise NoEligibleContext(
            "no candidate clears Top-3, assembled-recall and context-ratio gates"
        )
    return min(
        eligible,
        key=lambda candidate: (
            -candidate.assembled_recall,
            -candidate.rank_top3,
            candidate.max_total_memories,
            candidate.window_radius if candidate.window_radius is not None else 10**9,
            candidate.median_context_tokens,
            candidate.truncated_questions,
            candidate.name,
        ),
    )
