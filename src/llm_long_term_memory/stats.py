"""Paired-comparison statistics, in one place.

Four copies of the exact McNemar p-value had grown across the repository — in this
package's zero-yield reporting, in `scripts/batch_position_analyse.py`, and in two of
the `tools/` analysers. They agreed, which is luck rather than design: a p-value that
decides whether an experiment is promoted must not depend on which file computed it,
and four copies is four chances for one of them to drift into a chi-square
approximation, a one-sided tail, or a different treatment of zero discordant pairs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import comb


def exact_mcnemar(wins: int, losses: int) -> float:
    """Two-sided exact McNemar/binomial p-value for discordant pairs.

    Concordant pairs carry no information about a difference, so this is a sign test
    over the discordant ones. Exact rather than chi-square because the discordant
    counts in this repository's gates run to single figures, where the asymptotic form
    is not to be trusted. With no discordant pairs there is nothing to test, and the
    answer is 1.0 rather than undefined.
    """
    if wins < 0 or losses < 0:
        raise ValueError(f"wins and losses are counts: got {wins}, {losses}")
    n = wins + losses
    if not n:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_outcomes(
    baseline: Mapping[str, bool],
    candidate: Mapping[str, bool],
    ids: Sequence[str],
) -> dict:
    """The win/loss/tie decomposition both arms are judged on, plus its p-value.

    `ids` is passed explicitly rather than inferred from the mappings: a comparison
    must be over the registered question set, not over whichever ids both files happen
    to contain. A missing id is an error here, not a silently dropped pair.
    """
    missing = [qid for qid in ids if qid not in baseline or qid not in candidate]
    if missing:
        raise KeyError(f"{len(missing)} id(s) absent from an arm, first: {missing[0]!r}")
    wins = [qid for qid in ids if candidate[qid] and not baseline[qid]]
    losses = [qid for qid in ids if baseline[qid] and not candidate[qid]]
    return {
        "n": len(ids),
        "old_correct": sum(bool(baseline[qid]) for qid in ids),
        "new_correct": sum(bool(candidate[qid]) for qid in ids),
        "wins": wins,
        "losses": losses,
        "ties": len(ids) - len(wins) - len(losses),
        "net": len(wins) - len(losses),
        "exact_mcnemar_p": exact_mcnemar(len(wins), len(losses)),
    }
