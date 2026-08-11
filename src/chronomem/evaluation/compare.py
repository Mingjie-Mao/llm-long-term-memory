"""Paired comparison between two variants.

Comparing headline accuracies is the wrong test here, and measurably so: the same
configuration re-run scores differently, because `temperature=0` does not make a
hosted LLM deterministic and because the judge's borderline calls move too. On 50
questions, four flips is eight percentage points — larger than any improvement this
project is likely to produce.

The fix is not a bigger sample first; it is a better test. Both variants answer the
*same* questions, so the runs are paired, and a paired test throws away the noise
they share. Questions both got right and questions both got wrong carry no
information about which is better — only the disagreements do:

    b01 = A wrong, B right      (B's wins)
    b10 = A right, B wrong      (B's losses)

Under the null hypothesis that the variants are equally good, each disagreement is
a coin flip, so the count of B's wins among `b01 + b10` trials is Binomial(n, 0.5).
That is McNemar's test in its exact form, which is the right one at these counts —
the chi-square approximation is unreliable below about 25 discordant pairs, and
this project will usually have fewer.

The practical consequence: a variant can raise the headline number and still fail
this test, and that is the honest outcome to report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

from .harness import RunReport


def _binom_two_sided(wins: int, n: int) -> float:
    """Exact two-sided binomial p-value for `wins` successes in `n` fair trials."""
    if n == 0:
        return 1.0
    probs = [comb(n, k) for k in range(n + 1)]
    total = float(sum(probs))
    observed = probs[wins] / total
    # Two-sided: sum every outcome no more likely than the one observed.
    tail = sum(p for p in probs if p / total <= observed + 1e-12)
    return min(1.0, tail / total)


@dataclass(slots=True)
class PairedComparison:
    variant_a: str
    variant_b: str
    n_paired: int
    both_right: int
    both_wrong: int
    b_wins: int
    """A wrong, B right."""
    b_losses: int
    """A right, B wrong."""
    p_value: float
    win_ids: list[str] = field(default_factory=list)
    loss_ids: list[str] = field(default_factory=list)

    @property
    def discordant(self) -> int:
        return self.b_wins + self.b_losses

    @property
    def net(self) -> int:
        return self.b_wins - self.b_losses

    @property
    def accuracy_delta(self) -> float:
        return self.net / self.n_paired if self.n_paired else 0.0

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def verdict(self) -> str:
        if self.n_paired == 0:
            return "no questions in common"
        if self.discordant == 0:
            return "the two variants answered identically on every question"
        direction = "better" if self.net > 0 else "worse"
        if self.significant:
            return (
                f"{self.variant_b} is {direction} than {self.variant_a}: "
                f"{self.b_wins}W-{self.b_losses}L on {self.discordant} disagreements, "
                f"p={self.p_value:.3f}"
            )
        return (
            f"no detectable difference: {self.b_wins}W-{self.b_losses}L on "
            f"{self.discordant} disagreements, p={self.p_value:.3f}. A "
            f"{self.accuracy_delta:+.1%} gap in headline accuracy is within what "
            f"re-running the same configuration produces."
        )


def compare(a: RunReport, b: RunReport) -> PairedComparison:
    by_a = {r.question_id: r.correct for r in a.results}
    by_b = {r.question_id: r.correct for r in b.results}
    shared = sorted(set(by_a) & set(by_b))

    both_right = both_wrong = 0
    wins: list[str] = []
    losses: list[str] = []
    for qid in shared:
        ra, rb = by_a[qid], by_b[qid]
        if ra and rb:
            both_right += 1
        elif not ra and not rb:
            both_wrong += 1
        elif rb:
            wins.append(qid)
        else:
            losses.append(qid)

    return PairedComparison(
        variant_a=a.variant,
        variant_b=b.variant,
        n_paired=len(shared),
        both_right=both_right,
        both_wrong=both_wrong,
        b_wins=len(wins),
        b_losses=len(losses),
        p_value=_binom_two_sided(len(wins), len(wins) + len(losses)),
        win_ids=wins,
        loss_ids=losses,
    )


@dataclass(slots=True)
class Variability:
    """Spread across repeated runs of one unchanged configuration.

    This is the yardstick every reported difference has to clear. Without it a
    table of accuracies is a list of numbers with unknown precision.
    """

    variant: str
    accuracies: list[float]

    @property
    def n_runs(self) -> int:
        return len(self.accuracies)

    @property
    def mean(self) -> float:
        return sum(self.accuracies) / self.n_runs if self.n_runs else 0.0

    @property
    def spread(self) -> float:
        return max(self.accuracies) - min(self.accuracies) if self.accuracies else 0.0

    @property
    def stdev(self) -> float:
        if self.n_runs < 2:
            return 0.0
        m = self.mean
        return (sum((x - m) ** 2 for x in self.accuracies) / (self.n_runs - 1)) ** 0.5

    def summary(self) -> str:
        runs = ", ".join(f"{a:.1%}" for a in self.accuracies)
        return (
            f"`{self.variant}` re-run {self.n_runs}x with nothing changed: {runs} "
            f"(mean {self.mean:.1%}, spread {self.spread:.1%}, sd {self.stdev:.1%}). "
            f"Differences below this are not evidence."
        )
