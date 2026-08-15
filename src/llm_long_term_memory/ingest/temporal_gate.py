"""The gate Stage B has to pass before an ingest is worth its quota.

Four numbers, because the two ways of getting temporal resolution wrong are not the
same mistake and a single accuracy figure hides both.

    key consistency      do the two halves of one change land on the same key? If
                         they do not, no supersede can fire however good the rest is.
    replacement recall   of the real changes, how many are marked `replaces`? Misses
                         leave a stale fact beside a current one.
    false supersede      of the pairs that should coexist, how many were marked
                         `replaces`? Each one deletes a fact that is still true.
    cross-run stability  does the same input give the same keys twice? An unstable
                         keyer produces a store whose timelines depend on ingest
                         order, which no downstream measurement can control for.

Only the last two are symmetric in cost, and they are not symmetric with each other:
a missed replacement leaves the system no worse than a flat list, while a false one
makes it confidently wrong. Hence the asymmetric thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .keying import Keying, UpdateOp

THRESHOLDS = {
    "key_consistency": 0.97,
    "replacement_recall": 0.92,
    "false_supersede": 0.015,
    "cross_run_stability": 0.95,
}


@dataclass(slots=True)
class PairOutcome:
    attribute: str
    expected: str
    before: Keying
    after: Keying

    @property
    def same_key(self) -> bool:
        return self.before.temporal_key == self.after.temporal_key

    @property
    def actual(self) -> str:
        return str(self.after.update_op)

    @property
    def correct(self) -> bool:
        return self.actual == self.expected


@dataclass
class GateReport:
    outcomes: list[PairOutcome] = field(default_factory=list)
    stability: float | None = None

    def _of(self, expected: str) -> list[PairOutcome]:
        return [o for o in self.outcomes if o.expected == expected]

    @property
    def key_consistency(self) -> float:
        """Measured on the pairs that describe one attribute — replacements and
        removals. Coexistence pairs may legitimately share a key or not."""
        relevant = self._of("replaces") + self._of("removes")
        return sum(o.same_key for o in relevant) / len(relevant) if relevant else 0.0

    @property
    def replacement_recall(self) -> float:
        real = self._of("replaces")
        return sum(o.actual == "replaces" for o in real) / len(real) if real else 0.0

    @property
    def false_supersede(self) -> float:
        """Denominator is every pair that should *not* replace.

        A benchmark of real changes alone would score a keyer that always says
        `replaces` at 100%, so the coexistence and ambiguous cases are what make
        this number mean anything.
        """
        safe = self._of("coexists")
        return sum(o.actual == "replaces" for o in safe) / len(safe) if safe else 0.0

    @property
    def removal_recall(self) -> float:
        real = self._of("removes")
        return sum(o.actual == "removes" for o in real) / len(real) if real else 0.0

    def passes(self) -> dict[str, bool]:
        out = {
            "key_consistency": self.key_consistency >= THRESHOLDS["key_consistency"],
            "replacement_recall": self.replacement_recall >= THRESHOLDS["replacement_recall"],
            "false_supersede": self.false_supersede <= THRESHOLDS["false_supersede"],
        }
        if self.stability is not None:
            out["cross_run_stability"] = self.stability >= THRESHOLDS["cross_run_stability"]
        return out

    @property
    def gate_open(self) -> bool:
        return all(self.passes().values())

    def failures(self) -> list[PairOutcome]:
        return [o for o in self.outcomes if not o.correct or not o.same_key]

    def to_dict(self) -> dict:
        """Persist the gate result beside usage, not only as terminal output."""
        return {
            "thresholds": THRESHOLDS,
            "key_consistency": self.key_consistency,
            "replacement_recall": self.replacement_recall,
            "false_supersede": self.false_supersede,
            "removal_recall": self.removal_recall,
            "cross_run_stability": self.stability,
            "gate_open": self.gate_open,
            "outcomes": [
                {
                    "attribute": outcome.attribute,
                    "expected": outcome.expected,
                    "before": {
                        "temporal_key": outcome.before.temporal_key,
                        "update_op": str(outcome.before.update_op),
                        "object": outcome.before.object,
                    },
                    "after": {
                        "temporal_key": outcome.after.temporal_key,
                        "update_op": str(outcome.after.update_op),
                        "object": outcome.after.object,
                    },
                    "same_key": outcome.same_key,
                    "correct": outcome.correct,
                }
                for outcome in self.outcomes
            ],
        }


def score(
    pairs, keyings: list[tuple[Keying, Keying]], stability: float | None = None
) -> GateReport:
    return GateReport(
        outcomes=[
            PairOutcome(p.attribute, p.expected, before, after)
            for p, (before, after) in zip(pairs, keyings, strict=True)
        ],
        stability=stability,
    )


def stability_between(first: list[Keying], second: list[Keying]) -> float:
    """Share of facts given the same key and operation on both runs."""
    if not first:
        return 0.0
    agree = sum(
        1
        for a, b in zip(first, second, strict=True)
        if a.temporal_key == b.temporal_key and a.update_op == b.update_op
    )
    return agree / len(first)


def unusable(keyings: list[Keying]) -> int:
    """Facts keyed in a way temporal resolution cannot act on at all."""
    return sum(1 for k in keyings if not k.trackable or k.update_op is UpdateOp.NONE)
