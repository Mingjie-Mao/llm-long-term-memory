"""Whether Stage B structured the facts well enough for P4 to work.

Stage A decides whether a fact is written down at all; Stage B decides how it is
keyed. Those are different failures with different consequences, and after the
two-stage split they are separately measurable for the first time.

Three metrics, because "predicate accuracy" alone does not say whether temporal
resolution can function:

    predicate accuracy      is the relation right? Decides how facts are organised.
    temporal-key consistency   do the before and after of one change land on the
                            same `(subject, predicate)`? If they do not, supersede
                            never fires and the timeline is never built. This is
                            bounded above by the fallthrough rate: a fact keyed
                            `states` collides with nothing by construction.
    false supersede rate    were two facts that could both be true marked as
                            replacing one another? The worst outcome of the three,
                            because it silently retires a fact that still holds and
                            removes it from every later query.

The third is the one to protect. A missing timeline leaves the system no worse than
naive retrieval; a wrong one makes it confidently wrong, which is the failure this
project exists to fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chronomem.ingest.schemas import is_single_valued
from chronomem.store import Memory


@dataclass(slots=True)
class PredicateCase:
    text: str
    expected: str
    actual: str = ""

    @property
    def correct(self) -> bool:
        return self.actual == self.expected


@dataclass
class PredicateReport:
    cases: list[PredicateCase] = field(default_factory=list)
    fallthrough_total: int = 0
    memories_total: int = 0

    @property
    def accuracy(self) -> float:
        return sum(c.correct for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def fallthrough_rate(self) -> float:
        """Share keyed `states`. Not an error in itself — it is the safe default —
        but it is the ceiling on how much of the store temporal resolution can
        ever see."""
        return self.fallthrough_total / self.memories_total if self.memories_total else 0.0

    def confusions(self) -> list[tuple[str, str, str]]:
        return [(c.text, c.expected, c.actual) for c in self.cases if not c.correct]


@dataclass(slots=True)
class ChangePair:
    """One attribute, stated before and after it changed."""

    before: str
    after: str
    attribute: str


@dataclass
class ConsistencyReport:
    pairs: list[tuple[ChangePair, str, str]] = field(default_factory=list)

    @property
    def consistent(self) -> int:
        return sum(1 for _, b, a in self.pairs if b == a)

    @property
    def rate(self) -> float:
        return self.consistent / len(self.pairs) if self.pairs else 0.0

    @property
    def resolvable(self) -> int:
        """Consistent *and* on a key temporal resolution will act on.

        Agreeing on `states` is not a success: both sides land on a key that
        supersedes nothing, so the change is recorded and then ignored.
        """
        return sum(
            1 for pair, b, a in self.pairs if b == a and b != "states" and is_single_valued(b)
        )

    @property
    def resolvable_rate(self) -> float:
        return self.resolvable / len(self.pairs) if self.pairs else 0.0

    def failures(self) -> list[tuple[str, str, str, str]]:
        return [(p.attribute, p.before, b, a) for p, b, a in self.pairs if b != a or b == "states"]


@dataclass(slots=True)
class Supersession:
    earlier: Memory
    later: Memory

    @property
    def key(self) -> str:
        return f"{self.earlier.subject}/{self.earlier.predicate}"


@dataclass
class SupersedeReport:
    checked: list[tuple[Supersession, bool, str]] = field(default_factory=list)
    """(supersession, is_legitimate, reason)"""

    @property
    def total(self) -> int:
        return len(self.checked)

    @property
    def false_count(self) -> int:
        return sum(1 for _, ok, _ in self.checked if not ok)

    @property
    def false_rate(self) -> float:
        return self.false_count / self.total if self.total else 0.0

    def offenders(self) -> list[tuple[str, str, str, str]]:
        return [
            (s.key, s.earlier.object or s.earlier.content, s.later.object or s.later.content, why)
            for s, ok, why in self.checked
            if not ok
        ]


def collect_supersessions(memories: list[Memory]) -> list[Supersession]:
    by_id = {m.id: m for m in memories}
    out = []
    for m in memories:
        if m.status == "superseded" and m.superseded_by in by_id:
            out.append(Supersession(earlier=m, later=by_id[m.superseded_by]))
    return out


def judge_supersession(pair: Supersession) -> tuple[bool, str]:
    """Structural checks only — no model call.

    Catches the two shapes that made the v1 arity list wrong: a supersede on a
    predicate that is not single-valued, and one between values that are not
    competing at all. It cannot catch every semantic error, which is why the
    offenders list is printed for reading rather than only counted.
    """
    predicate = pair.earlier.predicate or ""

    if not is_single_valued(predicate) and not pair.later.replaces_previous:
        return False, f"{predicate!r} is multi-valued and nothing signalled a replacement"

    earlier_obj = (pair.earlier.object or "").strip().lower()
    later_obj = (pair.later.object or "").strip().lower()
    if earlier_obj and earlier_obj == later_obj:
        return False, "same value on both sides — a restatement, not a change"
    if not earlier_obj or not later_obj:
        return False, "one side has no object, so nothing was actually compared"

    return True, "single-valued key with differing values"
