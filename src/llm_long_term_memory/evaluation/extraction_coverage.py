"""Answer coverage: does the extracted store still contain the answer?

This isolates extraction quality from retrieval quality. If a question's answer was
never written into memory, no amount of work on ranking, temporal resolution, or
packing can recover it — the information left the system at ingestion time. Running
this before the full ingest is what stops 1,920 requests being spent on a prompt
that silently drops the details the benchmark asks about.

It runs against the `oracle` variant, which contains only the evidence sessions, so
a handful of questions costs a handful of requests.

**What this metric can and cannot see.** It is a string match, so it only detects
answers that appear verbatim in the store. Most of LongMemEval's gold answers are
not like that — they are *computed*: the difference between two dates ("3 weeks"),
the sum of two durations ("an hour and a half"), an ordering ("the chili post came
first"), a price difference ("$270"). In every one of those cases the component
facts were extracted correctly, exact dates included, and the metric still scores a
miss.

So it is a strict lower bound, and the number itself is not the target. What it is
genuinely good at is catching regressions in *specific-detail retention* — whether
"10 hours", "$45", "The Glass Menagerie" survive extraction or get generalised
away. That is a real and common failure mode, it is invisible end-to-end until it
has already cost accuracy, and this catches it for a few requests. Read movement in
the number, not its level, and never tune the extraction prompt to raise it past
the point where the details are being kept.


This measures extraction against the benchmark's gold answers, so it is
evaluation, not ingestion, and it lived under `ingest/` only because it is run
before an ingest. Nothing in `ingest/` imports the benchmark any more; this was
the last thing there that did, and moving it made the statement true rather than
nearly true. `lltm ingest coverage` is unchanged: where the code lives is not
where the operator looks for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from llm_long_term_memory.evaluation.datasets.longmemeval import Instance
from llm_long_term_memory.store import Memory

_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")
_WORD = re.compile(r"[a-z0-9]+")

# Categories where a string match cannot decide the question, so including them
# in the headline number measures the metric rather than the extractor:
#
#   multi-session   the gold is computed from facts in several sessions. "an hour
#                   and a half" is the sum of a 30-minute commute and a one-hour
#                   routine; both are extracted, neither contains the answer.
#   ...-preference  the gold is a paragraph describing a preferred response style,
#                   not a fact that could appear in a memory at all.
#
# Extraction is still run for these — they are reported separately so a real
# regression in them stays visible.
UNMEASURABLE_TYPES = frozenset({"multi-session", "single-session-preference"})


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _searchable(m: Memory) -> str:
    """Everything the store actually knows about a memory, not just its prose.

    `occurred_at` matters here: a temporal question's gold answer is often a date
    ("March 2026") that the content sentence never spells out because it is held in
    the structured column instead. Searching only `content` scores the store as
    having lost information it is in fact holding.
    """
    parts = [m.content, m.subject or "", m.predicate or "", m.object or ""]
    if m.occurred_at:
        parts += [
            m.occurred_at.strftime("%Y %m %d"),
            m.occurred_at.strftime("%B %Y"),
            m.occurred_at.strftime("%B %d %Y"),
            m.occurred_at.strftime("%b"),
        ]
    return " ".join(parts)


def _source_text(instance: Instance) -> str:
    turns = [turn.content for session in instance.sessions for turn in session.turns]
    return " ".join(turns)


def answer_present(gold: str, memories: list[Memory]) -> bool:
    """Whether any memory plausibly carries the gold answer."""
    gold_norm = _norm(gold)
    if not gold_norm:
        return False
    haystack = _norm(" || ".join(_searchable(m) for m in memories))

    if gold_norm in haystack:
        return True

    # Numeric answers ("25", "45 minutes") are the common case and the one most
    # often lost, so check the digits independently of the surrounding words.
    gold_numbers = _NUM.findall(gold)
    if gold_numbers:
        found = set(_NUM.findall(haystack))
        if all(n in found for n in gold_numbers):
            return True

    # Multi-word answers where the memory reorders or splits the phrase.
    tokens = gold_norm.split()
    if len(tokens) > 1:
        hits = sum(1 for t in tokens if t in haystack.split())
        return hits / len(tokens) >= 0.8
    return False


def source_literal_present(gold: str, instance: Instance) -> bool:
    """Whether the gold is literally available in the lossless source sessions.

    This is a diagnostic ceiling for literal answers, not an answer-support claim:
    derived answers can be fully supported even when no source turn contains the
    final wording. The difference from `answer_present` exposes losses introduced
    by the structured representation itself.
    """
    gold_norm = _norm(gold)
    if not gold_norm:
        return False
    haystack = _norm(_source_text(instance))
    if gold_norm in haystack:
        return True
    gold_numbers = _NUM.findall(gold)
    if gold_numbers:
        found = set(_NUM.findall(haystack))
        if all(number in found for number in gold_numbers):
            return True
    tokens = gold_norm.split()
    if len(tokens) > 1:
        hits = sum(1 for token in tokens if token in haystack.split())
        return hits / len(tokens) >= 0.8
    return False


@dataclass(slots=True)
class CoverageCase:
    question_id: str
    question_type: str
    question: str
    gold: str
    covered: bool
    n_memories: int
    source_literal_covered: bool = False
    memories: list[str] = field(default_factory=list)

    @property
    def structured_literal_covered(self) -> bool:
        """The original `covered` field, named precisely for new reports."""
        return self.covered


@dataclass(slots=True)
class CoverageReport:
    cases: list[CoverageCase] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.cases)

    @property
    def measurable(self) -> list[CoverageCase]:
        return [c for c in self.cases if c.gold and c.question_type not in UNMEASURABLE_TYPES]

    @property
    def rate(self) -> float:
        """Coverage over the categories string matching can actually judge.

        This is the number to act on. `rate_all` includes categories where the gold
        answer is computed or is a free-text paragraph, and a low score there says
        nothing about whether extraction kept the underlying facts.
        """
        cases = self.measurable
        return sum(c.covered for c in cases) / len(cases) if cases else 0.0

    @property
    def rate_all(self) -> float:
        answerable = [c for c in self.cases if c.gold]
        return sum(c.covered for c in answerable) / len(answerable) if answerable else 0.0

    @property
    def source_literal_rate(self) -> float:
        cases = self.measurable
        return sum(c.source_literal_covered for c in cases) / len(cases) if cases else 0.0

    @property
    def memories_per_session(self) -> float:
        total = sum(c.n_memories for c in self.cases)
        return total / self.n if self.n else 0.0

    def by_type(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for c in self.cases:
            if not c.gold:
                continue
            n, ok = out.get(c.question_type, (0, 0))
            out[c.question_type] = (n + 1, ok + int(c.covered))
        return dict(sorted(out.items()))

    def misses(self) -> list[CoverageCase]:
        return [c for c in self.cases if c.gold and not c.covered]


def evaluate_coverage(instances: list[Instance], extract_fn, on_case=None) -> CoverageReport:
    """`extract_fn(sessions) -> list[Memory]` keeps this testable without an API."""
    report = CoverageReport()
    for inst in instances:
        if inst.is_abstention:
            continue  # no gold answer to look for
        memories = extract_fn(inst.sessions)
        case = CoverageCase(
            question_id=inst.question_id,
            question_type=inst.question_type,
            question=inst.question,
            gold=inst.answer,
            covered=answer_present(inst.answer, memories),
            n_memories=len(memories),
            source_literal_covered=source_literal_present(inst.answer, inst),
            memories=[m.content for m in memories],
        )
        report.cases.append(case)
        if on_case:
            on_case(case, report)
    return report
