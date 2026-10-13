"""Stage D of research/ROADMAP.md: find the earlier fact a new fact may update.

`TemporalResolver` only compares facts under one `(user, subject, predicate)`. On the
fact-pair set (research/fact-pairs-v1.json) 13 of the 21 real replacements sit under two
different keys — `painting_projects` and `painting_projects_completed`, `support_group`
and `support_groups` — so the resolver never sees them together. Candidate matching
widens the comparison to the same user's facts that read alike; a separate decision
(replace, coexist, terminate, unrelated, uncertain) must confirm a relation before any
validity changes. Matching alone never closes a fact.

Candidates are always the asking user's own memories: a fact is never compared with,
let alone superseded by, another tenant's.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from llm_long_term_memory.store import Memory


@dataclass(frozen=True, slots=True)
class Candidate:
    memory: Memory
    similarity: float
    same_key: bool


def _when(memory: Memory) -> datetime:
    """The resolver's clock: when the fact happened if stated, else when it was said.

    Ordering a pair by any other date than `TemporalResolver` uses lets the two disagree
    about which value is later — "five sessions last year" said in October is, to the
    resolver, earlier than "three sessions" said in May.
    """
    return memory.occurred_at or memory.valid_from or memory.ingested_at or datetime.min


def candidate_predecessors(
    new: Memory,
    pool: Iterable[Memory],
    vector_of: Callable[[str], np.ndarray | None],
    k: int = 5,
    *,
    same_subject: bool = False,
    roles: tuple[str, ...] | None = None,
    earlier_only: bool = True,
) -> list[Candidate]:
    """The `k` memories of `new`'s own user most similar to it, best first."""
    target = vector_of(new.id)
    if target is None:
        return []
    scored: list[Candidate] = []
    for memory in pool:
        if memory.id == new.id or memory.user_id != new.user_id:
            continue
        if earlier_only and _when(memory) > _when(new):
            continue
        if same_subject and (memory.subject or "") != (new.subject or ""):
            continue
        if roles and memory.source_role not in roles:
            continue
        vector = vector_of(memory.id)
        if vector is None:
            continue
        scored.append(
            Candidate(
                memory=memory,
                similarity=float(np.dot(target, vector)),
                same_key=(memory.subject, memory.predicate) == (new.subject, new.predicate),
            )
        )
    scored.sort(key=lambda c: (-c.similarity, c.memory.id))
    return scored[:k]


TRANSITION_PROMPT_VERSION = "transition-judge-v2"

Relation = Literal["replace", "terminate", "coexist", "unrelated", "uncertain"]

# v2 is symmetric: it never says which fact is newer, because arrival order is not
# event order. The dates shown decide which value is current, so a fact that arrives
# late is judged against the same pair as one that arrived on time.
TRANSITION_SYSTEM = """\
You maintain a user's long-term memory. One fact has just been learned (FACT). For each
OTHER fact listed, decide how the two relate. Compare them by the dates shown; a fact
may be dated before or after FACT.

- replace: the same attribute of the same person or thing with a different value, so
  only the later-dated one is current. Counts and totals so far, schedules, statuses,
  levels, records, preferences, where something is kept.
- terminate: one of them says the other's state has ended, without a new value.
- coexist: both can be true at once. Another trip, purchase, visit, meeting or item is a
  separate event or thing, not a new value of the old one, even when it reads alike.
  The same thing restated with more or less detail also coexists.
- unrelated: about something else.
- uncertain: the facts alone do not settle it.

Choose replace only when you are confident one value stopped being current."""


class TransitionDecision(BaseModel):
    candidate: str = Field(description="The other fact's label, e.g. E1")
    reason: str = Field(description="One short sentence")
    relation: Relation


class TransitionVerdict(BaseModel):
    decisions: list[TransitionDecision]


def _line(label: str, memory: Memory) -> str:
    when = _when(memory)
    stamp = "date unknown" if when == datetime.min else f"{when:%Y-%m-%d}"
    return f"{label} ({stamp}): {memory.content}"


def transition_prompt(new: Memory, candidates: list[Candidate]) -> str:
    others = "\n".join(_line(f"E{i}", c.memory) for i, c in enumerate(candidates, start=1))
    return f"{_line('FACT', new)}\n\nOTHER facts:\n{others}\n"


def decide(client, model: str, new: Memory, candidates: list[Candidate]) -> dict[str, Relation]:
    """The relation of `new` to each candidate, by memory id. Unlabelled or unparsable
    decisions come back as 'uncertain', which changes nothing."""
    if not candidates:
        return {}
    completion = client.generate(
        role="extractor",
        model=model,
        prompt=transition_prompt(new, candidates),
        system=TRANSITION_SYSTEM,
        temperature=0.0,
        max_output_tokens=1024,
        schema=TransitionVerdict,
    )
    labels = {f"E{i}": c.memory.id for i, c in enumerate(candidates, start=1)}
    result = {c.memory.id: "uncertain" for c in candidates}
    try:
        verdict = TransitionVerdict.model_validate_json(completion.text)
    except ValueError:
        return result
    for decision in verdict.decisions:
        if decision.candidate in labels:
            result[labels[decision.candidate]] = decision.relation
    return result


def apply_update(store, resolver, a: Memory, b: Memory) -> tuple[Memory, Memory]:
    """Apply a confirmed `replace` between `a` and `b`; return (earlier, later).

    The later-dated fact moves onto the earlier fact's key and is marked as replacing
    its value; the resolver then rebuilds that key from every fact on it, which is what
    makes the result independent of arrival order. Facts are never compared across
    users.
    """
    if a.user_id != b.user_id:
        raise ValueError("an update never crosses users")
    earlier, later = sorted((a, b), key=lambda m: (_when(m), m.id))
    store.link_update(
        later.id,
        subject=earlier.subject or "",
        predicate=earlier.predicate or "",
        target_object=(earlier.object or earlier.content or "").strip(),
    )
    resolver.resolve_memories([store.get(later.id)])
    return earlier, later


# ---------------------------------------------------------------- pairwise (v3)

PAIR_PROMPT_VERSION = "transition-pair-v3"

# v3 judges one pair at a time, always shown earlier-then-later by the resolver's clock.
# The request therefore does not depend on which fact arrived last, so a decision made
# when the later fact arrives is the decision a late-arriving earlier fact would get
# (the v2 list judge changed its mind with the anchor on 2 of 21 pairs).
PAIR_SYSTEM = """\
You maintain a user's long-term memory. Decide how two facts about the same user relate.
Fact A is dated no later than fact B.

- replace: the same attribute of the same person or thing with a different value, so B's
  value is current and A's is not. Counts and totals so far, schedules, statuses,
  levels, records, preferences, where something is kept.
- terminate: B says A's state has ended, without a new value.
- coexist: both can be true at once. Another trip, purchase, visit, meeting or item is a
  separate event or thing, not a new value of the old one, even when it reads alike.
  The same thing restated with more or less detail also coexists.
- unrelated: about something else.
- uncertain: the facts alone do not settle it.

Choose replace only when you are confident A's value stopped being current."""


class PairVerdict(BaseModel):
    reason: str = Field(description="One short sentence")
    relation: Relation


def ordered(a: Memory, b: Memory) -> tuple[Memory, Memory]:
    """(earlier, later) by the resolver's clock, ties broken by id."""
    return tuple(sorted((a, b), key=lambda m: (_when(m), m.id)))  # type: ignore[return-value]


def pair_prompt(a: Memory, b: Memory) -> str:
    earlier, later = ordered(a, b)
    return f"{_line('A', earlier)}\n{_line('B', later)}\n"


def judge_pair(client, model: str, a: Memory, b: Memory) -> Relation:
    """How the later of `a` and `b` relates to the earlier; 'uncertain' if unparsable."""
    if a.user_id != b.user_id:
        raise ValueError("facts of different users are never compared")
    completion = client.generate(
        role="extractor",
        model=model,
        prompt=pair_prompt(a, b),
        system=PAIR_SYSTEM,
        temperature=0.0,
        max_output_tokens=256,
        schema=PairVerdict,
    )
    try:
        return PairVerdict.model_validate_json(completion.text).relation
    except ValueError:
        return "uncertain"


def related_candidates(
    new: Memory,
    pool: Iterable[Memory],
    vector_of: Callable[[str], np.ndarray | None],
    *,
    k: int = 5,
    threshold: float = 0.65,
    roles: tuple[str, ...] = ("user",),
) -> list[Candidate]:
    """Facts of any date similar enough to `new` to be worth a judgment.

    0.65 kept all 21 annotated replacements while selecting 16% of user-stated memories
    on dev100 (results/analysis/fact-pairs-v1.candidates.md); read off the same data, so
    a planning value rather than a validated one.
    """
    if roles and new.source_role not in roles:
        return []
    ranked = candidate_predecessors(new, pool, vector_of, k=k, roles=roles, earlier_only=False)
    return [c for c in ranked if c.similarity >= threshold]


class UpdateLinker:
    """Stage D in the write path: judge new user-stated facts against similar facts of
    any date, and apply confirmed replacements. Runs after the batch is written and
    resolved; each unordered pair is judged at most once per batch."""

    version = PAIR_PROMPT_VERSION

    def __init__(self, client, model: str, *, threshold: float = 0.65, k: int = 5) -> None:
        self.client = client
        self.model = model
        self.threshold = threshold
        self.k = k

    @classmethod
    def identity_for(cls, threshold: float = 0.65, k: int = 5) -> str:
        return f"{cls.version}:t{threshold:.2f}:k{k}"

    @property
    def identity(self) -> str:
        return self.identity_for(self.threshold, self.k)

    def link(self, store, index, resolver, memories: list[Memory]) -> dict[str, int]:
        stats = {"judgments": 0, "linked": 0}
        seen: set[frozenset[str]] = set()
        for written in memories:
            if written.source_role != "user":
                continue
            pool = store.iter_all(written.user_id)
            vectors = index.vectors_for({m.id for m in pool})
            for cand in related_candidates(
                written, pool, vectors.get, k=self.k, threshold=self.threshold
            ):
                pair = frozenset((written.id, cand.memory.id))
                if pair in seen:
                    continue
                seen.add(pair)
                current, other = store.get(written.id), store.get(cand.memory.id)
                stats["judgments"] += 1
                if judge_pair(self.client, self.model, current, other) == "replace":
                    apply_update(store, resolver, current, other)
                    stats["linked"] += 1
        return stats
