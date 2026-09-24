"""Temporal resolution: deciding which competing fact is currently true.

This is the layer the baselines are missing. Given

    Jan  I use TensorFlow.
    Mar  I'm learning PyTorch.
    Aug  I've switched completely to PyTorch.

both of them hand the model all three statements and let it guess. On LongMemEval-S
that showed up as 23.1% (full context) and 38.5% (naive RAG) on temporal-reasoning
questions — the worst category for both, and why P4 was promoted ahead of P3.

**Timelines are rebuilt, not patched.** The obvious implementation compares each
incoming fact against the current head of its key and supersedes the loser. It
breaks as soon as ingestion order stops matching event order, which on a batched
pipeline is immediately:

  * A January fact arriving after an August one would make TensorFlow current again.
  * A March fact arriving after Jan and Aug have already resolved needs January's
    `valid_to` moved from August back to March — but January is no longer active,
    so a comparison against the head cannot even see it.

So resolution reads every memory for a key, superseded ones included, sorts by
`occurred_at` — the time the fact states, else the time it was said — and writes
the intervals that timeline implies. That makes the
operation idempotent and order-independent: the same set of facts produces the same
timeline no matter what sequence they arrived in.

**Restatement is not change.** Saying "I use PyTorch" in March and again in August
did not move anything. Consecutive mentions of the same value collapse into one
interval owned by the *earliest* of them — so "when did you switch to PyTorch?" is
answered by March — and the later restatements are folded into the owner rather than
counted as changes.

There is no LLM call here. Ordering by `occurred_at` is arithmetic over data the
extractor already attached, and putting a model call on this path would charge a
request per memory for a decision that does not need one. Facts without a date are
reported and left alone rather than guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from llm_long_term_memory.ingest.schemas import is_single_valued
from llm_long_term_memory.store import Memory, MemoryStore


@dataclass(slots=True)
class TimelineChange:
    memory_id: str
    action: str  # 'superseded' | 'restatement' | 'promoted' | 'revalidated'
    superseded_by: str | None = None
    valid_to: datetime | None = None

    def __str__(self) -> str:
        target = f" -> {self.superseded_by}" if self.superseded_by else ""
        return f"{self.memory_id} {self.action}{target}"


@dataclass(slots=True)
class ResolutionStats:
    keys_examined: int = 0
    keys_resolved: int = 0
    superseded: int = 0
    """Real changes of value: one fact replacing a different one."""
    restatements: int = 0
    """Repeats of a value already in force. Folded into the owning interval."""
    promoted: int = 0
    """Rows restored to current. Resolution is not monotonic — ingesting an older
    fact can demote a head, and a corrected re-run has to be able to undo that."""
    revalidated: int = 0
    """Intervals whose end moved because a fact landed in the middle of the chain."""
    repaired: int = 0
    """Rows released from a supersede that should never have happened, because the
    predicate is no longer treated as single-valued. A nonzero count on a re-run
    means an earlier arity call was wrong and this store had true facts hidden."""
    skipped_undated: int = 0
    skipped_ambiguous: int = 0
    """Replacement signals that had more than one live predecessor. Choosing one by
    row order would hide a potentially true fact, so the resolver abstained."""
    changes: list[TimelineChange] = field(default_factory=list)
    terminated: int = 0
    """Termination events retained as history and excluded from current state."""

    @property
    def writes(self) -> int:
        return len(self.changes)


def _value(memory: Memory) -> str:
    return (memory.object or memory.content or "").strip().lower()


_QUANTITY = re.compile(r"\b\d[\d,]*\b")


def _quantities(memory: Memory) -> set[str]:
    return {q.replace(",", "") for q in _QUANTITY.findall(memory.content or "")}


def _is_restatement(memory: Memory, owner: Memory) -> bool:
    """Is `memory` really just `owner` said again?

    A restatement is folded away — marked superseded, pointing at the owner, and
    invisible to retrieval. That is right for "I live in Sydney" said twice and
    badly wrong for anything carrying information the owner does not have, because
    the fold is silent and the memory never comes back.

    `_value` compares `object`, which is the *predicate's* value and not the
    sentence's. Two memories can share one and differ in everything the question
    turns on. On dev50 this cost `92a0aa75`: both memories keyed to
    `(user, job_title)` with object "Senior Marketing Specialist", one saying two
    years and four months of marketing experience and the other three years and
    nine months at the company. Same object, so the later one was folded as a
    repeat — and the answer needed the difference between them.

    It is rare. Across the clean P10 store exactly one fold is lossy in this way —
    31 memories carry a `superseded_by`, but 30 of those had their value genuinely
    change and were superseded rather than folded. `superseded_by` is set by both
    paths, and counting it without comparing `_value` first overstates the problem
    by thirty. Re-resolving the whole store under this change moves two rows.

    So numbers gate the fold. It is a narrow test and deliberately so — it fires
    on the case where silent loss is demonstrable and leaves genuine repetition
    alone. It does not make `_value` right, and it does not fix `92a0aa75`, whose
    root cause is upstream: two facts about different durations should never have
    been keyed to `job_title` at all. This stops the loss; the keying is Stage B's
    to fix.
    """
    return _value(owner) == _value(memory) and not (_quantities(memory) - _quantities(owner))


def _is_removal(memory: Memory) -> bool:
    """A removal mentions the old value, but is never a restatement of it."""
    return memory.update_op == "removes"


_REPLACED_WITHOUT_SUCCESSOR = re.compile(
    r"\breplaced\s+(?:their|his|her|the|my)\b(?![^.]*\bwith\b)", re.IGNORECASE
)


def _operation(memory: Memory) -> str:
    """Return the explicit transition, with one conservative legacy repair.

    Before `target_object` existed, Stage B sometimes labelled "replaced their Nike
    shoes" as REPLACE and put Nike in `object`. The sentence names only the value
    that ended; treating it as a successor produced the measured Adidas/Nike false
    supersession. This narrow rule makes old stores replayable without guessing a
    new value. New extraction is instructed to emit REMOVES + target_object.
    """
    if (
        memory.update_op == "replaces"
        and not memory.target_object
        and _REPLACED_WITHOUT_SUCCESSOR.search(memory.content or "")
    ):
        return "removes"
    if memory.update_op == "coexists" and memory.replaces_previous:
        return "replaces"
    return memory.update_op


def _target_value(memory: Memory) -> str:
    fallback = memory.object if _operation(memory) == "removes" else ""
    return (memory.target_object or fallback or "").strip().lower()


def _target(active: list[Memory], transition: Memory) -> Memory | None:
    """Identify exactly one predecessor; ambiguity must never hide a true fact."""
    wanted = _target_value(transition)
    if wanted:
        matches = [item for item in active if _value(item) == wanted]
        return matches[0] if len(matches) == 1 else None
    return active[0] if len(active) == 1 else None


def as_of(memories: list[Memory], when: datetime) -> list[Memory]:
    """Which of these facts were in force at `when`.

    The query the baselines cannot answer at all: "what framework was I using in
    April?" needs the interval a fact held, not just whether it is current.
    """
    out = []
    for m in memories:
        start = m.valid_from or m.occurred_at
        if start is None or start > when:
            continue
        if m.valid_to is not None and m.valid_to <= when:
            continue
        out.append(m)
    return out


class TemporalResolver:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    # ------------------------------------------------------------------ entry

    def resolve_all(self, user_id: str) -> ResolutionStats:
        stats = ResolutionStats()
        for subject, predicate in self.store.predicate_keys(user_id):
            self._resolve_key(user_id, subject, predicate, stats)
        return stats

    def resolve_everything(self) -> ResolutionStats:
        """Full pass over every namespace in the store.

        `resolve_all` takes one user id; this walks them all. The distinction is not
        cosmetic — the corpus is 50 independent simulated users (D25), so
        `resolve_all("user")` against this store examines zero keys and reports
        success, which is how a whole-store repair pass can silently do nothing.
        """
        stats = ResolutionStats()
        for user_id in self.store.user_ids():
            for subject, predicate in self.store.predicate_keys(user_id):
                self._resolve_key(user_id, subject, predicate, stats)
        return stats

    def resolve_memories(self, memories: list[Memory]) -> ResolutionStats:
        """Resolve only the keys touched by a batch — the ingestion hot path."""
        stats = ResolutionStats()
        seen: set[tuple[str, str, str]] = set()
        for m in memories:
            if not (m.subject and m.predicate):
                continue
            key = (m.user_id, m.subject, m.predicate)
            if key in seen:
                continue
            seen.add(key)
            self._resolve_key(m.user_id, m.subject, m.predicate, stats)
        return stats

    # ------------------------------------------------------------- one key

    def _resolve_key(
        self, user_id: str, subject: str, predicate: str, stats: ResolutionStats
    ) -> None:
        memories = self.store.find_by_predicate(
            user_id, subject, predicate, include_superseded=True
        )
        if not memories:
            return
        if len(memories) == 1:
            only = memories[0]
            if _operation(only) == "removes" and only.occurred_at is not None:
                self._make_historical(only, stats)
            return

        # A key is resolvable if its predicate is structurally single-valued, or if
        # the user's own wording marked a statement as replacing an earlier one.
        # Without the second clause, "switched completely to PyTorch" would not
        # supersede "uses TensorFlow", because `uses_framework` is not on the arity
        # list — you can, in general, use two frameworks (D20).
        if not (is_single_valued(predicate) or any(m.replaces_previous for m in memories)):
            # Repair, not merely abstain. If this key was resolved under an earlier,
            # wrong arity call, its facts are sitting superseded and invisible to
            # retrieval. Correcting the predicate list has to actually give them
            # back, or one bad call is permanent in every store already built — and
            # rebuilding a store costs hours of quota.
            for memory in memories:
                if memory.status == "superseded":
                    self.store.mark_current(memory.id)
                    stats.repaired += 1
                    stats.changes.append(TimelineChange(memory.id, "repaired"))
            return

        stats.keys_examined += 1

        # `occurred_at`, not `event_time`: a fact that states no time still has a
        # place on the timeline — the conversation it was said in. Reading
        # `event_time` here would make every memory undated the moment the two
        # were separated, and the whole resolver would go quiet.
        dated = [m for m in memories if m.occurred_at is not None]
        stats.skipped_undated += len(memories) - len(dated)
        if len(dated) < 2:
            return

        dated.sort(key=lambda m: (m.occurred_at, m.id))

        # Collapse consecutive equal values into runs. The earliest member owns the
        # interval; the rest are restatements of a value already in force.
        runs: list[list[Memory]] = []
        for m in dated:
            if (
                runs
                and not _is_removal(runs[-1][0])
                and not _is_removal(m)
                and _is_restatement(m, runs[-1][0])
            ):
                runs[-1].append(m)
            else:
                runs.append([m])

        # Work from successors rather than blindly closing each row with the next
        # row in event order. If coexisting facts have accumulated, a later
        # `replaces_previous` signal does not say which one it replaces. Row order is
        # not evidence: choosing the immediately preceding Walmart trip once hid the
        # still-relevant weekly grocery average. The safe direction is to abstain and
        # leave both visible until extraction supplies a less ambiguous key.
        owners = [run[0] for run in runs]
        active: list[Memory] = []
        successor: dict[str, Memory] = {}
        historical: set[str] = set()
        for owner in owners:
            operation = _operation(owner)
            if operation == "removes":
                previous = _target(active, owner)
                if previous is not None:
                    active.remove(previous)
                    successor[previous.id] = owner
                elif active:
                    stats.skipped_ambiguous += 1
                historical.add(owner.id)
                continue
            if operation == "replaces" and active:
                previous = _target(active, owner)
                if previous is not None:
                    active.remove(previous)
                    successor[previous.id] = owner
                else:
                    stats.skipped_ambiguous += 1
            active.append(owner)

        for owner in owners:
            winner = successor.get(owner.id)
            if winner is None:
                if owner.id in historical:
                    self._make_historical(owner, stats)
                else:
                    self._make_current(owner, stats)
            else:
                self._supersede(owner, winner, stats)

        # Fold only after the owner's final interval is known, so a restatement gets
        # the same valid_to as the value it repeats.
        for owner, run in zip(owners, runs, strict=True):
            for repeat in run[1:]:
                self._fold(repeat, owner, stats)

    # --------------------------------------------------------------- writes
    #
    # Each of these is a no-op when the store already holds the desired state.
    # That is what makes re-resolution idempotent: a settled timeline produces no
    # writes and no counter movement, so `superseded` reports changes made rather
    # than changes present.

    def _supersede(self, loser: Memory, winner: Memory, stats: ResolutionStats) -> None:
        already = (
            loser.status == "superseded"
            and loser.superseded_by == winner.id
            and loser.valid_to == winner.occurred_at
        )
        if already:
            return
        # Distinguish "this fact is newly retired" from "its interval just moved
        # because something landed in the middle of the chain".
        rewired = loser.status == "superseded"
        self.store.mark_superseded(loser.id, winner.id, winner.occurred_at)
        if rewired:
            stats.revalidated += 1
        else:
            stats.superseded += 1
        stats.changes.append(
            TimelineChange(
                loser.id,
                "revalidated" if rewired else "superseded",
                winner.id,
                winner.occurred_at,
            )
        )

    def _fold(self, repeat: Memory, owner: Memory, stats: ResolutionStats) -> None:
        """A restatement points at the interval owner without ending it."""
        if repeat.status == "superseded" and repeat.superseded_by == owner.id:
            return
        self.store.mark_superseded(repeat.id, owner.id, owner.valid_to or repeat.occurred_at)
        self.store.set_validity(repeat.id, owner.valid_to)
        stats.restatements += 1
        stats.changes.append(TimelineChange(repeat.id, "restatement", owner.id))

    def _make_current(self, memory: Memory, stats: ResolutionStats) -> None:
        if memory.status == "active" and memory.valid_to is None:
            return
        self.store.mark_current(memory.id)
        stats.promoted += 1
        stats.changes.append(TimelineChange(memory.id, "promoted"))

    def _make_historical(self, memory: Memory, stats: ResolutionStats) -> None:
        at = memory.occurred_at
        if at is None:
            return
        if memory.status == "historical" and memory.valid_to == at:
            return
        self.store.mark_historical(memory.id, at)
        stats.terminated += 1
        stats.changes.append(TimelineChange(memory.id, "terminated", valid_to=at))
