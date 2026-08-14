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
`event_time`, and writes the intervals that timeline implies. That makes the
operation idempotent and order-independent: the same set of facts produces the same
timeline no matter what sequence they arrived in.

**Restatement is not change.** Saying "I use PyTorch" in March and again in August
did not move anything. Consecutive mentions of the same value collapse into one
interval owned by the *earliest* of them — so "when did you switch to PyTorch?" is
answered by March — and the later restatements are folded into the owner rather than
counted as changes.

There is no LLM call here. Ordering by `event_time` is arithmetic over data the
extractor already attached, and putting a model call on this path would charge a
request per memory for a decision that does not need one. Facts without a date are
reported and left alone rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from chronomem.ingest.schemas import is_single_valued
from chronomem.store import Memory, MemoryStore


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
    changes: list[TimelineChange] = field(default_factory=list)

    @property
    def writes(self) -> int:
        return len(self.changes)


def _value(memory: Memory) -> str:
    return (memory.object or memory.content or "").strip().lower()


def _is_removal(memory: Memory) -> bool:
    """A removal mentions the old value, but is never a restatement of it."""
    return memory.update_op == "removes"


def as_of(memories: list[Memory], when: datetime) -> list[Memory]:
    """Which of these facts were in force at `when`.

    The query the baselines cannot answer at all: "what framework was I using in
    April?" needs the interval a fact held, not just whether it is current.
    """
    out = []
    for m in memories:
        start = m.valid_from or m.event_time
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
        if len(memories) < 2:
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

        dated = [m for m in memories if m.event_time is not None]
        stats.skipped_undated += len(memories) - len(dated)
        if len(dated) < 2:
            return

        dated.sort(key=lambda m: (m.event_time, m.id))

        # Collapse consecutive equal values into runs. The earliest member owns the
        # interval; the rest are restatements of a value already in force.
        runs: list[list[Memory]] = []
        for m in dated:
            if (
                runs
                and not _is_removal(runs[-1][0])
                and not _is_removal(m)
                and _value(runs[-1][0]) == _value(m)
            ):
                runs[-1].append(m)
            else:
                runs.append([m])

        for i, run in enumerate(runs):
            owner, repeats = run[0], run[1:]
            nxt = runs[i + 1][0] if i + 1 < len(runs) else None

            # Resolvability is decided per key, but *closing* a fact is decided per
            # successor. Without this check a single `replaces` anywhere on a key
            # made the whole chain resolvable and then retired every consecutive
            # pair on it — including successors that explicitly said `coexists`.
            # Observed live: "averaging $100 per week on groceries" was retired by
            # "spent $75 at Walmart last Saturday", two facts that are both true,
            # because some third fact on `grocery_spending` had signalled a change.
            #
            # Stage B's gate scores 0% false supersede in isolation; the loss was
            # entirely in the integration, which is why the per-fact verdict has to
            # be honoured here rather than summarised into a per-key one.
            if nxt is None or not nxt.replaces_previous:
                self._make_current(owner, stats)
            else:
                self._supersede(owner, nxt, stats)

            for repeat in repeats:
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
            and loser.valid_to == winner.event_time
        )
        if already:
            return
        # Distinguish "this fact is newly retired" from "its interval just moved
        # because something landed in the middle of the chain".
        rewired = loser.status == "superseded"
        self.store.mark_superseded(loser.id, winner.id, winner.event_time)
        if rewired:
            stats.revalidated += 1
        else:
            stats.superseded += 1
        stats.changes.append(
            TimelineChange(
                loser.id, "revalidated" if rewired else "superseded", winner.id, winner.event_time
            )
        )

    def _fold(self, repeat: Memory, owner: Memory, stats: ResolutionStats) -> None:
        """A restatement points at the interval owner without ending it."""
        if repeat.status == "superseded" and repeat.superseded_by == owner.id:
            return
        self.store.mark_superseded(repeat.id, owner.id, owner.valid_to or repeat.event_time)
        self.store.set_validity(repeat.id, owner.valid_to)
        stats.restatements += 1
        stats.changes.append(TimelineChange(repeat.id, "restatement", owner.id))

    def _make_current(self, memory: Memory, stats: ResolutionStats) -> None:
        if memory.status == "active" and memory.valid_to is None:
            return
        self.store.mark_current(memory.id)
        stats.promoted += 1
        stats.changes.append(TimelineChange(memory.id, "promoted"))
