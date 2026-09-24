"""Storage abstractions.

`MemoryStore` is a Protocol rather than an ABC so that a second backend (pgvector,
added in V2) can be written without inheriting from the SQLite implementation. The
same test suite runs against every backend — see tests/test_store.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

MemoryType = Literal["semantic", "episodic", "preference", "procedural", "profile"]
MemoryStatus = Literal["active", "superseded", "historical", "evicted"]


@dataclass(slots=True)
class Memory:
    """A single stored fact.

    `token_count` is the packer's knapsack weight and is required at construction
    time; callers compute it with the tokenizer they will actually pack against.
    """

    id: str
    user_id: str
    type: MemoryType
    content: str
    token_count: int

    subject: str | None = None
    """Who or what the fact is ABOUT — not who said it. "Andy wore a blue shirt",
    stated by the user, has subject 'andy'. See `source_role`."""
    predicate: str | None = None
    object: str | None = None
    target_object: str | None = None
    """The prior value explicitly targeted by a transition.

    `object` is the value established by ADD/COEXIST/REPLACE. `target_object` is
    the value closed by REPLACE/TERMINATE. Keeping them separate prevents a
    sentence such as "replaced the Nike shoes" from turning Nike into the new
    current value merely because it is the only value named in the sentence.
    """

    source_role: str = "user"
    """Who said it: user | assistant | system. Orthogonal to `subject`. Keeping
    these separate is what makes "what did you recommend?" expressible — it filters
    on the speaker, while "what does Andy wear?" filters on the subject."""

    scope: str | None = None
    """Why this is worth retaining: profile | preference | plan | recommendation |
    commitment | shared_context | event. Drives retrieval filters and the
    inspector's grouping."""

    importance: float = 0.5
    confidence: float = 1.0

    event_time: datetime | None = None
    """When the event happened, **as the fact itself states it**. None when the fact
    states no time, which is the common case and is deliberately not filled in.

    It used to be the session's date for every memory, which made "I moved last July"
    dated to the day it was mentioned and made a restatement of an old event look
    newer than the original. A guessed date is worse than an absent one because it
    ranks: the temporal layer already reports undated facts and leaves them alone,
    which is the correct outcome. See `observed_at` for the time that is always known.
    """

    observed_at: datetime | None = None
    """When the statement was made — the conversation's date.

    Always known, because it comes from the corpus rather than from the text. Rows
    written before this column existed carry it in `event_time` instead, and the
    migration copies it across; on those rows the two are equal and there is no way
    to tell a stated time from an assumed one, because the old code never recorded
    the difference.
    """

    event_time_expression: str | None = None
    """Date words retained from the extracted fact. The source session and span
    identify the original conversation; this field never invents missing wording."""
    event_time_source_expression: str | None = None
    """Date words copied from the matched source turn, when the anchor contains one.
    Kept distinct because the extracted fact may paraphrase or omit the user's words."""
    event_time_estimate: datetime | None = None
    """Nominal date for an imprecise expression; never used to close a timeline."""
    event_time_precision: str | None = None
    """day | approximate_day | month | unresolved. NULL on old rows."""

    valid_from: datetime | None = None
    valid_to: datetime | None = None
    ingested_at: datetime | None = None

    update_op: str = "coexists"
    """Stage B's verdict on what this fact does to earlier ones on its key.
    New resolution reads this operation directly. `replaces_previous` remains for
    old stores and one-stage extractor compatibility only."""

    replaces_previous: bool = False
    """The user explicitly framed this as replacing an earlier statement. Lets the
    temporal resolver act on keys whose predicate is not structurally single-valued."""

    superseded_by: str | None = None
    status: MemoryStatus = "active"

    strength: float = 1.0
    access_count: int = 0
    last_accessed_at: datetime | None = None
    strength_updated_at: datetime | None = None

    source_session_id: str | None = None
    source_turn_index: int | None = None
    source_char_start: int | None = None
    source_char_end: int | None = None
    entities: list[str] = field(default_factory=list)

    @property
    def occurred_at(self) -> datetime | None:
        """The best available time for the event: what was stated, else when it was said.

        This is what ordering and recency want. Reading `event_time` directly now
        answers a narrower question — "did the user give this fact a time?" — and the
        two must not be confused, because the whole point of separating them is that
        one is evidence and the other is an assumption.
        """
        return self.event_time or self.observed_at

    @property
    def event_time_is_stated(self) -> bool:
        """Whether the time came from the fact rather than from its conversation."""
        return self.event_time is not None

    @property
    def is_current(self) -> bool:
        return self.status == "active" and self.valid_to is None


@dataclass(slots=True)
class Turn:
    id: str
    session_id: str
    turn_index: int
    role: Literal["user", "assistant"]
    content: str
    ts: datetime


@dataclass(slots=True)
class Session:
    id: str
    user_id: str
    started_at: datetime
    source: str | None = None
    turns: list[Turn] = field(default_factory=list)


@dataclass(slots=True)
class LexicalHit:
    """A BM25 hit. `score` is raw FTS5 bm25() output, which is negative and
    unbounded; normalization to [0,1] happens in the retrieval layer (P3) so that
    the raw number stays inspectable in the demo's score-breakdown panel."""

    memory_id: str
    score: float


@runtime_checkable
class MemoryStore(Protocol):
    """Persistence contract. Deliberately excludes ranking: scoring lives in the
    retrieval layer so that swapping backends cannot change eval results."""

    def initialize(self) -> None: ...

    def add_session(self, session: Session) -> None: ...

    def add_memories(self, memories: list[Memory]) -> None:
        """Batch insert. Batched because ingestion writes thousands of rows and
        per-row transactions dominate wall-clock on SQLite."""
        ...

    def get(self, memory_id: str) -> Memory | None: ...

    def get_many(self, memory_ids: list[str]) -> list[Memory]: ...

    def iter_active(self, user_id: str) -> list[Memory]: ...

    def find_by_predicate(
        self, user_id: str, subject: str, predicate: str, include_superseded: bool = False
    ) -> list[Memory]:
        """Memories sharing a (subject, predicate) key.

        `include_superseded` is what lets P4 rebuild a whole timeline rather than
        compare against the current head — necessary because ingestion order is not
        event order.
        """
        ...

    def user_ids(self) -> list[str]:
        """Every namespace present. Whole-store passes must walk all of them."""
        ...

    def memory_ids(self) -> set[str]:
        """Every durable memory id, independent of tenant or lifecycle status."""
        ...

    def session_ids_for_user(self, user_id: str) -> set[str]:
        """Session ids belonging to one namespace."""
        ...

    def session_ids(self) -> set[str]:
        """Every archived source session id, including resumable in-flight work."""
        ...

    def predicate_keys(self, user_id: str) -> list[tuple[str, str]]:
        """Every (subject, predicate) present, for a full re-resolution pass."""
        ...

    def search_lexical(
        self, user_id: str, query: str, limit: int, include_superseded: bool = False
    ) -> list[LexicalHit]: ...

    def mark_superseded(self, memory_id: str, superseded_by: str, valid_to: datetime) -> None: ...

    def remove_memories(self, memory_ids: list[str]) -> int:
        """Undo the memory half of a write that failed after they were persisted."""
        ...

    def restore_memory_states(
        self, states: list[tuple[str, str, str | None, datetime | None, datetime | None]]
    ) -> None:
        """Put back (status, superseded_by, valid_from, valid_to) captured before resolution."""
        ...

    def mark_current(self, memory_id: str) -> None:
        """Reopen a memory as the live value of its key.

        Resolution is not monotonic: ingesting an older fact can demote the current
        head, and a corrected re-run has to be able to promote one back.
        """
        ...

    def mark_historical(self, memory_id: str, at: datetime) -> None:
        """Retain a transition event without making it a current attribute value."""
        ...

    def set_validity(self, memory_id: str, valid_to: datetime | None) -> None: ...

    def record_access(
        self, memory_ids: list[str], at: datetime, reinforcement: float = 0.30
    ) -> None:
        """Reinforcement half of decay (P5): bumps access_count and last_accessed_at."""
        ...

    def set_strengths(self, strengths: dict[str, float], at: datetime | None = None) -> None:
        """Persist decayed strengths without changing retrieval metadata."""
        ...

    def mark_evicted(self, memory_ids: list[str]) -> None:
        """Hide low-value active memories while preserving their provenance."""
        ...

    def add_evidence(self, memory_id: str, source_memory_ids: list[str]) -> None:
        """Record the raw memories supporting a consolidated memory."""
        ...

    def evidence_for(self, memory_id: str) -> list[str]:
        """Return source-memory IDs in a stable order for inspection and auditing."""
        ...

    def evidence_source_ids(self) -> set[str]:
        """Sources already folded into a consolidation, used to prevent re-folding."""
        ...

    def turns_for_session(self, session_id: str) -> list[Turn]:
        """Return the lossless source turns in their original order."""
        ...

    def search_turns_in_sessions(
        self, user_id: str, query: str, session_ids: set[str], limit: int = 3
    ) -> list[Turn]:
        """BM25 search restricted to already-located source sessions."""
        ...

    def count(self, user_id: str | None = None, status: MemoryStatus | None = None) -> int: ...

    def close(self) -> None: ...


@runtime_checkable
class VectorIndex(Protocol):
    """Semantic half of hybrid retrieval.

    Exact search only. Approximate indexes (HNSW/IVF) would introduce recall noise
    that is indistinguishable from a regression in the memory algorithm, which
    would make the ablation table unreadable. At <1M vectors exact search is fast
    enough that there is nothing to buy.
    """

    def add(self, ids: list[str], vectors: object) -> None: ...

    def search(self, vector: object, limit: int) -> list[tuple[str, float]]: ...

    def save(self) -> None: ...

    def __len__(self) -> int: ...
