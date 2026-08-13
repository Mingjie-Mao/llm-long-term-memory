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
MemoryStatus = Literal["active", "superseded", "evicted"]


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
    predicate: str | None = None
    object: str | None = None

    importance: float = 0.5
    confidence: float = 1.0

    event_time: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    ingested_at: datetime | None = None

    replaces_previous: bool = False
    """The user explicitly framed this as replacing an earlier statement. Lets the
    temporal resolver act on keys whose predicate is not structurally single-valued."""

    superseded_by: str | None = None
    status: MemoryStatus = "active"

    strength: float = 1.0
    access_count: int = 0
    last_accessed_at: datetime | None = None

    source_session_id: str | None = None
    entities: list[str] = field(default_factory=list)

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

    def predicate_keys(self, user_id: str) -> list[tuple[str, str]]:
        """Every (subject, predicate) present, for a full re-resolution pass."""
        ...

    def search_lexical(self, user_id: str, query: str, limit: int) -> list[LexicalHit]: ...

    def mark_superseded(self, memory_id: str, superseded_by: str, valid_to: datetime) -> None: ...

    def mark_current(self, memory_id: str) -> None:
        """Reopen a memory as the live value of its key.

        Resolution is not monotonic: ingesting an older fact can demote the current
        head, and a corrected re-run has to be able to promote one back.
        """
        ...

    def set_validity(self, memory_id: str, valid_to: datetime | None) -> None: ...

    def record_access(self, memory_ids: list[str], at: datetime) -> None:
        """Reinforcement half of decay (P5): bumps access_count and last_accessed_at."""
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
