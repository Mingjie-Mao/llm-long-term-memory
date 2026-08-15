"""Request and response models for the v1 API.

Deliberately not the domain dataclasses. `Memory` carries fields the wire has no
business exposing (internal strength bookkeeping) and omits things a client needs
(the reason a memory was rejected). Keeping them separate is also what lets the
storage schema change without breaking clients.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    """Where a memory came from. Every memory has one; this is the product's
    central claim, so it is a required part of the response rather than a debug
    extra."""

    session_id: str | None = None
    turn_index: int | None = None
    char_start: int | None = None
    char_end: int | None = None


class MemoryOut(BaseModel):
    id: str
    content: str
    type: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    source_role: str = "user"
    scope: str | None = None
    status: str = "active"
    importance: float = 0.5
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    superseded_by: str | None = None
    source: SourceRef = Field(default_factory=SourceRef)

    @classmethod
    def of(cls, memory: Any) -> MemoryOut:
        return cls(
            id=memory.id,
            content=memory.content,
            type=memory.type,
            subject=memory.subject,
            predicate=memory.predicate,
            object=memory.object,
            source_role=memory.source_role,
            scope=memory.scope,
            status=memory.status,
            importance=memory.importance,
            valid_from=memory.valid_from,
            valid_to=memory.valid_to,
            superseded_by=memory.superseded_by,
            source=SourceRef(
                session_id=memory.source_session_id,
                turn_index=memory.source_turn_index,
                char_start=memory.source_char_start,
                char_end=memory.source_char_end,
            ),
        )


class RetrievalSignalsOut(BaseModel):
    semantic: float
    bm25: float
    recency: float
    importance: float
    entity: float


class ScoredMemoryOut(MemoryOut):
    score: float
    signals: RetrievalSignalsOut
    rerank_score: float | None = None

    @classmethod
    def of_hit(cls, hit: Any) -> ScoredMemoryOut:
        base = MemoryOut.of(hit.memory)
        return cls(
            **base.model_dump(),
            score=hit.score,
            signals=RetrievalSignalsOut(**hit.signals.to_dict()),
            rerank_score=hit.rerank_score,
        )


class RejectedOut(BaseModel):
    """Why a candidate did not come back. Absence without explanation is
    indistinguishable from a retrieval bug, which is the thing this project is
    supposed to make visible."""

    memory_id: str
    reason: str
    superseded_by: str | None = None
    content: str = ""
    score: float | None = None


class SearchRequest(BaseModel):
    user_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    limit: int | None = Field(default=None, ge=1, le=100)
    include_superseded: bool = False
    explain: bool = False
    """Also return current memories that lost on rank. Powers the inspector's
    "why was this not used?" panel."""


class SearchResponse(BaseModel):
    memories: list[ScoredMemoryOut]
    rejected: list[RejectedOut] = Field(default_factory=list)
    candidates_considered: int = 0


class MessageRequest(BaseModel):
    user_id: str = Field(min_length=1)
    role: Literal["user", "assistant"] = "user"
    content: str = Field(min_length=1)
    session_id: str | None = None


class MessageResponse(BaseModel):
    session_id: str
    turn_index: int
    memories: list[MemoryOut]
    usage: dict[str, Any] = Field(default_factory=dict)


class MemoryListResponse(BaseModel):
    memories: list[MemoryOut]
    total: int
    limit: int
    offset: int


class EvidenceOut(BaseModel):
    session_id: str
    turn_index: int
    role: str
    text: str
    char_start: int | None = None
    char_end: int | None = None


class MemoryDetailResponse(BaseModel):
    memory: MemoryOut
    evidence: EvidenceOut | None = None


class TimelineResponse(BaseModel):
    """A supersession chain, oldest first. `valid_to` on all but the last entry is
    what turns a pile of contradictory facts into a history."""

    subject: str
    predicate: str
    entries: list[MemoryOut]


class RawTurnOut(BaseModel):
    session_id: str
    turn_index: int
    role: str
    content: str


class RawSearchResponse(BaseModel):
    """What the raw-conversation fallback can recover for a query.

    Structured memory is a lossy compression; this is the ground truth it was
    compressed from, and the answer path only reaches it when the compression turns
    out to have dropped what was asked for.
    """

    turns: list[RawTurnOut]


class RecordedEvidenceOut(BaseModel):
    session_id: str
    turn_index: int
    role: str
    text: str


class GoldenRunResponse(BaseModel):
    """A recorded run, explicitly labelled as recorded.

    `is_current` is the field that keeps this honest: a recording made under a
    different store or prompt version is reported stale rather than presented as
    what the system does now.
    """

    name: str
    namespace: str
    query: str
    answer: str
    answer_status: str
    fallback_level: str
    fallback_reason: str | None = None
    evidence: list[RecordedEvidenceOut] = Field(default_factory=list)
    memories_selected: int = 0
    candidates_considered: int = 0
    judge_verdict: str | None = None
    judge_reason: str | None = None
    recorded_at: str = ""
    runs: int = 1
    versions: dict[str, str | None] = Field(default_factory=dict)
    note: str = ""
    is_current: bool = True
    """False when the live process no longer matches the fingerprint recorded."""


class AnswerRequest(BaseModel):
    user_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    limit: int | None = Field(default=None, ge=1, le=100)


class SelectedMemoryOut(BaseModel):
    id: str | None = None
    content: str = ""
    score: float | None = None
    scope: str | None = None
    source_role: str | None = None


class AnswerResponse(BaseModel):
    answer: str
    top_k: int = 0
    selected: list[SelectedMemoryOut] = Field(default_factory=list)
    """Exactly the memories the answerer received. The inspector renders these rather
    than running its own retrieval, so what is inspected is what was answered from."""
    answer_status: str | None = None
    fallback_level: str = "none"
    fallback_reason: str | None = None
    fallback_turns: list[str] = Field(default_factory=list)
    memories_selected: int = 0
    candidates_considered: int = 0
    context_tokens: int = 0
    latency_ms: float = 0.0
    answerer_calls: int = 1


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    store: str
    memories: int
    search_available: bool = True
    detail: str | None = None
