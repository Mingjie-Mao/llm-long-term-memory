"""FastAPI application.

Handlers translate HTTP to `MemoryService` calls and back. They contain no domain
logic: anything a handler decides is a decision the MCP server and the inspector
would have to re-implement, and the point of the service layer is that they do not.

Run it:

    uvicorn llm_long_term_memory.api.app:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from .logging import RequestLoggingMiddleware, configure_logging, logger, set_fingerprint
from .models import (
    AnswerRequest,
    AnswerResponse,
    EvidenceOut,
    GoldenRunResponse,
    HealthResponse,
    MemoryDetailResponse,
    MemoryListResponse,
    MemoryOut,
    MessageRequest,
    MessageResponse,
    RawSearchResponse,
    RawTurnOut,
    RejectedOut,
    ScoredMemoryOut,
    SearchRequest,
    SearchResponse,
    TimelineResponse,
)
from .service import EncoderUnavailable, MemoryNotFound, MemoryService, NamespaceRequired

_service: MemoryService | None = None


def get_service() -> MemoryService:
    if _service is None:  # pragma: no cover - guarded by lifespan
        raise HTTPException(status_code=503, detail="service not initialised")
    return _service


def set_service(service: MemoryService | None) -> None:
    """Injection point for tests, which build a service over a temp store rather
    than the configured one."""
    global _service
    _service = service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One composition root. Building these per request would reload a transformer
    # on every call.
    configure_logging()
    # Only close what we opened. An injected service belongs to whoever injected it —
    # closing it here left a shut store behind a live global, so a second startup in
    # the same process (two TestClients, an embedded host) failed on a closed
    # connection.
    owns_service = _service is None
    if owns_service:
        set_service(MemoryService())
    set_fingerprint(_service.fingerprint())

    # Warm the encoder here rather than on the first request. It is lazily loaded so
    # that CLI commands which never embed do not pay for torch, but in a service that
    # laziness just moves a 12s model load onto whichever user arrives first — and
    # makes the p95 in the logs a property of process age rather than of the system.
    # A read-only deployment without the embedding extra is allowed to skip it;
    # `search` will fail loudly there anyway.
    try:
        _service.encoder.encode_one("warmup")
    except EncoderUnavailable as exc:
        logger().warning("encoder_unavailable", detail=str(exc))

    logger().info(
        "startup",
        store=_service.store_name,
        memories=_service.store.count(),
        answerer=_service.config.models.answerer,
        config=_service.fingerprint(),
        write_path="enabled" if _service.extractor else "disabled (no extractor)",
        live_answer="enabled" if getattr(_service, "answerer", None) else "disabled (no key)",
    )
    yield
    if owns_service and _service is not None:
        _service.close()
        set_service(None)


app = FastAPI(
    title="llm-long-term-memory",
    version="1.0.0",
    summary="A persistent memory layer for LLM agents.",
    lifespan=lifespan,
)
app.add_middleware(RequestLoggingMiddleware)


@app.exception_handler(NamespaceRequired)
async def _namespace_required(_request, exc: NamespaceRequired):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/", include_in_schema=False)
def inspector():
    """The inspector, served from the same process as the API it reads.

    A single static file with no build step and no dependencies: it is a client of
    the documented endpoints, so anything it can show, an integrator can fetch.
    """
    # no-store, because the page is edited during development and a cached copy
    # silently shows stale behaviour — which cost a confused debugging round when a
    # fixed timeline renderer appeared not to have changed.
    return FileResponse(
        Path(__file__).parent / "static" / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.exception_handler(EncoderUnavailable)
async def _encoder_unavailable(_request, exc: EncoderUnavailable):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/healthz", response_model=HealthResponse)
def healthz(service: MemoryService = Depends(get_service)) -> HealthResponse:
    """Reports `degraded` when the read path cannot serve a query.

    A health check that returns ok while `search` returns 500 is worse than no
    health check: it tells an orchestrator to send traffic to a container that
    cannot answer it. Still a 200, because the process is alive and the browse and
    timeline endpoints work — the status field carries the distinction.
    """
    available = service.encoder_available
    return HealthResponse(
        status="ok" if available else "degraded",
        store=service.store_name,
        memories=service.store.count(),
        search_available=available,
        detail=None
        if available
        else "semantic search unavailable: the `embed` extra is not installed",
    )


@app.get("/v1/config")
def read_config(service: MemoryService = Depends(get_service)) -> dict[str, Any]:
    """The manifest of what this process is running. Never includes credentials."""
    return service.manifest()


@app.post("/v1/memories/search", response_model=SearchResponse)
def search(request: SearchRequest, service: MemoryService = Depends(get_service)) -> SearchResponse:
    result = service.search(
        request.user_id,
        request.query,
        limit=request.limit,
        include_superseded=request.include_superseded,
        explain=request.explain,
    )
    return SearchResponse(
        memories=[ScoredMemoryOut.of_hit(hit) for hit in result.memories],
        rejected=[
            RejectedOut(
                memory_id=r.memory_id,
                reason=r.reason,
                superseded_by=r.superseded_by,
                content=r.content,
                score=r.score,
            )
            for r in result.rejected
        ],
        candidates_considered=result.candidates_considered,
    )


@app.get("/v1/memories", response_model=MemoryListResponse)
def list_memories(
    user_id: str = Query(min_length=1),
    status: str | None = None,
    type: str | None = None,
    scope: str | None = None,
    source_role: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: MemoryService = Depends(get_service),
) -> MemoryListResponse:
    memories, total = service.list_memories(
        user_id,
        status=status,
        memory_type=type,
        scope=scope,
        source_role=source_role,
        limit=limit,
        offset=offset,
    )
    return MemoryListResponse(
        memories=[MemoryOut.of(m) for m in memories],
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get("/v1/memories/{memory_id}", response_model=MemoryDetailResponse)
def get_memory(
    memory_id: str,
    user_id: str = Query(min_length=1),
    service: MemoryService = Depends(get_service),
) -> MemoryDetailResponse:
    try:
        memory = service.get(user_id, memory_id)
    except MemoryNotFound:
        # 404 for a memory in another namespace too: distinguishing "not yours" from
        # "does not exist" would leak whether an id is in use.
        raise HTTPException(status_code=404, detail="memory not found") from None
    evidence = service.evidence(user_id, memory_id)
    return MemoryDetailResponse(
        memory=MemoryOut.of(memory),
        evidence=EvidenceOut(**evidence) if evidence else None,
    )


@app.post("/v1/raw/search", response_model=RawSearchResponse)
def raw_search(
    request: SearchRequest, service: MemoryService = Depends(get_service)
) -> RawSearchResponse:
    """The fallback layer, queryable directly. Memory first — this is the 'when
    needed' half, and seeing it separately is what makes the distinction legible."""
    turns = service.raw_search(request.user_id, request.query, limit=request.limit or 3)
    return RawSearchResponse(
        turns=[
            RawTurnOut(
                session_id=t.session_id,
                turn_index=t.turn_index,
                role=t.role,
                content=t.content,
            )
            for t in turns
        ]
    )


@app.get("/v1/golden/{name}", response_model=GoldenRunResponse)
def golden_run(name: str, service: MemoryService = Depends(get_service)) -> GoldenRunResponse:
    """A recorded demonstration run. Never executes a model.

    The inspector shows this by default so a demo page costs nothing to open and
    shows the same thing every time — while `is_current` prevents it from passing
    off a stale recording as present behaviour.
    """
    try:
        run, is_current = service.golden_run(name)
    except MemoryNotFound:
        raise HTTPException(status_code=404, detail=f"no recorded run named {name!r}") from None
    return GoldenRunResponse(**run.to_dict(), is_current=is_current)


@app.post("/v1/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest, service: MemoryService = Depends(get_service)) -> AnswerResponse:
    """Run the full answer path live. Costs quota, so it is never called on page load."""
    try:
        result = service.answer(request.user_id, request.query, limit=request.limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    return AnswerResponse(**result)


@app.get("/v1/timeline", response_model=TimelineResponse)
def timeline(
    user_id: str = Query(min_length=1),
    subject: str = Query(min_length=1),
    predicate: str = Query(min_length=1),
    service: MemoryService = Depends(get_service),
) -> TimelineResponse:
    entries = service.timeline(user_id, subject, predicate)
    return TimelineResponse(
        subject=subject,
        predicate=predicate,
        entries=[MemoryOut.of(m) for m in entries],
    )


@app.post("/v1/messages", response_model=MessageResponse)
def add_message(
    request: MessageRequest, service: MemoryService = Depends(get_service)
) -> MessageResponse:
    try:
        result = service.add_message(
            request.user_id, request.role, request.content, request.session_id
        )
    except RuntimeError as exc:
        # No extractor configured is a deployment state, not a client error.
        raise HTTPException(status_code=503, detail=str(exc)) from None
    return MessageResponse(
        session_id=result["session_id"],
        turn_index=result["turn_index"],
        memories=[MemoryOut.of(m) for m in result["memories"]],
        usage=result["usage"],
    )


@app.delete("/v1/memories/{memory_id}", status_code=204)
def forget(
    memory_id: str,
    user_id: str = Query(min_length=1),
    service: MemoryService = Depends(get_service),
) -> None:
    try:
        service.forget(user_id, memory_id)
    except MemoryNotFound:
        raise HTTPException(status_code=404, detail="memory not found") from None
