"""A public playground over the parts of the memory engine that cost nothing to run.

**Why this lives outside `src/`.** The v2 experiment freeze hashes
`src/llm_long_term_memory/**/*.py`, `src/**/*.sql`, `scripts/*.py`, `pyproject.toml`
and `uv.lock` file by file. Editing any of them makes the in-flight `dev100` store
unable to prove it was produced under the frozen system — which has already cost this
project 408 extractor calls once. This module therefore *imports* the engine and adds
nothing to it, lives in a directory the freeze does not glob, and uses only
dependencies `pyproject.toml` already declares. Nothing here may be moved into `src/`
before the final test run.

**Why only half the engine.** Two operations need a model: turning prose into facts
(the extractor) and turning memories into an answer (the answerer). Both draw on a
free-tier pool of 500 requests per day that the experiments are currently spending.
Everything else the engine does — storing a fact, closing the previous value on the
same key, ranking, saying *why* a memory was rejected, BM25 over the raw archive,
rebuilding a supersession chain — is local computation with no API call and no
per-request cost. That is also the half a reader does not believe from a static page,
so it is the half worth serving live first.

**Identity is issued, never accepted.** The engine reads `user_id` from the request
body, which the README lists under its limitations. A public deployment cannot do
that: anyone could type another namespace and read it. Here the server mints an
opaque namespace, signs it, and hands back a token; every later call derives the
namespace from the token's signature. The client never names a namespace and cannot
forge one without the secret.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_long_term_memory.embed import Encoder
from llm_long_term_memory.ingest.schemas import (
    SINGLE_VALUED_PREDICATES,
    normalize_predicate,
)
from llm_long_term_memory.retrieve import HybridRetriever
from llm_long_term_memory.store import (
    Memory,
    NumpyFlatIndex,
    Session,
    SQLiteMemoryStore,
    Turn,
    scoped_session_id,
)
from llm_long_term_memory.temporal.resolve import TemporalResolver

# --------------------------------------------------------------------------- limits
#
# Deliberately small. This is a portfolio playground, not a service: the numbers are
# sized so that one visitor cannot make the page unusable for the next one, and so
# that a loop in someone's console cannot fill the disk.

SESSION_TTL = timedelta(hours=1)
MAX_FACTS_PER_SESSION = 30
MAX_TURNS_PER_SESSION = 30
MAX_CHARS = 2_000
MAX_SESSIONS_PER_IP_PER_HOUR = 10
MAX_REQUESTS_PER_IP_PER_MINUTE = 60

DEMO_STORE = Path(os.environ.get("LLTM_DEMO_STORE", "stores/demo-playground"))
# Not derived from the experiment secret and not reused anywhere: a demo token grants
# access to a throwaway namespace and nothing else. Regenerated on restart, which
# invalidates outstanding tokens — acceptable for sessions that live an hour.
SECRET = os.environ.get("LLTM_DEMO_SECRET", secrets.token_hex(32)).encode()

_PREDICATE_OK = re.compile(r"^[a-z0-9_]{1,40}$")


# ------------------------------------------------------------------------- identity


def _sign(namespace: str) -> str:
    mac = hmac.new(SECRET, namespace.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{namespace}.{mac}"


def _verify(token: str) -> str:
    """Return the namespace a token proves, or raise.

    `compare_digest` rather than `==` so that a wrong signature costs the same time as
    a right one; the namespace is short and guessable otherwise.
    """
    namespace, _, mac = token.partition(".")
    if not namespace or not mac:
        raise HTTPException(401, "malformed session token")
    if not hmac.compare_digest(_sign(namespace), token):
        raise HTTPException(401, "invalid session token")
    return namespace


# ------------------------------------------------------------------------ rate limit


class Limiter:
    """Fixed-window counters held in memory.

    In memory because the alternative is a dependency, and this is one process. It
    resets on restart, which is the correct failure direction for a demo: a restart
    should not lock anyone out.
    """

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window: float) -> None:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                retry = int(window - (now - q[0])) + 1
                raise HTTPException(
                    429,
                    detail={
                        "error": "rate_limited",
                        "message": "Too many requests from this address.",
                        "retry_after_seconds": retry,
                    },
                )
            q.append(now)


limiter = Limiter()


def client_ip(request: Request) -> str:
    # Cloudflare and most proxies set this; falling back to the socket peer is right
    # for local runs and wrong behind an unknown proxy, which is why the limit is a
    # backstop rather than a security control.
    fwd = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for")
    return (fwd.split(",")[0].strip() if fwd else None) or (
        request.client.host if request.client else "unknown"
    )


# --------------------------------------------------------------------------- engine


@dataclass
class Playground:
    store: SQLiteMemoryStore
    index: NumpyFlatIndex
    encoder: Encoder
    retriever: HybridRetriever
    resolver: TemporalResolver
    created: dict[str, datetime]
    encoder_id: str = "unknown"

    def namespace_alive(self, namespace: str) -> bool:
        made = self.created.get(namespace)
        return made is not None and datetime.now() - made < SESSION_TTL


pg: Playground | None = None


def engine() -> Playground:
    if pg is None:  # pragma: no cover - set in lifespan
        raise HTTPException(503, "playground not initialised")
    return pg


def session_namespace(request: Request, x_demo_token: str = Header(default="")) -> str:
    limiter.check(f"ip:{client_ip(request)}", MAX_REQUESTS_PER_IP_PER_MINUTE, 60)
    namespace = _verify(x_demo_token)
    if not engine().namespace_alive(namespace):
        raise HTTPException(
            410,
            detail={
                "error": "session_expired",
                "message": "This demo session has expired. Start a new one.",
            },
        )
    return namespace


# ------------------------------------------------------------------------ hard delete


def hard_delete(store: SQLiteMemoryStore, namespace: str) -> dict[str, int]:
    """Physically remove a demo namespace.

    The product's `forget` marks a memory evicted and never erases it, because
    provenance is the point. That is right for the product and wrong here: a stranger
    may type something personal into a public box, and "we kept it but hid it" is not
    an answer. So the demo deletes rows, and the core API's semantics are left alone.
    """
    # Two things this must not do, both learned by doing them.
    #
    # **Do not touch `memories_fts` / `turns_fts` directly.** They are FTS5
    # *external content* tables, and the schema already carries AFTER DELETE triggers
    # that retire the right index entry. Issuing `DELETE FROM memories_fts` in
    # addition corrupts the index outright — SQLite then reports "database disk image
    # is malformed" on the next statement.
    #
    # **Do not open a second connection.** The store holds an open handle in WAL mode
    # and is single-writer by design; a second writer on the same file is the failure
    # this project already documents. Reusing the store's connection keeps one writer
    # even though it means reaching for a private attribute.
    db = store._conn
    db.execute("PRAGMA foreign_keys = ON")
    sessions = [r[0] for r in db.execute("SELECT id FROM sessions WHERE user_id = ?", (namespace,))]
    memories = [r[0] for r in db.execute("SELECT id FROM memories WHERE user_id = ?", (namespace,))]
    turns = db.execute(
        "SELECT count(*) FROM turns WHERE session_id IN "
        "(SELECT id FROM sessions WHERE user_id = ?)",
        (namespace,),
    ).fetchone()[0]

    db.execute("DELETE FROM memories WHERE user_id = ?", (namespace,))
    db.execute(
        "DELETE FROM turns WHERE session_id IN (SELECT id FROM sessions WHERE user_id = ?)",
        (namespace,),
    )
    db.execute("DELETE FROM sessions WHERE user_id = ?", (namespace,))
    db.execute("DELETE FROM entities WHERE user_id = ?", (namespace,))
    db.commit()

    return {"sessions": len(sessions), "memories": len(memories), "turns": turns, "ids": memories}


def drop_from_index(index: NumpyFlatIndex, ids: list[str]) -> int:
    """Remove vectors for deleted memories.

    `NumpyFlatIndex` is append-only — it has `add` and `save` and no delete — and it
    lives in `src/`, which the experiment freeze hashes, so a delete method cannot be
    added there before the final run. Rewriting the two arrays from outside is the
    remaining option.

    Leaving them would not corrupt a search: `get_many` drops ids the store no longer
    holds and the retriever filters by namespace anyway. It would corrupt the promise.
    An embedding is derived from what a stranger typed into a public box, and "we
    deleted the row but kept the vector" is not the hard delete the page claims.
    """
    if not ids:
        return 0
    doomed = set(ids)
    keep = [i for i, mid in enumerate(index._ids) if mid not in doomed]
    dropped = len(index._ids) - len(keep)
    if not dropped:
        return 0
    index._vectors = index._vectors[keep] if keep else np.zeros((0, index.dim), dtype=np.float32)
    index._ids = [index._ids[i] for i in keep]
    index.save()
    return dropped


def sweep(p: Playground) -> int:
    """Delete every namespace past its hour. Called on each session creation, which is
    the only moment the store is guaranteed to be growing."""
    dead = [ns for ns, made in p.created.items() if datetime.now() - made >= SESSION_TTL]
    for ns in dead:
        drop_from_index(p.index, hard_delete(p.store, ns)["ids"])
        p.created.pop(ns, None)
    return len(dead)


# ----------------------------------------------------------------------------- wire


class FactIn(BaseModel):
    predicate: str = Field(description="snake_case key, e.g. lives_in")
    object: str = Field(description="the value, e.g. Melbourne")
    content: str = Field(description="how a person would say it")
    subject: str = "user"
    scope: str | None = "profile"
    source_role: str = "user"
    event_time: str | None = Field(default=None, description="ISO date; defaults to now")
    replaces_previous: bool = Field(
        default=False,
        description=(
            "Whether this statement closes the earlier value on the same key. It is a "
            "per-fact verdict, not a per-key one: a single `replaces` anywhere on a key "
            "once retired every consecutive pair on it, including successors that "
            "explicitly coexist."
        ),
    )


class TurnIn(BaseModel):
    role: str = "assistant"
    content: str
    session_id: str | None = None


class QueryIn(BaseModel):
    query: str
    limit: int = 10


app = FastAPI(title="LLTM playground", docs_url="/demo/docs", openapi_url="/demo/openapi.json")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.environ.get("LLTM_DEMO_ORIGINS", "*").split(",") if o],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["content-type", "x-demo-token"],
)


ONNX_DIR = Path(__file__).resolve().parent / "onnx-encoder"


def build_encoder() -> tuple[Any, str]:
    """Prefer the ONNX encoder; fall back to the torch one it was exported from.

    The fallback exists so a checkout without the exported model still runs. It is a
    fallback and not a choice: on a 512MB instance the torch path is the one that
    does not fit.
    """
    if (ONNX_DIR / "model.onnx").exists():
        try:
            from onnx_encoder import OnnxEncoder

            return OnnxEncoder(ONNX_DIR), "onnx:all-MiniLM-L6-v2"
        except Exception as exc:  # pragma: no cover - depends on the host
            print(f"onnx encoder unavailable ({exc}); falling back to torch")
    return Encoder(), "torch:all-MiniLM-L6-v2"


@app.on_event("startup")
def _start() -> None:
    global pg
    DEMO_STORE.parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteMemoryStore(DEMO_STORE.with_suffix(".db"))
    store.initialize()
    encoder, encoder_id = build_encoder()

    # A vector index is only meaningful under the encoder that produced it. This
    # project has already shipped one store built by two different generations of a
    # component and had to label every result from it diagnostic; the same mistake is
    # cheaper to prevent than to detect. The demo store is ephemeral by design, so the
    # right response to a mismatch is to discard it rather than to reconcile it.
    previous = store.get_meta("demo_encoder")
    index_path = DEMO_STORE.parent / f"{DEMO_STORE.name}-index"
    if previous and previous != encoder_id:
        print(f"encoder changed {previous} -> {encoder_id}; discarding the demo index")
        for suffix in (".npy", ".ids.json"):
            index_path.with_suffix(suffix).unlink(missing_ok=True)
        db = store._conn
        for table in ("memories", "turns", "sessions", "entities"):
            db.execute(f"DELETE FROM {table}")
        db.commit()
    store.set_meta("demo_encoder", encoder_id)

    index = NumpyFlatIndex(index_path, dim=encoder.dim)
    pg = Playground(
        store=store,
        index=index,
        encoder=encoder,
        retriever=HybridRetriever(store, index, weights={"semantic": 1.0}, candidate_limit=50),
        resolver=TemporalResolver(store),
        created={},
        encoder_id=encoder_id,
    )


@app.get("/demo/health")
def health(p: Playground = Depends(engine)) -> dict[str, Any]:
    return {
        "status": "ok",
        "live_sessions": sum(1 for ns in p.created if p.namespace_alive(ns)),
        "llm_required": False,
        "encoder": p.encoder_id,
        "capabilities": [
            "add_fact",
            "supersession",
            "search",
            "explain",
            "timeline",
            "raw_search",
            "hard_delete",
        ],
        "session_ttl_minutes": int(SESSION_TTL.total_seconds() // 60),
        "single_valued_predicates": sorted(SINGLE_VALUED_PREDICATES),
    }


@app.post("/demo/session")
def new_session(request: Request, p: Playground = Depends(engine)) -> dict[str, Any]:
    limiter.check(f"new:{client_ip(request)}", MAX_SESSIONS_PER_IP_PER_HOUR, 3600)
    swept = sweep(p)
    namespace = f"demo_{uuid.uuid4().hex[:12]}"
    p.created[namespace] = datetime.now()
    return {
        "token": _sign(namespace),
        "expires_in_seconds": int(SESSION_TTL.total_seconds()),
        "swept_expired_sessions": swept,
        "limits": {
            "facts": MAX_FACTS_PER_SESSION,
            "turns": MAX_TURNS_PER_SESSION,
            "max_chars": MAX_CHARS,
        },
    }


@app.delete("/demo/session", status_code=200)
def end_session(ns: str = Depends(session_namespace), p: Playground = Depends(engine)) -> dict:
    removed = hard_delete(p.store, ns)
    removed["vectors"] = drop_from_index(p.index, removed.pop("ids"))
    p.created.pop(ns, None)
    return {"deleted": True, "rows": removed}


def _serialise(m: Memory) -> dict[str, Any]:
    return {
        "id": m.id,
        "content": m.content,
        "subject": m.subject,
        "predicate": m.predicate,
        "object": m.object,
        "scope": m.scope,
        "source_role": m.source_role,
        "status": m.status,
        "superseded_by": m.superseded_by,
        "event_time": m.event_time.isoformat() if m.event_time else None,
        "valid_from": m.valid_from.isoformat() if m.valid_from else None,
        "valid_to": m.valid_to.isoformat() if m.valid_to else None,
        "replaces_previous": m.replaces_previous,
        "single_valued_key": bool(m.predicate)
        and normalize_predicate(m.predicate) in SINGLE_VALUED_PREDICATES,
    }


@app.post("/demo/facts")
def add_fact(
    fact: FactIn,
    ns: str = Depends(session_namespace),
    p: Playground = Depends(engine),
) -> dict[str, Any]:
    """Write one fact and re-resolve its timeline. No model is called."""
    if len(fact.content) > MAX_CHARS or len(fact.object) > 200:
        raise HTTPException(413, "too long for a demo")
    predicate = normalize_predicate(fact.predicate)
    if not _PREDICATE_OK.match(predicate):
        raise HTTPException(422, "predicate must be snake_case letters, digits and underscores")
    if sum(1 for _ in p.store.iter_all(ns)) >= MAX_FACTS_PER_SESSION:
        raise HTTPException(429, f"a demo session holds at most {MAX_FACTS_PER_SESSION} facts")

    when = datetime.fromisoformat(fact.event_time) if fact.event_time else datetime.now()
    before = {m.id: m.status for m in p.store.iter_all(ns)}

    memory = Memory(
        id=f"m_{uuid.uuid4().hex[:10]}",
        user_id=ns,
        type="semantic",
        content=fact.content,
        token_count=len(fact.content.split()),
        subject=fact.subject,
        predicate=predicate,
        object=fact.object,
        source_role=fact.source_role,
        scope=fact.scope,
        event_time=when,
        valid_from=when,
        ingested_at=datetime.now(),
        replaces_previous=fact.replaces_previous,
        update_op="replaces" if fact.replaces_previous else "coexists",
    )
    p.store.add_memories([memory])
    p.index.add([memory.id], p.encoder.encode([memory.content]))
    p.index.save()

    stats = p.resolver.resolve_all(ns)
    after = list(p.store.iter_all(ns))
    changed = [_serialise(m) for m in after if before.get(m.id) not in (None, m.status)]
    return {
        "created": _serialise(memory),
        "superseded_now": stats.superseded,
        "changed_by_this_write": changed,
        "memories": [_serialise(m) for m in after],
    }


@app.post("/demo/turns")
def add_turn(
    turn: TurnIn,
    ns: str = Depends(session_namespace),
    p: Playground = Depends(engine),
) -> dict[str, Any]:
    """Store a raw conversation turn verbatim, so the archive has something to recover
    from. Storing is free; turning it into facts is the extractor's job and is not
    part of this half."""
    if len(turn.content) > MAX_CHARS:
        raise HTTPException(413, "too long for a demo")
    external = turn.session_id or "demo-chat"
    sid = scoped_session_id(ns, external)
    existing = p.store.get_session(sid)
    idx = len(existing.turns) if existing else 0
    if idx >= MAX_TURNS_PER_SESSION:
        raise HTTPException(429, f"a demo session holds at most {MAX_TURNS_PER_SESSION} turns")
    now = datetime.now()
    p.store.add_session(
        Session(
            id=sid,
            user_id=ns,
            started_at=existing.started_at if existing else now,
            source="demo",
            turns=[
                *(existing.turns if existing else []),
                Turn(
                    id=f"{sid}:{idx}",
                    session_id=sid,
                    turn_index=idx,
                    role="assistant" if turn.role == "assistant" else "user",
                    content=turn.content,
                    ts=now,
                ),
            ],
        )
    )
    return {"session_id": external, "turn_index": idx}


@app.post("/demo/search")
def search(
    q: QueryIn,
    ns: str = Depends(session_namespace),
    p: Playground = Depends(engine),
) -> dict[str, Any]:
    """Real hybrid retrieval, with the rejections and their reasons.

    `rejected` is the field no comparable system exposes and the reason this endpoint
    is worth serving live: an absence with a reason is the difference between "the
    ranking chose not to" and "something is broken".
    """
    if not q.query.strip():
        raise HTTPException(422, "empty query")
    started = time.perf_counter()
    vec = p.encoder.encode([q.query])[0]
    hits = p.retriever.retrieve(vec, q.query, ns, temporal=True, limit=q.limit)
    everything = list(p.store.iter_all(ns))
    kept = {h.memory.id for h in hits}

    rejected = []
    for m in everything:
        if m.id in kept:
            continue
        if m.status == "superseded":
            rejected.append(
                {
                    "id": m.id,
                    "content": m.content,
                    "reason": "superseded",
                    "superseded_by": m.superseded_by,
                }
            )
        elif m.status == "active":
            rejected.append({"id": m.id, "content": m.content, "reason": "below_rank"})

    return {
        "query": q.query,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "memories": [
            {**_serialise(h.memory), "score": round(h.score, 4), "signals": h.signals.to_dict()}
            for h in hits
        ],
        "rejected": rejected,
        "candidates_considered": len(everything),
    }


@app.get("/demo/memories")
def memories(ns: str = Depends(session_namespace), p: Playground = Depends(engine)) -> dict:
    rows = [_serialise(m) for m in p.store.iter_all(ns)]
    return {
        "active": [r for r in rows if r["status"] == "active"],
        "superseded": [r for r in rows if r["status"] == "superseded"],
        "total": len(rows),
    }


@app.get("/demo/timeline")
def timeline(
    predicate: str,
    subject: str = "user",
    ns: str = Depends(session_namespace),
    p: Playground = Depends(engine),
) -> dict[str, Any]:
    chain = p.store.find_by_predicate(
        ns, subject, normalize_predicate(predicate), include_superseded=True
    )
    chain.sort(key=lambda m: (m.event_time or datetime.min, m.id))
    entries = [_serialise(m) for m in chain]
    # Only claim a replacement when an entry actually names the next as its successor.
    # Rendering an arrow between every consecutive pair invented a history on
    # multi-valued keys — facts that coexist shown as a chain of replacements.
    for a, b in pairwise(entries):
        a["replaced_by_next"] = a["superseded_by"] == b["id"]
    if entries:
        entries[-1]["replaced_by_next"] = False
    return {
        "subject": subject,
        "predicate": normalize_predicate(predicate),
        "single_valued_key": normalize_predicate(predicate) in SINGLE_VALUED_PREDICATES,
        "entries": entries,
    }


@app.post("/demo/raw/search")
def raw_search(
    q: QueryIn,
    ns: str = Depends(session_namespace),
    p: Playground = Depends(engine),
) -> dict[str, Any]:
    """BM25 over the stored turns — the layer the answerer falls back to when a memory
    kept the gist and dropped the detail. Free, because it is an index lookup."""
    started = time.perf_counter()
    turns = p.store.search_turns(ns, q.query, limit=min(q.limit, 5))
    return {
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "turns": [
            {
                "session_id": t.session_id.split(":")[-1],
                "turn_index": t.turn_index,
                "role": t.role,
                "content": t.content,
            }
            for t in turns
        ],
    }
