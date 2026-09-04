"""The service layer: one composition root, no HTTP.

Everything the API can do lives here as plain Python, so the same object backs the
REST handlers, the MCP tools, and the inspector without any of them growing a second
copy of the domain logic. HTTP concerns — status codes, request parsing — stay in
`app.py`.

Two rules this module exists to enforce:

**One set of resources per process.** The store, the vector index, and the encoder
are expensive to build (the encoder loads a transformer) and cheap to share. A
handler that constructed them per request would reload a model on every call.

**Namespace isolation is not optional.** Every read takes a `user_id` and the store
filters on it. This mirrors the benchmark harness, where the namespace is the
question id and cross-namespace leakage would have silently inflated every result.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from llm_long_term_memory.config import ExperimentConfig, Settings
from llm_long_term_memory.evaluation.judge import JUDGE_PROMPT_VERSION
from llm_long_term_memory.evaluation.runners.base import ANSWER_PROMPT_VERSION
from llm_long_term_memory.retrieve import EvidenceHydrator, HybridRetriever
from llm_long_term_memory.store import (
    Memory,
    NumpyFlatIndex,
    Session,
    SQLiteMemoryStore,
    Turn,
    external_session_id,
    scoped_session_id,
)


class NamespaceRequired(ValueError):
    """A stateful call arrived without a user_id."""


class MemoryNotFound(LookupError):
    pass


class EncoderUnavailable(RuntimeError):
    """The embedding model is not installed, so semantic retrieval cannot run.

    Distinguished from a generic failure because it is a *deployment* fact with a
    specific remedy, and because the alternative — a bare 500 — told a user nothing
    while `/healthz` simultaneously reported the service healthy."""


@dataclass(frozen=True, slots=True)
class RejectedMemory:
    """A candidate that was retrieved and then not returned.

    This is the field no comparable system exposes, and it is the whole point of the
    inspector: "superseded by m_8f2a" explains an absence that would otherwise look
    like a retrieval failure.

    Two reasons, and they mean opposite things:

    * `superseded` — the fact was true and no longer is. Returning it would be wrong.
    * `below_rank` — the fact is current but scored outside `limit`. Returning it
      would have been affordable; the ranking chose not to. This is the one worth
      staring at when an answer is wrong.
    """

    memory_id: str
    reason: str
    superseded_by: str | None = None
    content: str = ""
    score: float | None = None


@dataclass(slots=True)
class SearchResult:
    memories: list[Any]
    rejected: list[RejectedMemory]
    candidates_considered: int


class MemoryService:
    """Owns the store, index, encoder and retriever for the lifetime of a process."""

    def __init__(
        self,
        # The product, not the baseline. `baselines.yaml` leaves the conditional
        # raw-conversation fallback off, which is right for a comparison arm and
        # wrong for the thing being served: the demo the README opens with recovers
        # a dropped URL from the archive, and under the baseline config that path is
        # unreachable. dev50: 56.0% without it, 72.0% with (results/table.md).
        config_path: str | Path = "configs/fallback.yaml",
        # The clean P10 store: one extractor generation, 2,348/2,348 sessions, and
        # what every formal number is measured on. `two-stage-hydrated` mixes two
        # generations and its results stay labelled diagnostic.
        store_name: str = "two-stage-p10",
        settings: Settings | None = None,
        encoder=None,
        extractor=None,
    ) -> None:
        self.settings = settings or Settings()
        self.config = ExperimentConfig.from_yaml(config_path)
        self.store_name = store_name
        # Both the inspector and the answerer read this. Two independent defaults is
        # how the panel came to explain a different memory set from the one the
        # answerer saw.
        self.top_k = self.config.service.top_k

        self.store = SQLiteMemoryStore(self.settings.store_dir / f"{store_name}.db")
        self.store.initialize()

        # Injected in tests. Left lazy in production so that importing the app does
        # not pull torch.
        self._encoder = encoder
        # An injected encoder dictates the index width, because the two must agree
        # and the injected one is the concrete fact. Reading `.dim` off the lazy
        # production encoder would defeat the laziness by loading the model here.
        dim = getattr(encoder, "dim", None) if encoder is not None else None
        self.index = NumpyFlatIndex(
            self.settings.store_dir / f"{store_name}-index",
            dim=dim or self.config.models.embedding_dim,
        )
        # Only the write path needs this, and it needs an API key. Read-only
        # deployments and the whole test suite work without one.
        self.extractor = extractor
        # Live answering is opt-in for the same reason the write path is: it needs a
        # credential, and the read-only service must work without one. Built on first
        # use rather than here, so constructing the service neither requires a key nor
        # forces the encoder to load.
        self._answerer = None

        reranker = None
        if self.config.retrieval.rerank.enabled:
            from llm_long_term_memory.retrieve import CrossEncoderReranker

            reranker = CrossEncoderReranker(
                model_name=self.config.retrieval.rerank.model,
                candidates=self.config.retrieval.rerank.candidates,
                batch_size=self.config.retrieval.rerank.batch_size,
            )

        self.retriever = HybridRetriever(
            self.store,
            self.index,
            weights=self.config.retrieval.weights.model_dump(),
            candidate_limit=self.config.retrieval.candidate_limit,
            recency_halflife_days=self.config.retrieval.recency_halflife_days,
            use_strength=self.config.decay.enabled,
            reranker=reranker,
        )
        self.hydrator = EvidenceHydrator(
            self.store,
            neighbouring_sentences=self.config.hydration.neighbouring_sentences,
            allocation=self.config.hydration.allocation,
        )

    @property
    def encoder(self):
        if self._encoder is None:
            try:
                from llm_long_term_memory.embed import Encoder
            except ImportError as exc:
                raise EncoderUnavailable(
                    "semantic retrieval needs the `embed` extra: "
                    "uv sync --extra embed, or build the image with "
                    'EXTRAS="api,llm,embed"'
                ) from exc

            self._encoder = Encoder(self.config.models.embedder)
        return self._encoder

    @property
    def encoder_available(self) -> bool:
        """Whether the read path can actually serve a query.

        Checked without constructing the model: importing is enough to know, and
        loading it here would undo the laziness the property exists for.
        """
        if self._encoder is not None:
            return True
        import importlib.util

        return importlib.util.find_spec("sentence_transformers") is not None

    @property
    def answerer(self):
        """The live answer path, or None when no credential is configured.

        Reuses `MemoryRunner` — the same object the benchmark harness drives — so a
        live answer in the inspector goes through exactly the code that produced the
        recorded results. A second implementation here would let the demo and the
        measurements drift apart, which is the one thing a demo must not do.
        """
        if self._answerer is None and self.settings.has_api_key:
            from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
            from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
            from llm_long_term_memory.llm.client import GeminiClient

            quota = QuotaManager(
                state_dir=self.settings.store_dir / "quota",
                default=Limits(
                    rpm=self.config.quota.rpm,
                    tpm=self.config.quota.tpm,
                    rpd=self.config.quota.rpd,
                ),
            )
            quota.load_learned()
            client = GeminiClient(
                self.settings.require_api_key(), quota=quota, usage=UsageTracker()
            )
            self._answerer = MemoryRunner(
                client,
                model=self.config.models.answerer,
                encoder=self.encoder,
                store=self.store,
                index=self.index,
                temporal=self.config.temporal_resolution,
                top_k=self.top_k,
                retrieval_weights=self.config.retrieval.weights.model_dump(),
                candidate_limit=self.config.retrieval.candidate_limit,
                recency_halflife_days=self.config.retrieval.recency_halflife_days,
                raw_fallback=self.config.fallback.enabled,
                raw_fallback_max_turns=self.config.fallback.max_turns,
                raw_fallback_max_chars=self.config.fallback.max_chars,
            )
        return self._answerer

    # ------------------------------------------------------------------ reads

    def search(
        self,
        user_id: str,
        query: str,
        limit: int | None = None,
        include_superseded: bool = False,
        explain: bool = False,
    ) -> SearchResult:
        """Rank memories for a query within one namespace.

        `include_superseded=False` applies temporal resolution: only facts currently
        in force are returned, and the ones that were replaced come back under
        `rejected` rather than vanishing.
        """
        self._require_namespace(user_id)
        vector = self.encoder.encode_one(query)
        hits, trace = self.retriever.retrieve_with_trace(
            vector,
            query,
            user_id,
            temporal=not include_superseded,
            limit=limit or self.top_k,
        )

        rejected: list[RejectedMemory] = []
        seen: set[str] = set()
        if not include_superseded:
            # Not derived from the candidate trace: temporal filtering removes
            # superseded rows before they ever become candidates, so they are not in
            # it. Walk each returned memory's own key chain instead, which also
            # scopes the answer to what is relevant — "you are seeing Sydney because
            # Canberra was replaced by it" — rather than listing every dead fact in
            # the namespace.
            for hit in hits:
                current = hit.memory
                if not (current.subject and current.predicate):
                    continue
                for other in self.store.find_by_predicate(
                    user_id, current.subject, current.predicate, include_superseded=True
                ):
                    if (
                        other.status == "superseded"
                        and other.superseded_by == current.id
                        and other.id not in seen
                    ):
                        seen.add(other.id)
                        rejected.append(
                            RejectedMemory(
                                memory_id=other.id,
                                reason="superseded",
                                superseded_by=other.superseded_by,
                                content=other.content,
                            )
                        )

        if explain:
            # Current facts that lost on rank rather than on validity. This is the
            # half of "why was this not used?" that a superseded list cannot answer,
            # and the half that indicts the ranking rather than the timeline.
            returned = {hit.memory.id for hit in hits}
            for candidate_id in trace.candidate_ids:
                if candidate_id in returned or candidate_id in seen:
                    continue
                memory = self.store.get(candidate_id)
                if memory is None:
                    continue
                seen.add(candidate_id)
                rejected.append(
                    RejectedMemory(
                        memory_id=candidate_id,
                        reason="below_rank",
                        content=memory.content,
                    )
                )
        return SearchResult(
            memories=hits,
            rejected=rejected,
            candidates_considered=len(trace.candidate_ids),
        )

    def golden_run(self, name: str) -> tuple[Any, bool]:
        """A recorded run plus whether it still describes this process.

        Returns `(run, is_current)`. Staleness is not a warning to bury: a recorded
        answer from a different store or prompt shown as the current behaviour is the
        same failure as an unversioned result table, and this project has already
        been bitten by prompt drift making numbers incomparable.
        """
        from .golden import fingerprint, load_runs, store_state

        run = load_runs().get(name)
        if run is None:
            raise MemoryNotFound(name)
        current = fingerprint(
            run.namespace,
            run.query,
            {
                "answer_prompt": ANSWER_PROMPT_VERSION,
                "judge_prompt": JUDGE_PROMPT_VERSION,
                "extractor": self.store.get_meta("extractor_version"),
            },
            store_state(self.store, run.namespace),
        )
        return run, current == run.fingerprint

    def answer(self, user_id: str, query: str, limit: int | None = None) -> dict[str, Any]:
        """Run the full answer path live. Requires an answerer, so requires a key.

        Never called on page load — the inspector shows a recorded run by default and
        only reaches this when a visitor asks for it. A demo that spends quota on
        every refresh is a demo nobody can leave running.
        """
        self._require_namespace(user_id)
        if self.answerer is None:
            raise RuntimeError(
                "no answerer configured; set GEMINI_API_KEY to enable live answering"
            )
        from llm_long_term_memory.evaluation.datasets.longmemeval import Instance

        started = perf_counter()
        instance = Instance(
            question_id=user_id,
            question_type="live",
            question=query,
            answer="",
            question_date=datetime.now().strftime("%Y/%m/%d"),
            sessions=[],
            answer_session_ids=[],
        )
        # A caller-supplied limit must actually apply. Accepting one, ignoring it, and
        # then reporting a count computed under a different k is how the inspector
        # ended up explaining a memory set the answerer never saw.
        runner = self.answerer
        previous_k = runner.top_k
        if limit and limit != previous_k:
            runner.top_k = limit
        try:
            result = runner.answer(instance)
        finally:
            runner.top_k = previous_k

        notes = result.notes
        # The memories the answerer actually received, returned so the inspector can
        # render *this* run rather than deriving its own retrieval alongside it.
        selected = [
            {
                "id": entry.get("memory_id"),
                "content": entry.get("content", ""),
                "score": entry.get("score"),
                "scope": entry.get("scope"),
                "source_role": entry.get("source_role"),
            }
            for entry in (notes.get("retrieval") or [])
        ]
        return {
            "answer": result.text,
            "top_k": limit or previous_k,
            "selected": selected,
            "answer_status": notes.get("answer_status"),
            "fallback_level": notes.get("fallback_level", "none"),
            "fallback_reason": notes.get("fallback_reason"),
            "fallback_turns": notes.get("fallback_turns", []),
            "memories_selected": len(selected),
            "candidates_considered": notes.get("candidates_considered", 0),
            "context_tokens": result.context_tokens,
            "latency_ms": round((perf_counter() - started) * 1000, 1),
            "answerer_calls": 2 if notes.get("fallback_level", "none") != "none" else 1,
        }

    def raw_search(self, user_id: str, query: str, limit: int = 3) -> list[Any]:
        """BM25 over the raw conversation archive for one namespace.

        Exposed on its own so the inspector can show what the fallback layer *would*
        recover, without spending an answerer call. In the answer path this is
        reached only when structured memory reports itself insufficient.
        """
        self._require_namespace(user_id)
        return self.store.search_turns(user_id, query, limit=limit)

    def get(self, user_id: str, memory_id: str) -> Memory:
        self._require_namespace(user_id)
        memory = self.store.get(memory_id)
        # Checked rather than trusted: returning another namespace's memory because
        # the caller guessed an id is the one bug this layer must not have.
        if memory is None or memory.user_id != user_id:
            raise MemoryNotFound(memory_id)
        return memory

    def list_memories(
        self,
        user_id: str,
        status: str | None = None,
        memory_type: str | None = None,
        scope: str | None = None,
        source_role: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Memory], int]:
        self._require_namespace(user_id)
        memories = [
            m
            for m in self.store.iter_all(user_id)
            if (status is None or m.status == status)
            and (memory_type is None or m.type == memory_type)
            and (scope is None or m.scope == scope)
            and (source_role is None or m.source_role == source_role)
        ]
        memories.sort(key=lambda m: (m.valid_from or m.ingested_at or datetime.min, m.id))
        return memories[offset : offset + limit], len(memories)

    def timeline(self, user_id: str, subject: str, predicate: str) -> list[Memory]:
        """The supersession chain for one key, oldest first.

        This is what makes a changed fact legible: Canberra → Sydney with the date
        the move was stated, rather than one row silently overwriting another.
        """
        self._require_namespace(user_id)
        chain = self.store.find_by_predicate(user_id, subject, predicate, include_superseded=True)
        return sorted(chain, key=lambda m: (m.valid_from or m.ingested_at or datetime.min, m.id))

    def evidence(self, user_id: str, memory_id: str) -> dict[str, Any] | None:
        """The source turn a memory was extracted from, with its span."""
        memory = self.get(user_id, memory_id)
        result = self.hydrator.hydrate([memory], max_tokens=self.config.hydration.max_tokens)
        if not result.evidence:
            return None
        item = result.evidence[0]
        return {
            "session_id": external_session_id(item.session_id),
            "turn_index": item.turn_index,
            "role": item.role,
            "text": item.text,
            "char_start": memory.source_char_start,
            "char_end": memory.source_char_end,
        }

    # ----------------------------------------------------------------- writes

    def add_message(
        self, user_id: str, role: str, content: str, session_id: str | None = None
    ) -> dict[str, Any]:
        """Persist one turn and extract memories from it.

        Requires an extractor, which requires an API key. Read-only deployments can
        run the whole service without one.
        """
        self._require_namespace(user_id)
        if self.extractor is None:
            raise RuntimeError(
                "no extractor configured; set GEMINI_API_KEY to enable the write path"
            )

        now = datetime.now()
        external_id = session_id or f"s_{uuid.uuid4().hex[:12]}"
        session_id = scoped_session_id(user_id, external_id)
        existing = self.store.get_session(session_id)
        turn_index = len(existing.turns) if existing else 0

        self.store.add_session(
            Session(
                id=session_id,
                user_id=user_id,
                started_at=existing.started_at if existing else now,
                source="api",
                turns=[
                    *(existing.turns if existing else []),
                    Turn(
                        id=f"{session_id}:{turn_index}",
                        session_id=session_id,
                        turn_index=turn_index,
                        role=role,
                        content=content,
                        ts=now,
                    ),
                ],
            )
        )

        outcome = self.extractor.extract_turn(
            user_id=user_id, session_id=session_id, role=role, content=content, now=now
        )
        if outcome.memories:
            self.store.add_memories(outcome.memories)
            self.index.add(
                [m.id for m in outcome.memories],
                self.encoder.encode([m.content for m in outcome.memories]),
            )
            self.index.save()
        return {
            "session_id": external_id,
            "turn_index": turn_index,
            "memories": outcome.memories,
            "usage": outcome.usage,
        }

    def forget(self, user_id: str, memory_id: str) -> None:
        """Mark a memory evicted. Never a hard delete: provenance is the product,
        and a removed row cannot explain why it is gone."""
        memory = self.get(user_id, memory_id)
        self.store.mark_evicted([memory.id])

    # ---------------------------------------------------------------- support

    @staticmethod
    def _require_namespace(user_id: str) -> None:
        if not user_id or not user_id.strip():
            raise NamespaceRequired("user_id is required on every stateful call")

    def manifest(self) -> dict[str, Any]:
        """What this process is actually running. Read-only, and safe to expose:
        model ids and weights, never credentials."""
        return {
            "store": self.store_name,
            "memories": self.store.count(),
            "config": {
                "name": self.config.name,
                "answerer": self.config.models.answerer,
                "extractor": self.config.models.extractor,
                "embedder": self.config.models.embedder,
                "temporal_resolution": self.config.temporal_resolution,
                "retrieval": {
                    "weights": self.config.retrieval.weights.model_dump(),
                    "top_k": getattr(self, "top_k", self.config.service.top_k),
                    "benchmark_top_k": self.config.retrieval.top_k,
                    "candidate_limit": self.config.retrieval.candidate_limit,
                    "rerank": self.config.retrieval.rerank.model_dump(),
                },
            },
        }

    def fingerprint(self) -> str:
        """A short identity for the running configuration, stamped on every log
        line. Enough to tell two deployments apart when their logs are pooled."""
        r = self.config.retrieval
        # Some read-only inspector tests intentionally construct the smallest
        # possible service without running the expensive composition root.
        live_top_k = getattr(self, "top_k", self.config.service.top_k)
        return (
            f"{self.config.name}/{self.store_name}"
            f"/top_k={live_top_k}/rerank={'on' if r.rerank.enabled else 'off'}"
            f"/temporal={'on' if self.config.temporal_resolution else 'off'}"
        )

    def close(self) -> None:
        self.store.close()
