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

import hashlib
import json
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime, timedelta
from functools import wraps
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any

from llm_long_term_memory.answering import ANSWER_PROMPT_VERSION
from llm_long_term_memory.api.budget import AccountBudgets, BudgetExceeded
from llm_long_term_memory.config import ExperimentConfig, Settings
from llm_long_term_memory.retrieve import EvidenceHydrator, HybridRetriever
from llm_long_term_memory.store import (
    ErasureJournal,
    Memory,
    NumpyFlatIndex,
    Session,
    SQLiteMemoryStore,
    Turn,
    external_session_id,
    scoped_session_id,
)


class WriteInProgress(RuntimeError):
    """Another call already claimed this idempotency key and has not finished."""


class IdempotencyConflict(RuntimeError):
    """This idempotency key was claimed for a different request body."""


# How long a claimed key may stay pending before a retry may take it over. A crash cannot
# run the failure path, so some window is required or a key held by a dead process is
# unusable forever. Set well above any write this service performs: taking over a claim
# that is still running would write twice, and a caller waiting is the cheaper mistake.
PENDING_WRITE_TAKEOVER = timedelta(minutes=15)


def _request_fingerprint(role: str, content: str, session_id: str | None) -> str:
    """What the key was claimed for.

    Everything that changes what gets written, and nothing that does not: the same message
    retried into the same session is the same request whatever its transport details. A
    hash rather than the body itself, because this is stored and the body is user content.
    """
    payload = json.dumps([role, content, session_id], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable(value):
    """Memories are dataclasses; the stored replay has to round-trip through JSON.

    `Memory` is a slots dataclass, so it has no `__dict__`. Reading one through `vars()`
    fell through to `str()`, and the replay of a successful write came back as a list of
    reprs — which the response model then tried to read attributes off, answering the
    retry with a 500. The retry is the one request idempotency exists to make safe, so
    the field order here matters: dataclasses first, `__dict__` only after.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


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


@dataclass(slots=True)
class TurnExtractionOutcome:
    memories: list[Memory]
    usage: dict[str, Any]


class LiveTurnExtractor:
    """Adapt the frozen batch extractor to one live conversation turn.

    The REST and MCP surfaces accept one turn, while the research extractor accepts
    a list of sessions. Keeping this adapter explicit prevents tests from inventing
    an ``extract_turn`` method that no production class implements.

    **Preceding turns are shown, and this is not optional.** A turn arrives on its own
    and refers to what came before it:

        assistant: I'd recommend the Sony WH-1000XM6.
        user:      I bought the Sony one you recommended yesterday.

    Extracting the second turn alone can only produce "the user bought the Sony one",
    which names nothing and answers nothing later. The batch ingestion path never had
    this problem — it sees a whole conversation at once — so the loss is specific to
    the live surface and invisible in every benchmark number this project reports.

    The preceding turns are passed as part of the same session, which is also how the
    extractor already reads a conversation, so no prompt changes and no extractor
    version moves. The cost is that facts from those turns are extracted again; they
    are already in the store from when those turns were written, and the deduplicator
    this class already runs is what drops them. `usage["duplicates_dropped"]` therefore
    rises with the window, which is the honest place to read the price.
    """

    CONTEXT_TURNS = 4
    """How many preceding turns to show. Four covers the ordinary case — a
    recommendation, an acknowledgement, a follow-up, a reply — without turning every
    live write into a re-extraction of the whole conversation. It is a class attribute
    so a deployment can lower it to 0 and get exactly the previous behaviour.
    """

    def __init__(self, batch_extractor, deduplicator, encoder, usage) -> None:
        self.batch_extractor = batch_extractor
        self.deduplicator = deduplicator
        self.encoder = encoder
        self.usage = usage

    def extract_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        now: datetime,
        context_turns: Sequence[Any] = (),
    ) -> TurnExtractionOutcome:
        from llm_long_term_memory.conversation import ConversationSession, ConversationTurn
        from llm_long_term_memory.llm import UsageTracker

        started = len(self.usage.records)
        self.batch_extractor.user_id = user_id
        window = list(context_turns)[-self.CONTEXT_TURNS :] if self.CONTEXT_TURNS else []
        extracted = self.batch_extractor.extract(
            [
                ConversationSession(
                    session_id=session_id,
                    date=now.strftime("%Y-%m-%d %H:%M"),
                    turns=[
                        *(ConversationTurn(role=t.role, content=t.content) for t in window),
                        ConversationTurn(role=role, content=content),
                    ],
                )
            ]
        )
        vectors = self.encoder.encode([memory.content for memory in extracted.memories])
        deduped = self.deduplicator.process(extracted.memories, vectors)
        for current, _previous in deduped.updates:
            current.replaces_previous = True
            current.update_op = "replaces"

        per_turn = UsageTracker(records=list(self.usage.records[started:]))
        summary = per_turn.summary()
        summary.update(
            {
                "duplicates_dropped": deduped.duplicates,
                "updates_detected": len(deduped.updates),
                "bad_session_index": extracted.dropped_bad_index,
            }
        )
        return TurnExtractionOutcome(deduped.kept, summary)


def _synchronised(method):
    """Serialize access to the process-owned SQLite connection and numpy index."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        lock = getattr(self, "_lock", None)
        if lock is None:  # legacy/minimal injected fixtures built with __new__
            return method(self, *args, **kwargs)
        with lock:
            return method(self, *args, **kwargs)

    return wrapped


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
        self._lock = RLock()
        self.settings = settings or Settings()
        self.config = ExperimentConfig.from_yaml(config_path)
        self.store_name = store_name
        # Both the inspector and the answerer read this. Two independent defaults is
        # how the panel came to explain a different memory set from the one the
        # answerer saw.
        self.top_k = self.config.service.top_k

        self.store = SQLiteMemoryStore(self.settings.store_dir / f"{store_name}.db")
        self.store.initialize()
        self._budgets: AccountBudgets | None = None
        self._erasures: ErasureJournal | None = None

        # Injected in tests. Left lazy in production so that importing the app does
        # not pull torch.
        self._encoder = encoder
        self._encoder_error: str | None = None
        # An injected encoder dictates the index width, because the two must agree
        # and the injected one is the concrete fact. Reading `.dim` off the lazy
        # production encoder would defeat the laziness by loading the model here.
        dim = getattr(encoder, "dim", None) if encoder is not None else None
        self.index = NumpyFlatIndex(
            self.settings.store_dir / f"{store_name}-index",
            dim=dim or self.config.models.embedding_dim,
        )
        # SQLite and the flat index are separate resources. Refuse to serve a
        # partially committed generation instead of silently losing memories or
        # returning orphan vectors. Empty/empty is a valid fresh service.
        self.index.validate_ids(self.store.memory_ids())
        # Only the write path needs this, and it needs an API key. Read-only
        # deployments and the whole test suite work without one.
        self.extractor = extractor
        self._live_client_instance = None
        self._live_usage = None
        self._live_quota = None
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
        if self.config.temporal_resolution:
            from llm_long_term_memory.temporal import TemporalResolver

            self.resolver = TemporalResolver(self.store)
        else:
            self.resolver = None

    @property
    def encoder(self):
        encoder_error = getattr(self, "_encoder_error", None)
        if encoder_error is not None:
            raise EncoderUnavailable(encoder_error)
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
        if getattr(self, "_encoder_error", None) is not None:
            return False
        if self._encoder is not None:
            return True
        import importlib.util

        return importlib.util.find_spec("sentence_transformers") is not None

    def mark_encoder_unavailable(self, detail: str) -> None:
        self._encoder_error = f"semantic search unavailable: {detail}"

    @property
    def encoder_error(self) -> str | None:
        return self._encoder_error

    @property
    def write_available(self) -> bool:
        settings = getattr(self, "settings", None)
        return self.extractor is not None or bool(settings and settings.has_api_key)

    @property
    def live_answer_available(self) -> bool:
        settings = getattr(self, "settings", None)
        return self._answerer is not None or bool(settings and settings.has_api_key)

    def _get_live_client(self):
        if self._live_client_instance is None:
            from llm_long_term_memory.llm import Limits, QuotaManager, UsageTracker
            from llm_long_term_memory.llm.client import GeminiClient

            self._live_quota = QuotaManager(
                state_dir=self.settings.store_dir / "quota",
                default=Limits(
                    rpm=self.config.quota.rpm,
                    tpm=self.config.quota.tpm,
                    rpd=self.config.quota.rpd,
                ),
            )
            self._live_quota.load_learned()
            self._live_usage = UsageTracker()
            self._live_client_instance = GeminiClient(
                self.settings.require_api_key(), quota=self._live_quota, usage=self._live_usage
            )
        return self._live_client_instance

    def _build_live_extractor(self) -> LiveTurnExtractor:
        try:
            from llm_long_term_memory.ingest import Deduplicator, Extractor, TwoStageExtractor
        except ImportError as exc:
            raise RuntimeError(
                "the write path needs the `llm` extra: uv sync --extra llm --extra embed"
            ) from exc

        client = self._get_live_client()
        batch_extractor = (
            TwoStageExtractor(client, self.config.models.extractor)
            if self.config.ingest.two_stage
            else Extractor(client, self.config.models.extractor)
        )
        deduplicator = Deduplicator(
            client,
            self.config.models.extractor,
            self.encoder,
            store=self.store,
            index=self.index,
            threshold=self.config.ingest.dedupe_similarity_threshold,
        )
        return LiveTurnExtractor(batch_extractor, deduplicator, self.encoder, self._live_usage)

    @property
    def answerer(self):
        """The live answer path, or None when no credential is configured.

        Uses the stable product `AnswerEngine`. Evaluation runners may wrap the same
        retrieval and answering primitives with gold-only diagnostics, but the live
        API does not depend on benchmark instances or judge prompts.
        """
        if self._answerer is None and self.settings.has_api_key:
            from llm_long_term_memory.runtime import AnswerEngine

            client = self._get_live_client()
            self._answerer = AnswerEngine(
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

    @_synchronised
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
        self._require_limit(limit, maximum=100)
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

    @_synchronised
    def golden_run(self, name: str) -> tuple[Any, bool]:
        """A recorded run plus whether it still describes this process.

        Returns `(run, is_current)`. Staleness is not a warning to bury: a recorded
        answer from a different store or prompt shown as the current behaviour is the
        same failure as an unversioned result table, and this project has already
        been bitten by prompt drift making numbers incomparable.
        """
        from llm_long_term_memory.evaluation.judge import JUDGE_PROMPT_VERSION

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

    @_synchronised
    def answer(self, user_id: str, query: str, limit: int | None = None) -> dict[str, Any]:
        """Run the full answer path live. Requires an answerer, so requires a key.

        Never called on page load — the inspector shows a recorded run by default and
        only reaches this when a visitor asks for it. A demo that spends quota on
        every refresh is a demo nobody can leave running.
        """
        self._require_namespace(user_id)
        self._require_limit(limit, maximum=100)
        if self.answerer is None:
            raise RuntimeError(
                "no answerer configured; set GEMINI_API_KEY to enable live answering"
            )
        from llm_long_term_memory.conversation import AnswerRequest

        started = perf_counter()
        # This used to build a benchmark `Instance` with `answer=""` and
        # `question_type="live"`, which is a gold-answer record with the gold left
        # blank: a live question is not a benchmark row, and the two fields it had to
        # invent were the giveaway. The request carries what answering needs and
        # nothing about what the answer should be.
        request = AnswerRequest(
            question=query,
            asked_on=datetime.now().strftime("%Y/%m/%d"),
            user_id=user_id,
        )
        # A caller-supplied limit must actually apply. Accepting one, ignoring it, and
        # then reporting a count computed under a different k is how the inspector
        # ended up explaining a memory set the answerer never saw.
        runner = self.answerer
        model = getattr(runner, "model", self.config.models.answerer)
        self.budgets.check(user_id, model, "answer")
        records = getattr(getattr(getattr(runner, "client", None), "usage", None), "records", None)
        records = records if isinstance(records, list) else None
        attempt_start = len(records) if records is not None else None
        try:
            result = runner.answer_request(request, limit=limit)
        except Exception:
            self._charge_answer(user_id, model, records, attempt_start, None)
            raise
        self._charge_answer(user_id, model, records, attempt_start, result)

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
            "top_k": limit or runner.top_k,
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

    def _charge_answer(self, user_id, model, records, started, result) -> None:
        """Charge all answer attempts, including retries and raw-source recovery.

        The live runner shares its client's UsageTracker. An empty slice means the
        request stopped before contacting the provider, so it costs nothing. Injected
        runners without a tracker use the answer's reported tokens as a fallback,
        matching the conservative failed-call accounting on the write path.
        """
        if records is not None:
            for attempt in records[started:]:
                self.budgets.record(
                    user_id,
                    attempt.model or model,
                    attempt.role or "answer",
                    input_tokens=attempt.input_tokens,
                    output_tokens=attempt.output_tokens,
                    ok=attempt.ok,
                )
            return
        self.budgets.record(
            user_id,
            model,
            "answer",
            input_tokens=getattr(result, "prompt_tokens", 0),
            output_tokens=getattr(result, "output_tokens", 0),
            calls=2
            if result is not None and result.notes.get("fallback_level", "none") != "none"
            else 1,
            ok=result is not None,
        )

    @_synchronised
    def raw_search(self, user_id: str, query: str, limit: int = 3) -> list[Any]:
        """BM25 over the raw conversation archive for one namespace.

        Exposed on its own so the inspector can show what the fallback layer *would*
        recover, without spending an answerer call. In the answer path this is
        reached only when structured memory reports itself insufficient.
        """
        self._require_namespace(user_id)
        self._require_limit(limit, maximum=100)
        return self.store.search_turns(user_id, query, limit=limit)

    @_synchronised
    def get(self, user_id: str, memory_id: str) -> Memory:
        self._require_namespace(user_id)
        memory = self.store.get(memory_id)
        # Checked rather than trusted: returning another namespace's memory because
        # the caller guessed an id is the one bug this layer must not have.
        if memory is None or memory.user_id != user_id:
            raise MemoryNotFound(memory_id)
        return memory

    @_synchronised
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

    @_synchronised
    def timeline(self, user_id: str, subject: str, predicate: str) -> list[Memory]:
        """The supersession chain for one key, oldest first.

        This is what makes a changed fact legible: Canberra → Sydney with the date
        the move was stated, rather than one row silently overwriting another.
        """
        self._require_namespace(user_id)
        chain = self.store.find_by_predicate(user_id, subject, predicate, include_superseded=True)
        return sorted(chain, key=lambda m: (m.valid_from or m.ingested_at or datetime.min, m.id))

    @_synchronised
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

    @property
    def erasures(self) -> ErasureJournal:
        """The erasure journal for this service's store, beside it rather than in it.

        Bound to the current store for the same reason `budgets` is: a service whose store
        is swapped would otherwise keep writing tombstones next to the old one, where no
        restore of the new one will ever look for them.
        """
        journal = ErasureJournal.beside(self.store.path)
        if self._erasures is None or self._erasures.path != journal.path:
            self._erasures = journal
        return self._erasures

    @property
    def budgets(self) -> AccountBudgets:
        """Per-account spend, bound to whichever store this service currently holds.

        Distinct from the process-wide provider limiter in `llm/rate_limiter.py`: that one
        stops the deployment exhausting the provider, this one stops one namespace
        exhausting the deployment.

        Rebuilt when the store is swapped rather than captured in `__init__`. A service
        whose store is replaced after construction — which the harness does, and which a
        restore would do — otherwise keeps a ledger pointed at the old connection, and the
        first budget check after the swap reads a closed database.
        """
        if self._budgets is None or self._budgets.store is not self.store:
            self._budgets = AccountBudgets(self.store)
        return self._budgets

    @_synchronised
    def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
        session_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Persist one turn and extract memories from it.

        Requires an extractor, which requires an API key. Read-only deployments can
        run the whole service without one.

        **Why a key matters here specifically.** The turn is persisted *before* extraction
        runs, and `turn_index` is `len(existing.turns)`. A provider failure followed by an
        ordinary client retry therefore appended the same turn again at the next index,
        and extraction ran over the duplicate. With a key the retry replays the first
        reply instead of writing a second turn.

        The key is claimed *before* the write, not after. Claiming it afterwards would
        leave the window it exists to close: two concurrent retries would both find no
        key, both write, and only then race to record one.
        """
        self._require_namespace(user_id)
        if idempotency_key is not None:
            idempotency_key = idempotency_key.strip()
            if not idempotency_key or len(idempotency_key) > 200:
                raise ValueError("idempotency_key must be 1-200 characters")
        # Everything that can refuse the request without calling anyone is settled before a
        # key is claimed or anything is charged. A session id scoped to another user (the
        # caller's 422) and a deployment with no extractor (a 503) used to be charged to the
        # account as a failed provider call and to leave the key 'failed', as though an
        # attempt had run; a client retrying its own 422 spent its daily budget on requests
        # that never left the process.
        if session_id is not None:
            scoped_session_id(user_id, session_id)
        self._require_extractor()
        if idempotency_key is not None:
            replay = self._claim_write_key(
                user_id, idempotency_key, _request_fingerprint(role, content, session_id)
            )
            if replay is not None:
                return replay
        # After the claim, so a replay of a finished write is served free even to an account
        # that is now over budget; before any provider call, so an account over budget is
        # stopped here rather than by a provider 429 that every other tenant would also see.
        try:
            self.budgets.check(user_id, self.config.models.extractor, "extract")
        except BudgetExceeded as refusal:
            # The claim is released, not left pending. Nothing ran, so nothing is charged —
            # but a claim held past its 429 answered the caller's retry, once the cap was
            # lifted or the day rolled over, with "still in progress" for the whole
            # takeover window.
            if idempotency_key is not None:
                self.store.record_write_failure(
                    user_id, idempotency_key, f"{type(refusal).__name__}: {refusal}"
                )
            raise
        started = self._provider_attempts()
        try:
            reply = self._add_message(user_id, role, content, session_id)
        except Exception as failure:
            # Charged for what reached the provider; see `_charge_write`. A budget that
            # excused failures is one a retry loop walks straight through, and a retry loop
            # is what runs when things are already going wrong.
            self._charge_write(user_id, started, None)
            # The claim must not outlive the attempt. Before this, a provider error left
            # the key 'pending' for good: every later retry read the placeholder and was
            # told a write was still in progress, so the one thing the key was supposed to
            # make safe — retrying a failed write — was the thing it made impossible.
            if idempotency_key is not None:
                self.store.record_write_failure(
                    user_id, idempotency_key, f"{type(failure).__name__}: {failure}"
                )
            raise
        self._charge_write(user_id, started, reply)
        if idempotency_key is not None:
            self.store.record_write_reply(
                user_id, idempotency_key, json.dumps(reply, default=_jsonable)
            )
        return reply

    def _claim_write_key(self, user_id: str, key: str, fingerprint: str) -> dict[str, Any] | None:
        """Claim the key, or return the reply a previous identical write produced.

        Returns None when the caller should go ahead and write. Raises when the key
        cannot be claimed: either another attempt holds it, or it was claimed for a
        different request.
        """
        existing = self.store.write_key(user_id, key)
        if existing is None:
            if not self.store.remember_write(
                user_id, key, json.dumps({"pending": True}), fingerprint
            ):
                # Another call claimed it between the read and the write. Its reply may
                # still be in flight, so the caller retries rather than being handed a
                # half-written answer.
                raise WriteInProgress("a write with this idempotency_key is already in progress")
            return None

        # A key claimed for one request may not answer a different one. Replaying the
        # first reply would tell the caller their second message was stored when it never
        # was, which is a silent data loss the caller cannot detect. NULL fingerprints
        # come from rows written before the column existed and are not second-guessed.
        recorded = existing.get("fingerprint")
        if recorded is not None and recorded != fingerprint:
            raise IdempotencyConflict(
                "this idempotency_key was used for a different request; "
                "use a new key for a new message"
            )

        state = existing.get("state") or "done"
        if state == "done":
            replay = json.loads(existing["response"])
            replay["idempotent_replay"] = True
            return replay
        if state == "pending" and not self._claim_is_stale(existing):
            raise WriteInProgress("a write with this idempotency_key is still in progress")
        # 'failed', or a 'pending' claim old enough that the process holding it is gone.
        # A compare-and-swap on the row as it was read, so two retries racing on one orphan
        # cannot both take it over.
        if not self.store.take_over_write(
            user_id,
            key,
            fingerprint,
            expected_state=state,
            expected_claimed_at=existing.get("claimed_at"),
        ):
            raise WriteInProgress("another call took over this idempotency_key first")
        return None

    def _claim_is_stale(self, row: dict[str, Any]) -> bool:
        """Whether a pending claim is an orphan rather than a live request.

        A crash cannot run the failure path, so without a takeover window a key held by a
        dead process stays unusable forever. The window is deliberately longer than any
        write should take: taking over a claim that is actually still running would write
        twice, which is worse than making the caller wait.
        """
        claimed_at = row.get("claimed_at")
        if not claimed_at:
            # Claimed before this column existed. Treat as stale: such a row can only have
            # been left behind by the version that had no failure path at all.
            return True
        try:
            claimed = datetime.fromisoformat(claimed_at)
        except ValueError:
            return True
        if claimed.tzinfo is None:
            # Claims are stamped in UTC. A naive stamp predates that, and was local time.
            claimed = claimed.astimezone()
        return datetime.now(UTC) - claimed > PENDING_WRITE_TAKEOVER

    def _require_extractor(self) -> None:
        """Build the live extractor on first use, or say why the write path is unavailable."""
        if self.extractor is None:
            if not self.settings.has_api_key:
                raise RuntimeError(
                    "no extractor configured; set GEMINI_API_KEY to enable the write path"
                )
            self.extractor = self._build_live_extractor()

    def _provider_attempts(self) -> int | None:
        """How many provider attempts the extractor has on record, or None if it keeps none.

        The live extractor shares the provider client's `UsageTracker`, which records every
        attempt — retries and failures included — as it happens. Writes are serialised per
        process, so the records after this mark belong to the write about to run.
        """
        records = getattr(getattr(self.extractor, "usage", None), "records", None)
        return len(records) if isinstance(records, list) else None

    def _charge_write(
        self, user_id: str, started: int | None, reply: dict[str, Any] | None
    ) -> None:
        """Put one write's provider spend on the account's ledger; `reply` is None on failure.

        Where the extractor keeps per-attempt records, the charge is exactly the attempts
        made during this write, grouped by model and role, with their tokens and failures.
        A failed write used to be charged one call and no tokens whatever it had sent, while
        the provider client retries rate limits and dropped connections inside a single
        write: a retry storm, the thing a budget exists to stop, went through at a fraction
        of its cost. Records with nothing after the mark mean nothing reached the provider,
        and nothing is charged.
        """
        model = self.config.models.extractor
        if started is not None:
            groups: dict[tuple[str, str], list[Any]] = {}
            for attempt in self.extractor.usage.records[started:]:
                key = (attempt.model or model, attempt.role or "extract")
                groups.setdefault(key, []).append(attempt)
            for (attempt_model, role), attempts in groups.items():
                self.budgets.record(
                    user_id,
                    attempt_model,
                    role,
                    calls=len(attempts),
                    failed_calls=sum(1 for attempt in attempts if not attempt.ok),
                    input_tokens=sum(attempt.input_tokens for attempt in attempts),
                    output_tokens=sum(attempt.output_tokens for attempt in attempts),
                )
            if groups or reply is None:
                return
        elif reply is None:
            # An extractor that keeps no records cannot say what it sent. One failed call is
            # charged rather than none: under-charging is the direction a budget must not err.
            self.budgets.record(user_id, model, "extract", ok=False)
            return

        usage = reply.get("usage") or {}
        # From `by_role`, not from `total_tokens`: the ledger keeps input and output apart
        # because they are priced apart. `RoleStats` carries no model, so the configured one
        # stands in; it is the model this path calls.
        charged = 0
        for role_stats in (usage.get("by_role") or {}).values():
            calls = int(role_stats.get("calls") or 0)
            self.budgets.record(
                user_id,
                model,
                role_stats.get("role") or "extract",
                input_tokens=int(role_stats.get("input_tokens") or 0),
                output_tokens=int(role_stats.get("output_tokens") or 0),
                calls=calls,
                # The failures, not every call of a role that had one: `failed_calls` is how
                # an operator tells a retry storm from a single retry.
                failed_calls=int(role_stats.get("failures") or 0),
            )
            charged += calls
        if not charged:
            # A write that extracted nothing still went through the service. Recording a
            # zero-token call keeps "this namespace made a request" true in the ledger;
            # dropping it would let a caller that always extracts nothing run uncounted.
            self.budgets.record(user_id, model, "extract")

    def _resolution_snapshot(
        self, memories: list[Memory]
    ) -> list[tuple[str, str, str | None, datetime | None, datetime | None]]:
        """The rows resolution may edit, as they stand now.

        The resolver rebuilds whole `(user_id, subject, predicate)` keys, so the rows it
        can touch are exactly the ones already stored under the keys this batch carries.
        Reading them before the write is what makes a rollback possible at all.
        """
        keys = {
            (m.user_id, m.subject, m.predicate)
            for m in memories
            if m.user_id and m.subject and m.predicate
        }
        return [
            (row.id, row.status, row.superseded_by, row.valid_from, row.valid_to)
            for user_id, subject, predicate in sorted(keys)
            for row in self.store.find_by_predicate(
                user_id, subject, predicate, include_superseded=True
            )
        ]

    def _add_message(
        self, user_id: str, role: str, content: str, session_id: str | None
    ) -> dict[str, Any]:
        """The write itself, with no idempotency or budget bookkeeping.

        Expects `_require_extractor` to have run; `add_message` calls it before claiming.
        """
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

        try:
            outcome = self.extractor.extract_turn(
                user_id=user_id,
                session_id=session_id,
                role=role,
                content=content,
                now=now,
                # The turns already on this session, so a pronoun or "the Sony one"
                # can be resolved against what it refers to.
                context_turns=existing.turns if existing else (),
            )
        except Exception:
            # The turn is written before extraction so that a success cannot leave memories
            # without the text they came from. The cost is this rollback: a provider
            # failure would otherwise leave a turn nothing was extracted from, and the
            # retry would append it a second time at the next index. Undoing it is what
            # lets a failed write be retried at all rather than wedged forever.
            self.store.remove_turn(f"{session_id}:{turn_index}")
            raise
        if outcome.memories:
            for memory in outcome.memories:
                memory.user_id = user_id
                memory.source_session_id = session_id
                memory.source_turn_index = turn_index
            # Captured before anything is written, because resolution is not invertible:
            # it closes validity intervals and points existing rows at a successor, and
            # nothing in the result says what those rows held before.
            prior = self._resolution_snapshot(outcome.memories)
            self.store.add_memories(outcome.memories)
            try:
                self.index.add(
                    [m.id for m in outcome.memories],
                    self.encoder.encode([m.content for m in outcome.memories]),
                )
                self.index.save()
                if self.resolver is not None:
                    resolution = self.resolver.resolve_memories(outcome.memories)
                    outcome.usage["superseded"] = resolution.superseded
                    outcome.usage["ambiguous_replacements"] = resolution.skipped_ambiguous
            except Exception:
                # Extraction succeeding is not the write succeeding. Encoding, the index
                # save and the resolver all run after the memories are durable, and a
                # failure in any of them used to leave them there while the caller was
                # told the write failed — so the retry, which re-extracts from scratch,
                # wrote the same facts a second time under new ids.
                #
                # SQLite first, and unconditionally. Resolution is undone before the
                # memories are deleted because a pre-existing row may already point at
                # one of them, and `superseded_by` is a plain reference that a delete
                # would leave dangling.
                self.store.restore_memory_states(prior)
                self.store.remove_memories([m.id for m in outcome.memories])
                self.store.remove_turn(f"{session_id}:{turn_index}")
                try:
                    self.index.remove([m.id for m in outcome.memories])
                    self.index.save()
                except Exception:
                    # The index is frequently the thing that just failed — a full disk
                    # fails the rollback's save exactly as it failed the write's — so
                    # this cannot be allowed to abandon the undo or to replace the
                    # original error with a less informative one. It is safe to skip:
                    # `get_many` drops ids it cannot find, so a vector with no row
                    # behind it is filtered out at hydration and costs space, not
                    # correctness. An orphan memory would not be survivable; an orphan
                    # vector is.
                    pass
                raise
        return {
            "session_id": external_id,
            "turn_index": turn_index,
            "memories": outcome.memories,
            "usage": outcome.usage,
        }

    @_synchronised
    def export_user(self, user_id: str) -> dict[str, Any]:
        """Everything held for one namespace, in one document.

        A data-subject export has to be complete to mean anything, so this reads the
        three planes that actually hold user content — memories including evicted and
        superseded ones, the raw conversation turns memories were extracted from, and the
        evidence links between them — rather than the retrievable subset the product
        normally shows. A memory the caller can no longer retrieve is still their data.
        """
        self._require_namespace(user_id)
        memories = self.store.iter_all(user_id)
        sessions = []
        for session_id in sorted(self.store.session_ids_for_user(user_id)):
            turns = self.store.turns_for_session(session_id)
            sessions.append(
                {
                    "session_id": external_session_id(session_id),
                    "turns": [
                        {
                            "turn_index": turn.turn_index,
                            "role": turn.role,
                            "content": turn.content,
                            "ts": turn.ts.isoformat() if turn.ts else None,
                        }
                        for turn in turns
                    ],
                }
            )
        return {
            "user_id": user_id,
            "exported_at": datetime.now().isoformat(),
            "store_fingerprint": self.fingerprint(),
            "counts": {
                "memories": len(memories),
                "sessions": len(sessions),
                "turns": sum(len(s["turns"]) for s in sessions),
            },
            "memories": [
                {
                    "id": m.id,
                    "type": m.type,
                    "content": m.content,
                    "subject": m.subject,
                    "predicate": m.predicate,
                    "object": m.object,
                    "target_object": m.target_object,
                    "update_op": m.update_op,
                    "scope": m.scope,
                    "status": m.status,
                    "source_role": m.source_role,
                    # Both times, and the caller can tell them apart: `event_time` is
                    # null unless the fact stated one, `observed_at` is when it was
                    # said. An export that showed only the first would look like the
                    # timeline had gone blank.
                    "event_time": m.event_time.isoformat() if m.event_time else None,
                    "observed_at": m.observed_at.isoformat() if m.observed_at else None,
                    "valid_from": m.valid_from.isoformat() if m.valid_from else None,
                    "valid_to": m.valid_to.isoformat() if m.valid_to else None,
                    "superseded_by": m.superseded_by,
                    "ingested_at": m.ingested_at.isoformat() if m.ingested_at else None,
                    "source_session_id": (
                        external_session_id(m.source_session_id) if m.source_session_id else None
                    ),
                    "source_turn_index": m.source_turn_index,
                    "evidence_memory_ids": self.store.evidence_for(m.id),
                }
                for m in memories
            ],
            "sessions": sessions,
        }

    @_synchronised
    def erase_user(self, user_id: str) -> dict[str, Any]:
        """Physically remove a namespace from every online data plane.

        Distinct from `forget`, and the distinction is the product's: `forget` is a soft
        delete because provenance is the point and a removed row cannot explain why it is
        gone. That reasoning does not survive a data-subject erasure request, where the
        requirement is that the data stop existing. So this is a second operation rather
        than a flag, and the two are named for what they do.

        The vector index is a separate resource from SQLite and cannot join the same
        transaction. Rows go first: a vector with no memory is an orphan the loader
        already tolerates and the next save removes, while a memory whose vector is gone
        would still be listed and still be searchable by id. Failing in the safer
        direction is the most this can offer without a distributed transaction, and it is
        recorded here rather than implied.
        """
        self._require_namespace(user_id)
        removed = self.store.hard_delete_user(user_id)
        memory_ids = list(removed.pop("ids", []))
        vectors = self.index.remove(memory_ids)
        if vectors:
            self.index.save()
        # Recorded after the rows are gone, so the journal never claims an erasure that
        # did not happen. The backups taken before this moment still hold the data, and
        # this line is what a restore replays to stop it coming back; without it, erasure
        # is only true until the next time someone restores.
        self.erasures.record(
            user_id,
            memories=int(removed.get("memories", 0)),
            sessions=int(removed.get("sessions", 0)),
            turns=int(removed.get("turns", 0)),
        )
        return {"user_id": user_id, **removed, "vectors": vectors}

    @_synchronised
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

    @staticmethod
    def _require_limit(limit: int | None, *, maximum: int) -> None:
        if limit is not None and not 1 <= limit <= maximum:
            raise ValueError(f"limit must be between 1 and {maximum}")

    @_synchronised
    def manifest(self) -> dict[str, Any]:
        """What this process is actually running. Read-only, and safe to expose:
        model ids and weights, never credentials."""
        return {
            "store": self.store_name,
            "memories": self.store.count(),
            "capabilities": {
                "search": self.encoder_available,
                "write": self.write_available,
                "live_answer": self.live_answer_available,
            },
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

    @_synchronised
    def close(self) -> None:
        self.store.close()
