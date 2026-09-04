"""Answer from the extracted memory store rather than from raw sessions.

The first variant that is actually ChronoMem. Two knobs, so the ablation can
isolate what temporal resolution is worth:

    temporal=False   retrieve over every memory, superseded ones included. This is
                     the store's version of what naive RAG does — all three of
                     "uses TensorFlow", "learning PyTorch", "switched to PyTorch"
                     come back and the model picks.
    temporal=True    retrieve only what is currently in force, and tell the model
                     the validity window of what it is being given.

The retrieval itself is identical in both. Any difference in the results table is
attributable to the timeline, not to ranking.
"""

from __future__ import annotations

from datetime import datetime
from time import perf_counter

import numpy as np

from llm_long_term_memory.embed import Encoder
from llm_long_term_memory.evaluation.datasets.longmemeval import Instance
from llm_long_term_memory.llm.client import GeminiClient
from llm_long_term_memory.retrieve import (
    EvidenceHydrator,
    HybridRetriever,
    HydrationResult,
    RetrievedMemory,
    SessionBudget,
    build_coherent_context,
    render_evidence,
)
from llm_long_term_memory.store import (
    Memory,
    MemoryStore,
    NumpyFlatIndex,
    external_session_id,
)

from .base import ANSWER_PROMPT_VERSION, ANSWER_SYSTEM, Answer, AnswerVerdict
from .reasoning import (
    REASONED_ANSWER_PROMPT_VERSION,
    REASONED_ANSWER_SYSTEM,
    ReasonedAnswerVerdict,
    reasoning_kind,
    render_reasoned_prompt,
)

_TEMPLATE = """\
Here is what is known about the user, drawn from their chat history.

{context}

Today's date is {date}.

Question: {question}
"""

TEMPORAL_NOTE = (
    "Each fact is shown with the period it was true for. `(since <date>)` means it "
    "is still true now; `(<date> to <date>)` means it was replaced and is no longer "
    "current. Answer with what is true now unless the question asks about the past."
)


def _external_session_ids(values) -> set[str]:
    return {external_session_id(value) for value in values if value}


def _session_coverage(evidence: set[str], values) -> float | None:
    """Fraction of all gold sessions present, while keeping legacy any-hit recall."""
    if not evidence:
        return None
    present = evidence & _external_session_ids(values)
    return len(present) / len(evidence)


# Scope -> heading, in the order they are shown. Ordering is semantic rather than
# alphabetical: what the user wants and is planning comes before what happened and
# what was previously suggested.
#
# Grouping exists because an undifferentiated bullet list makes every memory look
# like a candidate answer. A model told "User preferences: dislikes raw fish" knows
# to constrain a restaurant recommendation; the same line in a flat list reads as a
# fact it should either quote or ignore.
_SCOPE_HEADINGS: tuple[tuple[str, str], ...] = (
    ("profile", "About the user"),
    ("preference", "User preferences"),
    ("plan", "Current plans"),
    ("event", "Past events"),
    ("recommendation", "Previously recommended by the assistant"),
    ("commitment", "The assistant agreed to"),
    ("shared_context", "Other people and things discussed"),
)
# Pre-P10 memories carry no scope. They are not dropped or guessed at — an
# unlabelled group is honest about what is known.
_UNSCOPED_HEADING = "Other things known about the user"


def render_grouped(memories: list[Memory], temporal: bool) -> str:
    """Group memories by scope under headings, preserving rank order within a group.

    Falls back to a plain list when nothing carries a scope, so a store built before
    P10 renders exactly as it did before.
    """
    if not any(memory.scope for memory in memories):
        return "\n".join(render_memory(memory, temporal) for memory in memories)

    by_scope: dict[str, list[Memory]] = {}
    for memory in memories:
        by_scope.setdefault(memory.scope or "", []).append(memory)

    blocks: list[str] = []
    for scope, heading in _SCOPE_HEADINGS:
        group = by_scope.pop(scope, None)
        if group:
            lines = "\n".join(render_memory(m, temporal) for m in group)
            blocks.append(f"{heading}:\n{lines}")
    # Anything left: unscoped memories, plus any scope the model invented that the
    # validator let through.
    leftovers = [m for group in by_scope.values() for m in group]
    if leftovers:
        lines = "\n".join(render_memory(m, temporal) for m in leftovers)
        blocks.append(f"{_UNSCOPED_HEADING}:\n{lines}")
    return "\n\n".join(blocks)


def render_memory(memory: Memory, temporal: bool) -> str:
    if not temporal:
        return f"- {memory.content}"

    start = memory.valid_from or memory.event_time
    if start and memory.valid_to:
        window = f"({start:%Y-%m-%d} to {memory.valid_to:%Y-%m-%d}, no longer current)"
    elif start:
        window = f"(since {start:%Y-%m-%d})"
    else:
        window = "(date unknown)"
    return f"- {memory.content} {window}"


class MemoryRunner:
    def __init__(
        self,
        client: GeminiClient,
        model: str,
        encoder: Encoder,
        store: MemoryStore,
        index: NumpyFlatIndex,
        temporal: bool = False,
        top_k: int = 20,
        retrieval_weights: dict[str, float] | None = None,
        candidate_limit: int = 50,
        recency_halflife_days: float = 30.0,
        decay_enabled: bool = False,
        decay_halflife_days: float = 60.0,
        reinforcement: float = 0.30,
        evidence_hydration: bool = False,
        adaptive_reasoning_hydration: bool = False,
        hydration_neighbouring_sentences: int = 1,
        hydration_max_tokens: int = 800,
        hydration_allocation: str = "ranked",
        token_budget: int = 0,
        utility_model=None,
        type_floors: dict[str, float] | None = None,
        max_output_tokens: int = 512,
        chars_per_token: float = 4.6,
        reranker=None,
        raw_fallback: bool = False,
        raw_fallback_max_turns: int = 3,
        raw_fallback_max_chars: int = 2400,
        session_budget: SessionBudget | None = None,
        oracle_session_context: bool = False,
        answer_policy: str = "v2",
    ) -> None:
        self.client = client
        self.model = model
        self.encoder = encoder
        self.store = store
        self.index = index
        self.temporal = temporal
        self.top_k = top_k
        # Budget packing reorders and drops memories by predicted utility, which is
        # exactly what a coherent context is built to prevent. Combining them yields
        # a context that is neither shape while reporting as both, so it is refused
        # rather than silently resolved in favour of whichever runs last.
        if session_budget is not None and token_budget:
            raise ValueError(
                "session_budget and token_budget both set: packing reorders the "
                "memories a coherent context orders deliberately. Choose one."
            )
        self.session_budget = session_budget
        if oracle_session_context and session_budget is None:
            raise ValueError("oracle_session_context requires a session_budget")
        self.oracle_session_context = oracle_session_context
        if answer_policy not in {"v2", "reasoned_v3"}:
            raise ValueError(f"unknown answer policy {answer_policy!r}")
        self.answer_policy = answer_policy
        self.answer_system = (
            REASONED_ANSWER_SYSTEM if answer_policy == "reasoned_v3" else ANSWER_SYSTEM
        )
        self.answer_prompt_version = (
            REASONED_ANSWER_PROMPT_VERSION
            if answer_policy == "reasoned_v3"
            else ANSWER_PROMPT_VERSION
        )
        self.verdict_schema = (
            ReasonedAnswerVerdict if answer_policy == "reasoned_v3" else AnswerVerdict
        )
        self.retriever = HybridRetriever(
            store,
            index,
            weights=retrieval_weights or {"semantic": 1.0},
            candidate_limit=candidate_limit,
            recency_halflife_days=recency_halflife_days,
            use_strength=decay_enabled,
            reranker=reranker,
        )
        self.decay_enabled = decay_enabled
        self.decay_halflife_days = decay_halflife_days
        self.reinforcement = reinforcement
        self.evidence_hydration = evidence_hydration
        self.adaptive_reasoning_hydration = adaptive_reasoning_hydration
        self.hydration_max_tokens = hydration_max_tokens
        self.hydrator = EvidenceHydrator(
            store,
            neighbouring_sentences=hydration_neighbouring_sentences,
            chars_per_token=chars_per_token,
            allocation=hydration_allocation,
        )
        # 0 disables packing and keeps the plain top-k truncation, so the P6 rows
        # are a change of selection policy against an otherwise identical pipeline.
        self.token_budget = token_budget
        self.utility_model = utility_model
        self.type_floors = type_floors
        self.max_output_tokens = max_output_tokens
        self.chars_per_token = chars_per_token
        # A plain attribute, not a property: the CLI overrides it so that the
        # results file is named after the config variant rather than after the flag.
        self.name = "chronomem" if temporal else "chronomem_no_temporal"
        # Read from the store, so a result row records the extractor that actually
        # wrote the data rather than whatever version is checked out today. `None`
        # for stores built before version stamping.
        # Off by default: it changes what a variant *is*, so it must be an ablation
        # row rather than a silent upgrade to every existing result.
        self.fallback = None
        self.raw_fallback_max_chars = max(1, raw_fallback_max_chars)
        if raw_fallback:
            from llm_long_term_memory.retrieve.fallback import RawFallback

            self.fallback = RawFallback(store, max_turns=raw_fallback_max_turns)
        self.extractor_version = (
            store.get_meta("extractor_version") if hasattr(store, "get_meta") else None
        )
        # What the answers were produced *from*. The extractor version alone does not
        # separate a pilot run against a partly-ingested store from a formal run
        # against the finished one — same code, different evidence — so the memory
        # count is part of the identity. `run_eval` refuses to resume across a change
        # in this string.
        self.store_fingerprint = f"{self.extractor_version or 'unversioned'}@{store.count()}"

    def prepare(self, instance: Instance) -> None:
        """Nothing per-instance: the store is built once by `lltm ingest`.

        This is the structural difference from the baselines, which rebuild an index
        for every question. It is also why this variant's cost is not visible in the
        per-question latency — ingestion is paid once, up front.
        """

    def _retrieve(self, query: np.ndarray, question: str, namespace: str) -> list[RetrievedMemory]:
        return self._retrieve_with_trace(query, question, namespace)[0]

    def _retrieve_with_trace(self, query: np.ndarray, question: str, namespace: str):
        return self.retriever.retrieve_with_trace(
            query,
            question,
            namespace,
            temporal=self.temporal,
            limit=self.top_k,
        )

    def retrieve(self, instance: Instance) -> list[RetrievedMemory]:
        """Retrieve exactly the candidates used by normal answering."""
        query = self.encoder.encode_one(instance.question)
        return self._retrieve(query, instance.question, instance.question_id)

    def _pack(self, candidates: list[Memory], retrieved: dict[str, RetrievedMemory], query: str):
        """Select under a token budget instead of truncating at top_k.

        Utilities come from the predictor when one is supplied and from the
        retrieval score otherwise. The fallback is not a placeholder: "pack by
        relevance" is the control the utility-aware packer has to beat, and running
        both through the same knapsack is what isolates the utility signal from the
        packing.
        """
        from llm_long_term_memory.influence import build as build_features
        from llm_long_term_memory.pack import pack

        if self.utility_model is None:
            utilities = [retrieved[m.id].score for m in candidates]
        else:
            import numpy as np

            rows = [
                build_features(
                    m,
                    query=query,
                    retrieval_score=retrieved[m.id].score,
                    rank=i,
                    neighbours=candidates,
                ).as_list()
                for i, m in enumerate(candidates)
            ]
            utilities = self.utility_model.predict(np.array(rows, dtype=float)).tolist()

        return pack(candidates, utilities, self.token_budget, type_floors=self.type_floors)

    def answer(self, instance: Instance) -> Answer:
        now = datetime.now()
        if self.decay_enabled:
            from llm_long_term_memory.lifecycle import apply_decay

            apply_decay(
                self.store,
                instance.question_id,
                now=now,
                halflife_days=self.decay_halflife_days,
            )
        retrieval_started = perf_counter()
        query_vector = self.encoder.encode_one(instance.question)
        retrieved, trace = self._retrieve_with_trace(
            query_vector, instance.question, instance.question_id
        )
        retrieval_latency_ms = (perf_counter() - retrieval_started) * 1000
        selected = [hit.memory for hit in retrieved]
        retrieved_by_id = {hit.memory.id: hit for hit in retrieved}

        coherent = None
        if self.session_budget is not None:
            # One namespace read, grouped once. Retrieval found the memories; this
            # decides which conversations they belong to and hands over each one
            # whole, in the order it happened.
            by_session: dict[str, list[Memory]] = {}
            for memory in self.store.iter_all(instance.question_id):
                if memory.source_session_id:
                    by_session.setdefault(memory.source_session_id, []).append(memory)
            forced_sessions = None
            if self.oracle_session_context:
                internal_by_external = {
                    external_session_id(session_id): session_id for session_id in by_session
                }
                forced_sessions = [
                    internal_by_external[session_id]
                    for session_id in instance.answer_session_ids
                    if session_id in internal_by_external
                ]
            coherent = build_coherent_context(
                retrieved,
                lambda sid: by_session.get(sid, []),
                self.session_budget,
                forced_session_ids=forced_sessions,
            )
            selected = list(coherent.memories)
            # Memories a coherent session contributes were not necessarily retrieved,
            # so they have no hit to report signals from. Recorded as absent rather
            # than fabricated at zero, which would read as "scored and lost".
            retrieved_by_id = {
                memory.id: retrieved_by_id[memory.id]
                for memory in selected
                if memory.id in retrieved_by_id
            }

        assembly_started = perf_counter()
        packed = None
        if self.token_budget:
            packed = self._pack(selected, retrieved_by_id, instance.question)
            selected = packed.selected

        operation = (
            reasoning_kind(instance.question) if self.answer_policy == "reasoned_v3" else None
        )
        hydrate_for_reasoning = self.adaptive_reasoning_hydration and operation in {
            "temporal",
            "multi_session_aggregation",
            "current_state",
        }
        context, hydration = self._assemble_context(selected, force_hydration=hydrate_for_reasoning)
        assembly_latency_ms = (perf_counter() - assembly_started) * 1000
        if self.fallback is not None:
            completion, answer_text, verdict, raw_evidence = self._answer_with_fallback(
                instance, context, selected
            )
        else:
            completion = self._complete(instance, context)
            answer_text, verdict, raw_evidence = completion.text.strip(), None, None
        if self.decay_enabled:
            self.store.record_access(
                [memory.id for memory in selected], now, reinforcement=self.reinforcement
            )

        evidence = set(instance.answer_session_ids)
        candidate_sessions = _external_session_ids(trace.candidate_session_ids)
        ranked_sessions = _external_session_ids(hit.memory.source_session_id for hit in retrieved)
        selected_sessions = _external_session_ids(m.source_session_id for m in selected)
        hydrated_sessions = _external_session_ids(e.session_id for e in hydration.evidence)
        return Answer(
            text=answer_text,
            context_tokens=int(
                (
                    len(context)
                    + (
                        len(raw_evidence.render(max_chars=self.raw_fallback_max_chars))
                        if raw_evidence and raw_evidence.used
                        else 0
                    )
                )
                / self.chars_per_token
            ),
            prompt_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=completion.api_latency_ms,
            retrieved_ids=[m.id for m in selected],
            notes={
                "top_k": self.top_k,
                "temporal": self.temporal,
                "retrieval_weights": self.retriever.weights,
                "retrieval": [
                    {
                        "memory_id": memory.id,
                        "score": retrieved_by_id[memory.id].score,
                        "strength": retrieved_by_id[memory.id].strength,
                        "signals": retrieved_by_id[memory.id].signals.to_dict(),
                    }
                    for memory in selected
                    if memory.id in retrieved_by_id
                ],
                # Which context shape produced this row. A result that does not name
                # its arm cannot be compared with one that does, and this project has
                # already shipped a store whose meta claimed the wrong extractor.
                "context_shape": (
                    "coherent-oracle"
                    if coherent is not None and self.oracle_session_context
                    else ("coherent" if coherent is not None else "flat")
                ),
                "coherent": None
                if coherent is None
                else {
                    "sessions": list(coherent.sessions),
                    "dropped_sessions": list(coherent.dropped_sessions),
                    "truncated": coherent.truncated,
                    "memories": len(coherent.memories),
                    "unretrieved": sum(1 for m in coherent.memories if m.id not in retrieved_by_id),
                    "budget": {
                        "max_sessions": self.session_budget.max_sessions,
                        "window_radius": self.session_budget.window_radius,
                        "max_total_memories": self.session_budget.max_total_memories,
                        "aggregate": self.session_budget.aggregate,
                        "session_order": self.session_budget.session_order,
                        "include_superseded": self.session_budget.include_superseded,
                    },
                    "oracle_sessions": self.oracle_session_context,
                },
                "decay_enabled": self.decay_enabled,
                "superseded_shown": sum(1 for m in selected if m.status != "active"),
                "token_budget": self.token_budget,
                "packed_utilisation": packed.utilisation if packed else None,
                "dropped_negative": packed.dropped_negative if packed else None,
                "source_session_recalled": bool(evidence & selected_sessions),
                "evidence_recalled": bool(evidence & selected_sessions),
                # Staged recall. One end-of-pipeline number cannot distinguish "never
                # found it" from "found it and then dropped it", and those call for
                # opposite fixes. Production stages are nested. In the explicitly
                # labelled oracle arm, `selected` can recover a gold session that
                # `ranked` missed; that non-production ceiling is intentionally not
                # interpreted as a ranking fix.
                "recall_stages": {
                    "candidates": bool(evidence & candidate_sessions),
                    "ranked": bool(evidence & ranked_sessions),
                    "selected": bool(evidence & selected_sessions),
                    "hydrated": bool(evidence & hydrated_sessions)
                    if self.evidence_hydration or hydrate_for_reasoning
                    else None,
                },
                "recall_coverage": {
                    "candidates": _session_coverage(evidence, candidate_sessions),
                    "ranked": _session_coverage(evidence, ranked_sessions),
                    "selected": _session_coverage(evidence, selected_sessions),
                    "hydrated": _session_coverage(evidence, hydrated_sessions)
                    if self.evidence_hydration or hydrate_for_reasoning
                    else None,
                },
                "all_source_sessions_recalled": (
                    _session_coverage(evidence, selected_sessions) == 1.0 if evidence else None
                ),
                "candidates_considered": len(trace.candidate_ids),
                "reranked": trace.reranked,
                "evidence_hydration": self.evidence_hydration,
                "adaptive_reasoning_hydration": self.adaptive_reasoning_hydration,
                "hydration_applied": bool(hydration.evidence),
                "hydrated_memory_ids": [item.memory_id for item in hydration.evidence],
                "hydrated_tokens": hydration.tokens,
                "hydration_missing_anchors": hydration.missing_anchors,
                "hydration_skipped_for_budget": hydration.skipped_for_budget,
                "hydration_redundant_anchors": hydration.redundant_anchors,
                "hydration_eligible_sessions": hydration.eligible_sessions,
                "hydration_sessions": hydration.hydrated_sessions,
                "hydration_session_coverage": (
                    hydration.hydrated_sessions / hydration.eligible_sessions
                    if hydration.eligible_sessions
                    else None
                ),
                "answer_status": verdict.status if verdict else None,
                "answer_policy": self.answer_policy,
                "reasoning_kind": operation,
                "answer_confidence": getattr(verdict, "confidence", None),
                "answer_evidence_summary": getattr(verdict, "evidence_summary", None),
                "answer_calculation": getattr(verdict, "calculation", None),
                "fallback_level": raw_evidence.level if raw_evidence else "none",
                "fallback_reason": (
                    raw_evidence.reason if raw_evidence and raw_evidence.used else None
                ),
                "fallback_turns": [
                    f"{external_session_id(t.session_id)}:{t.turn_index}"
                    for t in (raw_evidence.turns if raw_evidence else [])
                ],
                "retrieval_latency_ms": retrieval_latency_ms,
                "assembly_latency_ms": assembly_latency_ms,
                "answerer_api_latency_ms": completion.api_latency_ms,
            },
        )

    def answer_with_memories(self, instance: Instance, memories: list[Memory]) -> str:
        """Answer a fixed memory set without retrieval or lifecycle side effects."""
        context, _ = self._assemble_context(memories)
        completion = self._complete(instance, context)
        return completion.text.strip()

    def _assemble_context(
        self, memories: list[Memory], *, force_hydration: bool = False
    ) -> tuple[str, HydrationResult]:
        body = render_grouped(memories, self.temporal)
        context = f"{TEMPORAL_NOTE}\n\n{body}" if self.temporal else body
        hydration = HydrationResult()
        if self.evidence_hydration or force_hydration:
            hydration = self.hydrator.hydrate(memories, max_tokens=self.hydration_max_tokens)
            if hydration.evidence:
                context = (
                    f"Structured memories:\n{context}\n\n"
                    f"Verbatim source evidence:\n{render_evidence(hydration.evidence)}"
                )
        return context, hydration

    _FALLBACK_TEMPLATE = """\
Your earlier attempt could not answer this from structured memory alone:
{reason}

Here are the original conversation turns those memories were extracted from. They
are verbatim, so the exact wording, links and figures are present.

{evidence}

Today's date is {date}.

Question: {question}
"""

    def _answer_with_fallback(self, instance: Instance, context: str, selected: list[Memory]):
        """One LLM call when memory suffices, two when it does not.

        The first call returns a verdict rather than prose. Only `need_source` pays
        for raw evidence and a second call — which is what keeps this from becoming
        naive RAG with extra steps.
        """
        first = self._complete(instance, context, structured=True)
        try:
            verdict = self.verdict_schema.model_validate_json(first.text)
        except ValueError:
            # A model that ignored the schema still produced an answer; treat the
            # raw text as one rather than failing the question.
            return first, first.text.strip(), None, None

        if verdict.status == "answer" or self.fallback is None:
            return first, verdict.answer.strip() or first.text.strip(), verdict, None

        evidence = self.fallback.recover(
            instance.question_id,
            verdict.source_query or instance.question,
            selected if verdict.status == "need_source" else [],
        )
        if not evidence.used:
            # Nothing in the archive either. Abstention is correct here, and is a
            # measured strength worth protecting.
            return first, (verdict.answer or "I do not know.").strip(), verdict, evidence

        rendered_evidence = evidence.render(max_chars=self.raw_fallback_max_chars)
        if self.answer_policy == "reasoned_v3":
            prompt = render_reasoned_prompt(
                f"{context}\n\nThe first pass requested source because: "
                f"{verdict.reason or 'the requested detail was missing'}\n\n"
                f"Verbatim source evidence:\n{rendered_evidence}",
                instance.question_date,
                instance.question,
            )
        else:
            prompt = self._FALLBACK_TEMPLATE.format(
                reason=verdict.reason or "the requested detail was missing",
                evidence=rendered_evidence,
                date=instance.question_date,
                question=instance.question,
            )
        second = self.client.generate(
            role="answerer",
            model=self.model,
            prompt=prompt,
            system=self.answer_system,
            temperature=0.0,
            max_output_tokens=self.max_output_tokens,
            est_input_tokens=int(len(prompt) / self.chars_per_token),
        )
        return second, second.text.strip(), verdict, evidence

    def _complete(self, instance: Instance, context: str, structured: bool = False):
        prompt = (
            render_reasoned_prompt(context, instance.question_date, instance.question)
            if self.answer_policy == "reasoned_v3"
            else _TEMPLATE.format(
                context=context,
                date=instance.question_date,
                question=instance.question,
            )
        )
        completion = self.client.generate(
            role="answerer",
            model=self.model,
            prompt=prompt,
            system=self.answer_system,
            temperature=0.0,
            max_output_tokens=self.max_output_tokens,
            est_input_tokens=int(len(prompt) / self.chars_per_token),
            **({"schema": AnswerVerdict} if structured else {}),
        )
        return completion
