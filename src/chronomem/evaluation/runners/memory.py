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

from chronomem.embed import Encoder
from chronomem.evaluation.datasets.longmemeval import Instance
from chronomem.llm.client import GeminiClient
from chronomem.retrieve import (
    EvidenceHydrator,
    HybridRetriever,
    HydrationResult,
    RetrievedMemory,
    render_evidence,
)
from chronomem.store import Memory, MemoryStore, NumpyFlatIndex

from .base import ANSWER_SYSTEM, Answer

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
        hydration_neighbouring_sentences: int = 1,
        hydration_max_tokens: int = 800,
        token_budget: int = 0,
        utility_model=None,
        type_floors: dict[str, float] | None = None,
        max_output_tokens: int = 512,
        chars_per_token: float = 4.6,
    ) -> None:
        self.client = client
        self.model = model
        self.encoder = encoder
        self.store = store
        self.index = index
        self.temporal = temporal
        self.top_k = top_k
        self.retriever = HybridRetriever(
            store,
            index,
            weights=retrieval_weights or {"semantic": 1.0},
            candidate_limit=candidate_limit,
            recency_halflife_days=recency_halflife_days,
            use_strength=decay_enabled,
        )
        self.decay_enabled = decay_enabled
        self.decay_halflife_days = decay_halflife_days
        self.reinforcement = reinforcement
        self.evidence_hydration = evidence_hydration
        self.hydration_max_tokens = hydration_max_tokens
        self.hydrator = EvidenceHydrator(
            store,
            neighbouring_sentences=hydration_neighbouring_sentences,
            chars_per_token=chars_per_token,
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

    def prepare(self, instance: Instance) -> None:
        """Nothing per-instance: the store is built once by `chronomem ingest`.

        This is the structural difference from the baselines, which rebuild an index
        for every question. It is also why this variant's cost is not visible in the
        per-question latency — ingestion is paid once, up front.
        """

    def _retrieve(self, query: np.ndarray, question: str, namespace: str) -> list[RetrievedMemory]:
        return self.retriever.retrieve(
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
        from chronomem.influence import build as build_features
        from chronomem.pack import pack

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
            from chronomem.lifecycle import apply_decay

            apply_decay(
                self.store,
                instance.question_id,
                now=now,
                halflife_days=self.decay_halflife_days,
            )
        retrieval_started = perf_counter()
        retrieved = self.retrieve(instance)
        retrieval_latency_ms = (perf_counter() - retrieval_started) * 1000
        selected = [hit.memory for hit in retrieved]
        retrieved_by_id = {hit.memory.id: hit for hit in retrieved}

        assembly_started = perf_counter()
        packed = None
        if self.token_budget:
            packed = self._pack(selected, retrieved_by_id, instance.question)
            selected = packed.selected

        context, hydration = self._assemble_context(selected)
        assembly_latency_ms = (perf_counter() - assembly_started) * 1000
        completion = self._complete(instance, context)
        if self.decay_enabled:
            self.store.record_access(
                [memory.id for memory in selected], now, reinforcement=self.reinforcement
            )

        evidence = set(instance.answer_session_ids)
        return Answer(
            text=completion.text.strip(),
            context_tokens=int(len(context) / self.chars_per_token),
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
                ],
                "decay_enabled": self.decay_enabled,
                "superseded_shown": sum(1 for m in selected if m.status != "active"),
                "token_budget": self.token_budget,
                "packed_utilisation": packed.utilisation if packed else None,
                "dropped_negative": packed.dropped_negative if packed else None,
                "source_session_recalled": bool(
                    evidence & {m.source_session_id for m in selected if m.source_session_id}
                ),
                "evidence_recalled": bool(
                    evidence & {m.source_session_id for m in selected if m.source_session_id}
                ),
                "evidence_hydration": self.evidence_hydration,
                "hydrated_memory_ids": [item.memory_id for item in hydration.evidence],
                "hydrated_tokens": hydration.tokens,
                "hydration_missing_anchors": hydration.missing_anchors,
                "hydration_skipped_for_budget": hydration.skipped_for_budget,
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

    def _assemble_context(self, memories: list[Memory]) -> tuple[str, HydrationResult]:
        body = "\n".join(render_memory(memory, self.temporal) for memory in memories)
        context = f"{TEMPORAL_NOTE}\n\n{body}" if self.temporal else body
        hydration = HydrationResult()
        if self.evidence_hydration:
            hydration = self.hydrator.hydrate(memories, max_tokens=self.hydration_max_tokens)
            if hydration.evidence:
                context = (
                    f"Structured memories:\n{context}\n\n"
                    f"Verbatim source evidence:\n{render_evidence(hydration.evidence)}"
                )
        return context, hydration

    def _complete(self, instance: Instance, context: str):
        prompt = _TEMPLATE.format(
            context=context, date=instance.question_date, question=instance.question
        )
        completion = self.client.generate(
            role="answerer",
            model=self.model,
            prompt=prompt,
            system=ANSWER_SYSTEM,
            temperature=0.0,
            max_output_tokens=self.max_output_tokens,
            est_input_tokens=int(len(prompt) / self.chars_per_token),
        )
        return completion
