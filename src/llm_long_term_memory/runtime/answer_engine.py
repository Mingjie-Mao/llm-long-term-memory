"""Stable product answer path, independent of benchmark runners.

The evaluation package may wrap this engine and attach gold-only diagnostics, but the
live API must not construct a benchmark runner or import a judge to answer a user.

What it answers with is named by a profile (`ANSWER_PROFILES`). `t4` is the arm with
the best exposed-development result (heldout100, three runs: 87.67% against v1's
81.00%); `tests/test_product_t4_equivalence.py` holds the engine to the exact requests
the evaluated `two_stage_raw_primary_t4` arm sends, so the measured system and the
served one cannot drift apart again.
"""

from __future__ import annotations

from datetime import datetime
from time import perf_counter
from typing import Any

from llm_long_term_memory.answering import (
    ANSWER_SYSTEM,
    COUNT_LATEST_SYSTEM,
    Answer,
    AnswerVerdict,
    NotedAnswerVerdict,
    asks_about_time,
    asks_for_aggregate,
    count_time_system,
)
from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.ingest.extract import _parse_date
from llm_long_term_memory.retrieve import (
    EvidenceHydrator,
    HybridRetriever,
    HydrationResult,
    render_evidence,
)
from llm_long_term_memory.retrieve.fallback import RawFallback
from llm_long_term_memory.runtime.context import (
    ANSWER_TEMPLATE,
    FALLBACK_TEMPLATE,
    TEMPORAL_NOTE,
    render_context,
)
from llm_long_term_memory.store import Memory, MemoryStore, VectorIndex, external_session_id

__all__ = ["ANSWER_PROFILES", "TEMPORAL_NOTE", "AnswerEngine", "render_context"]

# Engine keyword arguments per profile. `top_k` is left to the caller: the service's
# inspector and answerer share one value (`ServiceConfig.top_k`).
ANSWER_PROFILES: dict[str, dict[str, Any]] = {
    # Frozen v2's answer path: structured memory, raw turns only on a fallback verdict.
    # About 600 context tokens per question; 72% on the frozen test100.
    "v2": {
        "raw_fallback": True,
        "raw_fallback_max_turns": 3,
        "raw_fallback_max_chars": 2400,
    },
    # Raw-primary t4: v2 plus source sentences behind each memory, 4,000 tokens of the
    # user's own turns found by the question (user turns first, the period a question
    # names searched by time, relative dates resolved), a latest-value rule, and notes
    # before answering counts and date questions. About 6,300 context tokens.
    "t4": {
        "raw_fallback": True,
        "raw_fallback_max_turns": 3,
        "raw_fallback_max_chars": 2400,
        "evidence_hydration": True,
        "raw_primary_tokens": 4000,
        "raw_primary_time_notes": True,
        "raw_primary_dated": True,
        "raw_primary_prefer_user": True,
        "raw_primary_time_window": True,
        "answer_policy": "v2_clt",
    },
}


class AnswerEngine:
    """Retrieve tenant-scoped memory and answer, with conditional raw fallback."""

    def __init__(
        self,
        client,
        *,
        model: str,
        encoder,
        store: MemoryStore,
        index: VectorIndex,
        temporal: bool = True,
        top_k: int = 10,
        retrieval_weights: dict[str, float] | None = None,
        candidate_limit: int = 50,
        recency_halflife_days: float = 30.0,
        raw_fallback: bool = False,
        raw_fallback_max_turns: int = 3,
        raw_fallback_max_chars: int = 2400,
        max_output_tokens: int = 512,
        chars_per_token: float = 4.6,
        answer_policy: str = "v2",
        evidence_hydration: bool = False,
        hydration_neighbouring_sentences: int = 1,
        hydration_max_tokens: int = 800,
        hydration_allocation: str = "ranked",
        raw_primary_tokens: int = 0,
        raw_primary_time_notes: bool = False,
        raw_primary_dated: bool = False,
        raw_primary_prefer_user: bool = False,
        raw_primary_time_window: bool = False,
    ) -> None:
        raw_flags = (
            raw_primary_time_notes,
            raw_primary_dated,
            raw_primary_prefer_user,
            raw_primary_time_window,
        )
        if any(raw_flags) and not raw_primary_tokens:
            raise ValueError("raw_primary_* options need raw_primary_tokens")
        if answer_policy not in {"v2", "v2_clt"}:
            raise ValueError(f"unknown answer policy {answer_policy!r}")
        self.client = client
        self.model = model
        self.encoder = encoder
        self.store = store
        self.temporal = temporal
        self.top_k = top_k
        self.answer_policy = answer_policy
        self.max_output_tokens = max_output_tokens
        self.chars_per_token = chars_per_token
        self.raw_fallback_max_chars = raw_fallback_max_chars
        self.evidence_hydration = evidence_hydration
        self.hydration_max_tokens = hydration_max_tokens
        self.raw_primary_tokens = raw_primary_tokens
        self.raw_primary_time_notes = raw_primary_time_notes
        self.raw_primary_dated = raw_primary_dated
        self.raw_primary_prefer_user = raw_primary_prefer_user
        self.raw_primary_time_window = raw_primary_time_window
        self.retriever = HybridRetriever(
            store,
            index,
            weights=retrieval_weights or {"semantic": 1.0},
            candidate_limit=candidate_limit,
            recency_halflife_days=recency_halflife_days,
        )
        self.hydrator = EvidenceHydrator(
            store,
            neighbouring_sentences=hydration_neighbouring_sentences,
            chars_per_token=chars_per_token,
            allocation=hydration_allocation,
        )
        self.fallback = (
            RawFallback(store, max_turns=raw_fallback_max_turns) if raw_fallback else None
        )

    @classmethod
    def from_profile(cls, profile: str, client, **kwargs) -> AnswerEngine:
        if profile not in ANSWER_PROFILES:
            raise ValueError(f"unknown answer profile {profile!r}")
        return cls(client, **{**ANSWER_PROFILES[profile], **kwargs})

    def _answer_mode(self, question: str):
        """System prompt, verdict schema and output limit for one question.

        Under v2_clt a count or total, or a question about dates, durations or order, is
        answered from notes and gets room for them; every other question keeps v2's
        verdict and output limit.
        """
        if self.answer_policy != "v2_clt":
            return ANSWER_SYSTEM, AnswerVerdict, self.max_output_tokens, None
        counting = asks_for_aggregate(question)
        timing = asks_about_time(question)
        if not (counting or timing):
            return count_time_system(False, False), AnswerVerdict, self.max_output_tokens, None
        mode = "_".join(m for m, on in (("count", counting), ("time", timing)) if on) + "_notes"
        return (
            count_time_system(counting, timing),
            NotedAnswerVerdict,
            max(1024, self.max_output_tokens),
            mode,
        )

    def _context(self, memories: list[Memory]) -> tuple[str, HydrationResult]:
        context = render_context(memories, self.temporal)
        hydration = HydrationResult()
        if self.evidence_hydration:
            hydration = self.hydrator.hydrate(memories, max_tokens=self.hydration_max_tokens)
            if hydration.evidence:
                context = (
                    f"Structured memories:\n{context}\n\n"
                    f"Verbatim source evidence:\n{render_evidence(hydration.evidence)}"
                )
        return context, hydration

    def answer_request(
        self,
        request: AnswerRequest,
        *,
        limit: int | None = None,
        evidence_session_ids: tuple[str, ...] = (),
    ) -> Answer:
        del evidence_session_ids  # benchmark-only labels never affect product behavior
        system, schema, max_output_tokens, answer_mode = self._answer_mode(request.question)
        asked_on: datetime | None = _parse_date(request.asked_on) if request.asked_on else None
        started = perf_counter()
        query = self.encoder.encode_one(request.question)
        retrieved, trace = self.retriever.retrieve_with_trace(
            query,
            request.question,
            request.user_id,
            temporal=self.temporal,
            limit=limit or self.top_k,
            as_of=asked_on,
        )
        retrieval_ms = (perf_counter() - started) * 1000
        memories = [hit.memory for hit in retrieved]
        context, hydration = self._context(memories)
        excerpts = None
        if self.raw_primary_tokens:
            from llm_long_term_memory.retrieve.excerpts import archive_excerpts

            dated = self.raw_primary_dated or self.raw_primary_time_window
            # Scoped by `request.user_id` inside `search_turns`: another tenant's turns
            # cannot be selected (tests/test_product_raw_primary.py).
            excerpts = archive_excerpts(
                self.store,
                request.user_id,
                request.question,
                self.raw_primary_tokens,
                chars_per_token=self.chars_per_token,
                asked_on=asked_on if dated or self.raw_primary_time_notes else None,
                time_notes=self.raw_primary_time_notes,
                prefer_user=self.raw_primary_prefer_user,
                time_window=self.raw_primary_time_window,
            )
            block = excerpts.render()
            if block:
                context = f"{context}\n\n{block}" if context else block
        prompt = ANSWER_TEMPLATE.format(
            context=context or "No relevant long-term memory was found.",
            date=request.asked_on,
            question=request.question,
        )
        first = self.client.generate(
            role="answerer",
            model=self.model,
            prompt=prompt,
            system=system,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            est_input_tokens=int(len(prompt) / self.chars_per_token),
            **({"schema": schema} if self.fallback else {}),
        )
        answer_text = first.text.strip()
        evidence = None
        verdict = None
        if self.fallback is not None:
            try:
                verdict = schema.model_validate_json(first.text)
            except ValueError:
                verdict = None
            if verdict is not None:
                if verdict.status == "answer":
                    answer_text = verdict.answer.strip() or "I do not know."
                else:
                    evidence = self.fallback.recover(
                        request.user_id,
                        verdict.source_query or request.question,
                        memories if verdict.status == "need_source" else [],
                    )
                    if evidence.used:
                        fallback_prompt = FALLBACK_TEMPLATE.format(
                            reason=verdict.reason or "the requested detail was missing",
                            evidence=evidence.render(max_chars=self.raw_fallback_max_chars),
                            date=request.asked_on,
                            question=request.question,
                        )
                        first = self.client.generate(
                            role="answerer",
                            model=self.model,
                            prompt=fallback_prompt,
                            # The second pass carries no schema, so it never gets a
                            # prompt that asks for notes it has nowhere to put.
                            system=COUNT_LATEST_SYSTEM
                            if self.answer_policy == "v2_clt"
                            else ANSWER_SYSTEM,
                            temperature=0.0,
                            max_output_tokens=max_output_tokens,
                            est_input_tokens=int(len(fallback_prompt) / self.chars_per_token),
                        )
                        answer_text = first.text.strip()
                    else:
                        answer_text = verdict.answer.strip() or "I do not know."
        if answer_text.startswith(("{", "```")):
            answer_text = "I do not know."
        return Answer(
            text=answer_text or "I do not know.",
            context_tokens=int(len(context) / self.chars_per_token),
            prompt_tokens=first.input_tokens,
            output_tokens=first.output_tokens,
            latency_ms=first.api_latency_ms,
            retrieved_ids=[memory.id for memory in memories],
            notes={
                "top_k": limit or self.top_k,
                "temporal": self.temporal,
                "retrieval_weights": self.retriever.weights,
                "retrieval": [
                    {
                        "memory_id": hit.memory.id,
                        "score": hit.score,
                        "strength": hit.strength,
                        "signals": hit.signals.to_dict(),
                    }
                    for hit in retrieved
                ],
                "answer_policy": self.answer_policy,
                "answer_mode": answer_mode,
                "answer_status": verdict.status if verdict else None,
                "fallback_level": evidence.level if evidence else "none",
                # Which archived turns the fallback attached, so a caller can show them.
                # Reporting only: the answer and its prompt are unchanged.
                "fallback_reason": evidence.reason if evidence and evidence.used else None,
                "fallback_turns": [
                    f"{external_session_id(t.session_id)}:{t.turn_index}"
                    for t in (evidence.turns if evidence else [])
                ],
                "candidates_considered": len(trace.candidate_ids),
                "retrieval_latency_ms": retrieval_ms,
                "hydrated_memory_ids": [item.memory_id for item in hydration.evidence],
                "raw_primary_turns": len(excerpts.turns) if excerpts else 0,
                "raw_primary_time_notes": self.raw_primary_time_notes,
            },
        )
