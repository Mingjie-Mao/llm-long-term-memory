"""Stable product answer path, independent of benchmark runners.

The evaluation package may wrap this engine and attach gold-only diagnostics, but the
live API must not construct a benchmark runner or import a judge to answer a user.
"""

from __future__ import annotations

from datetime import datetime
from time import perf_counter

from llm_long_term_memory.answering import ANSWER_SYSTEM, Answer, AnswerVerdict
from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.ingest.extract import _parse_date
from llm_long_term_memory.retrieve import HybridRetriever
from llm_long_term_memory.retrieve.fallback import RawFallback
from llm_long_term_memory.store import Memory, MemoryStore, VectorIndex

_TEMPLATE = """\
Here is what is known about the user, drawn from their chat history.

{context}

Today's date is {date}.

Question: {question}
"""

_FALLBACK_TEMPLATE = """\
Your earlier attempt could not answer this from structured memory alone:
{reason}

Here are the original conversation turns those memories were extracted from. They
are verbatim, so the exact wording, links and figures are present.

{evidence}

Today's date is {date}.

Question: {question}
"""

TEMPORAL_NOTE = (
    "Each fact is shown with the period it was true for. `(since <date>)` means it "
    "is still true now; `(<date> to <date>)` means it was replaced and is no longer "
    "current. Answer with what is true now unless the question asks about the past."
)

_SCOPE_HEADINGS: tuple[tuple[str, str], ...] = (
    ("profile", "About the user"),
    ("preference", "User preferences"),
    ("plan", "Current plans"),
    ("event", "Past events"),
    ("recommendation", "Previously recommended by the assistant"),
    ("commitment", "The assistant agreed to"),
    ("shared_context", "Other people and things discussed"),
)
_UNSCOPED_HEADING = "Other things known about the user"


def render_memory(memory: Memory, temporal: bool) -> str:
    if not temporal:
        return f"- {memory.content}"
    start = memory.valid_from or memory.occurred_at
    if start and memory.valid_to:
        window = f"({start:%Y-%m-%d} to {memory.valid_to:%Y-%m-%d}, no longer current)"
    elif start:
        window = f"(since {start:%Y-%m-%d})"
    else:
        window = "(date unknown)"
    return f"- {memory.content} {window}"


def render_timelines(memories: list[Memory], temporal: bool) -> tuple[str, list[Memory]]:
    if not temporal:
        return "", list(memories)
    chains: dict[tuple[str, str], list[Memory]] = {}
    for memory in memories:
        if memory.subject and memory.predicate:
            chains.setdefault((memory.subject, memory.predicate), []).append(memory)
    blocks: list[str] = []
    claimed: set[str] = set()
    for (subject, predicate), group in chains.items():
        if len(group) < 2:
            continue
        ordered = sorted(
            group,
            key=lambda item: (item.valid_from or item.occurred_at or datetime.max, item.id),
        )
        lines = []
        for memory in ordered:
            when = memory.valid_from or memory.occurred_at
            stamp = f"{when:%Y-%m-%d}" if when else "date unknown"
            mark = "[CURRENT]" if memory.status == "active" and not memory.valid_to else "[was]"
            lines.append(f"    {stamp}  {memory.content} {mark}")
            claimed.add(memory.id)
        label = f"{subject} {predicate}".replace("_", " ").strip()
        blocks.append(f"{label} — how this changed over time:\n" + "\n".join(lines))
    remaining = [memory for memory in memories if memory.id not in claimed]
    return "\n\n".join(blocks), remaining


def render_context(memories: list[Memory], temporal: bool) -> str:
    timelines, remaining = render_timelines(memories, temporal)
    prefix = [timelines] if timelines else []
    if not any(memory.scope for memory in remaining):
        flat = "\n".join(render_memory(memory, temporal) for memory in remaining)
        body = "\n\n".join([*prefix, flat]) if flat else "\n\n".join(prefix)
        return f"{TEMPORAL_NOTE}\n\n{body}" if temporal else body

    by_scope: dict[str, list[Memory]] = {}
    for memory in remaining:
        by_scope.setdefault(memory.scope or "", []).append(memory)
    blocks: list[str] = []
    for scope, heading in _SCOPE_HEADINGS:
        group = by_scope.pop(scope, None)
        if group:
            lines = "\n".join(render_memory(memory, temporal) for memory in group)
            blocks.append(f"{heading}:\n{lines}")
    leftovers = [memory for group in by_scope.values() for memory in group]
    if leftovers:
        lines = "\n".join(render_memory(memory, temporal) for memory in leftovers)
        blocks.append(f"{_UNSCOPED_HEADING}:\n{lines}")
    body = "\n\n".join([*prefix, *blocks])
    return f"{TEMPORAL_NOTE}\n\n{body}" if temporal else body


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
    ) -> None:
        self.client = client
        self.model = model
        self.encoder = encoder
        self.store = store
        self.temporal = temporal
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.chars_per_token = chars_per_token
        self.raw_fallback_max_chars = raw_fallback_max_chars
        self.retriever = HybridRetriever(
            store,
            index,
            weights=retrieval_weights or {"semantic": 1.0},
            candidate_limit=candidate_limit,
            recency_halflife_days=recency_halflife_days,
        )
        self.fallback = (
            RawFallback(store, max_turns=raw_fallback_max_turns) if raw_fallback else None
        )

    def answer_request(
        self,
        request: AnswerRequest,
        *,
        limit: int | None = None,
        evidence_session_ids: tuple[str, ...] = (),
    ) -> Answer:
        del evidence_session_ids  # benchmark-only labels never affect product behavior
        started = perf_counter()
        query = self.encoder.encode_one(request.question)
        retrieved, trace = self.retriever.retrieve_with_trace(
            query,
            request.question,
            request.user_id,
            temporal=self.temporal,
            limit=limit or self.top_k,
            as_of=_parse_date(request.asked_on) if request.asked_on else None,
        )
        retrieval_ms = (perf_counter() - started) * 1000
        memories = [hit.memory for hit in retrieved]
        context = render_context(memories, self.temporal)
        prompt = _TEMPLATE.format(
            context=context or "No relevant long-term memory was found.",
            date=request.asked_on,
            question=request.question,
        )
        first = self.client.generate(
            role="answerer",
            model=self.model,
            prompt=prompt,
            system=ANSWER_SYSTEM,
            schema=AnswerVerdict if self.fallback else None,
            temperature=0.0,
            max_output_tokens=self.max_output_tokens,
            est_input_tokens=int(len(prompt) / self.chars_per_token),
        )
        answer_text = first.text.strip()
        evidence = None
        verdict = None
        if self.fallback is not None:
            try:
                verdict = AnswerVerdict.model_validate_json(first.text)
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
                        fallback_prompt = _FALLBACK_TEMPLATE.format(
                            reason=verdict.reason or "the requested detail was missing",
                            evidence=evidence.render(max_chars=self.raw_fallback_max_chars),
                            date=request.asked_on,
                            question=request.question,
                        )
                        first = self.client.generate(
                            role="answerer",
                            model=self.model,
                            prompt=fallback_prompt,
                            system=ANSWER_SYSTEM,
                            temperature=0.0,
                            max_output_tokens=self.max_output_tokens,
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
                "answer_status": verdict.status if verdict else None,
                "fallback_level": evidence.level if evidence else "none",
                "candidates_considered": len(trace.candidate_ids),
                "retrieval_latency_ms": retrieval_ms,
            },
        )
