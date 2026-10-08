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

from llm_long_term_memory.answering import (
    COUNT_LATEST_PROMPT_VERSION,
    COUNT_LATEST_SYSTEM,
    COUNT_NOTES_SYSTEM,
    COUNT_TIME_PROMPT_VERSION,
    NOTED_ANSWER_PROMPT_VERSION,
    NOTED_ANSWER_SYSTEM,
    NotedAnswerVerdict,
    asks_about_time,
    asks_for_aggregate,
    count_time_system,
)
from llm_long_term_memory.conversation import AnswerRequest
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
from llm_long_term_memory.runtime.grounded_answering import (
    PROMPT_VERSION as GROUNDED_PROMPT_VERSION,
)
from llm_long_term_memory.runtime.grounded_answering import (
    SYSTEM as GROUNDED_SYSTEM,
)
from llm_long_term_memory.runtime.grounded_answering import (
    GroundedAnswerer,
    GroundedAnswererV2,
    GroundedAnswererV3,
    GroundedAnswererV4,
    GroundedAnswererV5,
    GroundedAnswererV6,
    GroundedAnswererV7,
    GroundedAnswererV8,
    GroundedAnswererV9,
    GroundedAnswererV10,
    GroundedAnswererV11,
    GroundedAnswererV12,
    GroundedAnswererV13,
    GroundedAnswererV14,
    GroundedAnswererV15,
    GroundedAnswererV16,
    GroundedAnswererV17,
    GroundedAnswererV18,
    GroundedVerdict,
    GroundedVerdictV6,
    GroundedVerdictV7,
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
from .synthesis import (
    SYNTHESIS_ANSWER_PROMPT_VERSION,
    SYNTHESIS_ANSWER_SYSTEM,
    SYNTHESIS_ENUMERATE_ANSWER_SYSTEM,
    SYNTHESIS_ENUMERATE_PROMPT_VERSION,
    EnumeratingSynthesisVerdict,
    SynthesisVerdict,
    compute,
    missing_field_is_narration,
)
from .v2c import V2C_ANSWER_PROMPT_VERSION, V2C_ANSWER_SYSTEM
from .v2d import (
    V2D_ANSWER_PROMPT_VERSION,
    V2D_ANSWER_SYSTEM,
    V2DVerdict,
)
from .v2d import (
    compute as compute_v2d,
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


def _as_of(question_date: str) -> datetime | None:
    """The question's date, or None when it cannot be read.

    None falls back to the wall clock, which is what every caller got before. A
    guessed date would be worse than no date: it would make recency confidently wrong
    rather than visibly absent.
    """
    from llm_long_term_memory.ingest.extract import _parse_date

    return _parse_date(question_date) if question_date else None


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


def label_memories(memories: list[Memory]) -> dict[str, str]:
    """Stable citation handles in rank order: M1, M2, ... one per memory in context.

    Positional rather than derived from `memory.id`: the model has to copy these, and a
    sixteen-character hex id is a transcription error waiting to happen, while the
    position is also the reading order the prompt asks it to work through.
    """
    return {memory.id: f"M{i}" for i, memory in enumerate(memories, start=1)}


def render_grouped(
    memories: list[Memory],
    temporal: bool,
    timelines_enabled: bool = True,
    labels: dict[str, str] | None = None,
    mark_unstated: bool = False,
) -> str:
    """Group memories by scope under headings, preserving rank order within a group.

    Falls back to a plain list when nothing carries a scope, so a store built before
    P10 renders exactly as it did before.
    """
    # Chains first, so a superseded value is never also listed flat under a scope
    # heading — seeing it twice, once marked "no longer current" and once not, is
    # worse than either rendering alone.
    timelines, memories = render_timelines(memories, temporal and timelines_enabled, labels)
    tag = (lambda m: labels.get(m.id, "")) if labels else (lambda m: "")
    prefix = [timelines] if timelines else []

    if not any(memory.scope for memory in memories):
        flat = "\n".join(
            render_memory(memory, temporal, tag(memory), mark_unstated) for memory in memories
        )
        return "\n\n".join([*prefix, flat]) if flat else "\n\n".join(prefix)

    by_scope: dict[str, list[Memory]] = {}
    for memory in memories:
        by_scope.setdefault(memory.scope or "", []).append(memory)

    blocks: list[str] = []
    for scope, heading in _SCOPE_HEADINGS:
        group = by_scope.pop(scope, None)
        if group:
            lines = "\n".join(render_memory(m, temporal, tag(m), mark_unstated) for m in group)
            blocks.append(f"{heading}:\n{lines}")
    # Anything left: unscoped memories, plus any scope the model invented that the
    # validator let through.
    leftovers = [m for group in by_scope.values() for m in group]
    if leftovers:
        lines = "\n".join(render_memory(m, temporal, tag(m), mark_unstated) for m in leftovers)
        blocks.append(f"{_UNSCOPED_HEADING}:\n{lines}")
    return "\n\n".join([*prefix, *blocks])


def render_timelines(
    memories: list[Memory], temporal: bool, labels: dict[str, str] | None = None
) -> tuple[str, list[Memory]]:
    """Render supersession chains as chains, and return what was not part of one.

    The store has already decided which value of a key is current: it closed the old
    one's validity window and set its status. Flat rendering throws that away and asks
    the model to rediscover it from two adjacent bullet points that differ only in a
    parenthesised date — and on `train150` it does not: a question about where a guitar
    was serviced was answered "you *plan* to send it", with both the plan and the
    completion in context.

    So a key with more than one value is rendered as one block, in event order, with
    the current value marked. No new memory is fetched and no new field is computed;
    this is the same data the flat renderer already had.

    Only for `temporal=True`. Without dates there is no order to show, and grouping
    undated values would assert a sequence the store never established.
    """
    if not temporal:
        return "", list(memories)

    chains: dict[tuple[str, str], list[Memory]] = {}
    for memory in memories:
        if memory.subject and memory.predicate:
            chains.setdefault((memory.subject, memory.predicate), []).append(memory)

    blocks: list[str] = []
    claimed: set[str] = set()
    for (subject, predicate), group in chains.items():
        # One value is not a timeline. Rendering it as one would add a heading and a
        # "current" label to a fact nothing ever contradicted.
        if len(group) < 2:
            continue
        ordered = sorted(
            group,
            key=lambda m: (m.valid_from or m.occurred_at or datetime.max, m.id),
        )
        lines = []
        for memory in ordered:
            when = memory.valid_from or memory.occurred_at
            stamp = f"{when:%Y-%m-%d}" if when else "date unknown"
            mark = "[CURRENT]" if memory.status == "active" and not memory.valid_to else "[was]"
            # Chain members are labelled too. A count question can have a member that is
            # also part of a supersession chain, and an unlabelled one would be a member
            # the answerer has no way to cite.
            tag = f"[{labels[memory.id]}] " if labels and memory.id in labels else ""
            lines.append(f"    {stamp}  {tag}{memory.content} {mark}")
            claimed.add(memory.id)
        label = f"{subject} {predicate}".replace("_", " ").strip()
        blocks.append(f"{label} — how this changed over time:\n" + "\n".join(lines))

    remaining = [memory for memory in memories if memory.id not in claimed]
    return "\n\n".join(blocks), remaining


def render_memory(
    memory: Memory, temporal: bool, label: str = "", mark_unstated: bool = False
) -> str:
    """One memory as a context line.

    `mark_unstated` is the v2e difference and nothing else changes with it. `valid_from`
    is the date a fact was *stated*, so an undated fact has always rendered as
    "(since <the day it was mentioned>)" — which reads as the day it became true. On
    every one of the thirty temporal questions in reasoning-48 some selected evidence
    is undated, so a reader asked "how many days between X and Y" is routinely given a
    mention date wearing an event date's clothes. Saying "mentioned" instead is the
    whole mechanism.

    Off by default, because turning it on changes the context of every recorded arm.
    """
    # The label is a citation handle, not content. It goes first so that a model working
    # through the context in order reads it before the sentence it belongs to.
    tag = f"[{label}] " if label else ""
    if not temporal:
        return f"- {tag}{memory.content}"

    start = memory.valid_from or memory.occurred_at
    if start and memory.valid_to:
        window = f"({start:%Y-%m-%d} to {memory.valid_to:%Y-%m-%d}, no longer current)"
    elif start and mark_unstated and not memory.event_time_is_stated:
        window = f"(mentioned {start:%Y-%m-%d}; the fact states no date)"
    elif start:
        window = f"(since {start:%Y-%m-%d})"
    else:
        window = "(date unknown)"
    return f"- {tag}{memory.content} {window}"


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
        parallel_raw_windows: int = 0,
        parallel_raw_planned: bool = False,
        raw_primary_tokens: int = 0,
        raw_primary_only: bool = False,
        raw_primary_turn_index=None,
        raw_primary_dated: bool = False,
        raw_primary_memory_fusion: bool = False,
        raw_primary_fact_keys: bool = False,
        raw_primary_time_notes: bool = False,
        raw_primary_prefer_user: bool = False,
        raw_primary_time_window: bool = False,
        date_provenance: bool = False,
        session_budget: SessionBudget | None = None,
        oracle_session_context: bool = False,
        answer_policy: str = "v2",
        timeline_rendering: bool = True,
        scanner=None,
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
        if answer_policy not in {
            "v2",
            "v2_notes",
            "v2_cl",
            "v2_clt",
            "grounded_v1",
            "grounded_v2",
            "grounded_v3",
            "grounded_v4",
            "grounded_v5",
            "grounded_v6",
            "grounded_v7",
            "grounded_v8",
            "grounded_v9",
            "grounded_v10",
            "grounded_v11",
            "grounded_v12",
            "grounded_v13",
            "grounded_v14",
            "grounded_v15",
            "grounded_v16",
            "grounded_v17",
            "grounded_v18",
            "v2c",
            "v2d",
            "reasoned_v3",
            "synthesis_v4",
            "synthesis_v4_enumerate",
        }:
            raise ValueError(f"unknown answer policy {answer_policy!r}")
        self.answer_policy = answer_policy
        # Attempt 1 bundled three answerer changes and moved in two directions.
        # `current_state` is the one operation deterministic arithmetic never touches —
        # `compute` returns computed=False for it — so its -6.1 points on clean rows can
        # only come from the prompt or from this rendering. Making it switchable is what
        # separates them, and costs one arm rather than three.
        self.timeline_rendering = timeline_rendering
        # v4.1. Injected rather than constructed here so this module stays free of the
        # encoder the router needs, and so an arm without it is byte-identical.
        self.scanner = scanner
        self._scan_route = None
        # One table rather than three parallel conditionals: the previous shape let the
        # schema drift away from the parser without anything noticing.
        self.answer_system, self.answer_prompt_version, self.verdict_schema = {
            "v2": (ANSWER_SYSTEM, ANSWER_PROMPT_VERSION, AnswerVerdict),
            "v2_notes": (NOTED_ANSWER_SYSTEM, NOTED_ANSWER_PROMPT_VERSION, NotedAnswerVerdict),
            "v2_cl": (COUNT_LATEST_SYSTEM, COUNT_LATEST_PROMPT_VERSION, AnswerVerdict),
            "v2_clt": (COUNT_LATEST_SYSTEM, COUNT_TIME_PROMPT_VERSION, AnswerVerdict),
            "grounded_v1": (GROUNDED_SYSTEM, GROUNDED_PROMPT_VERSION, GroundedVerdict),
            "grounded_v2": (
                GroundedAnswererV2.system,
                GroundedAnswererV2.prompt_version,
                GroundedVerdict,
            ),
            "grounded_v3": (
                GroundedAnswererV3.system,
                GroundedAnswererV3.prompt_version,
                GroundedVerdict,
            ),
            "grounded_v4": (
                GroundedAnswererV4.system,
                GroundedAnswererV4.prompt_version,
                GroundedVerdict,
            ),
            "grounded_v5": (
                GroundedAnswererV5.system,
                GroundedAnswererV5.prompt_version,
                GroundedVerdict,
            ),
            "grounded_v6": (
                GroundedAnswererV6.system,
                GroundedAnswererV6.prompt_version,
                GroundedVerdictV6,
            ),
            "grounded_v7": (
                GroundedAnswererV7.system,
                GroundedAnswererV7.prompt_version,
                GroundedVerdictV7,
            ),
            "grounded_v8": (
                GroundedAnswererV8.system,
                GroundedAnswererV8.prompt_version,
                GroundedVerdictV7,
            ),
            "grounded_v9": (
                GroundedAnswererV9.system,
                GroundedAnswererV9.prompt_version,
                GroundedVerdictV7,
            ),
            "grounded_v10": (
                GroundedAnswererV10.system,
                GroundedAnswererV10.prompt_version,
                GroundedVerdictV7,
            ),
            "grounded_v11": (
                GroundedAnswererV11.system,
                GroundedAnswererV11.prompt_version,
                GroundedVerdictV7,
            ),
            "grounded_v12": (
                GroundedAnswererV12.system,
                GroundedAnswererV12.prompt_version,
                GroundedVerdictV7,
            ),
            "grounded_v13": (
                GroundedAnswererV13.system,
                GroundedAnswererV13.prompt_version,
                GroundedVerdictV7,
            ),
            "grounded_v14": (
                GroundedAnswererV14.system,
                GroundedAnswererV14.prompt_version,
                GroundedVerdictV7,
            ),
            "grounded_v15": (
                GroundedAnswererV15.system,
                GroundedAnswererV15.prompt_version,
                GroundedVerdictV7,
            ),
            "grounded_v16": (
                GroundedAnswererV16.system,
                GroundedAnswererV16.prompt_version,
                GroundedVerdictV7,
            ),
            "grounded_v17": (
                GroundedAnswererV17.system,
                GroundedAnswererV17.prompt_version,
                GroundedVerdictV7,
            ),
            "grounded_v18": (
                GroundedAnswererV18.system,
                GroundedAnswererV18.prompt_version,
                GroundedVerdictV7,
            ),
            "reasoned_v3": (
                REASONED_ANSWER_SYSTEM,
                REASONED_ANSWER_PROMPT_VERSION,
                ReasonedAnswerVerdict,
            ),
            "synthesis_v4": (
                SYNTHESIS_ANSWER_SYSTEM,
                SYNTHESIS_ANSWER_PROMPT_VERSION,
                SynthesisVerdict,
            ),
            "synthesis_v4_enumerate": (
                SYNTHESIS_ENUMERATE_ANSWER_SYSTEM,
                SYNTHESIS_ENUMERATE_PROMPT_VERSION,
                EnumeratingSynthesisVerdict,
            ),
            "v2c": (V2C_ANSWER_SYSTEM, V2C_ANSWER_PROMPT_VERSION, AnswerVerdict),
            "v2d": (V2D_ANSWER_SYSTEM, V2D_ANSWER_PROMPT_VERSION, V2DVerdict),
        }[answer_policy]
        # Labels are rendered only for the arm that cites them. An arm that carries them
        # without using them would differ from its control by a changed context and
        # nothing else, which is a second variable bought for no mechanism.
        self.label_context = answer_policy in {"synthesis_v4_enumerate", "v2d"}
        self._context_labels: set[str] = set()
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
        self._base_max_output_tokens = max_output_tokens
        self._answer_mode: str | None = None
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

        # v5.0. Raw conversation stops being a second pass paid for on demand and
        # becomes evidence the first call already has. The offline gate found three of
        # v2c's eleven failures were a sentence the extractor dropped from a session
        # retrieval had already selected — "I just received my new tennis racket
        # today", "up to 50 hours per week", "February 14th" — and in thirty-eight of
        # forty-eight questions no raw turn reached the reader at all.
        #
        # Deliberately `recover_local_detail`: it ranks turns inside the located
        # sessions and never falls through to another conversation. Archive-wide
        # retrieval is what v2b measured as noisy, and it stays where it is, behind the
        # answerer's own request.
        # v2e. Say which dates the facts state and which are only when they were
        # mentioned. Nothing else about the arm differs from v2c.
        self.date_provenance = date_provenance
        # Raw-primary arms (`results/prereg-raw-primary-heldout100-v1.md`): verbatim
        # turns found in the whole archive by the question, attached before the first
        # answer call. `raw_primary_only` drops the memory context and keeps only them.
        self.raw_primary_tokens = max(0, raw_primary_tokens)
        self.raw_primary_only = raw_primary_only
        # v2 of the arm (`results/prereg-raw-primary-dev100-v3.md`): fuse BM25 with a
        # dense turn ranking, and date each conversation relative to the question.
        # Both off by default, so the registered v1 arm is unchanged.
        self.raw_primary_turn_index = raw_primary_turn_index
        self.raw_primary_dated = raw_primary_dated
        # v4 (`results/prereg-raw-primary-dev100-v4.md`): the turns the retrieved
        # memories point at vote in the fusion, and turns are ranked lexically under
        # their text plus the facts anchored to them.
        self.raw_primary_memory_fusion = raw_primary_memory_fusion
        self.raw_primary_fact_keys = raw_primary_fact_keys
        self.raw_primary_time_notes = raw_primary_time_notes
        self.raw_primary_prefer_user = raw_primary_prefer_user
        self.raw_primary_time_window = raw_primary_time_window
        if raw_primary_only and not self.raw_primary_tokens:
            raise ValueError("raw_primary_only needs raw_primary_tokens")
        self.parallel_raw = None
        self.parallel_raw_planned = parallel_raw_planned
        self.parallel_raw_windows = max(0, parallel_raw_windows)
        if self.parallel_raw_windows:
            from llm_long_term_memory.retrieve.fallback import RawFallback

            self.parallel_raw = RawFallback(store, max_turns=self.parallel_raw_windows)
        self.extractor_version = (
            store.get_meta("extractor_version") if hasattr(store, "get_meta") else None
        )
        # What the answers were produced *from*. The extractor version alone does not
        # separate a pilot run against a partly-ingested store from a formal run
        # against the finished one — same code, different evidence — so the memory
        # count is part of the identity. `run_eval` refuses to resume across a change
        # in this string.
        self.store_fingerprint = f"{self.extractor_version or 'unversioned'}@{store.count()}"
        self.grounded = (
            {
                "grounded_v1": GroundedAnswerer,
                "grounded_v2": GroundedAnswererV2,
                "grounded_v3": GroundedAnswererV3,
                "grounded_v4": GroundedAnswererV4,
                "grounded_v5": GroundedAnswererV5,
                "grounded_v6": GroundedAnswererV6,
                "grounded_v7": GroundedAnswererV7,
                "grounded_v8": GroundedAnswererV8,
                "grounded_v9": GroundedAnswererV9,
                "grounded_v10": GroundedAnswererV10,
                "grounded_v11": GroundedAnswererV11,
                "grounded_v12": GroundedAnswererV12,
                "grounded_v13": GroundedAnswererV13,
                "grounded_v14": GroundedAnswererV14,
                "grounded_v15": GroundedAnswererV15,
                "grounded_v16": GroundedAnswererV16,
                "grounded_v17": GroundedAnswererV17,
                "grounded_v18": GroundedAnswererV18,
            }[answer_policy](
                client,
                model=model,
                encoder=encoder,
                store=store,
                turn_index=raw_primary_turn_index,
                fallback=self.fallback,
                chars_per_token=chars_per_token,
                max_output_tokens=max_output_tokens,
            )
            if answer_policy
            in {
                "grounded_v1",
                "grounded_v2",
                "grounded_v3",
                "grounded_v4",
                "grounded_v5",
                "grounded_v6",
                "grounded_v7",
                "grounded_v8",
                "grounded_v9",
                "grounded_v10",
                "grounded_v11",
                "grounded_v12",
                "grounded_v13",
                "grounded_v14",
                "grounded_v15",
                "grounded_v16",
                "grounded_v17",
                "grounded_v18",
            }
            else None
        )

    def prepare(self, instance: Instance) -> None:
        """Nothing per-instance: the store is built once by `lltm ingest`.

        This is the structural difference from the baselines, which rebuild an index
        for every question. It is also why this variant's cost is not visible in the
        per-question latency — ingestion is paid once, up front.
        """

    def _retrieve(self, query: np.ndarray, question: str, namespace: str) -> list[RetrievedMemory]:
        return self._retrieve_with_trace(query, question, namespace)[0]

    def _retrieve_with_trace(
        self,
        query: np.ndarray,
        question: str,
        namespace: str,
        limit: int | None = None,
        as_of: datetime | None = None,
    ):
        ranked, trace = self.retriever.retrieve_with_trace(
            query,
            question,
            namespace,
            temporal=self.temporal,
            limit=limit or self.top_k,
            as_of=as_of,
        )
        if self.scanner is None:
            return ranked, trace
        # The scan runs after ranking and truncation on purpose: it enumerates a set the
        # ranking was never going to complete, and letting it compete inside the ranking
        # would mean scoring enumerated members against retrieved ones on a similarity
        # they do not have.
        outcome = self.scanner.expand(question, namespace, ranked)
        self._scan_route = {
            "routed": outcome.route.routed,
            "predicted_class": outcome.route.predicted_class,
            "margin": round(outcome.route.margin, 4),
            "added": outcome.added,
        }
        return outcome.memories, trace

    def retrieve(self, instance: Instance) -> list[RetrievedMemory]:
        """Retrieve exactly the candidates used by normal answering."""
        query = self.encoder.encode_one(instance.question)
        return self._retrieve(query, instance.question, instance.store_namespace)

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

    def answer(self, instance: Instance, *, limit: int | None = None) -> Answer:
        """The evaluation interface: one benchmark instance in, one scored answer out.

        Everything the engine below needs is the question, the day it is asked and the
        tenant it is asked of. `answer_session_ids` is gold and is passed separately,
        because only two things use it — the validation-only oracle context, and the
        recall note the report is written from — and neither is part of answering.
        """
        return self.answer_request(
            AnswerRequest(
                question=instance.question,
                asked_on=instance.question_date,
                user_id=instance.store_namespace,
            ),
            limit=limit,
            evidence_session_ids=tuple(instance.answer_session_ids),
        )

    def answer_request(
        self,
        request: AnswerRequest,
        *,
        limit: int | None = None,
        evidence_session_ids: tuple[str, ...] = (),
    ) -> Answer:
        """Answer a live question. No benchmark object reaches this path.

        This is the seam the product uses: the API used to fabricate an `Instance`
        with an empty gold answer to get here, which made the answer path unreadable
        without the benchmark and put a field named `answer` — meaning the correct
        one — inside a live request.
        """
        # Reset per question, not per runner. A derivation left over from the previous
        # question would be recorded against this one, and it would look plausible.
        self._computation: dict | None = None
        self._answer_was_raw_structure = False
        self._scan_route = None
        if self.answer_policy in {"v2_cl", "v2_clt"}:
            # Per question: a count or total (and under v2_clt a question about dates,
            # durations or order) is answered from notes, with room for them; every
            # other question keeps v2's verdict and output limit.
            counting = asks_for_aggregate(request.question)
            timing = self.answer_policy == "v2_clt" and asks_about_time(request.question)
            noted = counting or timing
            self._answer_mode = (
                "_".join(m for m, on in (("count", counting), ("time", timing)) if on) + "_notes"
                if noted
                else "latest_only"
            )
            self.answer_system = (
                count_time_system(counting, timing)
                if self.answer_policy == "v2_clt"
                else COUNT_NOTES_SYSTEM
                if counting
                else COUNT_LATEST_SYSTEM
            )
            self.verdict_schema = NotedAnswerVerdict if noted else AnswerVerdict
            self.max_output_tokens = (
                max(1024, self._base_max_output_tokens) if noted else self._base_max_output_tokens
            )
        now = datetime.now()
        if self.decay_enabled:
            from llm_long_term_memory.lifecycle import apply_decay

            apply_decay(
                self.store,
                request.user_id,
                now=now,
                halflife_days=self.decay_halflife_days,
            )
        retrieval_started = perf_counter()
        query_vector = self.encoder.encode_one(request.question)
        retrieved, trace = self._retrieve_with_trace(
            query_vector,
            request.question,
            request.user_id,
            limit=limit,
            # The question's own date, not the wall clock. On this benchmark the
            # difference is total: on a corpus measured against today, 0 of 2,550 memories
            # score above 0.01 at the shipped half-life; measured against the corpus
            # itself, 182 do.
            as_of=_as_of(request.asked_on),
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
            for memory in self.store.iter_all(request.user_id):
                if memory.source_session_id:
                    by_session.setdefault(memory.source_session_id, []).append(memory)
            forced_sessions = None
            if self.oracle_session_context:
                internal_by_external = {
                    external_session_id(session_id): session_id for session_id in by_session
                }
                forced_sessions = [
                    internal_by_external[session_id]
                    for session_id in evidence_session_ids
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
            packed = self._pack(selected, retrieved_by_id, request.question)
            selected = packed.selected

        if self.grounded is not None:
            answer = self.grounded.answer(request, selected, query_vector)
            if self.decay_enabled:
                self.store.record_access(
                    [memory.id for memory in selected], now, reinforcement=self.reinforcement
                )
            calls = answer.notes.get("grounded_calls", [])
            final_sources = calls[-1]["evidence"]["sources"] if calls else []
            present = _external_session_ids(s["session_id"] for s in final_sources)
            answer.notes.update(
                {
                    "top_k": limit or self.top_k,
                    "temporal": self.temporal,
                    "evidence_recalled": bool(set(evidence_session_ids) & present),
                    "source_session_recalled": bool(set(evidence_session_ids) & present),
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "extractor_version": self.extractor_version,
                }
            )
            return answer

        v2c_plan = None
        if self.answer_policy in {"v2c", "v2d"}:
            from .v2c import plan as plan_v2c

            v2c_plan = plan_v2c(request.question, selected)
            selected = v2c_plan.memories

        operation = (
            reasoning_kind(request.question) if self.answer_policy == "reasoned_v3" else None
        )
        hydrate_for_reasoning = self.adaptive_reasoning_hydration and operation in {
            "temporal",
            "multi_session_aggregation",
            "current_state",
        }
        hydrate_for_detail = False
        context, hydration = self._assemble_context(
            selected, force_hydration=hydrate_for_reasoning or hydrate_for_detail
        )
        if v2c_plan and v2c_plan.context_note:
            context = f"{v2c_plan.context_note}\n\n{context}"
        detail_evidence = None
        if v2c_plan and v2c_plan.detail_hydration_reason and self.fallback is not None:
            detail_evidence = self.fallback.recover_local_detail(
                request.user_id, request.question, selected
            )
            if detail_evidence.used:
                context = (
                    f"{context}\n\nQuestion-specific verbatim source evidence:\n"
                    f"{detail_evidence.render(max_chars=self.raw_fallback_max_chars)}"
                )
        parallel_evidence = None
        if self.parallel_raw is not None:
            windows = self._parallel_raw_budget(request.question)
            if windows:
                self.parallel_raw.max_turns = windows
                parallel_evidence = self.parallel_raw.recover_local_detail(
                    request.user_id, request.question, selected
                )
                if parallel_evidence.used:
                    # Labelled as the conversation itself, and placed after the
                    # structured facts rather than instead of them. The point of the
                    # arm is that both are present on the first call; burying the
                    # memories under transcript would be a slide back into naive RAG,
                    # which `two_stage_hydrated` already measured as three times the
                    # context for no gain.
                    context = (
                        f"{context}\n\nVerbatim turns from those same conversations:\n"
                        f"{parallel_evidence.render(max_chars=self.raw_fallback_max_chars)}"
                    )
        raw_primary = None
        if self.raw_primary_tokens:
            from llm_long_term_memory.retrieve.excerpts import archive_excerpts

            raw_primary = archive_excerpts(
                self.store,
                request.user_id,
                request.question,
                self.raw_primary_tokens,
                chars_per_token=self.chars_per_token,
                turn_index=self.raw_primary_turn_index,
                query_vector=self.encoder.encode_one(request.question)
                if self.raw_primary_turn_index is not None
                else None,
                asked_on=_as_of(request.asked_on)
                if self.raw_primary_dated or self.raw_primary_time_window
                else None,
                memory_anchors=[
                    (hit.memory.source_session_id, hit.memory.source_turn_index)
                    for hit in retrieved
                    if hit.memory.source_session_id and hit.memory.source_turn_index is not None
                ]
                if self.raw_primary_memory_fusion
                else None,
                fact_keys=self.raw_primary_fact_keys,
                time_notes=self.raw_primary_time_notes,
                prefer_user=self.raw_primary_prefer_user,
                time_window=self.raw_primary_time_window,
            )
            block = raw_primary.render()
            if self.raw_primary_only:
                context = block
            elif block:
                context = f"{context}\n\n{block}"
        assembly_latency_ms = (perf_counter() - assembly_started) * 1000
        if self.fallback is not None or self.answer_policy == "v2d":
            completion, answer_text, verdict, raw_evidence = self._answer_with_fallback(
                request, context, selected
            )
        else:
            completion = self._complete(request, context)
            answer_text, verdict, raw_evidence = completion.text.strip(), None, None
        if self.decay_enabled:
            self.store.record_access(
                [memory.id for memory in selected], now, reinforcement=self.reinforcement
            )

        parallel_notes = {
            "raw_primary_tokens_budget": self.raw_primary_tokens,
            "raw_primary_only": self.raw_primary_only,
            "raw_primary_hybrid": self.raw_primary_turn_index is not None,
            "raw_primary_dated": self.raw_primary_dated,
            "raw_primary_memory_fusion": self.raw_primary_memory_fusion,
            "raw_primary_fact_keys": self.raw_primary_fact_keys,
            "raw_primary_time_notes": self.raw_primary_time_notes,
            "raw_primary_prefer_user": self.raw_primary_prefer_user,
            "raw_primary_time_window": self.raw_primary_time_window,
            "raw_primary_window_turns": raw_primary.window_turns if raw_primary else 0,
            "answer_mode": self._answer_mode,
            "raw_primary_turns": len(raw_primary.turns) if raw_primary else 0,
            "raw_primary_tokens": raw_primary.tokens if raw_primary else 0,
            "raw_primary_sessions": [external_session_id(s) for s in raw_primary.sessions]
            if raw_primary
            else [],
            "parallel_raw_windows": self.parallel_raw_windows,
            "parallel_raw_planned": self.parallel_raw_planned,
            "parallel_raw_level": parallel_evidence.level if parallel_evidence else None,
            "parallel_raw_turns": len(parallel_evidence.turns) if parallel_evidence else 0,
            "parallel_raw_sessions": sorted({turn.session_id for turn in parallel_evidence.turns})
            if parallel_evidence
            else [],
        }

        evidence = set(evidence_session_ids)
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
                "top_k": limit or self.top_k,
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
                    if self.evidence_hydration or hydrate_for_reasoning or hydrate_for_detail
                    else None,
                },
                "recall_coverage": {
                    "candidates": _session_coverage(evidence, candidate_sessions),
                    "ranked": _session_coverage(evidence, ranked_sessions),
                    "selected": _session_coverage(evidence, selected_sessions),
                    "hydrated": _session_coverage(evidence, hydrated_sessions)
                    if self.evidence_hydration or hydrate_for_reasoning or hydrate_for_detail
                    else None,
                },
                "all_source_sessions_recalled": (
                    _session_coverage(evidence, selected_sessions) == 1.0 if evidence else None
                ),
                # Ranked before the context budget chose among them. Without this the
                # rows cannot separate "retrieval never found the fact" from "retrieval
                # found it and composition dropped it", and that distinction is
                # unrecoverable after the run — the two call for opposite fixes.
                "ranked_memory_ids": [hit.memory.id for hit in retrieved],
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
                "v2c": None
                if v2c_plan is None
                else {
                    "suppressed_update_ids": v2c_plan.suppressed_update_ids,
                    "timeline_ids": v2c_plan.timeline_ids,
                    "detail_hydration_reason": v2c_plan.detail_hydration_reason,
                    "detail_hydration_applied": bool(detail_evidence and detail_evidence.used),
                    "detail_source_turns": [
                        f"{external_session_id(turn.session_id)}:{turn.turn_index}"
                        for turn in (detail_evidence.turns if detail_evidence else [])
                    ],
                },
                "reasoning_kind": operation,
                **parallel_notes,
                "answer_confidence": getattr(verdict, "confidence", None),
                "answer_evidence_summary": getattr(verdict, "evidence_summary", None),
                "answer_calculation": getattr(verdict, "calculation", None),
                # What the code computed, and from which operands. Without this a
                # v4 row cannot be audited: "3 items" and "3 items after folding two
                # spellings" are the same answer with different reasons to trust it.
                "synthesis_operation": getattr(verdict, "operation", None),
                "synthesis_computation": self._computation,
                # Recorded because a stop condition depends on it: an abstention that
                # became a computed number is the failure v4.0 is most likely to cause,
                # and it cannot be counted from a field nothing writes down.
                "synthesis_missing_field": getattr(verdict, "missing_field", None) or None,
                # Separates "the model named an absent operand" from "the model wrote its
                # deliberation into the field". Three of five populated values were the
                # second kind on v4.0-flat, and they blocked computations that should
                # have run — one signal covering two behaviours cannot be judged.
                "missing_field_was_narration": missing_field_is_narration(
                    getattr(verdict, "missing_field", "") or ""
                ),
                "answer_was_raw_structure": self._answer_was_raw_structure,
                # Present only for the v4.1 arm. `routed: false` is the expected value on
                # roughly three questions in four and is not a failure.
                "scan_route": self._scan_route,
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

    def _parallel_raw_budget(self, question: str) -> int:
        """How many verbatim turns to attach before the first call.

        Fixed by default, so the arm changes exactly one thing. The planned variant
        spends by what the question asks for: an aggregation has to enumerate members
        across conversations and a lookup needs one sentence, and the registered
        comparison between the two is whether the extra tokens buy anything.
        """
        if not self.parallel_raw_planned:
            return self.parallel_raw_windows
        kind = reasoning_kind(question)
        # Measured, not guessed. Replaying the three recoverable failures against the
        # committed store showed where each sentence sits in the source-local ranking:
        # the tennis-racket turn is reachable at two windows, the Valentine's Day turn
        # at three, and the "up to 50 hours per week" turn not until six — it is an
        # assistant summary that ranks sixth among the turns of the twelve sessions the
        # question selected. Rendering caps each turn at its query-relevant window, so
        # six windows cost about 2,500 characters rather than the 9,400 the raw turns
        # hold. An aggregation is also the kind that has to enumerate across
        # conversations, which is why it is the one that gets the wider budget.
        return {
            "direct": 2,
            "preference_application": 2,
            "temporal": 3,
            "current_state": 3,
            "multi_session_aggregation": 6,
        }.get(kind, self.parallel_raw_windows)

    def answer_with_memories(self, instance: Instance, memories: list[Memory]) -> str:
        """Answer a fixed memory set without retrieval or lifecycle side effects.

        Influence measurement removes one memory at a time and asks again, so this
        takes the set rather than finding it. It keeps the evaluation signature
        because that is the only thing that calls it.
        """
        context, _ = self._assemble_context(memories)
        completion = self._complete(
            AnswerRequest(
                question=instance.question,
                asked_on=instance.question_date,
                user_id=instance.store_namespace,
            ),
            context,
        )
        return completion.text.strip()

    def _assemble_context(
        self, memories: list[Memory], *, force_hydration: bool = False
    ) -> tuple[str, HydrationResult]:
        labels = label_memories(memories) if self.label_context else None
        # Recorded per question so `_apply_computation` can resolve citations against the
        # context this answer actually saw, rather than against whatever is in scope now.
        self._context_labels = set(labels.values()) if labels else set()
        body = render_grouped(
            memories, self.temporal, self.timeline_rendering, labels, self.date_provenance
        )
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

    def _readable_fallback_answer(self, text: str) -> str:
        """Apply the existing object/fence guard at every fallback return boundary.

        This detects output shape, not factual correctness. Keep the diagnostic
        separate from the final text so a successful computation is never discarded
        merely because the model's earlier answer field was structured.
        """
        text = text.strip()
        if text.startswith(("{", "```")):
            self._answer_was_raw_structure = True
            return "I do not know."
        return text or "I do not know."

    def _answer_with_fallback(self, request: AnswerRequest, context: str, selected: list[Memory]):
        """One LLM call when memory suffices, two when it does not.

        Both `need_source` and `no_evidence` can search the raw archive. A second
        model call is made only when that search recovers evidence.
        """
        first = self._complete(request, context, structured=True)
        try:
            verdict = self.verdict_schema.model_validate_json(first.text)
        except ValueError:
            # Preserve prose when the model ignored the schema, but never return
            # an object or fenced payload solely because validation failed.
            return first, self._readable_fallback_answer(first.text), None, None

        if verdict.status == "answer" or self.fallback is None:
            text = verdict.answer.strip() or first.text.strip()
            final = self._apply_computation(verdict, text)
            # Last resort, and a guarantee rather than a request. The prompt now asks for
            # `answer` on every operation, but a prompt is not a contract: when it is
            # empty and nothing was computed, the fallback text is the structure itself,
            # and emitting that is never the right answer to anything. An honest "I do
            # not know" is the truthful rendering of "the model produced no reply", and
            # it is scored as the abstention it is instead of as a wrong answer.
            # Two different questions, and conflating them cost 11 correct answers.
            #
            # The flag asks "did the model emit a structure at any point" — a diagnostic,
            # true even when `compute` or a populated `answer` produced a perfectly good
            # reply afterwards. The repair must ask something narrower: "is the text
            # about to be returned a structure". Driving a destructive action from the
            # diagnostic replaced 16 replies with an abstention on v4.0-flat2, and 11 of
            # them had been scored correct the run before — prose like "The user taking
            # painting classes came first."
            self._answer_was_raw_structure = text.lstrip().startswith(("{", "```"))
            final = self._readable_fallback_answer(final)
            # Measured on the *final* answer, not on the intermediate. The first version
            # of this flag asked "was `answer` empty and the raw text JSON", which is a
            # different question: `compute` then replaced that text with "5 days" and the
            # row was flagged as leaked while reading perfectly. On v4.0-flat it reported
            # 12 where 18 rows actually leaked — 4 false positives and 10 misses, wrong
            # in both directions. A model can also put JSON inside `answer` itself, which
            # the old check could not see at all.
            return first, final, verdict, None

        evidence = self.fallback.recover(
            request.user_id,
            verdict.source_query or request.question,
            selected if verdict.status == "need_source" else [],
        )
        if not evidence.used:
            # Nothing in the archive either. Abstention is correct here, and is a
            # measured strength worth protecting.
            return first, self._readable_fallback_answer(verdict.answer), verdict, evidence

        rendered_evidence = evidence.render(max_chars=self.raw_fallback_max_chars)
        if self.answer_policy == "reasoned_v3":
            prompt = render_reasoned_prompt(
                f"{context}\n\nThe first pass requested source because: "
                f"{verdict.reason or 'the requested detail was missing'}\n\n"
                f"Verbatim source evidence:\n{rendered_evidence}",
                request.asked_on,
                request.question,
            )
        elif self.answer_policy in {"v2c", "v2d"}:
            # v2c's query-time rules and deterministic source hydration live in
            # `context`. The generic fallback template used to discard both, so the
            # final answerer never saw the very repair that triggered the second pass.
            # Keep the complete first-pass evidence plan and add, rather than replace
            # it with, fallback evidence.
            prompt = _TEMPLATE.format(
                context=(
                    f"{context}\n\nThe first pass requested more source because: "
                    f"{verdict.reason or 'the requested detail was missing'}\n\n"
                    f"Additional verbatim source evidence:\n{rendered_evidence}\n\n"
                    "Final exact-reference reminder: if no candidate satisfies every "
                    "qualifier, do not force a match. State the source conflict and "
                    "distinguish the candidates."
                ),
                date=request.asked_on,
                question=request.question,
            )
        else:
            prompt = self._FALLBACK_TEMPLATE.format(
                reason=verdict.reason or "the requested detail was missing",
                evidence=rendered_evidence,
                date=request.asked_on,
                question=request.question,
            )
        second = self.client.generate(
            role="answerer",
            model=self.model,
            prompt=prompt,
            # The fallback call carries no schema, so a system prompt that demands a
            # structured verdict has nothing to parse it back out: under `synthesis_v4`
            # the model dutifully emitted JSON and it became the answer text on 9 of the
            # 18 leaked rows. The second pass only ever needs prose.
            system=(
                V2C_ANSWER_SYSTEM
                if self.answer_policy == "v2d"
                else ANSWER_SYSTEM
                if self.answer_policy in {"synthesis_v4", "synthesis_v4_enumerate"}
                else COUNT_LATEST_SYSTEM
                if self.answer_policy in {"v2_cl", "v2_clt"}
                else self.answer_system
            ),
            temperature=0.0,
            max_output_tokens=self.max_output_tokens,
            est_input_tokens=int(len(prompt) / self.chars_per_token),
        )
        return second, self._readable_fallback_answer(second.text), verdict, evidence

    def _apply_computation(self, verdict, text: str) -> str:
        """Let the code overwrite the model's arithmetic, where there is arithmetic.

        Only under `synthesis_v4`, and only when the operands were actually supplied.
        `compute` returns `computed=False` for a missing or malformed operand, and the
        model's own prose is kept in that case: a number derived from nothing would be
        worse than a guess, because it would arrive looking checked.

        `comparison` is computed but has no sentence of its own — knowing which date is
        earlier does not say which described event it belongs to — so its derivation is
        recorded and the wording is left alone.
        """
        if self.answer_policy == "v2d":
            result = compute_v2d(verdict, self._context_labels)
            self._computation = result.detail
            return result.answer if result.computed and result.answer else text
        if self.answer_policy not in {"synthesis_v4", "synthesis_v4_enumerate"}:
            return text
        result = compute(
            verdict,
            self._context_labels if self.answer_policy == "synthesis_v4_enumerate" else None,
        )
        self._computation = result.detail
        if result.computed and result.answer:
            return result.answer
        return text

    def _complete(self, request: AnswerRequest, context: str, structured: bool = False):
        prompt = (
            render_reasoned_prompt(context, request.asked_on, request.question)
            if self.answer_policy == "reasoned_v3"
            else _TEMPLATE.format(
                context=context,
                date=request.asked_on,
                question=request.question,
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
            # `self.verdict_schema`, not the base class. Sending the base schema while
            # parsing with the subclass is silent: every extra field carries a default,
            # so validation succeeds and the model is simply never asked. That is what
            # happened to the whole v3 line — `confidence` read 'medium' on 100% of rows
            # in every phase because it was the default, which made the registered
            # `no_new_confident_errors` gate pass without testing anything
            # (results/audit/v3-verdict-schema-never-sent-20260906.json).
            **({"schema": self.verdict_schema} if structured else {}),
        )
        return completion
