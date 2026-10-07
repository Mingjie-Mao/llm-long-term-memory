"""Exhaustive, tenant-scoped archive review under a per-page context budget.

Ranking cannot establish that an aggregate has every item. A caller opting into this
mode pays for bounded pages; reaching the cap is an explicit incomplete review.
Pages contain whole original turns and their dates. No summaries or benchmark labels.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace

from pydantic import BaseModel, Field

from llm_long_term_memory.ingest.event_time import temporal_evidence
from llm_long_term_memory.runtime.grounded_answering import (
    EvidenceLedger,
    EvidenceSource,
    add_turns,
)


class PageSelection(BaseModel):
    selected_sources: list[str] = Field(default_factory=list)
    uncertain_sources: list[str] = Field(default_factory=list)
    reviewed_sources: list[str] = Field(default_factory=list)
    scope_complete: bool = False


class PageSelectionError(ValueError):
    def __init__(self, response, cause="invalid_page_schema"):
        super().__init__(cause)
        self.response = response


@dataclass
class ArchivePages:
    pages: list[EvidenceLedger] = field(default_factory=list)
    omitted_turn_ids: list[str] = field(default_factory=list)
    total_turns: int = 0

    @property
    def complete(self):
        return not self.omitted_turn_ids


@dataclass(frozen=True)
class RawFragment(EvidenceSource):
    original_source_id: str = ""
    char_start: int = 0
    char_end: int = 0
    original_text_sha256: str = ""

    def render(self):
        return (
            super()
            .render()
            .replace("; said ", f"; original span {self.char_start}:{self.char_end}; said ", 1)
        )


def turn_sources(store, user_id, turn, *, max_tokens, chars_per_token, split_oversized):
    whole = EvidenceLedger(max_tokens=max_tokens, chars_per_token=chars_per_token)
    if add_turns(whole, store, user_id, [turn]):
        return whole.sources
    if not split_oversized:
        return []
    # Retain exact contiguous spans, not generated summaries or truncated evidence.
    roomy = EvidenceLedger(max_tokens=len(turn.content) + 4096, chars_per_token=1)
    add_turns(roomy, store, user_id, [turn])
    base = roomy.sources[0]
    span_budget = int(max_tokens * chars_per_token) - 512
    if span_budget <= 0:
        return []
    result, start = [], 0
    while start < len(turn.content):
        end = min(len(turn.content), start + span_budget)
        if end < len(turn.content):
            boundary = turn.content.rfind(" ", start, end)
            if boundary > start + span_budget // 2:
                end = boundary + 1
        session = store.get_session(turn.session_id)
        time = temporal_evidence(turn.content[start:end], session.started_at)

        def stamp(value):
            return value.strftime("%Y-%m-%d") if value else None

        result.append(
            RawFragment(
                **{
                    **base.__dict__,
                    "source_id": f"{turn.id}#span:{start}:{end}",
                    "text": turn.content[start:end],
                    "event_date": stamp(time.exact or time.estimate),
                    "date_expression": time.expression,
                    "date_precision": time.precision,
                },
                original_source_id=turn.id,
                char_start=start,
                char_end=end,
                original_text_sha256=hashlib.sha256(turn.content.encode()).hexdigest(),
            )
        )
        start = end
    return result


def archive_pages(
    store,
    user_id,
    *,
    max_tokens=6000,
    max_pages=32,
    chars_per_token=4.6,
    roles=None,
    split_oversized=False,
):
    if max_tokens <= 0 or max_pages <= 0:
        raise ValueError("page and context budgets must be positive")
    result = ArchivePages()
    page = EvidenceLedger(max_tokens=max_tokens, chars_per_token=chars_per_token)
    for sid in sorted(store.session_ids_for_user(user_id)):
        session = store.get_session(sid)
        if session is None or session.user_id != user_id:
            raise ValueError("archive page crossed tenant boundary")
        for turn in sorted(store.turns_for_session(sid), key=lambda t: t.turn_index):
            if roles is not None and turn.role not in roles:
                continue
            result.total_turns += 1
            sources = turn_sources(
                store,
                user_id,
                turn,
                max_tokens=max_tokens,
                chars_per_token=chars_per_token,
                split_oversized=split_oversized,
            )
            if not sources:
                result.omitted_turn_ids.append(turn.id)
            for source in sources:
                if len(result.pages) >= max_pages:
                    result.omitted_turn_ids.append(source.source_id)
                    continue
                if page.add(replace(source, id=f"E{len(page.sources) + 1}")):
                    continue
                if page.sources:
                    result.pages.append(page)
                    page = EvidenceLedger(max_tokens=max_tokens, chars_per_token=chars_per_token)
                if len(result.pages) >= max_pages or not page.add(replace(source, id="E1")):
                    result.omitted_turn_ids.append(source.source_id)
    if page.sources:
        result.pages.append(page)
    return result


SYSTEM = """Review original archived conversation evidence for one memory question.
This is evidence selection, not an answer. Select every potentially relevant source,
including competing values, historical updates, repeated event identity, start/end
events, uncertain dates and adjacent context needed to resolve pronouns. Never select
only the latest report when a question asks about the past or a total. Assistant advice
is not proof the user performed it. Do not calculate or guess missing facts.
Select primitive events BEFORE testing a cross-page relationship. For consecutive-day,
event-order, duration or total questions, retain all matching events and endpoints,
even if this page alone cannot establish the requested pair/order/total. Never discard
an event because its matching partner may be on another page. Uncertain potentially
matching events belong in uncertain_sources. Select first; copy the full receipt last.
reviewed_sources must list every evidence ID on THIS page exactly once. selected_sources
and uncertain_sources contain only IDs on this page. Include unresolved potentially
relevant sources in uncertain_sources. scope_complete concerns only this page; other
pages may contain additional events or adjacent spans of an oversized turn. A span
may lack context: retain it as uncertain if relevant. The original words are authoritative.
"""


def select_page(client, model, page, request, max_output_tokens=2048):
    context = page.render()
    reviewed_ids = [s.id for s in page.sources]
    receipt = json.dumps(reviewed_ids)
    response = client.generate(
        role="answerer",
        model=model,
        system=SYSTEM,
        prompt=(
            f"{context}\n\nQuestion date: {request.asked_on}\nQuestion: {request.question}"
            f"\nRequired reviewed_sources (copy this entire list exactly; "
            f"not only relevant sources or the tail of the page): {receipt}"
        ),
        schema=PageSelection,
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        est_input_tokens=int(len(context) / page.chars_per_token),
    )
    try:
        verdict = PageSelection.model_validate_json(response.text)
    except ValueError as exc:
        raise PageSelectionError(response) from exc
    ids = {s.id for s in page.sources}
    if not verdict.scope_complete:
        raise PageSelectionError(response, "page_review_incomplete")
    if set(verdict.reviewed_sources) != ids or len(verdict.reviewed_sources) != len(ids):
        raise PageSelectionError(response, "page_review_missing_or_duplicate_ids")
    if not set(verdict.selected_sources + verdict.uncertain_sources) <= ids:
        raise PageSelectionError(response, "page_selection_foreign_ids")
    selected = set(verdict.selected_sources + verdict.uncertain_sources)
    return [s.source_id for s in page.sources if s.id in selected], response, verdict
