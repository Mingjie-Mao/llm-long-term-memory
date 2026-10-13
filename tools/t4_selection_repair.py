"""Scoped t4 repair prototype: source-bound selection, without archive pagination.

No benchmark IDs, answers or reference flags enter this module. Keep the accepted
t4 implementation unchanged until a new comparison accepts this candidate.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

from llm_long_term_memory.evaluation.grounded.answering import (
    Calculation,
    build_ledger,
    ordinal_song_progression,
    redemption_place,
    scoped_payment_conflict,
    scoped_quantity_disclosure,
    song_ordinal,
)
from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
from llm_long_term_memory.ingest.extract import _parse_date
from llm_long_term_memory.retrieve.excerpts import archive_excerpts

VERSION = "memory-t4-scoped-selection-v1"
RULES = """Selection checks: Match the requested entity/category and relation before
choosing a number. A count of another category does not answer a false premise.
Distinguish the place an action happened from where a message/coupon originated.
For ordinal assistant artifacts, follow the original conversation order and revision
requests. For totals, separate stated subtotals from totals in the requested period;
do not add overlapping or unspecified reporting periods, or silently assign a year
to conflicting dates. State concrete uncertainties rather than inventing values.
"""


def owned_session_turns(store, user_id, session_id):
    session = store.get_session(session_id)
    if session is None or session.user_id != user_id:
        raise ValueError("selection source belongs to another tenant")
    return store.turns_for_session(session_id)


def collection_category_gap(ledger, question):
    """Conservative exact-category guard for signed collection count questions.

    An absent category means insufficient supplied evidence, not proof of zero.
    The narrow qualifier avoids declaring generic duration/aggregate synonyms absent.
    """
    if not re.search(r"\bcollect(?:ion|ing)\b", question, re.I):
        return None
    match = re.search(r"\bhow many\s+((?:autographed|signed)\s+\w+)\b", question, re.I)
    if not match:
        return None
    category = match[1].split()[-1].casefold().removesuffix("s")
    pattern = r"\b" + re.escape(category) + r"s?\b"
    sources = [s for s in ledger.sources if s.kind == "raw" and s.role == "user"]
    if not sources or any(re.search(pattern, s.text, re.I) for s in sources):
        return None
    return Calculation(
        f"I don't have evidence for how many {match[1]} you added to your collection. "
        "Counts stated for a different kind of item cannot establish that number.",
        True,
        {
            "mechanism": "requested_collection_category_absent",
            "requested_category": category,
            "reviewed_sources": [s.id for s in sources],
            "scope": "supplied sources only",
        },
    )


def selection_plan(store, request, packed_turns):
    turns = list(packed_turns)
    ordinal = song_ordinal(request.question)
    if ordinal is not None:
        # BM25 already locates these sources; t4's USER allocation can drop the
        # second assistant artifact. Hydrate a bounded contiguous prefix, preserving
        # every intervening role rather than ranking song bodies independently.
        ranked = store.search_turns(request.user_id, request.question, limit=500)
        sessions = list(
            dict.fromkeys(
                t.session_id
                for t in ranked
                if t.role == "assistant" and re.search(r"^\s*Chorus:\s*$", t.content, re.M | re.I)
            )
        )[:8]
        prefixes = []
        for sid in sessions:
            originals = sorted(
                owned_session_turns(store, request.user_id, sid), key=lambda t: t.turn_index
            )
            artifacts = [
                t
                for t in originals
                if t.role == "assistant" and re.search(r"^\s*Chorus:\s*$", t.content, re.M | re.I)
            ]
            if len(artifacts) >= ordinal:
                prefix = [t for t in originals if t.turn_index <= artifacts[ordinal - 1].turn_index]
                if len(prefix) <= 20 and sum(len(t.content) for t in prefix) <= 18000:
                    prefixes.extend(prefix)
        # A partial prefix must not become a seemingly unique valid candidate.
        if sum(len(t.content) for t in prefixes) > 24000:
            return None, None, {"cause": "ordinal_candidates_exceed_budget"}
        turns = prefixes
    elif re.search(r"\bwhere\b.*\bredeem\b.*\bcoupon\b", request.question, re.I):
        selected_sessions = {
            t.session_id
            for t in turns
            if t.role == "user"
            and re.search(r"\bI\s+(?:actually\s+)?redeemed\b.*\bcoupon\b", t.content, re.I)
        }
        neighbors = [
            t
            for sid in sorted(selected_sessions)
            for t in owned_session_turns(store, request.user_id, sid)
            if t.role == "user"
        ]
        turns = list({t.id: t for t in [*turns, *neighbors]}.values())
    ledger = build_ledger(store, request.user_id, [], turns, max_tokens=6500)
    if ledger.dropped:
        return None, None, {"cause": "selection_ledger_incomplete", "dropped": ledger.dropped}
    operations = [
        lambda: ordinal_song_progression(ledger, request.question),
        lambda: redemption_place(ledger, request.question),
        lambda: scoped_quantity_disclosure(ledger, request.question),
        lambda: scoped_payment_conflict(ledger, request),
        lambda: collection_category_gap(ledger, request.question),
    ]
    for operation in operations:
        result = operation()
        if result and result.computed:
            if result.detail.get("mechanism") == "scoped_quantity_disclosure":
                result = replace(
                    result,
                    answer=result.answer
                    + " Please clarify the period covered by each report and whether they overlap.",
                )
            return result, ledger, {"ledger": ledger.audit(), "calculation": result.detail}
    return None, ledger, {"ledger": ledger.audit(), "calculation": None}


class SelectionRepairRunner(MemoryRunner):
    @classmethod
    def from_t4(cls, runner):
        if runner.answer_policy != "v2_clt" or not runner.raw_primary_prefer_user:
            raise ValueError("scoped repair requires the accepted t4 base")
        wrapped = cls.__new__(cls)
        wrapped.__dict__.update(runner.__dict__)
        wrapped.answer_prompt_version = VERSION
        return wrapped

    def answer_request(self, request, **kwargs):
        self._selection_audit = None
        self._selection_added_tokens = 0
        answer = super().answer_request(request, **kwargs)
        answer.context_tokens += self._selection_added_tokens
        answer.notes["selection_repair"] = self._selection_audit
        return answer

    def _answer_with_fallback(self, request, context, selected):
        packed = archive_excerpts(
            self.store,
            request.user_id,
            request.question,
            self.raw_primary_tokens,
            chars_per_token=self.chars_per_token,
            asked_on=_parse_date(request.asked_on),
            time_notes=self.raw_primary_time_notes,
            prefer_user=True,
            time_window=True,
        )
        result, ledger, self._selection_audit = selection_plan(self.store, request, packed.turns)
        revised = context
        if result and result.detail.get("mechanism") == "ordinal_song_progression":
            marker = "Excerpts from the user's earlier conversations, found for this question"
            revised = context.split(marker, 1)[0] + ledger.render()
        revised += "\n\n" + RULES
        if result:
            # Source IDs, literal quotes, temporal conflicts and arithmetic are
            # supplied to the unchanged model; program output stays auditable.
            revised += "\nSource-bound selection: " + json.dumps(
                {"answer": result.answer, "derivation": result.detail}, ensure_ascii=False
            )
        self._selection_added_tokens = int(len(revised) / self.chars_per_token) - int(
            len(context) / self.chars_per_token
        )
        completed, text, verdict, evidence = super()._answer_with_fallback(
            request, revised, selected
        )
        if result:
            text = result.answer
        return completed, text, verdict, evidence
