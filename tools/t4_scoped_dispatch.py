"""Run accepted t4 unchanged unless a source-bound scoped repair is applicable."""

from __future__ import annotations

import re

from t4_event_count_repair import unresolved_event_references
from t4_savings_repair import SavingsRepairRunner, quoted_savings
from t4_selection_repair import selection_plan

from llm_long_term_memory.evaluation.grounded.answering import build_ledger
from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
from llm_long_term_memory.ingest.extract import _parse_date
from llm_long_term_memory.retrieve.excerpts import archive_excerpts

VERSION = "memory-t4-scoped-dispatch-v1"


def applicable_repair(runner, request):
    """Inspect only this tenant's sources; no identifiers or gold select the route."""
    packed = archive_excerpts(
        runner.store,
        request.user_id,
        request.question,
        runner.raw_primary_tokens,
        chars_per_token=runner.chars_per_token,
        asked_on=_parse_date(request.asked_on),
        time_notes=True,
        prefer_user=True,
        time_window=True,
    )
    if re.search(r"\bhow much\b.*\bsave\b", request.question, re.I):
        ledger = build_ledger(runner.store, request.user_id, [], packed.turns, max_tokens=6500)
        result = None if ledger.dropped else quoted_savings(ledger, request.question)
        if result:
            return result.detail["mechanism"]
    event, _ = unresolved_event_references(runner.store, request)
    if event:
        return event.detail["mechanism"]
    result, _, _ = selection_plan(runner.store, request, packed.turns)
    return result.detail["mechanism"] if result else None


class ScopedSavingsRepairRunner(SavingsRepairRunner):
    @classmethod
    def from_t4(cls, runner):
        wrapped = super().from_t4(runner)
        wrapped.answer_prompt_version = VERSION
        return wrapped

    def _answer_with_fallback(self, request, context, selected):
        mechanism = applicable_repair(self, request)
        if mechanism is None:
            self._selection_audit = {"active": False, "mechanism": None}
            # Delegating directly avoids the parent repair's unconditional RULES
            # suffix. Preserve the baseline's context, schema and fallback behavior.
            return MemoryRunner._answer_with_fallback(self, request, context, selected)
        return super()._answer_with_fallback(request, context, selected)
