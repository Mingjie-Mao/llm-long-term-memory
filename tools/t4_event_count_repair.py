"""Extend the scoped t4 prototype with literal unresolved personal-event references."""

from __future__ import annotations

import re

from t4_selection_repair import SelectionRepairRunner

from llm_long_term_memory.evaluation.grounded.answering import Calculation, build_ledger

VERSION = "memory-t4-scoped-selection-v2"


def scoped_event_category(question):
    match = re.search(r"\bhow many\s+(\w+)\s+(?:have I attended|did I attend)\b", question, re.I)
    if match and re.search(r"\bthis year\b", question, re.I):
        return match[1].casefold().removesuffix("s")
    return None


def unresolved_event_references(store, request):
    category = scoped_event_category(request.question)
    if not category:
        return None, None
    hits = store.search_turns(request.user_id, category, limit=500)
    turns = [t for t in hits if t.role == "user"]
    ledger = build_ledger(store, request.user_id, [], turns, max_tokens=6500)
    mentions = []
    owner = re.compile(r"\bmy\s+((?:[\w'-]+\s+){0,3}[\w'-]+)'s\s+" + re.escape(category), re.I)
    completed = (
        r"\b(?:attended|been to|went to|got back from|wore\b.+?\bto|bridesmaid|part of it)\b"
    )
    date = (
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December|\d{4}|(?:this|last) (?:year|month|week|weekend)|"
        r"today|yesterday)\b"
    )
    for source in ledger.sources:
        # Future plans and somebody else's described event are not attendance.
        if not re.search(completed, source.text, re.I):
            continue
        for match in owner.finditer(source.text):
            left = max(source.text.rfind(c, 0, match.start()) for c in ".!?") + 1
            right = min(
                [p for c in ".!?" if (p := source.text.find(c, match.end())) >= 0]
                or [len(source.text)]
            )
            clause = source.text[left:right]
            words = match[1].split()
            relation = words[-1].casefold()
            # A capitalized personal name already makes this relation identifiable.
            named = len(words) > 1 and words[-1][0].isupper()
            mentions.append(
                {
                    "source": source.id,
                    "source_id": source.source_id,
                    "session_id": source.session_id,
                    "quote": source.text,
                    "owner": match[1],
                    "relation": relation,
                    "named": named,
                    "dated": bool(re.search(date, clause, re.I)),
                    "event_clause": clause.strip(),
                }
            )
    repeated = {
        m["relation"]
        for m in mentions
        if not m["named"]
        and len(
            {n["session_id"] for n in mentions if not n["named"] and n["relation"] == m["relation"]}
        )
        > 1
    }
    undated = [m for m in mentions if not m["dated"]]
    # Do not reject every anaphora or every missing exact day. This concrete
    # conjunction distinguishes the unresolved total from the three named events.
    if not repeated or not undated:
        return None, None
    detail = {
        "mechanism": "unresolved_personal_event_references",
        "category": category,
        "missing_fields": ["event_identity", "event_date", "report_period", "attendance"],
        "repeated_unnamed_relations": sorted(repeated),
        "mentions": mentions,
        "ledger": ledger.audit(),
        "keyword_search_truncated": len(hits) == 500,
    }
    return Calculation(
        f"I cannot establish an exact total of {category}s attended this year from these records. "
        "Some unnamed event references may repeat earlier events, and another participation "
        "report does not specify when it happened or confirm attendance. Please clarify "
        "which references describe the same event, their dates/year, and whether you attended.",
        True,
        detail,
    ), ledger


class EventCountRepairRunner(SelectionRepairRunner):
    @classmethod
    def from_t4(cls, runner):
        wrapped = super().from_t4(runner)
        wrapped.answer_prompt_version = VERSION
        return wrapped

    def _answer_with_fallback(self, request, context, selected):
        result, ledger = unresolved_event_references(self.store, request)
        if result is None:
            return super()._answer_with_fallback(request, context, selected)
        marker = "Excerpts from the user's earlier conversations, found for this question"
        revised = (
            context.split(marker, 1)[0]
            + "Source identity/period check: "
            + result.answer
            + "\n"
            + ledger.render()
        )
        completed, _, verdict, evidence = super()._answer_with_fallback(request, revised, selected)
        self._selection_added_tokens += int(len(revised) / self.chars_per_token) - int(
            len(context) / self.chars_per_token
        )
        self._selection_audit = {"ledger": ledger.audit(), "calculation": result.detail}
        return completed, result.answer, verdict, evidence
