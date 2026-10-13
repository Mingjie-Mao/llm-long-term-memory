"""Source-bound quoted-price comparisons; future travel is not a purchase count."""

from __future__ import annotations

import json
import re
from decimal import Decimal

from t4_event_count_repair import EventCountRepairRunner

from llm_long_term_memory.evaluation.grounded.answering import Calculation, build_ledger
from llm_long_term_memory.ingest.extract import _parse_date
from llm_long_term_memory.retrieve.excerpts import archive_excerpts

VERSION = "memory-t4-scoped-selection-v3"


def quoted_savings(ledger, question):
    if re.search(
        r"\b(?:as of|back in|at that time|last year|last month)\b|"
        r"\bin\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December|\d{4})\b",
        question,
        re.I,
    ):
        return None  # Historical prices need an explicit temporal selection step.
    query = re.search(
        r"\bhow much\s+(?:will|would|do) I save\b.*?\b(?:taking|using)\s+"
        r"(?:(?:a|an|the)\s+)?(\w+)\b.*?\b(?:instead of|rather than)\s+"
        r"(?:(?:a|an|the)\s+)?(\w+)\b",
        question,
        re.I,
    )
    if not query or query[1].casefold() == query[2].casefold():
        return None
    chosen, original = query[1].casefold(), query[2].casefold()
    route = re.search(
        r"\bfrom\s+(?:(?:a|an|the|my)\s+)?(\w+)\s+to\s+"
        r"(?:(?:a|an|the|my)\s+)?(\w+)\b",
        question,
        re.I,
    )
    quotes = {chosen: [], original: []}
    for source in ledger.sources:
        if source.kind != "raw" or source.role != "user":
            continue
        # Compare reported prices, including plans. Do not invent actual purchases.
        for sentence in re.split(r"(?<=[.!?])\s+", source.text):
            if re.search(
                r"\b(?:imagine|suppose|if|not|don't|didn't|never|no longer)\b", sentence, re.I
            ):
                continue
            alternatives = [
                a for a in quotes if re.search(r"\b" + re.escape(a) + r"\b", sentence, re.I)
            ]
            prices = list(re.finditer(r"([$£€¥])\s*(\d+(?:,\d{3})*(?:\.\d+)?)", sentence))
            if len(alternatives) != 1 or len(prices) != 1:
                continue
            if route and not all(
                re.search(r"\b" + re.escape(w) + r"\b", sentence, re.I) for w in route.groups()
            ):
                continue
            p = prices[0]
            quotes[alternatives[0]].append(
                {
                    "source": source.id,
                    "source_id": source.source_id,
                    "quote": sentence,
                    "said": source.conversation_date,
                    "amount": p[2].replace(",", ""),
                    "currency": p[1],
                }
            )
    values = {
        a: {(q["currency"], Decimal(q["amount"])) for q in rows} for a, rows in quotes.items()
    }
    if any(len(v) != 1 for v in values.values()):
        return None
    chosen_unit, chosen_price = next(iter(values[chosen]))
    original_unit, original_price = next(iter(values[original]))
    if chosen_unit != original_unit:
        return None
    difference = original_price - chosen_price
    approximate = any(
        re.search(r"\b(?:about|around|roughly|approximately)\b", q["quote"], re.I)
        for rows in quotes.values()
        for q in rows
    )
    return Calculation(
        ("About " if approximate else "") + f"{chosen_unit}{difference:g} saved "
        f"({original_unit}{original_price:g} {original} minus "
        f"{chosen_unit}{chosen_price:g} {chosen}), "
        "using your quoted prices for this trip.",
        True,
        {
            "mechanism": "user_quote_savings",
            "quotes": quotes,
            "formula": "original minus chosen",
            "result": str(difference),
            "currency": chosen_unit,
            "approximate": approximate,
        },
    )


class SavingsRepairRunner(EventCountRepairRunner):
    @classmethod
    def from_t4(cls, runner):
        wrapped = super().from_t4(runner)
        wrapped.answer_prompt_version = VERSION
        return wrapped

    def _answer_with_fallback(self, request, context, selected):
        if not re.search(r"\bhow much\b.*\bsave\b", request.question, re.I):
            return super()._answer_with_fallback(request, context, selected)
        packed = archive_excerpts(
            self.store,
            request.user_id,
            request.question,
            self.raw_primary_tokens,
            chars_per_token=self.chars_per_token,
            asked_on=_parse_date(request.asked_on),
            time_notes=True,
            prefer_user=True,
            time_window=True,
        )
        ledger = build_ledger(self.store, request.user_id, [], packed.turns, max_tokens=6500)
        result = None if ledger.dropped else quoted_savings(ledger, request.question)
        if result is None:
            return super()._answer_with_fallback(request, context, selected)
        note = "\nQuoted-price comparison: Plans can supply prices for a future comparison; "
        note += "they are not an actual-purchase count. " + json.dumps(
            {"answer": result.answer, "derivation": result.detail}
        )
        completed, _, verdict, evidence = super()._answer_with_fallback(
            request, context + note, selected
        )
        self._selection_added_tokens += int(len(context + note) / self.chars_per_token) - int(
            len(context) / self.chars_per_token
        )
        self._selection_audit = {"ledger": ledger.audit(), "calculation": result.detail}
        return completed, result.answer, verdict, evidence
