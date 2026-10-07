import re
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.runtime.evidence_pages import (
    PageSelection,
    PageSelectionError,
    archive_pages,
    select_page,
)
from llm_long_term_memory.runtime.grounded_answering import GroundedAnswererV17
from llm_long_term_memory.store import NumpyFlatIndex, Session, SQLiteMemoryStore, Turn


def add(store, tenant, sid, bodies):
    turns = [
        Turn(f"{sid}:{i}", sid, i, role, text, datetime(2026, 10, 1))
        for i, (role, text) in enumerate(bodies)
    ]
    store.add_session(Session(sid, tenant, datetime(2026, 10, 1), "test", turns))
    return turns


@pytest.fixture
def store(tmp_path):
    db = SQLiteMemoryStore(tmp_path / "s.db")
    db.initialize()
    yield db
    db.close()


def test_whole_original_turns_are_partitioned_once_and_never_cross_tenants(store):
    own = add(
        store, "alice", "a", [("user", f"I bought item {i}. " + "word " * 25) for i in range(10)]
    )
    add(store, "bob", "b", [("user", "I bought a secret item.")])
    pages = archive_pages(store, "alice", max_tokens=100, max_pages=20)
    assert pages.complete and len(pages.pages) > 1
    sources = [s for page in pages.pages for s in page.sources]
    assert {s.source_id for s in sources} == {t.id for t in own}
    assert len(sources) == len(own)
    for page in pages.pages:
        assert len(page.render()) / page.chars_per_token <= 100
    assert [s.text for s in sources] == [t.content for t in own]


def test_page_limit_and_oversized_turn_are_explicit_incomplete_reviews(store):
    own = add(store, "alice", "a", [("user", "word " * 25)] * 3)
    pages = archive_pages(store, "alice", max_tokens=60, max_pages=1)
    assert not pages.complete and pages.omitted_turn_ids
    huge = archive_pages(store, "alice", max_tokens=2, max_pages=20)
    assert not huge.complete and set(huge.omitted_turn_ids) == {t.id for t in own}


@pytest.mark.parametrize("reviewed,selected", [([], []), (["E1", "E1"], []), (["E1"], ["E999"])])
def test_incomplete_or_foreign_selection_is_rejected(store, reviewed, selected):
    add(store, "alice", "a", [("user", "I paid $10.")])
    page = archive_pages(store, "alice").pages[0]
    client = SimpleNamespace(
        generate=lambda **kw: SimpleNamespace(
            text=PageSelection(
                reviewed_sources=reviewed, selected_sources=selected, scope_complete=True
            ).model_dump_json()
        )
    )
    with pytest.raises(PageSelectionError):
        select_page(client, "same-model", page, AnswerRequest("How much?", "2026-10-03", "alice"))


class SelectingClient:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["model"] == "same-lite"
        ids = re.findall(r"^\[(E\d+)\]", kwargs["prompt"], re.M)
        if kwargs["schema"] == PageSelection:
            text = PageSelection(
                reviewed_sources=ids, selected_sources=ids, scope_complete=True
            ).model_dump_json()
        else:
            operands = []
            for sid, body in re.findall(r"^\[(E\d+)\][^\n]*\n([^\n]+)", kwargs["prompt"], re.M):
                value = re.search(r"\$(\d+)", body)
                if value:
                    operands.append(
                        {"source": sid, "quote": body, "value": value[1], "unit": "$", "key": sid}
                    )
            text = kwargs["schema"](
                reviewed_sources=ids,
                observations=[
                    {"source": sid, "interpretation": "Actual purchase", "decision": "include"}
                    for sid in ids
                ],
                operation="sum",
                status="answer",
                scope_complete=True,
                operands=operands,
            ).model_dump_json()
        return SimpleNamespace(text=text, input_tokens=5, output_tokens=2, api_latency_ms=1)


def build(store, tmp_path, turns):
    index = NumpyFlatIndex(tmp_path / "turns", 2)
    index.add([t.id for t in turns], np.ones((len(turns), 2)))
    return GroundedAnswererV17(
        SelectingClient(), model="same-lite", store=store, turn_index=index, encoder=None
    )


def test_actual_reader_reviews_multiple_pages_then_computes_selected_original_items(
    store, tmp_path
):
    turns = add(
        store,
        "alice",
        "a",
        [("user", "I paid $10 for workshop A."), ("user", "I paid $20 for workshop B.")],
    )
    engine = build(store, tmp_path, turns)
    engine.review_page_tokens = 35
    answer = engine.answer(
        AnswerRequest("How much total money did I pay for workshops?", "2026-10-03", "alice"),
        [],
        np.ones(2),
    )
    assert answer.notes["archive_review"]["complete"]
    assert len(answer.notes["archive_review"]["pages"]) == 2
    assert "30" in answer.text
    assert answer.context_tokens <= 6000
    assert answer.prompt_tokens == 5 * len(engine.client.calls)
    assert answer.output_tokens == 2 * len(engine.client.calls)


def test_cap_stops_before_any_provider_call_and_does_not_give_incomplete_total(store, tmp_path):
    turns = add(store, "alice", "a", [("user", "I paid $10 for workshop A.")] * 3)
    engine = build(store, tmp_path, turns)
    engine.review_page_tokens, engine.max_review_pages = 35, 1
    answer = engine.answer(
        AnswerRequest("How much total money?", "2026-10-03", "alice"), [], np.ones(2)
    )
    assert not engine.client.calls
    assert answer.notes["evidence_gap"]["cause"] == "archive_page_budget_incomplete"
    assert answer.notes["answer_status"] == "need_source" and "10" not in answer.text


def test_selected_pool_overflow_is_explicit_and_never_runs_final_reader(store, tmp_path):
    turns = add(store, "alice", "a", [("user", "I paid $10. " + "word " * 900)] * 8)
    engine = build(store, tmp_path, turns)
    answer = engine.answer(
        AnswerRequest("How much total money?", "2026-10-03", "alice"), [], np.ones(2)
    )
    assert answer.notes["evidence_gap"]["cause"] == "selected_evidence_budget_incomplete"
    assert all(call["schema"] == PageSelection for call in engine.client.calls)
    assert answer.prompt_tokens == len(engine.client.calls) * 5


@pytest.mark.parametrize(
    "question,personal",
    [
        ("Which event happened first, the road trip or the arrival of a lens?", True),
        ("How many babies were born to friends and family members?", True),
        ("Can you remind me which beer you mentioned in our previous chat?", False),
        ("How much was allocated in the campaign plan?", False),
        ("How much did I spend on jewelry this month?", True),
    ],
)
def test_source_intent_is_distinct_from_first_person_phrasing(question, personal):
    from llm_long_term_memory.runtime.grounded_answering import personal_archive_review

    assert personal_archive_review(question) is personal


def test_oversized_user_turn_is_paged_as_exact_contiguous_spans(store):
    text = "I attended a workshop. " + "word " * 1200 + "I paid $20."
    turns = add(store, "alice", "a", [("user", text), ("assistant", "I suggest $999.")])
    add(store, "bob", "b", [("user", "secret " * 1000)])
    pages = archive_pages(store, "alice", roles=("user",), max_tokens=300, split_oversized=True)
    assert pages.complete
    fragments = [s for p in pages.pages for s in p.sources]
    assert len(fragments) > 1
    assert "".join(s.text for s in fragments) == text
    assert all(s.original_source_id == turns[0].id for s in fragments)
    assert all(text[s.char_start : s.char_end] == s.text for s in fragments)
    assert all(len(p.render()) / p.chars_per_token <= 300 for p in pages.pages)
    capped = archive_pages(
        store, "alice", roles=("user",), max_tokens=300, max_pages=1, split_oversized=True
    )
    assert not capped.complete and capped.omitted_turn_ids


def test_personal_pages_preserve_roles_and_refuse_assistant_numeric_operands(store, tmp_path):
    from llm_long_term_memory.runtime.grounded_answering import GroundedAnswererV18

    turns = add(
        store,
        "alice",
        "a",
        [("user", "I paid $10 for workshop A."), ("assistant", "I suggest $999 for workshop B.")],
    )
    index = NumpyFlatIndex(tmp_path / "turns", 2)
    index.add([t.id for t in turns], np.ones((len(turns), 2)))
    engine = GroundedAnswererV18(
        SelectingClient(), model="same-lite", store=store, turn_index=index, encoder=None
    )
    answer = engine.answer(
        AnswerRequest("How much total money did I pay for workshops?", "2026-10-03", "alice"),
        [],
        np.ones(2),
    )
    assert answer.notes["archive_review"]["role_scope"] == ("user",)
    assert all(
        "raw/assistant" not in c["prompt"]
        for c in engine.client.calls
        if c["schema"] is PageSelection
    )
    assert "999" not in answer.text
    assert answer.notes["archive_review"]["complete"]


def test_same_session_context_is_hydrated_before_summary_filler(store, tmp_path):
    from llm_long_term_memory.runtime.grounded_answering import (
        EvidenceLedger,
        GroundedAnswererV18,
        add_turns,
    )

    turns = add(
        store,
        "alice",
        "a",
        [
            ("user", "I shop at Store Z."),
            ("assistant", "You can use coupons there."),
            ("user", "I redeemed a $5 coupon on creamer yesterday."),
        ],
    )
    add(store, "bob", "b", [("user", "I shop at Secret Store.")])
    engine = build(store, tmp_path, turns)
    engine.__class__ = GroundedAnswererV18
    ledger = EvidenceLedger()
    add_turns(ledger, store, "alice", [turns[2]])
    assert (
        engine.hydrate_review_context(
            ledger, AnswerRequest("Where did I redeem it?", "2026-10-03", "alice")
        )
        == 2
    )
    assert {s.source_id for s in ledger.sources} == {t.id for t in turns}
    assert any("Store Z" in s.text for s in ledger.sources)


@pytest.mark.parametrize(
    "text,quote,valid",
    [
        ("I attended the workshop; it was a free event.", "it was a free event", True),
        ("I attended; the event was free.", "the event was free", True),
        ("It was not free.", "It was not free", False),
        ("If it was a free event, I would attend.", "it was a free event", False),
        (
            "I attended; it was a free event but I paid $20 registration.",
            "it was a free event",
            False,
        ),
        ("I paid $20 and got free food.", "free food", False),
    ],
)
def test_explicit_free_attendance_is_zero_cost_without_inventing_absent_item(text, quote, valid):
    from llm_long_term_memory.runtime.grounded_answering import EvidenceSource, explicit_free_price

    source = EvidenceSource("E1", "raw", "t", text, "user", "2026-10-01")
    assert explicit_free_price(source, quote, 0, "$") is valid
    assert not explicit_free_price(source, quote, 20, "$")


def test_free_price_calculation_is_opt_in_and_still_requires_user_source():
    import json

    from llm_long_term_memory.runtime.grounded_answering import (
        EvidenceLedger,
        EvidenceSource,
        GroundedVerdict,
        calculate,
    )

    source = EvidenceSource(
        "E1", "raw", "t", "I attended the workshop; it was a free event.", "user", "2026-10-01"
    )
    operand = dict(
        source="E1", quote="it was a free event", value="0", unit="$", key="workshop", time=""
    )
    verdict = GroundedVerdict(
        reviewed_sources=["E1"],
        scope_complete=True,
        operation="sum",
        status="answer",
        operand_records=[json.dumps(operand)],
    )
    ledger = EvidenceLedger(sources=[source])
    assert not calculate(verdict, ledger, "2026-10-03", quantity_binding=True).computed
    assert (
        calculate(verdict, ledger, "2026-10-03", quantity_binding=True, free_prices=True).answer
        == "0 $"
    )
    from dataclasses import replace

    ledger.sources = [replace(source, role="assistant")]
    assert (
        calculate(verdict, ledger, "2026-10-03", free_prices=True).detail["cause"]
        == "user_fact_requires_user_source"
    )


def test_past_event_future_month_conflict_discloses_payments_without_fabricating_year():
    from llm_long_term_memory.runtime.grounded_answering import (
        EvidenceLedger,
        EvidenceSource,
        GroundedAnswererV18,
        scoped_payment_conflict,
    )

    ledger = EvidenceLedger(
        sources=[
            EvidenceSource(
                "E1",
                "raw",
                "a",
                "I just attended a marketing workshop on March 15-16. I paid $500 to attend.",
                "user",
                "2023-02-26",
                session_id="a",
            ),
            EvidenceSource(
                "E2",
                "raw",
                "b",
                "I attended a writing workshop in November. I paid $200 to attend.",
                "user",
                "2023-02-26",
                session_id="b",
            ),
            EvidenceSource(
                "E3",
                "raw",
                "c",
                "I attended a mindfulness workshop on December 12. I paid $20 to attend.",
                "user",
                "2023-02-26",
                session_id="c",
            ),
            EvidenceSource(
                "E4",
                "raw",
                "d",
                "I just attended a workshop on March 15. I paid $999 to attend.",
                "assistant",
                "2023-02-26",
                session_id="d",
            ),
        ]
    )
    request = AnswerRequest(
        "How much money did I spend attending workshops in the last four months?",
        "2023-02-26",
        "alice",
    )
    result = scoped_payment_conflict(ledger, request)
    assert result.detail["reported_total"] == "720"
    assert not result.detail["scope_complete"]
    assert len(result.detail["temporal_conflicts"]) == 1
    assert "cannot confirm" in result.answer and "not assumed a corrected year" in result.answer
    engine = object.__new__(GroundedAnswererV18)
    gap = engine.completion_notes(ledger, request, None, result, result.answer)["evidence_gap"]
    assert gap["missing_fields"] == ["event_date", "event_year", "report_period"]
    assert gap["user_id"] == "alice"
    from dataclasses import replace

    ledger.sources[0] = replace(
        ledger.sources[0],
        text="I plan to attend a marketing workshop on March 15-16. I paid $500 to attend.",
    )
    assert scoped_payment_conflict(ledger, request) is None


def test_pronoun_operand_quote_expands_only_to_same_source_literal_member():
    from llm_long_term_memory.runtime.grounded_answering import (
        EvidenceLedger,
        EvidenceSource,
        GroundedAnswererV18,
        GroundedVerdictV7,
    )

    source = EvidenceSource(
        "E1",
        "raw",
        "a",
        "I wanted to resize my engagement ring. I got it a month ago.",
        "user",
        "2026-10-01",
    )
    response = GroundedVerdictV7(
        status="answer",
        scope_complete=True,
        operation="count",
        reviewed_sources=["E1"],
        observations=[dict(source="E1", interpretation="Acquired ring", decision="include")],
        operands=[
            dict(source="E1", quote="I got it a month ago.", value="engagement ring", key="ring")
        ],
    )
    engine = object.__new__(GroundedAnswererV18)
    parsed = engine.parse_verdict(
        response.model_dump_json(),
        EvidenceLedger(sources=[source]),
        AnswerRequest("How many pieces did I acquire?", "2026-10-03", "alice"),
    )
    import json

    assert json.loads(parsed.operand_records[0])["quote"] == source.text
    response.operands[0].value = "gold necklace"
    parsed = engine.parse_verdict(
        response.model_dump_json(),
        EvidenceLedger(sources=[source]),
        AnswerRequest("How many pieces did I acquire?", "2026-10-03", "alice"),
    )
    assert json.loads(parsed.operand_records[0])["quote"] == "I got it a month ago."


def test_payment_repetition_is_one_event_but_equal_price_distinct_events_survive():
    from dataclasses import replace

    from llm_long_term_memory.runtime.grounded_answering import (
        EvidenceLedger,
        EvidenceSource,
        scoped_payment_conflict,
    )

    a = EvidenceSource(
        "E1",
        "raw",
        "a",
        "I just attended a marketing workshop on March 15-16. I paid $500 to attend.",
        "user",
        "2023-02-26",
        session_id="a",
        turn_index=0,
    )
    b = EvidenceSource(
        "E2",
        "raw",
        "b",
        (
            "I attended a writing workshop in November. It was a two-day workshop, "
            "and I paid $200 to attend."
        ),
        "user",
        "2023-02-26",
        session_id="b",
        turn_index=0,
    )
    repeated = EvidenceSource(
        "E3",
        "raw",
        "c",
        "It was the one I attended in November, a two-day writing workshop. I paid $200 to attend.",
        "user",
        "2023-02-26",
        session_id="b",
        turn_index=4,
    )
    request = AnswerRequest(
        "How much money did I spend attending workshops in the last four months?",
        "2023-02-26",
        "alice",
    )
    ledger = EvidenceLedger(sources=[a, b, repeated])
    result = scoped_payment_conflict(ledger, request)
    assert result.detail["reported_total"] == "700" and len(result.detail["payments"]) == 2
    ledger.sources[2] = replace(
        repeated, text="I attended another writing workshop in December. I paid $200 to attend."
    )
    assert scoped_payment_conflict(ledger, request).detail["reported_total"] == "900"
    ledger.sources[2] = replace(
        repeated, text="I attended a photography workshop in November. I paid $200 to attend."
    )
    assert scoped_payment_conflict(ledger, request).detail["reported_total"] == "900"
    ledger.sources[0] = replace(
        a, text="If I just attended a marketing workshop on March 15-16, I paid $500 to attend."
    )
    assert scoped_payment_conflict(ledger, request) is None


def test_actual_jewelry_count_verification_excludes_storage_purchase(store, tmp_path):
    from llm_long_term_memory.runtime.grounded_answering import GroundedAnswererV18

    names = ["silver necklace", "emerald earrings", "engagement ring", "wooden dresser"]
    turns = add(store, "alice", "a", [("user", f"I acquired a {name}.") for name in names])

    class Client:
        def __init__(self):
            self.calls = []
            self.reads = 0

        def generate(self, **kw):
            self.calls.append(kw)
            ids = re.findall(r"^\[(E\d+)\]", kw["prompt"], re.M)
            if kw["schema"] is PageSelection:
                value = PageSelection(
                    reviewed_sources=ids, selected_sources=ids, scope_complete=True
                )
            else:
                self.reads += 1
                bindings = {
                    name: next(
                        sid
                        for sid, body in re.findall(
                            r"^\[(E\d+)\][^\n]*\n([^\n]+)", kw["prompt"], re.M
                        )
                        if name in body
                    )
                    for name in names
                }
                selected = names if self.reads == 1 else names[:3]
                if self.reads == 2:
                    assert "Storage furniture is outside" in kw["prompt"]
                value = kw["schema"](
                    operation="count",
                    status="answer",
                    scope_complete=True,
                    reviewed_sources=ids,
                    observations=[
                        dict(
                            source=bindings[n],
                            interpretation=n,
                            decision="include" if n in selected else "exclude",
                        )
                        for n in names
                    ],
                    operands=[
                        dict(source=bindings[n], quote=f"I acquired a {n}.", value=n, key=n)
                        for n in selected
                    ],
                )
            return SimpleNamespace(
                text=value.model_dump_json(), input_tokens=1, output_tokens=1, api_latency_ms=1
            )

    idx = NumpyFlatIndex(tmp_path / "turns", 2)
    idx.add([t.id for t in turns], np.ones((len(turns), 2)))
    engine = GroundedAnswererV18(
        Client(), model="same-lite", store=store, turn_index=idx, encoder=None
    )
    result = engine.answer(
        AnswerRequest("How many pieces of jewelry did I acquire?", "2026-10-03", "alice"),
        [],
        np.ones(2),
    )
    assert result.text == "3: silver necklace, emerald earrings, engagement ring"
    assert result.notes["grounded_calls"][0]["calculation"]["cause"] == "member_category_mismatch"
    assert engine.client.reads == 2
