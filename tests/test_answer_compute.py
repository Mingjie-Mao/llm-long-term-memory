from datetime import datetime

import numpy as np
import pytest

from llm_long_term_memory.answer_compute import (
    COMPUTE_RULES,
    CountedAnswerVerdict,
    CountedItem,
    DatedAnswerVerdict,
    DatedEvent,
    apply,
    compute_mode,
    first_number,
    parse_date,
)
from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
from llm_long_term_memory.store import Memory, NumpyFlatIndex, Session, SQLiteMemoryStore, Turn


@pytest.mark.parametrize(
    ("question", "mode"),
    [
        ("How many times did I ride rollercoasters from July to October?", "count"),
        ("How much money did I raise for charity in total?", "count"),
        ("How many weeks ago did I attend the festival?", "date"),
        ("How many days passed between the mass and the service?", "date"),
        ("How many days did it take for my backpack to arrive after I bought it?", "date"),
        ("Who became a parent first, Rachel or Alex?", "order"),
        ("Which mode of transport did I use most recently, a bus or a train?", "order"),
        # Differences, needs, rates and ages are not sums; lookups are untouched.
        ("How many points do I need to earn to redeem a free product?", None),
        ("How many more miles did I run than last week?", None),
        ("How many years will I be when my friend Rachel gets married?", None),
        ("What time do I usually go to the gym?", None),
        ("Which streaming service did I start using most recently?", None),
    ],
)
def test_compute_mode_is_narrow(question, mode):
    assert compute_mode(question) == mode


def _count(*items, answer, unit="", status="answer"):
    notes = [
        CountedItem(item=i, conversation_date="2023-07-01", user_words=i, amount=a, verdict=v)
        for i, a, v in items
    ]
    return CountedAnswerVerdict(notes=notes, unit=unit, status=status, answer=answer)


def test_count_overrides_only_inconsistent_arithmetic():
    v = _count(
        ("Mummy", 3, "COUNTS"),
        ("Xcelerator", 1, "COUNTS"),
        ("Space", 3, "COUNTS"),
        ("SeaWorld", 3, "COUNTS"),
        ("planned ride", 2, "OUTSIDE"),
        answer="8 times",
    )
    text, detail = apply("count", v, "8 times", "How many times...?", None)
    assert text.startswith("10") and detail["overrode"] and detail["total"] == 10
    same, detail = apply("count", v, "You rode 10 times.", "How many times...?", None)
    assert same == "You rode 10 times." and detail["overrode"] is False


def test_count_with_an_unstated_amount_is_a_known_part_never_a_total():
    v = _count(("a", 2, "COUNTS"), ("b", None, "COUNTS"), answer="2")
    text, detail = apply("count", v, "2", "How many?", None)
    assert text.startswith("At least 2") and "known part" in text and detail["partial"]


def test_count_leaves_abstentions_and_empty_counts_alone():
    declined = _count(answer="", status="no_evidence")
    assert apply("count", declined, "I do not know.", "How many?", None)[0] == "I do not know."
    nothing = _count(("x", 1, "OUTSIDE"), answer="You never mentioned any.")
    assert apply("count", nothing, "You never mentioned any.", "How many?", None)[0].startswith(
        "You never"
    )


def _dated(*events, answer="", status="answer"):
    return DatedAnswerVerdict(
        events=[DatedEvent(event=e, date=d, user_words=e) for e, d in events],
        status=status,
        answer=answer,
    )


def test_date_ago_is_counted_from_the_question_date():
    v = _dated(("Summer Nights festival", "2023-06-24"))
    q = "How many weeks ago did I attend the 'Summer Nights' festival?"
    text, detail = apply("date", v, "6 weeks ago", q, datetime(2023, 7, 15))
    assert text.startswith("3 weeks ago") and detail["value"] == 3


def test_date_interval_between_two_events_ignores_the_question_date():
    v = _dated(("ordered", "2023-03-01"), ("arrived", "2023-03-08"))
    q = "How many days did it take for my backpack to arrive after I bought it?"
    text, _ = apply("date", v, "5 days", q, datetime(2023, 5, 1))
    assert text.startswith("7 days")


def test_order_picks_the_earlier_event_and_respects_precision():
    v = _dated(("Alex adopted a baby girl", "2023-01"), ("Rachel's twins were born", "2023-02-12"))
    text, detail = apply(
        "order", v, "Rachel became a parent first.", "Who first, Rachel or Alex?", None
    )
    assert text.startswith("Alex adopted") and detail["overrode"]
    tie = _dated(("A", "2023-01"), ("B", "2023-01-20"))
    assert apply("order", tie, "A", "Which first, A or B?", None)[0] == "A"
    undated = _dated(("Ferrari model", "2023-02-01"), ("Porsche model", ""))
    assert apply("order", undated, "I do not know.", "Which first?", None)[0] == "I do not know."


def test_number_and_date_parsing():
    assert first_number("You raised $3,750 in total") == 3750
    assert first_number("three parties") == 3
    assert parse_date("2023-01")[1] == "month" and parse_date("2023/01/05")[1] == "day"
    assert parse_date("") is None


class Encoder:
    def encode_one(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)


class Client:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)

        class Completion:
            text = '{"status": "answer", "answer": "ok"}'
            input_tokens = output_tokens = 1
            api_latency_ms = 1.0

        return Completion()


@pytest.fixture
def store(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "m.db")
    store.initialize()
    started = datetime(2025, 5, 24)
    store.add_session(
        Session(
            "a", "u", started, turns=[Turn("a-0", "a", 0, "user", "I rode a kayak twice.", started)]
        )
    )
    store.add_memories(
        [
            Memory(
                id="m1",
                user_id="u",
                type="semantic",
                content="The user rode a kayak.",
                token_count=5,
                status="active",
                valid_from=started,
                ingested_at=started,
                source_session_id="a",
                source_turn_index=0,
            )
        ]
    )
    yield store
    store.close()


def _runner(policy, client, store, tmp_path):
    index = NumpyFlatIndex(tmp_path / f"i-{policy}", dim=2)
    index.add(["m1"], np.array([[1.0, 0.0]], dtype=np.float32))
    return MemoryRunner(
        client,
        model="m",
        encoder=Encoder(),
        store=store,
        index=index,
        top_k=20,
        temporal=True,
        evidence_hydration=True,
        raw_fallback=True,
        raw_primary_tokens=4000,
        raw_primary_dated=True,
        raw_primary_prefer_user=True,
        raw_primary_time_window=True,
        raw_primary_time_notes=True,
        answer_policy=policy,
    )


@pytest.mark.parametrize(
    "question",
    ["What boat do I have?", "When did I go kayaking?", "How many points do I need?"],
)
def test_untriggered_questions_send_exactly_the_t4_request(store, tmp_path, question):
    t4, t4c = Client(), Client()
    request = AnswerRequest(question=question, asked_on="2025-06-14", user_id="u")
    _runner("v2_clt", t4, store, tmp_path).answer_request(request)
    _runner("v2_clt_compute", t4c, store, tmp_path).answer_request(request)
    assert t4.calls == t4c.calls


def test_triggered_question_adds_only_its_rule_schema_and_room(store, tmp_path):
    t4, t4c = Client(), Client()
    request = AnswerRequest(
        question="How many times did I ride a kayak?", asked_on="2025-06-14", user_id="u"
    )
    _runner("v2_clt", t4, store, tmp_path).answer_request(request)
    _runner("v2_clt_compute", t4c, store, tmp_path).answer_request(request)
    a, b = t4.calls[0], t4c.calls[0]
    assert b["prompt"] == a["prompt"]
    assert b["system"] == f"{a['system']}\n\n{COMPUTE_RULES['count']}"
    assert b["schema"] is CountedAnswerVerdict and b["max_output_tokens"] == 2048


def test_record_only_keeps_the_model_answer_and_says_what_it_would_have_done():
    v = _count(("a", 3, "COUNTS"), ("b", 3, "COUNTS"), answer="5")
    text, detail = apply("count", v, "5", "How many?", None, override=False)
    assert text == "5" and detail["would_override"] and detail["total"] == 6
    assert detail["override_enabled"] is False


@pytest.mark.parametrize(
    ("question", "mode"),
    [
        ("How many model kits have I worked on or bought?", "count"),
        ("How many times did I ride rollercoasters from July to October?", "count"),
        ("How much money did I raise for charity in total?", None),
        ("How many days did it take me to finish the book?", None),
        ("How many minutes did I exceed my target time by?", None),
        ("How many weeks ago did I attend the festival?", None),
        ("Who became a parent first, Rachel or Alex?", None),
    ],
)
def test_v2_triggers_plain_counts_only(question, mode):
    from llm_long_term_memory.answer_compute import count_mode_v2

    assert count_mode_v2(question) == mode


def test_t4c2_records_and_never_substitutes(store, tmp_path):
    runner = _runner("v2_clt_compute2", Client(), store, tmp_path)
    assert runner.compute_override is False
    request = AnswerRequest(
        question="How many kayaks did I buy?", asked_on="2025-06-14", user_id="u"
    )
    runner.client.calls.clear()
    runner.answer_request(request)
    assert runner.client.calls[0]["schema"] is CountedAnswerVerdict
