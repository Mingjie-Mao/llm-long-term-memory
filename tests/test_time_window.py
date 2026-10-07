"""Time-aware retrieval for raw-primary t4: the period a question names, searched by time.

Registered in `results/prereg-raw-primary-t3-t4-heldout100-v1.md`. Off by default; when
on, only the asking tenant's turns may be packed.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from llm_long_term_memory.retrieve.excerpts import archive_excerpts
from llm_long_term_memory.retrieve.time_window import question_window, turn_dates
from llm_long_term_memory.store import Session, SQLiteMemoryStore, Turn

THURSDAY = date(2023, 6, 22)


@pytest.mark.parametrize(
    ("question", "window"),
    [
        (
            "Who did I go with to the music event last Saturday?",
            (date(2023, 6, 16), date(2023, 6, 18)),
        ),
        ("What did I do a week ago?", (date(2023, 6, 11), date(2023, 6, 19))),
        ("Where did I go last week?", (date(2023, 6, 11), date(2023, 6, 19))),
        ("What did I buy last month?", (date(2023, 5, 1), date(2023, 5, 31))),
    ],
)
def test_a_question_names_a_period(question, window):
    assert question_window(question, THURSDAY) == window


@pytest.mark.parametrize(
    "question",
    [
        "How many days passed between the two trips?",  # no period named
        "What did I do last week and last month?",  # two periods: ambiguous
    ],
)
def test_no_window_when_none_or_several(question):
    assert question_window(question, THURSDAY) is None


def test_a_turns_own_relative_date_counts():
    # Said on Tuesday 2023-06-20: "last Saturday" is 2023-06-17, widened by a day.
    assert turn_dates("I saw a concert last Saturday.", date(2023, 6, 20)) == [
        (date(2023, 6, 16), date(2023, 6, 18))
    ]


@pytest.fixture
def store(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "w.db")
    store.initialize()
    rows = [
        # The answer, in the asked period, sharing no word with the question.
        (
            "concert",
            "q1",
            datetime(2023, 6, 17),
            "Billie Eilish was amazing, I went with my parents.",
        ),
        # Lexically closer, outside the period.
        ("old", "q1", datetime(2023, 3, 1), "Any tips for a music event? I love music events."),
        # Another tenant, inside the period and lexically close.
        ("other", "q2", datetime(2023, 6, 17), "The music event last Saturday was with my sister."),
    ]
    for sid, user, started, content in rows:
        store.add_session(
            Session(sid, user, started, turns=[Turn(f"{sid}-0", sid, 0, "user", content, started)])
        )
    yield store
    store.close()


QUESTION = "Who did I go with to the music event last Saturday?"
ASKED = datetime(2023, 6, 22)


def test_off_by_default(store):
    found = archive_excerpts(store, "q1", QUESTION, 12, asked_on=ASKED)
    assert {t.session_id for t in found.turns} == {"old"}
    assert found.window_turns == 0


def test_on_packs_the_period_first_within_one_tenant(store):
    found = archive_excerpts(store, "q1", QUESTION, 30, asked_on=ASKED, time_window=True)
    ids = [t.session_id for t in found.turns]
    assert "concert" in ids
    assert "other" not in ids, "another tenant's turn must never be packed"
    assert found.window_turns == 1
