"""Relative dates resolved from each turn's own conversation date (raw-primary t1).

Registered in `results/prereg-raw-primary-time-notes-v1.md`. The v1 arm must render
exactly as before when the notes are off.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from llm_long_term_memory.retrieve.excerpts import archive_excerpts
from llm_long_term_memory.retrieve.time_notes import annotate
from llm_long_term_memory.store import Session, SQLiteMemoryStore, Turn

WEDNESDAY = date(2023, 5, 24)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a 5K run about a month ago", "about a month ago [≈ 2023-04-24]"),
        ("my router 3 weeks ago", "3 weeks ago [≈ 2023-05-03]"),
        ("in a league for about 2 months now", "for about 2 months now [since ≈ 2023-03-24]"),
        ("I finished it today", "today [= 2023-05-24 Wed]"),
        ("Yesterday I started", "Yesterday [= 2023-05-23 Tue]"),
        ("2 days ago", "2 days ago [= 2023-05-22]"),
        ("a few days ago", "a few days ago [≈ 2023-05-21]"),
        ("last Saturday", "last Saturday [= 2023-05-20 Sat]"),
        ("last weekend", "last weekend [= 2023-05-20 to 2023-05-21]"),
        ("this weekend", "this weekend [= 2023-05-27 to 2023-05-28]"),
        ("last week", "last week [= week of 2023-05-15 to 2023-05-21]"),
        ("this month", "this month [= 2023-05 (May 2023)]"),
        ("last December", "last December [= 2022-12]"),
        ("next March", "next March [= 2024-03]"),
    ],
)
def test_expressions_resolve_from_the_conversation_date(text, expected):
    annotated, count = annotate(text, WEDNESDAY)
    assert expected in annotated
    assert count == 1


@pytest.mark.parametrize(
    "text",
    [
        'my favourite song is "Yesterday"',  # a title, not a time
        "and Yesterday by the Beatles",
        "I finished the season in 14 days",  # a duration, not a date
        "this Friday",  # either side of the conversation
        "this May",
        "this past month",  # a span back from today, not a calendar month
    ],
)
def test_ambiguous_or_non_temporal_phrases_are_left_alone(text):
    assert annotate(text, WEDNESDAY) == (text, 0)


def test_the_words_are_kept_and_nothing_is_dated_without_a_conversation_date():
    text = "I set up the thermostat a month ago and the router 3 weeks ago."
    annotated, count = annotate(text, WEDNESDAY)
    assert count == 2
    assert annotated.replace(" [≈ 2023-04-24]", "").replace(" [≈ 2023-05-03]", "") == text
    assert annotate(text, None) == (text, 0)


def _store(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "t1.db")
    store.initialize()
    for sid, user, day, content in (
        ("run", "q1", 24, "I did a charity 5K run about a month ago."),
        ("other", "q2", 24, "I did a charity 5K run yesterday."),
    ):
        started = datetime(2023, 5, day, 16, 56)
        store.add_session(
            Session(sid, user, started, turns=[Turn(f"{sid}-0", sid, 0, "user", content, started)])
        )
    return store


def test_render_is_unchanged_when_notes_are_off(tmp_path):
    store = _store(tmp_path)
    try:
        plain = archive_excerpts(store, "q1", "charity run", 500).render()
        assert "[" not in plain
        assert "Conversation on 2023-05-24:" in plain
    finally:
        store.close()


def test_render_dates_turns_and_headers_within_one_tenant(tmp_path):
    store = _store(tmp_path)
    try:
        excerpts = archive_excerpts(
            store,
            "q1",
            "charity run",
            500,
            asked_on=datetime(2023, 6, 14),
            time_notes=True,
        )
        text = excerpts.render()
        assert "about a month ago [≈ 2023-04-24]" in text
        assert "(Wednesday; 21 days, about 3.0 weeks, before the question)" in text
        assert "yesterday" not in text, "another tenant's turn is never shown"
        assert {t.session_id for t in excerpts.turns} == {"run"}
    finally:
        store.close()


def test_weekend_phrases_said_on_a_weekend():
    sunday = date(2023, 4, 9)
    assert "[= 2023-04-08 to 2023-04-09]" in annotate("over the weekend", sunday)[0]
    assert "[= 2023-04-08 to 2023-04-09]" in annotate("over the weekend", date(2023, 4, 10))[0]
    assert annotate("this past weekend", sunday) == ("this past weekend", 0)


def test_only_the_users_turns_are_annotated(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "roles.db")
    store.initialize()
    started = datetime(2023, 5, 24)
    store.add_session(
        Session(
            "s",
            "q1",
            started,
            turns=[
                Turn("s-0", "s", 0, "user", "I ran a charity race yesterday.", started),
                Turn("s-1", "s", 1, "assistant", "How can I help with the race today?", started),
            ],
        )
    )
    try:
        text = archive_excerpts(store, "q1", "charity race", 500, time_notes=True).render()
        assert "yesterday [= 2023-05-23 Tue]" in text
        assert "race today?" in text
    finally:
        store.close()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "I've been working at NovaTech for about 4 years and 3 months now.",
            "for about 4 years and 3 months now [since ≈ 2019-02-24]",
        ),
        (
            "I've been working professionally for 9 years and use a notebook.",
            "for 9 years [since ≈ 2014-05-24]",
        ),
        ("She has been living there for two weeks", "for two weeks [since ≈ 2023-05-10]"),
    ],
)
def test_durations_up_to_the_conversation_get_a_start_date(text, expected):
    annotated, count = annotate(text, WEDNESDAY)
    assert expected in annotated
    assert count == 1
    assert annotated.replace(" [" + expected.split(" [")[1], "") == text


def test_a_past_duration_is_not_a_start_date():
    assert annotate("I went to Japan for two weeks.", WEDNESDAY)[1] == 0


def test_a_curly_apostrophe_counts():
    text = "I\u2019ve been with my current company for five years"
    assert "for five years [since ≈ 2018-05-24]" in annotate(text, WEDNESDAY)[0]


def test_an_offset_beyond_the_calendar_is_left_alone():
    text = "The pyramids were built 4500 years ago and the city 6000 years ago."
    assert annotate(text, WEDNESDAY) == (text, 0)
