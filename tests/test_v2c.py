from __future__ import annotations

from datetime import datetime

from llm_long_term_memory.evaluation.runners.v2c import plan
from llm_long_term_memory.store import Memory


def memory(
    id_: str,
    content: str,
    day: int,
    *,
    predicate: str,
    scope: str = "event",
    update_op: str = "coexists",
) -> Memory:
    when = datetime(2023, 1, day)
    return Memory(
        id=id_,
        user_id="u",
        type="semantic",
        content=content,
        token_count=10,
        ingested_at=when,
        event_time=when,
        valid_from=when,
        subject="user",
        predicate=predicate,
        object=content,
        scope=scope,
        update_op=update_op,
        replaces_previous=update_op == "replaces",
    )


def test_explicit_update_suppresses_older_related_value_across_neighbouring_keys():
    old = memory(
        "old",
        "The user set a personal best time of 27:12 in a charity 5K run.",
        5,
        predicate="running_achievements",
    )
    new = memory(
        "new",
        "The user's personal best time for a 5K run is 25:50.",
        12,
        predicate="running_personal_best",
        scope="profile",
        update_op="replaces",
    )

    outcome = plan("What was my personal best time in the charity 5K run?", [old, new])

    assert [item.id for item in outcome.memories] == ["new"]
    assert outcome.suppressed_update_ids == ["old"]
    assert "older related value was omitted" in outcome.context_note


def test_past_state_question_keeps_both_values():
    old = memory("old", "The user used TensorFlow for work.", 5, predicate="framework")
    new = memory(
        "new",
        "The user switched from TensorFlow to PyTorch for work.",
        12,
        predicate="work_framework",
        update_op="replaces",
    )

    outcome = plan("What framework did I use before I switched?", [old, new])

    assert [item.id for item in outcome.memories] == ["old", "new"]
    assert outcome.suppressed_update_ids == []


def test_exact_date_question_hydrates_when_memory_has_only_month_precision():
    coarse = memory(
        "coarse",
        'The user volunteered at the "Love is in the Air" dinner in February 2023.',
        20,
        predicate="volunteering",
    )

    outcome = plan("When did I volunteer at the fundraising dinner?", [coarse])

    assert "no day-level date" in outcome.detail_hydration_reason


def test_exact_date_question_does_not_hydrate_a_day_level_fact():
    exact = memory(
        "exact",
        "The user volunteered at the fundraising dinner on February 14th.",
        20,
        predicate="volunteering",
    )

    outcome = plan("When did I volunteer at the fundraising dinner?", [exact])

    assert outcome.detail_hydration_reason is None


def test_exact_prior_recommendation_name_hydrates_source_conversation():
    sibling = memory(
        "sibling",
        "The assistant recommended Escovitch Fish, made with snapper and vegetables.",
        20,
        predicate="food_recommendation",
        scope="recommendation",
    )

    outcome = plan(
        "What was the name of the dish you recommended with snapper and fruit?",
        [sibling],
    )

    assert "exact identity of a prior recommendation" in outcome.detail_hydration_reason
    assert "resolve every qualifier" in outcome.context_note
    assert "matches only some qualifiers" in outcome.context_note
    assert "do not force one" in outcome.context_note
    assert "actually recommended" in outcome.context_note


def test_generic_recommendation_question_does_not_force_detail_hydration():
    recommendation = memory(
        "recommendation",
        "The assistant recommended Escovitch Fish.",
        20,
        predicate="food_recommendation",
        scope="recommendation",
    )

    outcome = plan("What food did we discuss?", [recommendation])

    assert outcome.detail_hydration_reason is None


def test_sports_sequence_is_filtered_and_sorted_by_dates_in_content():
    nfl = memory(
        "nfl",
        "The user watched the NFL playoffs on 2023-01-15.",
        22,
        predicate="sports_viewing",
    )
    tennis = memory(
        "tennis",
        "The user takes tennis lessons every Thursday.",
        10,
        predicate="tennis_lessons",
    )
    nba = memory(
        "nba",
        "The user attended an NBA game on 2023-01-05.",
        5,
        predicate="event_attendance",
    )
    college = memory(
        "college",
        "The user watched the college football championship on 2023-01-14.",
        15,
        predicate="sports_viewing",
    )

    outcome = plan(
        "What is the order of the sports events I watched in January?",
        [nfl, tennis, nba, college],
    )

    assert outcome.timeline_ids == ["nba", "college", "nfl"]
    assert outcome.context_note.index("NBA") < outcome.context_note.index("college football")
    assert outcome.context_note.index("college football") < outcome.context_note.index("NFL")
    assert "tennis lessons" not in outcome.context_note
    assert "attending an event in person is one way of watching" in outcome.context_note
    assert "complete answer must include every event" in outcome.context_note
