"""`event_time` is what the fact says. `observed_at` is when it was said.

They used to be the same column, filled with the conversation's date, so a fact that
named no time was dated to the day it was mentioned and a restatement of an old event
looked newer than the event. `results/analysis/v5-offline-gate.md` attributes one of
v2c's reasoning-48 failures to exactly that: the dated "February 14" memory was not
selected, an undated restatement carried its March session's date, and the model
answered "zero days" where the gold is 29.

Separating them is representational. Ordering is unchanged, because everything that
asks "when did this happen" now reads `occurred_at`, which falls back to `observed_at`.
What is new is that the question "did the user give this fact a time?" can be asked at
all.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from llm_long_term_memory.ingest.event_time import stated_event_time, temporal_evidence
from llm_long_term_memory.ingest.structure import structure
from llm_long_term_memory.store import Memory

MARCH = datetime(2023, 3, 15)


def _memory(**kwargs) -> Memory:
    base = {"id": "m", "user_id": "u", "type": "semantic", "content": "c", "token_count": 1}
    return Memory(**{**base, **kwargs})


def test_a_fact_that_names_no_time_gets_no_event_time():
    assert stated_event_time("The user replaced their spark plugs.", MARCH) is None


def test_exact_relative_day_uses_observation_anchor_without_losing_words():
    evidence = temporal_evidence("The user moved yesterday.", MARCH)

    assert evidence.exact == datetime(2023, 3, 14)
    assert evidence.estimate == evidence.exact
    assert evidence.expression == "yesterday"
    assert evidence.precision == "day"


def test_month_offset_is_preserved_as_an_estimate_not_a_false_exact_day():
    anchor = datetime(2026, 9, 18)
    evidence = temporal_evidence("The user moved two months ago.", anchor)

    assert evidence.expression == "two months ago"
    assert evidence.estimate == datetime(2026, 7, 18)
    assert evidence.precision == "approximate_day"
    assert evidence.exact is None


def test_last_named_month_has_month_precision():
    evidence = temporal_evidence("The user moved last July.", datetime(2026, 9, 18))

    assert evidence.expression == "last July"
    assert evidence.estimate == datetime(2026, 7, 1)
    assert evidence.precision == "month"
    assert evidence.exact is None


def test_multiple_relative_dates_do_not_assign_one_to_the_whole_fact():
    evidence = temporal_evidence("The user moved last July and left yesterday.", MARCH)

    assert evidence.exact is None
    assert evidence.estimate is None
    assert evidence.precision == "unresolved"


def test_explicit_date_and_relative_phrase_do_not_choose_different_events_time():
    text = "The user moved on March 1, 2023, and left yesterday."
    evidence = temporal_evidence(text, MARCH)

    assert evidence.exact is None
    assert evidence.estimate is None
    assert evidence.expression == text
    assert evidence.precision == "unresolved"


def test_source_expression_survives_structuring_without_controlling_lifecycle():
    memory = structure(
        "The user moved two months ago.",
        session_id="s1",
        session_date="2026/09/18",
        user_id="alice",
    )

    assert memory.event_time is None
    assert memory.event_time_expression == "two months ago"
    assert memory.event_time_estimate == datetime(2026, 7, 18)
    assert memory.observed_at == datetime(2026, 9, 18)
    assert memory.occurred_at == memory.observed_at


def test_source_words_survive_even_when_extraction_omits_the_relative_phrase():
    from llm_long_term_memory.conversation import ConversationSession, ConversationTurn
    from llm_long_term_memory.ingest.provenance import attach_source_span

    session = ConversationSession(
        session_id="s1",
        date="2026/09/18",
        turns=[ConversationTurn(role="user", content="I moved to Sydney two months ago.")],
    )
    memory = structure(
        "The user moved to Sydney.",
        session_id="s1",
        session_date=session.date,
        user_id="alice",
    )

    attach_source_span(memory, session)

    assert memory.event_time_expression is None
    assert memory.event_time_source_expression == "two months ago"
    assert memory.event_time is None


def test_a_fact_that_names_a_date_without_a_year_takes_the_conversation_s():
    assert stated_event_time("Replaced the plugs on February 14.", MARCH) == datetime(2023, 2, 14)


def test_a_date_later_in_the_year_than_the_conversation_is_read_as_last_year():
    """ "December 30", said on 3 January, is a week ago and not a year away."""
    assert stated_event_time("Anniversary on December 30.", datetime(2023, 1, 3)) == datetime(
        2022, 12, 30
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Moved house on 2022-07-03.", datetime(2022, 7, 3)),
        ("Booked for February 14, 2024.", datetime(2024, 2, 14)),
        ("Booked for 14 February 2024.", datetime(2024, 2, 14)),
        ("Paid on Feb 3rd, 2021.", datetime(2021, 2, 3)),
    ],
)
def test_explicit_dates_are_read_in_the_forms_the_corpus_writes_them(text, expected):
    assert stated_event_time(text, MARCH) == expected


@pytest.mark.parametrize(
    "text",
    [
        "The user moved last July.",
        "The user left three weeks ago.",
        "Trip from February 10 to February 20.",
        "Version 3.10-02-01 of the tool.",
        "The user walked 416 steps.",
        "February 30 is not a date.",
    ],
)
def test_what_is_deliberately_left_undated(text):
    """A relative expression, a span and a non-date all stay None.

    Resolving them wrongly is worse than not resolving them: a wrong date ranks, sorts
    and supersedes, while an absent one is reported as undated and left alone.
    """
    assert stated_event_time(text, MARCH) is None


@pytest.mark.parametrize(
    "text",
    [
        "The party was about a month before May 21, 2023.",
        "The user pruned the rose bush about one month before May 20, 2023.",
        "The user bought a peace lily two weeks before May 20, 2023.",
        "The user has been saving since March 3, 2023.",
        "The user is on leave until 2023-06-01.",
        "The deposit is due prior to February 14, 2023.",
    ],
)
def test_a_date_an_offset_is_measured_from_is_not_the_event_s_time(text):
    """An anchor is not an answer.

    The extractor writes these constantly, because it resolves the user's "last month"
    by naming the conversation's date and leaving the offset in words. Reading the
    anchor as the event time was wrong on four of the first twenty dates this recovered
    from a real store, which is precisely the "a wrong date ranks" failure the module
    exists to avoid — so a sentence that measures from a date states no time at all.
    """
    assert stated_event_time(text, MARCH) is None


def test_an_ordinary_date_is_still_read_when_another_clause_mentions_before():
    """The guard looks just behind the date, not anywhere in the sentence."""
    assert stated_event_time(
        "The user finished the book on January 29, 2023, before starting another.", MARCH
    ) == datetime(2023, 1, 29)


def test_ordering_still_has_a_time_when_the_fact_gives_none():
    memory = _memory(observed_at=MARCH)

    assert memory.event_time is None
    assert memory.occurred_at == MARCH
    assert memory.event_time_is_stated is False


def test_a_stated_time_wins_over_the_conversation_s():
    memory = _memory(event_time=datetime(2023, 2, 14), observed_at=MARCH)

    assert memory.occurred_at == datetime(2023, 2, 14)
    assert memory.event_time_is_stated is True


def test_the_restatement_that_produced_the_wrong_duration_no_longer_outranks_the_original():
    """The gate16 failure, as two memories.

    The original states its date; the restatement, in a later conversation, does not.
    Before the split both carried their own session's date, so sorting by time put the
    restatement last and a duration measured from it came out as zero.
    """
    original = structure(
        "The user replaced their spark plugs with NGK plugs on February 14.",
        session_id="s1",
        session_date="2023/02/14",
        user_id="u",
    )
    restatement = structure(
        "The user replaced their spark plugs with NGK plugs.",
        session_id="s2",
        session_date="2023/03/15",
        user_id="u",
    )

    assert original.event_time == datetime(2023, 2, 14)
    assert restatement.event_time is None, "it names no date, so it asserts no date"
    assert restatement.observed_at == datetime(2023, 3, 15)
    # The distinction the answerer needs: one of these is evidence about when, the
    # other is only evidence that it was mentioned again.
    assert original.event_time_is_stated
    assert not restatement.event_time_is_stated


def test_validity_still_starts_when_the_fact_was_stated():
    """`valid_from` used to be the session date via `event_time`; it still is."""
    memory = structure(
        "The user owns a fern.", session_id="s1", session_date="2023/03/15", user_id="u"
    )

    assert memory.valid_from == datetime(2023, 3, 15)
    assert memory.valid_from == memory.observed_at


def test_an_existing_store_keeps_its_timeline_through_the_migration(tmp_path):
    """The backfill is a derivation, not a guess.

    The old code had exactly one source for `event_time` — the conversation's date —
    so on a pre-migration row it provably *is* the observation time. Copying it across
    leaves every ordering decision on that store where it was.
    """
    import sqlite3

    from llm_long_term_memory.store import SQLiteMemoryStore

    path = tmp_path / "old.db"
    store = SQLiteMemoryStore(path)
    store.initialize()
    store.add_memories([_memory(id="m1", observed_at=MARCH, event_time=MARCH, ingested_at=MARCH)])
    store.close()

    # Put the database back the way a pre-migration one looked.
    raw = sqlite3.connect(path)
    raw.execute("ALTER TABLE memories DROP COLUMN observed_at")
    raw.commit()
    raw.close()

    reopened = SQLiteMemoryStore(path)
    reopened.initialize()
    recovered = reopened.get("m1")
    reopened.close()

    assert recovered is not None
    assert recovered.observed_at == MARCH, "backfilled from the column that held it"
    assert recovered.event_time == MARCH, "left alone; nothing is destroyed to tidy a boundary"
    assert recovered.occurred_at == MARCH


def test_a_store_written_before_the_column_can_still_be_read_read_only(tmp_path):
    """A finished run must stay replayable.

    A read-only open skips migration, correctly, so the projection has to narrow to
    the columns the database actually has. Selecting the full list would turn every
    replay of an older store into `no such column`.
    """
    import sqlite3

    from llm_long_term_memory.store import SQLiteMemoryStore

    path = tmp_path / "old.db"
    store = SQLiteMemoryStore(path)
    store.initialize()
    store.add_memories([_memory(id="m1", observed_at=MARCH, event_time=MARCH, ingested_at=MARCH)])
    store.close()

    raw = sqlite3.connect(path)
    raw.execute("ALTER TABLE memories DROP COLUMN observed_at")
    raw.commit()
    raw.close()

    reader = SQLiteMemoryStore(path, read_only=True)
    reader.initialize()
    recovered = reader.get("m1")
    reader.close()

    assert recovered is not None
    assert recovered.event_time == MARCH
    assert recovered.observed_at is None, "the column is absent, and that is what None means"
    assert recovered.occurred_at == MARCH, "ordering falls back to what the row does have"


# --- v2e: saying which dates the facts state -----------------------------------------


def _rendered(memory, **kwargs) -> str:
    from llm_long_term_memory.evaluation.runners.memory import render_memory

    return render_memory(memory, temporal=True, **kwargs)


def _plugs(**kwargs):
    return _memory(content="The user replaced the plugs", **kwargs)


def test_v2c_rendering_is_untouched_by_default():
    """Every recorded arm reads its context exactly as it did."""
    undated = _plugs(event_time=None, observed_at=MARCH, valid_from=MARCH)

    assert _rendered(undated) == "- The user replaced the plugs (since 2023-03-15)"


def test_an_undated_fact_stops_claiming_the_day_it_was_mentioned():
    undated = _plugs(event_time=None, observed_at=MARCH, valid_from=MARCH)

    line = _rendered(undated, mark_unstated=True)

    assert "mentioned 2023-03-15" in line
    assert "states no date" in line
    assert "since" not in line


def test_a_dated_fact_reads_the_same_under_both():
    stated = _plugs(
        event_time=datetime(2023, 2, 14),
        observed_at=datetime(2023, 2, 14),
        valid_from=datetime(2023, 2, 14),
    )

    assert _rendered(stated) == _rendered(stated, mark_unstated=True)
    assert "since 2023-02-14" in _rendered(stated, mark_unstated=True)


def test_a_superseded_window_is_left_alone():
    """A closed interval already says what it is; the marker is for open ones."""
    closed = _plugs(
        event_time=None, observed_at=MARCH, valid_from=MARCH, valid_to=datetime(2023, 8, 1)
    )

    assert "no longer current" in _rendered(closed, mark_unstated=True)
    assert "mentioned" not in _rendered(closed, mark_unstated=True)


def test_a_fact_with_no_time_at_all_still_says_so():
    nothing = _plugs(event_time=None, observed_at=None, valid_from=None)

    assert "date unknown" in _rendered(nothing, mark_unstated=True)
