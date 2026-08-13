"""Extraction fidelity gate.

Most of these pin the *denominator*. Twice now a low score turned out to be the
metric counting things no extractor should keep — list markers as quantities,
sentence openers as proper nouns — and tuning a prompt against that would have
taught the extractor to hoard.
"""

from __future__ import annotations

from chronomem.evaluation.datasets.longmemeval import HaystackSession, HaystackTurn
from chronomem.ingest.fidelity import _extract_facets, score_sessions, user_assertions
from chronomem.store import Memory


def session(*turns: tuple[str, str]) -> HaystackSession:
    return HaystackSession(
        session_id="s1",
        date="2023/03/15 (Wed) 10:00",
        turns=[HaystackTurn(role=r, content=c) for r, c in turns],
    )


def mem(content: str) -> Memory:
    return Memory(id="m", user_id="u", type="semantic", content=content, token_count=5)


def test_a_pasted_numbered_list_is_not_forty_quantities():
    """The failure that made quantity recall read 20% when the extractor was
    correctly ignoring the content: a user pasted a numbered document."""
    text = "Here are the steps:\n1. Wash hands\n2. Rinse\n3. Repeat\n4. Dry"
    assert _extract_facets(text)["quantity"] == set()


def test_a_number_attached_to_a_noun_counts():
    facets = _extract_facets("I added 25 postcards and upgraded to 16GB of RAM.")
    assert "25" in facets["quantity"]
    assert "16" in facets["quantity"]


def test_sentence_openers_are_not_proper_nouns():
    assert _extract_facets("As I'm working late")["proper_noun"] == set()
    assert "evelyn hardcastle" in _extract_facets("I read Evelyn Hardcastle")["proper_noun"]


def test_questions_are_excluded_from_the_denominator():
    """A user asking about Osama bin Laden's height has stated nothing about
    themselves; penalising the extractor for dropping it rewards hoarding."""
    s = session(("user", "How tall was Osama bin Laden? I am 180cm myself."))
    kept = user_assertions(s)
    assert "180cm" in kept
    assert "Osama" not in kept


def test_assistant_turns_are_excluded():
    s = session(("user", "I run 5km."), ("assistant", "Try 10km next, or 15km."))
    kept = user_assertions(s)
    assert "5km" in kept and "10km" not in kept


def test_filler_durations_do_not_count():
    assert _extract_facets("Hang on a second, I moved three months ago.")["duration"] == {
        "three months"
    }


def test_recall_is_scored_across_all_memories_of_a_session():
    """A specific counts as retained if it survived into any memory from that
    session, not necessarily the one that also carries the surrounding fact."""
    s = session(("user", "I collected for three months and own 25 postcards."))
    memories = [mem("The user collected for three months."), mem("The user owns 25 postcards.")]
    report = score_sessions([(s, memories)])
    assert report.per_facet["duration"].recall == 1.0
    assert report.per_facet["quantity"].recall == 1.0


def test_a_dropped_specific_is_reported_with_an_example():
    s = session(("user", "I have been collecting for three months."))
    report = score_sessions([(s, [mem("The user collects things.")])])
    assert report.per_facet["duration"].recall == 0.0
    assert "three months" in report.missed_examples["duration"]


def test_empty_extraction_scores_zero_not_one():
    s = session(("user", "I own 25 postcards."))
    report = score_sessions([(s, [])])
    assert report.overall == 0.0
    assert report.memories_per_session == 0.0
