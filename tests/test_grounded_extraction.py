"""Extraction that cites its source, and the four ways it can fail to.

Detail retention is 36.6% overall and the first loss point for 10 of 14 analysed
failures, so this is the layer worth hardening. The instruction to stay atomic is
already in both shipped extractors and 36.6% is the number with it — what changes here
is what a memory must carry, not what the model is asked to try.
"""

from __future__ import annotations

import pytest

from llm_long_term_memory.ingest.grounded import (
    GroundedFact,
    GroundedResult,
    ground,
    locate,
    normalise,
    uncovered_specifics,
)

TURN = (
    "I stayed in Xinjiang for 12 days last July, and 3 of them I was sick "
    "and didn't leave the room."
)


def fact(**kw):
    return GroundedFact(
        content=kw.pop("content", "The user's Xinjiang trip lasted 12 days."),
        verbatim_span=kw.pop("verbatim_span", "I stayed in Xinjiang for 12 days"),
        turn_index=kw.pop("turn_index", 0),
        **kw,
    )


def test_a_span_that_is_in_the_turn_is_anchored_to_real_offsets():
    report = ground(GroundedResult(facts=[fact()]), [TURN])
    assert not report.rejected
    anchored = report.anchored[0]
    assert TURN[anchored.char_start : anchored.char_end] == "I stayed in Xinjiang for 12 days"


def test_a_paraphrase_is_rejected_rather_than_stored():
    """This is the whole point: extraction stops being "the model believes it extracted
    correctly" and becomes "every memory cites a turn"."""
    report = ground(
        GroundedResult(facts=[fact(verbatim_span="I spent twelve days in Xinjiang")]), [TURN]
    )
    assert not report.anchored
    assert "not in the turn" in report.rejected[0][1]


def test_spacing_may_differ_but_words_may_not():
    """A model that copies correctly can still normalise a double space. Anything looser
    than that would accept a paraphrase, which is the thing being guarded against."""
    report = ground(
        GroundedResult(facts=[fact(verbatim_span="I  stayed   in Xinjiang for 12 days")]), [TURN]
    )
    assert report.anchored
    assert TURN[report.anchored[0].char_start : report.anchored[0].char_end].startswith("I stayed")


def test_a_turn_that_does_not_exist_is_refused():
    report = ground(GroundedResult(facts=[fact(turn_index=9)]), [TURN])
    assert "turn 9 is not in this session" in report.rejected[0][1]


def test_a_typed_value_must_appear_in_its_own_span():
    """A typed number that disagrees with its evidence is worse than prose, because it
    arrives looking checked."""
    report = ground(
        GroundedResult(facts=[fact(attribute="duration", value="14", unit="days")]), [TURN]
    )
    assert not report.anchored
    assert "not in its own span" in report.rejected[0][1]

    ok = ground(GroundedResult(facts=[fact(attribute="duration", value="12", unit="days")]), [TURN])
    assert ok.anchored and not ok.rejected


def test_the_rejection_rate_is_reported_rather_than_hidden():
    """It is the measurement that says whether the model can do this at all."""
    report = ground(
        GroundedResult(facts=[fact(), fact(verbatim_span="something it never said")]), [TURN]
    )
    assert report.grounded_rate == pytest.approx(0.5)


def test_one_sentence_yields_one_record_per_fact():
    """A single record reading "the user travelled to Xinjiang last July" loses the 12
    days, the 3 days and the staying indoors. All three have to survive somewhere."""
    facts = [
        fact(attribute="duration", value="12", unit="days"),
        fact(
            content="The user was sick for 3 days of the trip.",
            verbatim_span="3 of them I was sick",
            attribute="illness_duration",
            value="3",
            unit="days",
        ),
        fact(
            content="The user did not leave the room while sick.",
            verbatim_span="didn't leave the room",
        ),
    ]
    report = ground(GroundedResult(facts=facts), [TURN])
    assert len(report.anchored) == 3 and not report.rejected
    assert uncovered_specifics(TURN, [a.fact.content for a in report.anchored]) == {}


def test_the_coverage_check_names_what_a_summary_dropped_and_costs_nothing():
    """The obvious design is a second model call asking "did you miss anything". This
    finds the same thing deterministically, so a repair request is spent only when
    something is actually missing."""
    missing = uncovered_specifics(TURN, ["The user travelled to Xinjiang last July."])
    # The turn says "3 of them", so 3 is a quantity there and only "12 days" reads as a
    # duration. The check reports what the source actually stated, not what a reader
    # would infer it meant.
    assert missing["duration"] == {"12 days"}
    assert missing["quantity"] == {"12", "3"}


def test_an_unresolvable_date_stays_null_rather_than_becoming_the_session_date():
    """`event_time` is used for ordering, so a guessed date reorders a timeline. Today
    every memory in a session shares the session's date — 4,714 sessions, 3,776 distinct
    timestamps."""
    undated = fact(time_expression="", event_time=None)
    assert undated.event_time is None
    dated = fact(time_expression="last July", event_time="2025-07")
    assert dated.event_time == "2025-07"


def test_the_prompt_forbids_dating_a_fact_by_when_it_was_mentioned():
    from llm_long_term_memory.ingest import grounded

    prompt = grounded._GROUNDED_PROMPT
    assert "Never use the session's date as `event_time` merely because" in prompt
    assert "leave it null" in prompt


def test_normalisation_stops_at_case_and_whitespace():
    """A span that only matches after aggressive rewriting is not evidence."""
    assert normalise("  I  Stayed\tin Xinjiang ") == "i stayed in xinjiang"
    assert locate("didn't leave", TURN) is not None
    assert locate("did not leave", TURN) is None


def test_a_unit_carried_by_an_earlier_clause_is_accepted():
    """ "I stayed 12 days, and 3 of them I was sick" states a duration of 3 days in a
    clause that never says "days". Holding the unit to the span would leave the model two
    options on a fact like that — invent a span, or drop the fact — and both are worse."""
    report = ground(
        GroundedResult(
            facts=[
                fact(
                    content="The user was sick for 3 days of the trip.",
                    verbatim_span="3 of them I was sick",
                    attribute="illness_duration",
                    value="3",
                    unit="days",
                )
            ]
        ),
        [TURN],
    )
    assert report.anchored and not report.rejected


def test_but_the_value_is_still_held_to_its_own_span():
    """The unit is anaphoric; the number is not. A number that is not in the evidence it
    cites is the failure the typed field exists to prevent."""
    report = ground(
        GroundedResult(facts=[fact(verbatim_span="3 of them I was sick", value="7", unit="days")]),
        [TURN],
    )
    assert not report.anchored
    assert "value '7' is not in its own span" in report.rejected[0][1]


def test_a_unit_invented_from_nowhere_is_still_refused():
    report = ground(
        GroundedResult(
            facts=[fact(verbatim_span="3 of them I was sick", value="3", unit="kilometres")]
        ),
        [TURN],
    )
    assert "nowhere in the turn" in report.rejected[0][1]
