"""The fact-coverage instrument, and the three ways it could be confidently wrong.

`source_session_recalled` says a labelled session survived; it does not say the needed
fact did, which is why the 98.3% / 55% gap could not be read as a reasoning-error rate.
This measures the fact itself — so the first thing it has to be is honest about what it
cannot decide.
"""

from __future__ import annotations

import pytest

from llm_long_term_memory.evaluation import beam_coverage as fc

CONVERSATION = fc.haystack(
    "I finally hit 58% on the practice quiz, up from 41%.",
    "The deposit was $9,800 and the report came to 7 pages.",
    "I moved my sessions to Schwab Intelligent Portfolios in June 2024.",
)


def item(text, conversation=CONVERSATION):
    return fc.classify(text, conversation)


def test_a_rubric_item_naming_no_distinctive_fact_is_undecidable_not_absent():
    """ "mention of cultural differences" cannot be checked by string matching, and
    scoring it absent would manufacture a loss."""
    assert item("LLM response should contain: mention of cultural differences").kind == (
        fc.UNDECIDABLE
    )


def test_a_bare_number_is_not_distinctive_enough_to_match_on():
    """A 500K-token conversation contains "26" somewhere whatever the question is."""
    assert fc.distinctive_tokens("LLM response should mention: 26") == set()
    assert fc.distinctive_tokens("LLM response should mention: 26 weeks") == {"26 week"}


def test_the_units_are_a_closed_list_so_the_next_word_is_not_read_as_one():
    """ "from February 20 till April 9" used to yield the quantity "20 till"."""
    assert "20 till" not in fc.distinctive_tokens("from February 20 till April 9")
    assert "february 20" in fc.distinctive_tokens("from February 20 till April 9")


def test_money_percentages_dates_and_quantities_all_normalise_to_one_spelling():
    tokens = fc.distinctive_tokens("$9,800 · 58 percent · June 30, 2024 · 7 pages")
    assert tokens == {"$9800", "58%", "june 30 2024", "7 page"}


def test_the_opening_word_is_not_mistaken_for_a_name():
    """Rubric items open with a capital because they open a sentence."""
    assert fc.distinctive_tokens("Discussing recipe scaling with Michele Rossi") == {
        "michele rossi"
    }


def test_a_fact_absent_from_the_whole_conversation_is_derived_not_lost():
    """A duration question's answer is computed. Filing it as extraction loss would make
    temporal reasoning read as a total extraction failure."""
    assert item("LLM response should state: 64 days").kind == fc.DERIVED
    assert item("LLM response should mention: 58%").kind == fc.EVIDENCE


def test_a_negative_requirement_is_set_aside_rather_than_inverted():
    """ "should avoid" is satisfied by absence; counting it with the rest would score a
    correct answer as a loss."""
    assert item("LLM response should avoid: mentioning the old price").kind == fc.NEGATIVE_ITEM


def test_the_ladder_reports_the_first_stage_that_lost_the_fact():
    rubric = ("LLM response should mention: 58%",)
    stages = {
        "source": CONVERSATION,
        "extracted": CONVERSATION,
        "retrieved": CONVERSATION,
        "context": fc.haystack("nothing relevant here"),
        "answer": fc.haystack("nothing relevant here"),
    }
    report = fc.question_report(fc.ladder(rubric, CONVERSATION, stages))
    assert report["required_facts"] == 1
    assert report["first_loss"] == "context"
    assert report["required_fact_coverage"] == 0.0
    assert report["answer_utilisation"] is None


def test_answer_utilisation_only_counts_facts_that_reached_the_context():
    """Layer 4 is "it was there and the answer did not use it". A fact that never arrived
    is layer 3's failure, and charging it to the answerer is the confusion this exists to
    end."""
    rubric = ("LLM response should mention: 58%", "LLM response should mention: 7 pages")
    everywhere = {stage: CONVERSATION for stage in fc.STAGES}
    everywhere["context"] = fc.haystack("I finally hit 58% on the practice quiz.")
    everywhere["answer"] = fc.haystack("you hit 58%")
    report = fc.question_report(fc.ladder(rubric, CONVERSATION, everywhere))
    assert report["required_facts"] == 2
    assert report["required_fact_coverage"] == 0.5
    assert report["answer_utilisation"] == 1.0


def test_a_question_with_nothing_decidable_reports_none_rather_than_zero():
    rubric = ("LLM response should contain: a friendly tone",)
    report = fc.question_report(fc.ladder(rubric, CONVERSATION, dict.fromkeys(fc.STAGES, "")))
    assert report["required_fact_coverage"] is None
    assert report["undecidable_items"] == 1


def test_the_four_layers_average_by_conversation_not_by_question():
    rows = [
        {
            "question_id": f"beam-100K-1-information_extraction-{n}",
            "question_type": "information_extraction",
            "notes": {
                "recall_stages": {"selected": True},
                "recall_coverage": {"selected": 1.0},
            },
        }
        for n in range(3)
    ] + [
        {
            "question_id": "beam-500K-2-information_extraction-0",
            "question_type": "information_extraction",
            "notes": {
                "recall_stages": {"selected": False},
                "recall_coverage": {"selected": 0.0},
            },
        }
    ]
    layers = fc.layers(rows, {})
    assert layers["any_source_session"]["conversations"] == 2
    assert layers["any_source_session"]["value"] == pytest.approx(0.5)
    assert layers["required_fact_coverage"]["value"] is None


def test_the_report_says_when_a_mandatory_layer_is_missing():
    """A metric reported only when someone remembers to is not mandatory."""
    from llm_long_term_memory.evaluation import beam_report

    rows = [
        {
            "question_id": "beam-100K-1-information_extraction-0",
            "question_type": "information_extraction",
            "notes": {"judge": {"grades": [{"item": 1, "score": 1.0, "reason": ""}]}},
        }
    ]
    declared = {
        "judge": {"correct_at": 0.5},
        "primary": {"abilities": ["information_extraction"]},
        "reported_separately": {"abilities": []},
    }
    report = beam_report.report(rows, declared)
    assert report["mandatory_secondary_complete"] is False
    assert "required_fact_coverage" in report["mandatory_secondary_missing"]


def test_a_fact_missing_from_its_own_labelled_source_is_not_extraction_loss():
    """MemTrace calls it an Annotation Error. Filing it under extraction would credit the
    pipeline with a failure the benchmark's own labels caused."""
    rubric = ("LLM response should mention: 58%",)
    stages = dict.fromkeys(fc.STAGES, "")
    report = fc.question_report(fc.ladder(rubric, CONVERSATION, stages))
    assert report["suspected_annotation_gap"] is True
    assert report["first_loss"] == "source"


def test_a_zero_score_on_an_answer_carrying_every_required_fact_is_flagged():
    """Three times on this project a low score was the metric rather than the system."""
    rubric = ("LLM response should mention: 58%",)
    stages = dict.fromkeys(fc.STAGES, CONVERSATION)
    assert fc.question_report(fc.ladder(rubric, CONVERSATION, stages), 0.0)["suspected_judge_error"]
    assert not fc.question_report(fc.ladder(rubric, CONVERSATION, stages), 1.0)[
        "suspected_judge_error"
    ]


def test_floors_may_key_on_the_field_that_actually_discriminates():
    """`Memory.type` comes from a handful of verb regexes and 89.8% of a real store falls
    through to `semantic`, so a floor on it reserves a slice for almost everything.
    `scope` is extractor-assigned and spreads across seven values."""
    from llm_long_term_memory.pack.budget import pack
    from llm_long_term_memory.store import Memory

    def memory(mid, scope, tokens=10):
        return Memory(
            id=mid, user_id="u", type="semantic", content=f"fact {mid}",
            token_count=tokens, scope=scope,
        )  # fmt: skip

    memories = [memory(f"m{i}", "event") for i in range(8)] + [memory("p1", "profile")]
    utilities = [0.9] * 8 + [0.1]
    # Keyed on type every memory is 'semantic', so the profile floor reserves nothing for
    # the profile fact and the low-utility one is crowded out.
    by_type = pack(memories, utilities, budget=50, type_floors={"profile": 0.3})
    by_scope = pack(
        memories, utilities, budget=50, type_floors={"profile": 0.3}, floor_field="scope"
    )
    assert "p1" not in [m.id for m in by_type.selected]
    assert "p1" in [m.id for m in by_scope.selected]


def test_an_unknown_floor_field_is_refused():
    from llm_long_term_memory.pack.budget import pack

    with pytest.raises(ValueError, match="floors key on"):
        pack([], [], budget=10, floor_field="importance")
