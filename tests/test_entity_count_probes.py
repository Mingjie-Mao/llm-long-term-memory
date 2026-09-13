"""The rebuilt count generator, against the four defects that inverted a sign.

Each test below is one of the failures found by re-reading the paid v4 rows: an intention
counted as a completed act, one row counted as one member when it named two, a question
whose scope contradicted its own verb, and a member no reader could check. The old set
shipped all four; this one must refuse or fix each.

Fixtures rather than the real store, so the properties are pinned where CI can see them —
`train150.db` is not in a clean checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

gen = pytest.importorskip("entity_count_probes")


def memory(mid, obj, *, scope="event", predicate="books_read"):
    """A row shaped like the store's, addressable by key the way sqlite3.Row is."""
    return {"id": mid, "object": obj, "scope": scope, "predicate": predicate}


def build(members, relation="read"):
    return gen.build_probe("ns", relation, members)


# ------------------------------------------------------------ intentions are not members


def test_a_plan_is_not_a_completed_act():
    """ "is looking for thriller recommendations" scored as a book read. 19 of 30
    development probes carried at least one member like it."""
    members = [
        memory("m1", "The Nightingale"),
        memory("m2", "Educated"),
        memory("m3", "Circe"),
        memory("m4", "thriller recommendations", scope="preference"),
        memory("m5", "a book about sailing", scope="plan"),
    ]
    probe = build(members)

    assert probe["answer"] == 3
    assert probe["evidence_memory_ids"] == ["m1", "m2", "m3"]


def test_a_question_with_only_intentions_is_refused_not_answered_zero():
    """Returning 0 would be a confident gold answer over a set that was never asked for."""
    members = [memory(f"m{i}", f"book {i}", scope="plan") for i in range(4)]

    with pytest.raises(gen.Refusal, match="intention rather than an act"):
        build(members)


def test_commitments_and_recommendations_are_intentions_too():
    """ "needs to reorder the medication tomorrow" is a promise, not a completed task."""
    assert {"plan", "preference", "commitment", "recommendation"} <= gen.INTENT_SCOPES
    assert not (gen.INTENT_SCOPES & gen.COMPLETED_SCOPES)


# ------------------------------------------------------------ one row is not one member


def test_a_row_naming_two_things_contributes_two():
    """ "has been reading The Huffington Post and Politico" is one memory and two
    publications; the old gold counted it once."""
    members = [
        memory("m1", "The Huffington Post and Politico"),
        memory("m2", "Circe"),
    ]
    probe = build(members)

    assert probe["answer"] == 3
    assert "The Huffington Post" in probe["entities"]
    assert "Politico" in probe["entities"]


def test_a_comma_list_splits_too():
    probe = build([memory("m1", "Dune, Neuromancer, Snow Crash")])

    assert probe["answer"] == 3


def test_a_separator_inside_a_name_does_not_split_it():
    """ "Simon & Schuster" is one publisher. A split that leaves a fragment too short to be
    a thing means the separator was part of the name."""
    assert gen.entities_of("Simon & Schuster") == ["Simon & Schuster"]


def test_two_rows_naming_the_same_thing_contribute_one():
    """Deduplication is the other half: over-counting is as wrong as under-counting, and
    the real gap runs in both directions."""
    members = [
        memory("m1", "The Nightingale"),
        memory("m2", "the nightingale"),
        memory("m3", "Circe"),
        memory("m4", "Dune"),
    ]
    probe = build(members)

    assert probe["answer"] == 3
    # Both rows stay as evidence: the reader must be able to see why they merged.
    assert set(probe["evidence_memory_ids"]) == {"m1", "m2", "m3", "m4"}


def test_deduplication_ignores_articles_and_case_only():
    assert gen.normalise_entity("The  Nightingale!") == gen.normalise_entity("nightingale")
    # And does not merge genuinely different members.
    assert gen.normalise_entity("Dune") != gen.normalise_entity("Dune Messiah")


# ------------------------------------------------------------ self-contradiction


def test_a_completed_act_question_over_intent_predicates_is_refused():
    """The old set shipped: "How many distinct dishes has the user cooked? Count only
    facts recorded under: recipe interest, recipe to make, recipes tried." The verb asks
    for what was cooked; the scope asks for what was wanted."""
    members = [memory(f"m{i}", f"dish {i}", predicate="recipe_to_make") for i in range(4)]

    with pytest.raises(gen.Refusal, match="intent predicates"):
        build(members, relation="cooked")


@pytest.mark.parametrize(
    "predicate",
    [
        "recipe interest",
        "planned trips",
        "wishlist items",
        "upcoming events",
        "travel goals",
        # Both spellings: predicates arrive raw from the store and printed in the
        # question, and `_` is a word character so `\bto` never matched the raw form.
        "recipe to make",
        "recipe_to_make",
    ],
)
def test_intent_predicate_names_are_detected(predicate):
    assert gen.scope_conflicts([predicate]) == [predicate]


@pytest.mark.parametrize(
    "predicate",
    [
        "books read",
        "camera gear",
        "places visited",
        # The reason intent words are listed rather than stemmed: `plan\w*` also matches
        # these, and refusing them would silently drop every "how many plants" question.
        "garden plants",
        "houseplants",
        "plant care tool",
    ],
)
def test_completed_predicate_names_are_not_flagged(predicate):
    assert gen.scope_conflicts([predicate]) == []


# ------------------------------------------------------------ unverifiable members


def test_a_measurement_is_not_a_member():
    """ "55 pages per day" is a reading rate. It reached the old gold as a book."""
    assert gen.entities_of("55 pages per day") == []

    members = [memory("m1", "55 pages per day"), memory("m2", "Dune")]
    with pytest.raises(gen.Refusal, match="no recoverable entity"):
        build(members)


def test_an_empty_object_is_refused_rather_than_guessed():
    with pytest.raises(gen.Refusal, match="no recoverable entity"):
        build([memory("m1", ""), memory("m2", "Dune"), memory("m3", "Circe")])


def test_a_relation_without_vetted_phrasing_is_refused():
    with pytest.raises(gen.Refusal, match="no vetted phrasing"):
        build([memory("m1", "x")], relation="not_a_relation")


# ------------------------------------------------------------ size and shape


def test_a_set_too_small_to_read_is_refused():
    """Below three, an off-by-one is indistinguishable from a miscount."""
    with pytest.raises(gen.Refusal, match="only 2 distinct"):
        build([memory("m1", "Dune"), memory("m2", "Circe")])


def test_a_set_too_large_to_check_is_refused():
    members = [memory(f"m{i}", f"book number {i}") for i in range(20)]

    with pytest.raises(gen.Refusal, match="past the size"):
        build(members)


def test_the_question_states_the_deduplication_rule_it_is_scored_on():
    """The answerer is graded on distinct things. A question that did not say so would be
    measuring whether the model guessed the convention."""
    probe = build([memory(f"m{i}", f"book {i}") for i in range(4)])

    assert "Count each distinct thing once" in probe["question"]
    assert "Count only facts recorded under" in probe["question"]


def test_a_time_window_is_stated_in_the_question_when_one_is_given():
    from datetime import datetime

    probe = gen.build_probe(
        "ns",
        "read",
        [memory(f"m{i}", f"book {i}") for i in range(4)],
        window=(datetime(2023, 1, 1), datetime(2023, 6, 30)),
    )

    assert "between 2023-01-01 and 2023-06-30" in probe["question"]


def test_the_derivation_names_the_scopes_it_counted():
    """A gold answer that cannot be re-derived from its own derivation is a label."""
    probe = build([memory(f"m{i}", f"book {i}") for i in range(4)])

    assert "DISTINCT entities" in probe["derivation"]
    assert "event" in probe["derivation"] and "profile" in probe["derivation"]


# ------------------------------------------------------------ the real set, when present


def test_the_generated_set_contains_no_intention_members():
    """Replay against the committed set when the private store is available."""
    import json
    import sqlite3

    generated = REPO / "results/analysis/entity-count-probes.json"
    store = REPO / "stores/train150.db"
    if not (generated.is_file() and store.is_file()):
        pytest.skip("the generated set and its store are local research artifacts")

    payload = json.loads(generated.read_text(encoding="utf-8"))
    connection = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    scopes = dict(connection.execute("SELECT id, scope FROM memories"))
    allowed = set(payload["completed_scopes"])

    offenders = [
        (probe["probe_id"], mid)
        for probe in payload["probes"]
        for mid in probe["evidence_memory_ids"]
        if scopes.get(mid) not in allowed
    ]
    assert not offenders, offenders[:5]
    assert all(p["answer"] == len(p["entities"]) for p in payload["probes"])
