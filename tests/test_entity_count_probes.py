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
from datetime import datetime
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


# ------------------------------------------------ splits the generator must not trust

# Every object below is real, taken from `stores/train150.db` by the split audit, and every
# one of them put a non-member into the shipped gold answer of `entity-count-probes.json`.
# The module docstring always said an uncertain split is refused; until these tests it was
# not, and the count and the entity list agreed with each other so nothing downstream
# could notice.


@pytest.mark.parametrize(
    ("obj", "invented"),
    [
        # entity_count_0007, "how many distinct places has the user visited?" — gold 5.
        ("MoMA, Dec 2023", "Dec 2023"),
        ("American Museum of Natural History, Jan 10 2024", "Jan 10 2024"),
        # entity_count_0052, "how many distinct events has the user attended?" — gold 9,
        # of which three were months and three were cities.
        ("TechFest, San Francisco, February 2023", "February 2023"),
        ("Walk for Hunger on May 21, 2023", "2023"),
        # entity_count_1e043500_owns, third emission — gold 3, one of its members a decade.
        ("1970s and 1980s cameras", "1970s"),
    ],
)
def test_an_object_that_carries_its_own_date_is_refused(obj, invented):
    """A trailing date means the object is a record, not a list. Counting its commas as
    members turned a month into a place the user visited."""
    assert invented in gen.entities_of(obj)
    assert "date" in (gen.split_refusal(obj) or "")

    members = [memory("m1", obj)] + [memory(f"m{i}", f"thing {i}") for i in range(2, 5)]
    with pytest.raises(gen.Refusal, match="records provenance"):
        build(members, relation="visited")


def test_the_whole_object_is_refused_not_just_its_date():
    """`'TechFest, San Francisco, February 2023'` also yields 'San Francisco', which is not
    an event either. Once an object is known to carry provenance its other commas are
    provenance too, so dropping only the date would leave the city counted."""
    assert gen.entities_of("TechFest, San Francisco, February 2023") == [
        "TechFest",
        "San Francisco",
        "February 2023",
    ]
    assert gen.split_refusal("TechFest, San Francisco, February 2023")


def test_a_clause_is_not_an_entity():
    """entity_count_0035 counted `'attended on March 22nd, 2023'` as two events attended."""
    assert gen.split_refusal("attended on March 22nd, 2023")
    assert gen.split_refusal("attended the History Museum lecture series")


def test_a_byline_means_the_separators_belong_to_a_title():
    """entity_count_0040: `'Milk and Filth by Carmen Giménez Smith'` is one poetry
    collection. The split made it two books, one of them called Milk."""
    assert gen.entities_of("Milk and Filth by Carmen Giménez Smith") == [
        "Milk",
        "Filth by Carmen Giménez Smith",
    ]
    assert "byline" in (gen.split_refusal("Milk and Filth by Carmen Giménez Smith") or "")


def test_companions_are_not_members():
    """entity_count_0029: `'basketball with Tom and Alex'` is one hobby and two people.
    Alex was counted as a hobby the user practises."""
    assert "Alex" in gen.entities_of("basketball with Tom and Alex")
    assert "companions" in (gen.split_refusal("basketball with Tom and Alex") or "")


@pytest.mark.parametrize(
    "obj",
    [
        # The splits that are right, and must keep working: refusing these would be the
        # quieter failure the generator's own comment warns about.
        "tofu, tempeh, seitan",
        "The Huffington Post and Politico",
        "grilled chicken breasts and bell peppers",
        "road bike, mountain bike, commuter bike",
        "Native Instruments Komplete Kontrol and Akai MPC Live 2",
        "Simon & Schuster",
        "Spaghetti Carbonara",
    ],
)
def test_a_sound_split_is_not_refused(obj):
    assert gen.split_refusal(obj) is None


# Real objects again, from the second emission of the set. Every one reached a shipped gold
# answer because the rules above looked only at members a separator produced.


@pytest.mark.parametrize(
    "obj",
    [
        # entity_count_c7cf7dfd_visited, "how many distinct places has the user
        # visited?" — gold 3: a vet visit's date, and one city twice through two itinerary
        # lines.
        "January 15 2023",
        "arrive Calgary June 14 2023 11:30 am",
        "depart Calgary June 19 2023 10:25 am",
        # entity_count_8ebdbe50_acquired, "how many distinct things has the user
        # acquired?" — gold 4, with one pair of sneakers counted twice through two receipts.
        "sneakers on 2023-04-10 for $80 from Amazon",
        "sneakers from Amazon on 2023-05-22",
        # entity_count_b759caee_attended, "how many distinct events has the user
        # attended?" — gold 5.
        "raised $150",
        "St. Mary's Church Nov 2022",
        "cycling event September 2022",
    ],
)
def test_a_record_that_never_split_is_refused_like_one_that_did(obj):
    """A dated day or month, an ISO date, a clock time or a price marks a record of an
    occasion. Two records of one occasion do not deduplicate, so each was counted."""
    assert len(gen.entities_of(obj)) == 1
    assert "date, time or price" in (gen.split_refusal(obj) or "")


@pytest.mark.parametrize(
    "obj",
    [
        # Kept, from the same set: a leading model year is part of a name, and so can a
        # bare year be. Refusing them would be the quieter failure.
        "1995 Honda Civic",
        "1957 Gibson Les Paul Goldtop guitar",
        "1923 Paris Eiffel Tower postcard",
        "1984",
        "St. Jerome's Laneway Festival",
        "Dr. Martens boots",
        "Holiday Market",
    ],
)
def test_a_year_or_a_month_like_word_inside_a_name_is_not_provenance(obj):
    assert gen.split_refusal(obj) is None


@pytest.mark.parametrize("word", ["decorating", "Marketing", "novels", "octopus", "junk"])
def test_a_word_that_only_starts_like_a_month_is_not_a_date(word):
    """The first month pattern matched any word beginning with a month's three letters, so
    a list of hobbies containing 'decorating' was refused as a dated record."""
    assert gen.split_refusal(f"knitting, {word}, pottery") is None


def test_a_refusal_is_not_a_silent_zero():
    """The refusal has to reach the caller as a refusal. A probe emitted with the bad
    member merely dropped would ship a confident gold answer over a set nobody chose."""
    members = [memory("m1", "MoMA, Dec 2023")] + [
        memory(f"m{i}", f"place {i}") for i in range(2, 6)
    ]
    with pytest.raises(gen.Refusal) as raised:
        build(members, relation="visited")
    assert "m1" in str(raised.value)


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


# ------------------------------------------------------------------ the time boundary


def dated(mid, obj, when, **kw):
    row = memory(mid, obj, **kw)
    row["event_time"] = when
    return row


def test_a_time_window_is_stated_in_the_question_when_one_is_given():
    probe = gen.build_probe(
        "ns",
        "read",
        [dated(f"m{i}", f"book {i}", datetime(2023, 3, i + 1)) for i in range(4)],
        window=(datetime(2023, 1, 1), datetime(2023, 6, 30)),
    )

    assert "between 2023-01-01 and 2023-06-30" in probe["question"]


def test_the_window_filters_the_answer_and_not_only_the_question():
    """The window used to reach the question text alone: the sentence said "count only
    facts dated between …" while the answer counted every member, so the gold answer
    contradicted its own question. Nothing downstream could see it — the count and the
    entity list agreed with each other."""
    members = [
        dated("in1", "Dune", datetime(2023, 2, 1)),
        dated("in2", "Circe", datetime(2023, 3, 1)),
        dated("in3", "Educated", datetime(2023, 4, 1)),
        dated("out", "Neuromancer", datetime(2024, 1, 1)),
    ]

    probe = gen.build_probe(
        "ns", "read", members, window=(datetime(2023, 1, 1), datetime(2023, 6, 30))
    )

    assert probe["answer"] == 3
    assert "Neuromancer" not in probe["entities"]
    assert "out" not in probe["evidence_memory_ids"]


def test_the_window_boundary_includes_both_ends():
    """An off-by-one at the edge is the failure a reader cannot check from the answer."""
    members = [
        dated("start", "Dune", datetime(2023, 1, 1)),
        dated("mid", "Circe", datetime(2023, 3, 1)),
        dated("end", "Educated", datetime(2023, 6, 30)),
        dated("after", "Neuromancer", datetime(2023, 7, 1)),
    ]

    probe = gen.build_probe(
        "ns", "read", members, window=(datetime(2023, 1, 1), datetime(2023, 6, 30))
    )

    assert probe["answer"] == 3
    assert set(probe["evidence_memory_ids"]) == {"start", "mid", "end"}


def test_a_member_that_cannot_be_dated_is_refused_not_counted():
    """A row the query forgot to select `event_time` for would otherwise be counted into a
    time-bounded answer as if it fell inside the window."""
    members = [dated(f"m{i}", f"book {i}", datetime(2023, 3, 1)) for i in range(3)]
    members.append(memory("undated", "Neuromancer"))

    with pytest.raises(gen.Refusal, match="no event time"):
        gen.build_probe("ns", "read", members, window=(datetime(2023, 1, 1), datetime(2023, 6, 30)))


def test_an_empty_window_is_refused_rather_than_answered_zero():
    """Zero is a real answer to "how many books in 2019?" and this generator must not give
    it: the set was chosen for having members, so an empty window means the window was
    wrong, not that the user read nothing."""
    members = [dated(f"m{i}", f"book {i}", datetime(2023, 3, 1)) for i in range(4)]

    with pytest.raises(gen.Refusal, match="no member falls inside"):
        gen.build_probe(
            "ns", "read", members, window=(datetime(2019, 1, 1), datetime(2019, 12, 31))
        )


def test_a_bounded_derivation_says_what_its_dates_actually_mean():
    """The engine's event time mostly inherits the session date. A bounded gold answer
    counts facts *stated* in the window, not acts that happened in it, and the derivation
    has to say so or the number reads as something the data cannot support."""
    probe = gen.build_probe(
        "ns",
        "read",
        [dated(f"m{i}", f"book {i}", datetime(2023, 3, 1)) for i in range(4)],
        window=(datetime(2023, 1, 1), datetime(2023, 6, 30)),
    )

    assert "inherits the session date" in probe["derivation"]


# --------------------------------------------------------------------------- aliases


def test_an_alias_is_counted_twice_and_that_is_the_documented_choice():
    """`normalise_entity` is deliberately blunt: it folds case and articles and nothing
    else. "MoMA" and "Museum of Modern Art" are one museum and two members here.

    Pinned rather than fixed. Merging by similarity would also merge "Dune" with "Dune
    Messiah", and an over-merge is indistinguishable from the under-counting this set
    exists to measure. Anyone reading a count must know it is a count of surface forms."""
    probe = build(
        [
            memory("m1", "MoMA"),
            memory("m2", "Museum of Modern Art"),
            memory("m3", "Tate Modern"),
        ],
        relation="visited",
    )

    assert probe["answer"] == 3
    assert gen.normalise_entity("MoMA") != gen.normalise_entity("Museum of Modern Art")


def test_case_and_article_differences_are_not_aliases():
    """The one form of alias that is safe to fold, and the set relies on it: the same name
    written twice must not become two members."""
    probe = build(
        [
            memory("m1", "The Nightingale"),
            memory("m2", "the  nightingale"),
            memory("m3", "Circe"),
            memory("m4", "Dune"),
        ]
    )

    assert probe["answer"] == 3


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


def test_a_probe_keeps_its_id_when_a_rule_change_removes_an_earlier_one(tmp_path):
    """Ids were positions in one emission, so a refusal added ahead of a probe renumbered it:
    one regeneration changed the id of 49 of the 55 probes it kept. A review decision
    recorded against an id would then attach to a different question the next time the set
    was built. An id is now the namespace and relation the probe asks about."""
    import random
    import sqlite3

    relation_map = tmp_path / "map.csv"
    relation_map.write_text("predicate_raw,relation_type\nbooks_read,read\n", encoding="utf-8")
    namespaces = ["ns_a", "ns_b", "ns_c", "ns_d"]
    order = sorted((namespace, "read") for namespace in namespaces)
    random.Random(7).shuffle(order)
    first = order[0][0]

    def store(path, *, refuse_first):
        connection = sqlite3.connect(path)
        connection.executescript(
            "CREATE TABLE memories (id TEXT, user_id TEXT, predicate TEXT, object TEXT, "
            "scope TEXT, status TEXT, subject TEXT, event_time TEXT);"
            "CREATE TABLE turns (id TEXT); CREATE TABLE sessions (id TEXT);"
        )
        for namespace in namespaces:
            titles = ["Dune", "Circe", "Emma"]
            if refuse_first and namespace == first:
                titles[0] = "Dune on 2023-06-05 for $5"
            for i, title in enumerate(titles):
                connection.execute(
                    "INSERT INTO memories VALUES "
                    "(?, ?, 'books_read', ?, 'event', 'active', 'user', NULL)",
                    (f"{namespace}-{i}", namespace, title),
                )
        connection.commit()
        connection.close()
        return path

    before = gen.generate(store(tmp_path / "a.db", refuse_first=False), relation_map, 7, 10)
    after = gen.generate(store(tmp_path / "b.db", refuse_first=True), relation_map, 7, 10)

    ids_before = {p["namespace"]: p["probe_id"] for p in before["probes"]}
    ids_after = {p["namespace"]: p["probe_id"] for p in after["probes"]}
    assert first in ids_before and first not in ids_after
    assert ids_after == {namespace: ids_before[namespace] for namespace in ids_after}


def test_no_member_of_the_generated_set_is_one_the_current_rules_refuse():
    """The committed set must be the one these rules produce. A rule tightened without
    regenerating leaves shipped gold answers built on objects it now refuses."""
    import json
    import sqlite3

    generated = REPO / "results/analysis/entity-count-probes.json"
    store = REPO / "stores/train150.db"
    if not (generated.is_file() and store.is_file()):
        pytest.skip("the generated set and its store are local research artifacts")

    payload = json.loads(generated.read_text(encoding="utf-8"))
    connection = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    objects = dict(connection.execute("SELECT id, object FROM memories"))
    connection.close()

    refused = [
        (probe["probe_id"], objects.get(mid), gen.split_refusal(objects.get(mid)))
        for probe in payload["probes"]
        for mid in probe["evidence_memory_ids"]
        if gen.split_refusal(objects.get(mid))
    ]
    assert not refused, refused[:5]


@pytest.mark.parametrize("table", ["memories", "turns", "sessions", "evidence"])
def test_editing_evidence_without_changing_row_counts_changes_its_fingerprint(table):
    import sqlite3

    with sqlite3.connect(":memory:") as connection:
        for name in ("memories", "turns", "sessions", "evidence"):
            connection.execute(f"CREATE TABLE {name} (id TEXT, content TEXT)")
            connection.execute(f"INSERT INTO {name} VALUES ('one', 'old evidence')")
        before = gen._fingerprint(connection)
        connection.execute(f"UPDATE {table} SET content='corrected evidence' WHERE id='one'")
        assert gen._fingerprint(connection) != before


def test_evidence_fingerprint_does_not_depend_on_insertion_order():
    import sqlite3

    digests = []
    for ids in (("one", "two"), ("two", "one")):
        with sqlite3.connect(":memory:") as connection:
            for table in ("memories", "turns", "sessions"):
                connection.execute(f"CREATE TABLE {table} (id TEXT, content TEXT)")
                connection.executemany(
                    f"INSERT INTO {table} VALUES (?, ?)", [(mid, f"fact {mid}") for mid in ids]
                )
            digests.append(gen._fingerprint(connection))
    assert digests[0] == digests[1]


def test_generation_binds_the_snapshot_it_read_even_when_a_wal_writer_commits(
    tmp_path, monkeypatch
):
    """A concurrent correction must not attach the new evidence hash to old questions."""
    import sqlite3

    store = tmp_path / "evidence.db"
    relation_map = tmp_path / "map.csv"
    relation_map.write_text("predicate_raw,relation_type\nbooks_read,read\n", encoding="utf-8")
    writer = sqlite3.connect(store)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.executescript(
            "CREATE TABLE memories (id TEXT, user_id TEXT, predicate TEXT, object TEXT, "
            "scope TEXT, status TEXT, subject TEXT);"
            "CREATE TABLE turns (id TEXT); CREATE TABLE sessions (id TEXT);"
        )
        writer.executemany(
            "INSERT INTO memories VALUES (?, 'ns', 'books_read', ?, 'event', 'active', 'user')",
            [(str(i), title) for i, title in enumerate(("Dune", "Circe", "Emma"))],
        )
        writer.commit()
        original_fingerprint = gen._fingerprint
        before = original_fingerprint(writer)

        def edit_while_hashing(reader):
            writer.execute("UPDATE memories SET object='Educated' WHERE id='0'")
            writer.commit()
            return original_fingerprint(reader)

        monkeypatch.setattr(gen, "_fingerprint", edit_while_hashing)
        first = gen.generate(store, relation_map, 7, 10)
        assert "Dune" in first["probes"][0]["entities"]
        assert first["store_fingerprint"] == before

        monkeypatch.setattr(gen, "_fingerprint", original_fingerprint)
        second = gen.generate(store, relation_map, 7, 10)
        assert "Educated" in second["probes"][0]["entities"]
        assert second["store_fingerprint"] != before
        assert second["store_fingerprint_kind"] == "sqlite-evidence-content-v1"
    finally:
        writer.close()
