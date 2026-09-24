"""Temporal resolution.

The first test is the project's acceptance case: the TensorFlow → PyTorch sequence
from the README. The rest are the ways a naive implementation gets it wrong, most of
which come from ingestion order having nothing to do with event order.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from llm_long_term_memory.store import Memory, SQLiteMemoryStore
from llm_long_term_memory.temporal import TemporalResolver, as_of


@pytest.fixture
def store(tmp_path):
    s = SQLiteMemoryStore(tmp_path / "t.db")
    s.initialize()
    yield s
    s.close()


def fact(
    mid: str,
    obj: str,
    month: int,
    predicate: str = "uses_framework",
    subject: str = "user",
    ingested_day: int = 1,
    replaces_previous: bool = True,
) -> Memory:
    return Memory(
        id=mid,
        user_id="u1",
        type="semantic",
        content=f"The user uses {obj}",
        token_count=5,
        subject=subject,
        predicate=predicate,
        object=obj,
        event_time=datetime(2023, month, 1),
        valid_from=datetime(2023, month, 1),
        ingested_at=datetime(2026, 1, ingested_day),
        replaces_previous=replaces_previous,
    )


def current(store, predicate: str = "uses_framework") -> list[str]:
    """Distinct values currently in force.

    Distinct, not row count: consecutive restatements of the same value are left
    active on purpose — saying "I use PyTorch" twice did not change anything, so
    neither row supersedes the other.
    """
    return sorted(
        {m.object.strip().lower() for m in store.find_by_predicate("u1", "user", predicate)}
    )


# ------------------------------------------------------------- acceptance case


def test_the_readme_case(store):
    """Jan TensorFlow, Mar learning PyTorch, Aug switched to PyTorch.

    Exactly one fact must be left in force, with the earlier ones closed and
    pointing at what replaced them.
    """
    store.add_memories(
        [
            fact("jan", "TensorFlow", 1),
            fact("mar", "PyTorch", 3),
            fact("aug", "PyTorch", 8),
        ]
    )
    stats = TemporalResolver(store).resolve_all("u1")

    assert current(store) == ["pytorch"]
    assert stats.superseded == 1

    jan = store.get("jan")
    assert jan.status == "superseded"
    assert jan.superseded_by == "mar"
    assert jan.valid_to == datetime(2023, 3, 1)

    # Mar and Aug state the same value, so nothing changed between them. March owns
    # the interval — "when did you switch to PyTorch?" is answered by March, not
    # August — and August is folded into it rather than counted as a second change.
    assert store.get("mar").status == "active"
    assert store.get("mar").valid_to is None

    assert store.get("aug").status == "superseded"
    assert store.get("aug").superseded_by == "mar", "folded back into the owner"
    assert stats.restatements == 1
    assert stats.superseded == 1, "one real change, not two"


def test_restating_the_same_value_is_not_a_change(store):
    """ "Lives in Canberra" in March and again in June did not move anyone."""
    store.add_memories(
        [
            fact("a", "Canberra", 3, predicate="lives_in"),
            fact("b", "Canberra", 6, predicate="lives_in"),
        ]
    )
    stats = TemporalResolver(store).resolve_all("u1")

    assert stats.superseded == 0, "no change of value means no supersede"
    assert stats.restatements == 1
    assert store.get("a").status == "active", "the earlier mention owns the interval"
    assert store.get("b").superseded_by == "a"


def test_a_repeat_carrying_a_new_number_is_not_folded_away(store):
    """The fold is silent and permanent, so it must not swallow new information.

    Both of these key to the same predicate with the same object, which is all
    `_value` compares — and the second carries a figure the first does not. On
    dev50 this is `92a0aa75`: two memories about the same job title, one saying
    two years and four months and the other three years and nine months, and the
    answer needed the difference. Folded, the second was marked superseded and
    became invisible to retrieval, which only returns active rows.

    Rare in practice — one case in the clean P10 store — but silent and permanent,
    which is why it is gated rather than tolerated.
    """
    a = fact("a", "Senior Marketing Specialist", 3, predicate="job_title")
    a.content = "The user has worked in marketing for two years and 4 months."
    b = fact("b", "Senior Marketing Specialist", 6, predicate="job_title")
    b.content = (
        "The user is a Senior Marketing Specialist with 3 years and 9 months at the company."
    )
    store.add_memories([a, b])

    stats = TemporalResolver(store).resolve_all("u1")

    assert stats.restatements == 0, "different figures are not a restatement"
    assert store.get("b").status != "superseded", "the later figure must stay retrievable"


def test_a_genuine_repeat_with_the_same_numbers_is_still_folded(store):
    """The counterweight. Gating on numbers must not stop the fold from doing its
    job, or every re-mention becomes a separate fact and the timeline fills with
    duplicates."""
    a = fact("a", "Canberra", 3, predicate="lives_in")
    a.content = "The user lives in Canberra, 2 hours from the coast."
    b = fact("b", "Canberra", 6, predicate="lives_in")
    b.content = "The user still lives in Canberra, about 2 hours from the coast."
    store.add_memories([a, b])

    stats = TemporalResolver(store).resolve_all("u1")

    assert stats.restatements == 1
    assert store.get("b").superseded_by == "a"


# --------------------------------------------------------- ingestion order


def test_an_older_fact_arriving_late_does_not_win(store):
    """The failure a pairwise 'newest row wins' implementation would produce.

    August is ingested first; January arrives afterwards. Order of arrival must not
    make TensorFlow current again.
    """
    store.add_memories([fact("aug", "PyTorch", 8, ingested_day=1)])
    TemporalResolver(store).resolve_all("u1")
    assert current(store) == ["pytorch"]

    store.add_memories([fact("jan", "TensorFlow", 1, ingested_day=2)])
    TemporalResolver(store).resolve_all("u1")

    assert current(store) == ["pytorch"]
    assert store.get("jan").status == "superseded"
    assert store.get("jan").valid_to == datetime(2023, 8, 1)


def test_a_fact_landing_in_the_middle_rewires_the_chain(store):
    """Jan and Aug resolve first; then a March fact arrives between them. Jan's
    `valid_to` must move from August to March — which pairwise comparison against
    the current head cannot do, because Jan is no longer active."""
    store.add_memories([fact("jan", "TensorFlow", 1), fact("aug", "JAX", 8)])
    TemporalResolver(store).resolve_all("u1")
    assert store.get("jan").valid_to == datetime(2023, 8, 1)

    store.add_memories([fact("mar", "PyTorch", 3)])
    TemporalResolver(store).resolve_all("u1")

    assert store.get("jan").valid_to == datetime(2023, 3, 1), "rewired to the new middle"
    assert store.get("jan").superseded_by == "mar"
    assert store.get("mar").valid_to == datetime(2023, 8, 1)
    assert store.get("mar").superseded_by == "aug"
    assert current(store) == ["jax"]


def test_resolution_is_idempotent(store):
    store.add_memories([fact("jan", "TensorFlow", 1), fact("aug", "PyTorch", 8)])
    resolver = TemporalResolver(store)

    first = resolver.resolve_all("u1")
    second = resolver.resolve_all("u1")

    assert first.superseded == 1
    assert second.superseded == 0, "a settled timeline must produce no further writes"
    assert current(store) == ["pytorch"]


def test_head_is_promoted_back_when_a_later_fact_is_removed(store):
    """Resolution is not monotonic; without promotion a key could only accumulate
    superseded rows."""
    store.add_memories([fact("jan", "TensorFlow", 1), fact("aug", "PyTorch", 8)])
    resolver = TemporalResolver(store)
    resolver.resolve_all("u1")
    assert store.get("jan").status == "superseded"

    # Corrupt the state: mark the true head as superseded by the older fact, the
    # shape a half-finished run or a since-fixed bug would leave behind.
    store.mark_superseded("aug", "jan", datetime(2023, 1, 1))
    assert store.get("aug").status == "superseded"

    stats = resolver.resolve_all("u1")

    assert stats.promoted >= 1
    assert store.get("aug").status == "active", "the chronological head is restored"
    assert store.get("aug").valid_to is None
    assert current(store) == ["pytorch"]


# ------------------------------------------------------------------- guards


def test_multi_valued_predicates_are_never_superseded(store):
    """Owning a Fitbit does not stop you owning a peace lily."""
    store.add_memories(
        [
            fact("a", "peace lily", 3, predicate="owns", replaces_previous=False),
            fact("b", "Fitbit Charge 3", 8, predicate="owns", replaces_previous=False),
        ]
    )
    stats = TemporalResolver(store).resolve_all("u1")

    assert stats.superseded == 0
    assert current(store, "owns") == ["fitbit charge 3", "peace lily"]


def test_undated_memories_are_left_alone_not_guessed_at(store):
    undated = Memory(
        id="nodate",
        user_id="u1",
        type="semantic",
        content="The user uses Keras",
        token_count=5,
        subject="user",
        predicate="uses_framework",
        object="Keras",
        event_time=None,
        ingested_at=datetime(2026, 1, 1),
    )
    store.add_memories([fact("jan", "TensorFlow", 1), fact("aug", "PyTorch", 8), undated])
    stats = TemporalResolver(store).resolve_all("u1")

    assert stats.skipped_undated == 1
    assert store.get("nodate").status == "active", "no date means no basis to close it"
    assert store.get("jan").status == "superseded"


def test_a_single_fact_for_a_key_is_untouched(store):
    store.add_memories([fact("only", "PyTorch", 5)])
    stats = TemporalResolver(store).resolve_all("u1")
    assert stats.superseded == 0
    assert store.get("only").status == "active"


def test_different_subjects_do_not_collide(store):
    store.add_memories(
        [
            fact("u", "Canberra", 3, predicate="lives_in", subject="user"),
            fact("s", "Osaka", 8, predicate="lives_in", subject="sister"),
        ]
    )
    TemporalResolver(store).resolve_all("u1")
    assert store.get("u").status == "active"
    assert store.get("s").status == "active"


def test_object_comparison_ignores_formatting(store):
    store.add_memories(
        [
            fact("a", "PyTorch", 3),
            fact("b", "  pytorch  ", 8),
        ]
    )
    stats = TemporalResolver(store).resolve_all("u1")
    assert stats.superseded == 0, "same value, differently written"


# ---------------------------------------------------------------- as-of query


def test_as_of_returns_what_was_true_then(store):
    store.add_memories([fact("jan", "TensorFlow", 1), fact("aug", "PyTorch", 8)])
    TemporalResolver(store).resolve_all("u1")

    everything = store.find_by_predicate("u1", "user", "uses_framework", include_superseded=True)

    april = as_of(everything, datetime(2023, 4, 1))
    assert [m.object for m in april] == ["TensorFlow"]

    december = as_of(everything, datetime(2023, 12, 1))
    assert [m.object for m in december] == ["PyTorch"]

    # Before anything was known.
    assert as_of(everything, datetime(2022, 1, 1)) == []


def test_resolve_memories_only_touches_keys_in_the_batch(store):
    store.add_memories(
        [
            fact("jan", "TensorFlow", 1),
            fact("aug", "PyTorch", 8),
            fact("c1", "Canberra", 3, predicate="lives_in"),
            fact("c2", "Sydney", 9, predicate="lives_in"),
        ]
    )
    resolver = TemporalResolver(store)
    stats = resolver.resolve_memories([store.get("jan")])

    assert stats.keys_examined == 1
    assert store.get("jan").status == "superseded"
    assert store.get("c1").status == "active", "untouched key not yet resolved"


# ------------------------------------------------------------------- repair


def test_a_wrong_arity_call_is_undone_on_re_resolution(store, monkeypatch):
    """The failure this actually caught: `has_goal` was on the single-valued list,
    so a real ingest chained unrelated goals into a supersede sequence and hid 68
    true facts. Fixing the list has to give them back — rebuilding a store costs
    hours of quota, so "you must re-ingest" is not an acceptable remedy.
    """
    import llm_long_term_memory.temporal.resolve as resolve_mod

    store.add_memories(
        [
            fact("g1", "read 20 books", 3, predicate="has_goal", replaces_previous=False),
            fact("g2", "marathon training", 5, predicate="has_goal", replaces_previous=False),
        ]
    )

    # Put the store into the damaged state directly. Constructing it through the
    # resolver no longer works, and that is the point: closing a fact now requires
    # its successor to say so, so a wrong arity call alone cannot retire anything.
    # The repair path still has to exist for stores damaged before that fix.
    from datetime import datetime

    store.mark_superseded("g1", "g2", datetime(2023, 5, 1))
    assert store.get("g1").status == "superseded", "the damaged state to repair"

    # Now with the corrected list.
    monkeypatch.setattr(resolve_mod, "is_single_valued", lambda p: False)
    stats = TemporalResolver(store).resolve_all("u1")

    assert stats.repaired == 1
    assert store.get("g1").status == "active"
    assert store.get("g1").valid_to is None
    assert sorted(current(store, "has_goal")) == ["marathon training", "read 20 books"]


def test_repair_is_idempotent(store, monkeypatch):
    import llm_long_term_memory.temporal.resolve as resolve_mod

    monkeypatch.setattr(resolve_mod, "is_single_valued", lambda p: False)
    store.add_memories(
        [
            fact("g1", "read 20 books", 3, predicate="has_goal", replaces_previous=False),
            fact("g2", "marathon training", 5, predicate="has_goal", replaces_previous=False),
        ]
    )
    resolver = TemporalResolver(store)
    assert resolver.resolve_all("u1").repaired == 0
    assert resolver.resolve_all("u1").repaired == 0


def test_goals_and_schedules_are_multi_valued():
    """Both were on the list and both were wrong: a person holds many goals and
    many scheduled events at once."""
    from llm_long_term_memory.ingest import is_single_valued

    assert not is_single_valued("has_goal")
    assert not is_single_valued("scheduled")
    assert not is_single_valued("owns")
    assert is_single_valued("lives_in")
    assert is_single_valued("works_as")


def test_a_coexisting_successor_does_not_close_its_predecessor(store):
    """Resolvability is per key; closing a fact is per successor.

    Found by running the real pipeline after Stage B's gate had scored 0% false
    supersede in isolation. One `replaces` anywhere on a key made the whole key
    resolvable, and the chain rewrite then retired every consecutive pair on it —
    including successors that had explicitly said `coexists`. Live example: "The
    user has been averaging around $100 per week on groceries" was retired by "The
    user spent around $75 at Walmart last Saturday".
    """
    store.add_memories(
        [
            fact("avg", "$100 per week", 3, predicate="grocery_spending", replaces_previous=False),
            fact(
                "trip", "$75 at Walmart", 5, predicate="grocery_spending", replaces_previous=False
            ),
            # A genuine change on the same key, which is what made the key resolvable.
            fact("moved", "$150 per week", 7, predicate="grocery_spending", replaces_previous=True),
        ]
    )
    TemporalResolver(store).resolve_all("u1")

    assert store.get("avg").status == "active", "a coexisting successor must not retire it"
    assert store.get("moved").status == "active"

    # There are two live predecessors and the replacement signal does not identify
    # which one it means. Event order is not a valid tie-breaker, so the resolver
    # abstains rather than hiding the Walmart trip or the weekly average.
    assert store.get("trip").status == "active"
    assert store.get("trip").superseded_by is None
    assert TemporalResolver(store).resolve_all("u1").skipped_ambiguous == 1


def test_a_replacing_successor_still_closes_its_predecessor(store):
    """The other half: the fix must not disable supersede altogether."""
    store.add_memories(
        [
            fact("old", "Canberra", 3, predicate="home_city", replaces_previous=False),
            fact("new", "Sydney", 8, predicate="home_city", replaces_previous=True),
        ]
    )
    TemporalResolver(store).resolve_all("u1")

    assert store.get("old").status == "superseded"
    assert store.get("old").superseded_by == "new"
    assert store.get("new").status == "active"


def test_a_removal_is_not_folded_into_the_value_it_ends(store):
    """Selling a Honda repeats its name but is not a second ownership statement."""
    owned = fact("owned", "Honda Civic", 3, predicate="car", replaces_previous=False)
    removed = fact("sold", "Honda Civic", 7, predicate="car", replaces_previous=True)
    removed.content = "The user sold their Honda Civic"
    removed.update_op = "removes"
    removed.target_object = "Honda Civic"
    removed.object = ""
    store.add_memories([owned, removed])

    TemporalResolver(store).resolve_all("u1")

    assert store.get("owned").status == "superseded"
    assert store.get("owned").superseded_by == "sold"
    assert store.get("sold").status == "historical"
    assert current(store, "car") == []


def test_a_termination_targets_one_coexisting_value_without_becoming_current(store):
    """Removing Nike must not hide Adidas or turn Nike into a successor value."""
    nike = fact(
        "nike",
        "Nike Air Zoom Pegasus 38",
        1,
        predicate="running_shoes",
        replaces_previous=False,
    )
    adidas = fact(
        "adidas", "Adidas Ultraboost 22", 2, predicate="running_shoes", replaces_previous=False
    )
    removed = fact("removed", "", 3, predicate="running_shoes", replaces_previous=True)
    removed.content = "The user replaced their Nike Air Zoom Pegasus 38 shoes."
    removed.update_op = "removes"
    removed.target_object = "Nike Air Zoom Pegasus 38"
    store.add_memories([nike, adidas, removed])

    TemporalResolver(store).resolve_all("u1")

    assert store.get("nike").status == "superseded"
    assert store.get("adidas").status == "active"
    assert store.get("removed").status == "historical"
    assert current(store, "running_shoes") == ["adidas ultraboost 22"]


def test_the_measured_legacy_nike_sentence_is_replayed_as_a_termination(store):
    """Old stores have no target_object and contain the known wrong REPLACE verdict."""
    adidas = fact(
        "adidas", "Adidas Ultraboost 22", 1, predicate="running_shoes", replaces_previous=False
    )
    legacy = fact(
        "legacy", "Nike Air Zoom Pegasus 38", 2, predicate="running_shoes", replaces_previous=True
    )
    legacy.content = "The user replaced their Nike Air Zoom Pegasus 38 shoes around February 10."
    legacy.update_op = "replaces"
    legacy.target_object = None
    store.add_memories([adidas, legacy])

    TemporalResolver(store).resolve_all("u1")

    assert store.get("adidas").status == "active"
    assert store.get("legacy").status == "historical"


def test_an_ambiguous_termination_hides_nothing(store):
    first = fact("first", "red shoes", 1, predicate="shoes", replaces_previous=False)
    second = fact("second", "blue shoes", 2, predicate="shoes", replaces_previous=False)
    removed = fact("removed", "", 3, predicate="shoes", replaces_previous=True)
    removed.content = "The user stopped wearing those shoes."
    removed.update_op = "removes"
    removed.target_object = None
    store.add_memories([first, second, removed])

    stats = TemporalResolver(store).resolve_all("u1")

    assert stats.skipped_ambiguous == 1
    assert current(store, "shoes") == ["blue shoes", "red shoes"]
    assert store.get("removed").status == "historical"
