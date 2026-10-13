from datetime import datetime

import numpy as np
import pytest

from llm_long_term_memory.store import Memory, Session, SQLiteMemoryStore, Turn
from llm_long_term_memory.temporal import TemporalResolver, as_of
from llm_long_term_memory.temporal.transition import (
    TransitionVerdict,
    apply_update,
    candidate_predecessors,
    decide,
    transition_prompt,
)


def _mem(mid, user, predicate, content, day, obj=None):
    when = datetime(2023, 1, day)
    return Memory(
        id=mid,
        user_id=user,
        type="semantic",
        content=content,
        token_count=5,
        subject="user",
        predicate=predicate,
        object=obj,
        status="active",
        valid_from=when,
        observed_at=when,
        ingested_at=when,
        source_session_id=f"{user}-s{day}",
        source_turn_index=0,
    )


VECTORS = {
    "old": np.array([1.0, 0.0]),
    "new": np.array([0.99, 0.14]),
    "other": np.array([0.0, 1.0]),
    "bob": np.array([1.0, 0.0]),
}


def test_candidates_stay_within_the_user_and_rank_by_similarity():
    old = _mem("old", "alice", "team_size", "leads 4 engineers", 1)
    new = _mem("new", "alice", "team_count", "leads 5 engineers", 9)
    other = _mem("other", "alice", "hobby", "likes chess", 2)
    bob = _mem("bob", "bob", "team_size", "leads 4 engineers", 1)
    ranked = candidate_predecessors(new, [old, other, bob], VECTORS.get, k=5)
    assert [c.memory.id for c in ranked] == ["old", "other"]  # bob never appears
    assert ranked[0].same_key is False
    later_only = candidate_predecessors(old, [new], VECTORS.get, k=5)
    assert later_only == []  # earlier_only by default
    assert candidate_predecessors(old, [new], VECTORS.get, k=5, earlier_only=False)


class Client:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)

        class Completion:
            pass

        c = Completion()
        c.text = self.text
        return c


def test_decide_maps_labels_and_defaults_to_uncertain():
    old = _mem("old", "alice", "team_size", "leads 4 engineers", 1)
    new = _mem("new", "alice", "team_count", "leads 5 engineers", 9)
    top = candidate_predecessors(new, [old], VECTORS.get, k=5)
    reply = TransitionVerdict.model_validate(
        {"decisions": [{"candidate": "E1", "reason": "count changed", "relation": "replace"}]}
    ).model_dump_json()
    client = Client(reply)
    assert decide(client, "m", new, top) == {"old": "replace"}
    assert "FACT (2023-01-09)" in client.calls[0]["prompt"]
    assert "E1 (2023-01-01)" in transition_prompt(new, top)
    assert decide(Client("not json"), "m", new, top) == {"old": "uncertain"}


@pytest.fixture
def store(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "m.db")
    store.initialize()
    for day in (1, 9):
        started = datetime(2023, 1, day)
        sid = f"alice-s{day}"
        store.add_session(
            Session(sid, "alice", started, turns=[Turn(f"{sid}-0", sid, 0, "user", "x", started)])
        )
    yield store
    store.close()


@pytest.mark.parametrize("arrival", [("old", "new"), ("new", "old")])
def test_a_confirmed_update_converges_whichever_fact_arrives_last(store, arrival):
    facts = {
        "old": _mem("old", "alice", "team_size", "leads 4 engineers", 1, obj="4"),
        "new": _mem("new", "alice", "team_count", "leads 5 engineers", 9, obj="5"),
    }
    store.add_memories([facts[arrival[0]]])
    store.add_memories([facts[arrival[1]]])
    resolver = TemporalResolver(store)
    earlier, later = apply_update(store, resolver, store.get(arrival[1]), store.get(arrival[0]))
    assert (earlier.id, later.id) == ("old", "new")
    old, new = store.get("old"), store.get("new")
    assert (new.subject, new.predicate) == ("user", "team_size")
    assert old.status == "superseded" and old.superseded_by == "new"
    assert new.status == "active"
    assert [m.id for m in as_of([old, new], datetime(2023, 1, 5))] == ["old"]


def test_an_update_never_crosses_users(store):
    a = _mem("a", "alice", "team_size", "leads 4", 1)
    b = _mem("b", "bob", "team_size", "leads 5", 9)
    with pytest.raises(ValueError, match="users"):
        apply_update(store, TemporalResolver(store), a, b)


def test_pairs_are_ordered_by_the_resolvers_clock(store):
    # Said later, but about an earlier time: the resolver will place it first, so the
    # update must too, or the two would close the wrong fact.
    told_late = _mem("late", "alice", "sessions", "attended five sessions last year", 9)
    told_late.event_time = datetime(2022, 6, 1)
    told_early = _mem("early", "alice", "sessions_count", "attended three sessions", 1)
    store.add_memories([told_late, told_early])
    earlier, later = apply_update(
        store, TemporalResolver(store), store.get("late"), store.get("early")
    )
    assert (earlier.id, later.id) == ("late", "early")
    assert store.get("late").status == "superseded"


def test_a_spelled_out_new_count_is_never_folded_as_a_restatement(store):
    three = _mem("three", "alice", "support_group", "attended three sessions", 1, obj="group")
    five = _mem("five", "alice", "support_group", "attended five sessions", 9, obj="group")
    five.replaces_previous = True
    store.add_memories([three, five])
    TemporalResolver(store).resolve_all("alice")
    assert store.get("five").status == "active"
    assert store.get("three").status == "superseded" and store.get("three").valid_to
