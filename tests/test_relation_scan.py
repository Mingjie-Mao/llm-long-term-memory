"""An exhaustive scan must preserve its subject, namespace and abstention boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_long_term_memory.retrieve.relation_router import Route
from llm_long_term_memory.retrieve.scan import RelationScanner
from llm_long_term_memory.store import Memory


def memory(mid, user="alice", subject="user", predicate="read"):
    return Memory(
        mid, user, "episodic", f"{subject} read {mid}", 5, subject=subject, predicate=predicate
    )


def scanner(memories, routed=True, max_added=40):
    route = Route(
        relations=("read",) if routed else (),
        predicted_class="read",
        similarity=0.8,
        margin=0.5 if routed else 0.01,
    )
    return RelationScanner(
        SimpleNamespace(iter_active=lambda namespace: memories),
        SimpleNamespace(route=lambda question: route),
        {"read": "read"},
        max_added=max_added,
    )


def test_scan_never_adds_another_subject_or_namespace():
    scan = scanner(
        [
            memory("user-book"),
            memory("assistant-book", subject="assistant"),
            memory("other-user", user="bob"),
            memory("other-relation", predicate="watched"),
        ]
    )
    outcome = scan.expand("How many books has the user read?", "alice", [])
    assert [hit.memory.id for hit in outcome.memories] == ["user-book"]
    assert outcome.added == 1


@pytest.mark.parametrize("routed,max_added", [(False, 40), (True, 0)])
def test_abstention_and_oversized_scan_preserve_the_original_ranking(routed, max_added):
    original = [SimpleNamespace(memory=memory("ranked"))]
    scan = scanner([memory("extra")], routed, max_added)
    outcome = scan.expand("books", "alice", original)
    assert outcome.memories is original
    assert outcome.added == 0


def test_existing_hits_are_not_added_twice():
    original = [SimpleNamespace(memory=memory("ranked"))]
    outcome = scanner([memory("ranked"), memory("extra")]).expand("books", "alice", original)
    assert [hit.memory.id for hit in outcome.memories] == ["ranked", "extra"]
    assert outcome.added == 1
