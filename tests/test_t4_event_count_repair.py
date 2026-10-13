import sys
from datetime import datetime
from pathlib import Path

import pytest

from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.store import Session, SQLiteMemoryStore, Turn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from t4_event_count_repair import EventCountRepairRunner, unresolved_event_references
from t4_selection_repair import SelectionRepairRunner


@pytest.fixture
def store(tmp_path):
    s = SQLiteMemoryStore(tmp_path / "s.db")
    s.initialize()
    yield s
    s.close()


def add(store, sid, text, user="alice", role="user"):
    at = datetime(2026, 10, 15)
    store.add_session(Session(sid, user, at, "test", [Turn(sid + ":0", sid, 0, role, text, at)]))


def request():
    return AnswerRequest("How many weddings have I attended in this year?", "2026-10-15", "alice")


def ambiguous(store, user="alice", role="user"):
    add(store, "a", "I've been to a few weddings; my cousin's wedding was in August.", user, role)
    add(store, "b", "I last wore the locket to my cousin's wedding last month.", user, role)
    add(
        store,
        "c",
        "My sister's wedding was amazing. It was special to be a part of it.",
        user,
        role,
    )


def test_indirect_attendance_and_undated_participation_are_not_forced_into_exact_total(store):
    ambiguous(store)
    result, ledger = unresolved_event_references(store, request())
    assert result.computed and "exact total" in result.answer and "clarify" in result.answer
    assert result.detail["repeated_unnamed_relations"] == ["cousin"]
    assert len(result.detail["mentions"]) == 3 and len(ledger.sources) == 3
    assert "3 weddings" not in result.answer and "5 weddings" not in result.answer


@pytest.mark.parametrize("user,role", [("bob", "user"), ("alice", "assistant")])
def test_other_users_and_assistant_illustrations_cannot_create_gap(store, user, role):
    ambiguous(store, user, role)
    assert unresolved_event_references(store, request()) == (None, None)


def test_all_named_dated_events_and_repeated_reports_are_not_rejected(store):
    add(store, "a", "I attended my cousin Mia's wedding in August.")
    add(store, "b", "I attended my friend Ana's wedding in September.")
    add(store, "c", "I attended my sister Bea's wedding in October.")
    assert unresolved_event_references(store, request()) == (None, None)


def test_unrelated_date_does_not_date_the_wedding(store):
    ambiguous(store)
    add(
        store,
        "d",
        "My birthday was in September. My brother's wedding was lovely. "
        "It was special to be a part of it.",
    )
    result, _ = unresolved_event_references(store, request())
    brother = next(m for m in result.detail["mentions"] if m["relation"] == "brother")
    assert not brother["dated"] and "birthday" not in brother["event_clause"]


def test_future_plans_and_undated_single_reference_do_not_force_a_gap(store):
    add(store, "a", "I will go to my cousin's wedding next month.")
    add(store, "b", "My sister's wedding was amazing. It was special to be a part of it.")
    assert unresolved_event_references(store, request()) == (None, None)


def test_event_sources_replace_irrelevant_raw_and_account_for_smaller_actual_context(
    store, monkeypatch
):
    ambiguous(store)
    candidate = object.__new__(EventCountRepairRunner)
    candidate.store = store
    candidate.chars_per_token = 4.6
    received = []

    def parent(self, request, context, selected):
        received.append(context)
        self._selection_added_tokens = 25
        return object(), "unverified five", None, None

    monkeypatch.setattr(SelectionRepairRunner, "_answer_with_fallback", parent)
    marker = "Excerpts from the user's earlier conversations, found for this question"
    original = "memory state\n" + marker + "irrelevant raw" * 2000
    _, text, _, _ = candidate._answer_with_fallback(request(), original, [])
    assert "irrelevant raw" not in received[0] and "memory state" in received[0]
    assert "last month" in received[0] and "August" in received[0]
    assert "exact total" in text
    assert candidate._selection_added_tokens == 25 + int(len(received[0]) / 4.6) - int(
        len(original) / 4.6
    )


def test_v3_coverage_recognizes_literal_original_relative_date_source():
    from t4_event_count_acceptance import source_coverage

    from llm_long_term_memory.evaluation.datasets.longmemeval import (
        HaystackSession,
        HaystackTurn,
        Instance,
    )

    text = "I attended the wedding last weekend."
    instance = Instance(
        "q",
        "multi-session",
        "when",
        "gold",
        "2023-10-15",
        [HaystackSession("s", "2023-10-15", [HaystackTurn("user", text, True)])],
        ["s"],
    )
    reader = {"answer": {"notes": {"actual_answer_calls": [{"request": {"prompt": text}}]}}}
    assert source_coverage(instance, reader)["complete"]
