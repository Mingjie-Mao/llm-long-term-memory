import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.evaluation.grounded.answering import build_ledger
from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
from llm_long_term_memory.store import NumpyFlatIndex, Session, SQLiteMemoryStore, Turn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from t4_selection_repair import (
    SelectionRepairRunner,
    collection_category_gap,
    owned_session_turns,
    selection_plan,
)


@pytest.fixture
def store(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.initialize()
    yield store
    store.close()


def add(store, sid, *texts, user="alice", date="2026-09-01"):
    at = datetime.fromisoformat(date)
    turns = [
        Turn(f"{sid}:{i}", sid, i, "user" if i % 2 == 0 else "assistant", text, at)
        for i, text in enumerate(texts)
    ]
    store.add_session(Session(sid, user, at, "test", turns))
    return turns


def req(question):
    return AnswerRequest(question, "2026-10-01", "alice")


def test_missing_collection_category_does_not_borrow_other_count_or_claim_zero(store):
    turns = add(store, "mine", "I collected 19 signed coins.")
    ledger = build_ledger(store, "alice", [], turns)
    gap = collection_category_gap(ledger, "How many signed posters are in my collection?")
    assert gap.computed and "don't have evidence" in gap.answer
    assert "19" not in gap.answer and "zero" not in gap.answer
    assert collection_category_gap(ledger, "How many signed coins are in my collection?") is None
    assert collection_category_gap(ledger, "How many weeks did I spend collecting coins?") is None


def test_song_prefix_recovers_second_artifact_with_revision_and_no_foreign_sources(store):
    originals = add(
        store,
        "mine",
        "Create a sad song with notes",
        "Chorus:\nG A B\nLyrics",
        "Make it romantic",
        "Chorus:\nC D E F\nDifferent lyrics",
    )
    add(
        store,
        "foreign",
        "Create a sad song with notes",
        "Chorus:\nA B C\nLyrics",
        "Make it romantic",
        "Chorus:\nF G A\nLyrics",
        user="bob",
    )
    result, ledger, _ = selection_plan(
        store,
        req("In our conversation, what was the chorus progression in the second song?"),
        originals[:2],
    )
    assert result.answer == "C D E F"
    assert [s.turn_index for s in ledger.sources] == [0, 1, 2, 3]
    assert all(s.session_id == "mine" for s in ledger.sources)
    with pytest.raises(ValueError, match="tenant"):
        owned_session_turns(store, "alice", "foreign")


def test_two_qualifying_song_sessions_are_not_silently_disambiguated(store):
    for sid, notes in [("a", "C D E"), ("b", "G A B")]:
        add(
            store,
            sid,
            "Create a song",
            "Chorus:\nG A B\nLyrics",
            "Revise it",
            "Chorus:\n" + notes + "\nLyrics",
        )
    result, _, _ = selection_plan(
        store, req("What was the chorus progression in our second song?"), []
    )
    assert result is None


def test_coupon_inference_uses_same_user_session_and_labels_uncertainty(store):
    turns = add(
        store,
        "mine",
        "I redeemed a coupon on milk, found in my email inbox.",
        "Nice",
        "I shop at Shop One frequently.",
    )
    add(store, "foreign", "I shop at Shop Two.", user="bob")
    result, _, _ = selection_plan(store, req("Where did I redeem the coupon?"), turns[:1])
    assert result.answer.startswith("Likely Shop One") and result.detail["inferred"]
    assert "Shop Two" not in result.answer and "email" not in result.answer


def test_period_gap_retains_literal_counts_and_requests_overlap_clarification(store):
    a = add(store, "a", "I attended three sessions of the art group.")
    b = add(
        store, "b", "The art group I attended last year helped. I remember attending five sessions."
    )
    result, _, _ = selection_plan(
        store, req("How many sessions of the art group did I attend?"), a + b
    )
    assert result.computed and "combined total cannot" in result.answer
    assert "overlap" in result.answer and len(result.detail["quantity_citations"]) == 2


def test_unrelated_lookup_has_no_deterministic_override(store):
    turns = add(store, "mine", "My kayak cost $640.")
    result, _, _ = selection_plan(store, req("How much did my kayak cost?"), turns)
    assert result is None


def test_repaired_reader_records_actual_extra_context_and_unchanged_model(store, tmp_path):
    add(store, "mine", "I collected 19 signed coins.")

    class Client:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            from llm_long_term_memory.llm.client import Completion

            self.calls.append(kwargs)
            return Completion(
                text='{"notes":"wrong category","status":"answer","answer":"19"}',
                model=kwargs["model"],
                input_tokens=100,
                output_tokens=10,
                thinking_tokens=0,
                attempts=1,
                api_latency_ms=1,
            )

    class Encoder:
        def encode_one(self, text):
            return np.ones(2, dtype=np.float32)

    base = MemoryRunner(
        Client(),
        "original-model",
        Encoder(),
        store,
        NumpyFlatIndex(tmp_path / "i", dim=2),
        answer_policy="v2_clt",
        raw_primary_tokens=4000,
        raw_primary_prefer_user=True,
        raw_primary_time_notes=True,
        raw_primary_time_window=True,
        raw_fallback=True,
    )
    candidate = SelectionRepairRunner.from_t4(base)
    answer = candidate.answer_request(req("How many signed posters are in my collection?"))
    assert "don't have evidence" in answer.text and len(candidate.client.calls) == 1
    call = candidate.client.calls[0]
    assert call["model"] == "original-model"
    assert "Source-bound selection" in call["prompt"]
    assert answer.context_tokens > 100
    assert answer.notes["selection_repair"]["calculation"]["requested_category"] == "poster"
    assert type(base) is MemoryRunner  # Installation does not mutate accepted t4.


@pytest.mark.parametrize("failed_gap", [None, "f9e8c073"])
@pytest.mark.parametrize(
    "module_name", ["t4_scoped_selection_acceptance", "t4_event_count_acceptance"]
)
def test_new_acceptance_keeps_original_gold_separate_from_two_faithful_gaps(
    tmp_path, monkeypatch, failed_gap, module_name
):
    import importlib
    from types import SimpleNamespace

    from llm_long_term_memory.llm.usage import UsageTracker

    tool = importlib.import_module(module_name)
    supported = 21 - len(tool.GAP_GOLD)
    ids = [*tool.GAP_GOLD, *(f"supported-{n}" for n in range(supported))]
    instances = [
        SimpleNamespace(question_id=q, question_type="multi-session", answer="gold") for q in ids
    ]
    readers = {q: {} for q in ids}
    originals = {q: {"correct": q not in tool.GAP_GOLD} for q in ids}
    gaps = {q: {"correct": q != failed_gap} for q in tool.GAP_GOLD}
    monkeypatch.setattr(tool, "REPO", tmp_path)
    monkeypatch.setattr(tool, "BASE", tmp_path / "rows")
    (tmp_path / "results/analysis").mkdir(parents=True)
    for suffix in ("readers.jsonl", "grades.jsonl", "gap-grades.jsonl"):
        tool.BASE.with_suffix("." + suffix).touch()
    monkeypatch.setattr(
        tool,
        "rows_by_question",
        lambda p: gaps if "gap-grades" in p.name else originals if "grades" in p.name else readers,
    )
    monkeypatch.setattr(tool, "source_coverage", lambda i, r: {"complete": True})
    monkeypatch.setattr(tool, "replace", lambda i, **kwargs: i)
    monkeypatch.setattr(
        tool,
        "combine",
        lambda i, r, g, *args: SimpleNamespace(correct=g["correct"], context_tokens=6400),
    )
    result = tool.report(instances, "new-inventory", UsageTracker())
    assert result["original_correct"] == result["supported_correct"] == supported
    assert result["supported_total"] == supported
    assert result["pass"] is (failed_gap is None)
    assert originals["f9e8c073"]["correct"] is False
