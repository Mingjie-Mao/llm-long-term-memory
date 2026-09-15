"""Free falsification of retrieval-side candidates, and the one claim it may not make."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

replay = pytest.importorskip("retrieval_replay")


def test_a_sweep_crosses_every_axis():
    configurations = replay.parse_sweep(["top_k=10,20", "rerank=false,true"])
    assert len(configurations) == 4
    assert {"top_k": 10, "rerank": False} in configurations
    assert {"top_k": 20, "rerank": True} in configurations


def test_no_sweep_is_one_default_configuration():
    assert replay.parse_sweep([]) == [{}]


class Runner:
    """Answers from a script, the way the stubbed answerer does in a replay."""

    def __init__(self, notes_by_id):
        self.notes_by_id = notes_by_id

    def answer(self, instance):
        from types import SimpleNamespace

        return SimpleNamespace(
            notes=self.notes_by_id[instance.question_id], context_tokens=100, text=""
        )


class Instance:
    def __init__(self, question_id, evidence):
        self.question_id = question_id
        self.answer_session_ids = evidence


def notes(selected, coverage, ranked=("m1",)):
    return {
        "recall_stages": {"selected": selected},
        "recall_coverage": {"selected": coverage},
        "ranked_memory_ids": list(ranked),
        "retrieval": [{"memory_id": "m1"}],
    }


def test_any_source_and_all_sources_are_reported_separately():
    """The whole point: a single source-recall number saturates and stops discriminating,
    while complete coverage keeps moving."""
    instances = [Instance("q1", ["s1", "s2"]), Instance("q2", ["s1"])]
    runner = Runner({"q1": notes(True, 0.5), "q2": notes(True, 1.0)})
    outcome = replay.replay(runner, instances)
    assert outcome["any_source_session"] == pytest.approx(1.0)
    assert outcome["all_source_sessions"] == pytest.approx(0.5)


def test_questions_without_labelled_evidence_are_left_out_of_recall():
    """An abstention question has no source session, and counting it as a recall failure
    would penalise exactly the behaviour it tests."""
    instances = [Instance("q1", []), Instance("q2", ["s1"])]
    runner = Runner({"q1": notes(False, None), "q2": notes(True, 1.0)})
    outcome = replay.replay(runner, instances)
    assert outcome["questions"] == 2
    assert outcome["questions_with_labelled_evidence"] == 1
    assert outcome["any_source_session"] == pytest.approx(1.0)


def test_the_tool_says_it_cannot_speak_for_the_answerer():
    """Layer 4 is out of reach with a stubbed answerer, and the artifact has to say so
    rather than leave a reader to assume a retrieval win is an accuracy win."""
    source = (REPO / "tools/retrieval_replay.py").read_text(encoding="utf-8")
    assert "retrieval only" in source
    assert "out of reach" in source


def test_a_store_writing_axis_cannot_be_swept_in_one_run():
    """Eviction and decay rewrite the store, so the second value of the axis measures
    what the first left behind. Two caps reported identical numbers that way, and the
    second was a no-op wearing the first one's result."""
    with pytest.raises(ValueError, match="one value per run"):
        replay.parse_sweep(["evict_to=200,400"])
    with pytest.raises(ValueError, match="one value per run"):
        replay.parse_sweep(["decay_halflife=30,60"])
    assert replay.parse_sweep(["evict_to=200"]) == [{"evict_to": 200}]


def test_an_unknown_axis_is_refused_rather_than_silently_ignored():
    with pytest.raises(ValueError, match="unknown axis"):
        replay.parse_sweep(["topk=10"])
