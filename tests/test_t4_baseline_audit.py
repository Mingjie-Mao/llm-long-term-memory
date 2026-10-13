import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import t4_baseline_audit as audit  # noqa: E402


def _record(**delivery):
    base = {"gold_turns": 2, "delivered_per_run": [2, 2, 2]}
    return {"is_abstention": False, "delivery": {**base, **delivery}}


def test_auto_layer_needs_every_run_to_deliver_every_gold_turn():
    assert audit.auto_layer(_record()) == "evidence_complete"
    assert audit.auto_layer(_record(delivered_per_run=[2, 1, 2])) == "evidence_incomplete"
    assert audit.auto_layer({**_record(), "is_abstention": True}) == "abstention"


def test_committed_report_is_fully_labelled_and_consistent():
    report = json.loads((REPO / "results/analysis/t4-baseline-v1.json").read_text(encoding="utf-8"))
    notes = json.loads((REPO / "research/t4-first-loss-notes.json").read_text(encoding="utf-8"))
    wrong = {r["question_id"] for r in report["wrong"]}
    assert report["unlabelled"] == [] and report["stale_notes"] == []
    assert wrong == {q for q in notes if not q.startswith("_")}
    assert len(report["regression_set"]) == 200
    assert all(r["delivery"]["replay_matches_recorded_turns"] for r in report["wrong"])
    allowed = {"answer", "retrieval", "grading", "gold_scope"}
    assert {notes[q]["first_loss"] for q in wrong} <= allowed
    # Ambiguous items are exactly the ones the notes mark, and never answer-layer targets.
    assert set(report["ambiguous"]) == {q for q in wrong if notes[q].get("ambiguous")}
    runs = [run["correct"] for split in report["baseline"].values() for run in split["runs"]]
    assert runs == [89, 86, 88, 88, 89, 88]
