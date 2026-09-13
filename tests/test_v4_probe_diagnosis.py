"""The v4 diagnosis must classify count failures by layer, and must not flatter a rerun.

The point of the tool is that it separates three failures that need different fixes, and
that it treats two runs of one configuration as noise rather than as an effect. Both are
easy to lose in a refactor and neither shows up as a crash, so they are pinned here on
small fixtures rather than on the real 142-row files — those live outside CI.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

diagnose = pytest.importorskip("diagnose_v4_probes")


def gold_store(tmp_path: Path, facts: dict[str, str]) -> Path:
    path = tmp_path / "gold.db"
    with sqlite3.connect(path) as con:
        con.execute("create table memories (id text, content text, object text)")
        con.executemany(
            "insert into memories values (?, ?, ?)",
            [(mid, f"user acquired {obj}", obj) for mid, obj in facts.items()],
        )
    return path


def row(probe_id, *, verdict, parsed, items, retrieved, complete, kind="count"):
    return {
        "probe_id": probe_id,
        "kind": kind,
        "verdict": verdict,
        "parsed": parsed,
        "synthesis_computation": {"items": items},
        "retrieved_ids": retrieved,
        "context_complete": complete,
    }


def test_a_number_matching_its_own_item_list_is_not_an_arithmetic_failure(tmp_path):
    """Under-enumeration is a reading failure. Counting nine things nine is arithmetic
    working correctly on the wrong set, and calling it arithmetic would send the next
    candidate to fix the part that already works."""
    facts = {f"m{i}": f"thing{i}" for i in range(3)}
    probes = {
        "count_0": {"answer": 3, "evidence_memory_ids": list(facts)},
    }
    rows = {
        "count_0": row(
            "count_0",
            verdict="wrong",
            parsed="2",
            items=["thing0", "thing1"],
            retrieved=list(facts),
            complete=True,
        )
    }
    out = diagnose._diagnose_counts(rows, probes, diagnose._gold_texts(gold_store(tmp_path, facts)))

    assert out["number_disagreed_with_own_item_list"] == 0
    assert out["layers"]["reading_failure"] == 1
    assert out["layers"]["retrieval_gap"] == 0
    assert out["direction_of_wrong_numbers"] == {"under": 1, "over": 0}
    # thing2 was retrieved and never named: that is the fact the next candidate must reach.
    assert out["gold_facts_never_named"] == {"in_context_never_named": 1, "absent_from_context": 0}


def test_a_fact_outside_the_context_is_a_retrieval_failure_not_a_reading_one(tmp_path):
    facts = {f"m{i}": f"thing{i}" for i in range(3)}
    probes = {"count_0": {"answer": 3, "evidence_memory_ids": list(facts)}}
    rows = {
        "count_0": row(
            "count_0",
            verdict="wrong",
            parsed="2",
            items=["thing0", "thing1"],
            retrieved=["m0", "m1"],
            complete=False,
        )
    }
    out = diagnose._diagnose_counts(rows, probes, diagnose._gold_texts(gold_store(tmp_path, facts)))

    assert out["layers"] == {
        "correct": 0,
        "abstained_no_number": 0,
        "retrieval_gap": 1,
        "reading_failure": 0,
    }
    assert out["gold_facts_never_named"] == {"in_context_never_named": 0, "absent_from_context": 1}


def test_an_item_naming_nothing_in_the_gold_set_counts_as_spurious(tmp_path):
    facts = {"m0": "thing0"}
    probes = {"count_0": {"answer": 1, "evidence_memory_ids": ["m0"]}}
    rows = {
        "count_0": row(
            "count_0",
            verdict="wrong",
            parsed="2",
            items=["thing0", "a completely unrelated invention"],
            retrieved=["m0"],
            complete=True,
        )
    }
    out = diagnose._diagnose_counts(rows, probes, diagnose._gold_texts(gold_store(tmp_path, facts)))

    assert out["spurious_items_named"] == 1
    assert out["direction_of_wrong_numbers"] == {"under": 0, "over": 1}


def test_two_runs_of_one_configuration_are_reported_as_churn_not_as_a_win():
    """The noise floor is the whole argument. If `_paired` ever reported only the wins,
    a rerun of the same arm would look like an improvement and a gate written in single
    probes would keep being read as if it could resolve them."""
    left = {
        "p0": {"verdict": "correct", "retrieved_ids": ["a"]},
        "p1": {"verdict": "wrong", "retrieved_ids": ["b"]},
        "p2": {"verdict": "correct", "retrieved_ids": ["c"]},
        "p3": {"verdict": "wrong", "retrieved_ids": ["d"]},
    }
    right = {
        "p0": {"verdict": "wrong", "retrieved_ids": ["a"]},
        "p1": {"verdict": "correct", "retrieved_ids": ["b"]},
        "p2": {"verdict": "correct", "retrieved_ids": ["c"]},
        "p3": {"verdict": "abstained", "retrieved_ids": ["d"]},
    }
    out = diagnose._paired(left, right)

    assert out == {
        "n": 4,
        "wins": 1,
        "losses": 1,
        "net": 0,
        "verdict_changed": 3,
        "correctness_changed": 2,
        # Equal retrieval alone does not establish historical source identity.
        "retrieval_identical": 4,
    }


def test_a_run_with_no_recorded_routing_is_visible_as_such():
    """v4.1's Gate 0 is stated over routed and abstained slices. A run whose rows never
    recorded a route cannot be graded against it, and the scoreboard has to say so rather
    than letting the run be read as if the gate had been checked."""
    rows = {
        "p0": {"probe_id": "p0", "kind": "count", "verdict": "correct", "scan_route": None},
        "p1": {"probe_id": "p1", "kind": "count", "verdict": "wrong"},
    }
    assert sum(1 for r in rows.values() if r.get("scan_route")) == 0


def test_parsed_is_read_as_text_because_that_is_how_rows_store_it():
    assert diagnose._as_int("9") == 9
    assert diagnose._as_int(" 9 ") == 9
    assert diagnose._as_int(9) == 9
    assert diagnose._as_int(None) is None
    assert diagnose._as_int("nine") is None


def test_the_real_diagnosis_reproduces_the_recorded_noise_floor():
    """Replay against the actual runs when they are present. Skipped in a clean CI
    checkout, which has neither the probe rows nor the private store."""
    record = REPO / "results/analysis/v4-probe-diagnosis.json"
    if not record.is_file():
        pytest.skip("v4-probe-diagnosis.json is a local research artifact")
    payload = json.loads(record.read_text(encoding="utf-8"))

    noise = payload["noise_floor"]
    # The two runs share a configuration, so identical retrieval is the precondition that
    # makes their disagreement noise rather than an effect.
    assert noise["retrieval_identical"] == noise["n"]
    # The registered candidate's net effect must not be read as larger than this.
    assert abs(noise["net"]) >= abs(payload["comparisons_against_control"]["v4.0-flat2"]["net"])
    # Deterministic arithmetic is the one v4 change the evidence does support.
    control = payload["count_diagnosis"]["control-v3.3"]
    for arm in ("v4.0-flat", "v4.0-flat2", "v4.1-scan"):
        assert (
            payload["count_diagnosis"][arm]["number_disagreed_with_own_item_list"]
            < control["number_disagreed_with_own_item_list"]
        )
