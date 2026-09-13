from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_v42_comparison import BudgetStopped, DurableClient, Meter, verify
from v42_protocol import (
    UNCITED_TWO_SIGMA_FACTS,
    analyse,
    append_json,
    exact_cluster_p,
    schedule,
    uncited_gold_facts,
    validate_rows,
)


def probes():
    return [
        {
            "probe_id": f"{kind}{i}",
            "kind": kind,
            "question": f"{kind}? {i}",
            "answer": "gold",
            "namespace": f"n{i}",
        }
        for kind, count in (
            ("count", 30),
            ("duration", 30),
            ("comparison", 33),
            ("current_state", 49),
        )
        for i in range(count)
    ]


def rows_fixture():
    ps = probes()
    by_id = {p["probe_id"]: p for p in ps}
    cells = schedule(ps)
    rows = [
        {
            **c,
            "freeze_id": "freeze",
            "mode": "live",
            "question": by_id[c["probe_id"]]["question"],
            "gold": "gold",
            "namespace": by_id[c["probe_id"]]["namespace"],
            "verdict": "wrong",
            "retrieved_ids": ["a", "b"],
            "evidence_found": 2,
            "evidence_needed": 2,
            "context_complete": True,
            "answer_was_raw_structure": False,
            "synthesis_computation": {"counted_from": "cited_labels"},
        }
        for c in cells
    ]
    return ps, cells, rows


def test_schedule_runs_404_cells_and_only_count_repeats():
    cells = schedule(probes())
    assert len(cells) == len({c["cell_id"] for c in cells}) == 404
    assert sum(c["kind"] == "count" for c in cells) == 180
    assert all(c["kind"] == "count" for c in cells if c["repeat"] > 1)
    for a, b in zip(cells[::2], cells[1::2], strict=True):
        assert a["probe_id"] == b["probe_id"] and a["repeat"] == b["repeat"]
        assert a["arm"] != b["arm"]
    assert schedule(probes()) == cells


@pytest.mark.parametrize(
    "change", ["repeat", "arm", "freeze_id", "mode", "retrieved_ids", "duplicate", "missing"]
)
def test_resume_refuses_identity_and_evidence_drift(change):
    ps, cells, rows = rows_fixture()
    if change == "duplicate":
        rows.insert(1, rows[0])
    elif change == "missing":
        rows.pop(0)
    elif change == "retrieved_ids":
        rows[1][change] = ["b", "a"]
    else:
        rows[1][change] = "changed"
    with pytest.raises(ValueError):
        validate_rows(rows, cells, ps, "freeze", "live")


def test_exact_cluster_distribution_matches_enumeration():
    deltas = [2, -1, 0, 1]
    null = [
        sum(a * b for a, b in zip(signs, deltas, strict=True))
        for signs in itertools.product((-1, 1), repeat=4)
    ]
    expected = sum(abs(x) >= abs(sum(deltas)) for x in null) / len(null)
    assert exact_cluster_p(deltas) == expected
    assert exact_cluster_p([1] * 6) == 0.03125
    assert exact_cluster_p([2, 1, 1, 1, 1]) == 0.0625


def test_majority_requires_two_correct_and_does_not_count_repeats_as_new_probes():
    ps, cells, rows = rows_fixture()
    for r in rows:
        if (
            r["arm"] == "candidate"
            and r["kind"] == "count"
            and int(r["probe_id"][5:]) < 6
            and r["repeat"] <= 2
        ):
            r["verdict"] = "correct"
    result = analyse(rows, cells, ps, "freeze", "live")
    assert result["net_count"] == 6
    assert result["count_probes"] == 30
    assert result["outcome"] == "promote_to_next_development_comparison"
    ps[1]["namespace"] = ps[0]["namespace"]
    for r in rows:
        if r["probe_id"] == "count1":
            r["namespace"] = ps[0]["namespace"]
    assert analyse(rows, cells, ps, "freeze", "live")["outcome"] == "inconclusive"


def test_incomplete_data_has_no_performance_conclusion():
    ps, cells, rows = rows_fixture()
    result = analyse(rows[:17], cells, ps, "freeze", "live")
    assert result["outcome"] == "incomplete"
    assert "net_count" not in result


def test_budget_survives_restart_and_counts_failed_requests(tmp_path):
    path = tmp_path / "requests.jsonl"
    m = Meter(path, max_requests=2, token_limit=100)
    with m.measure("answerer", "fake") as box:
        box["input_tokens"] = 10
    with pytest.raises(RuntimeError), m.measure("answerer", "fake"):
        raise RuntimeError("simulated failed request")
    resumed = Meter(path, max_requests=2, token_limit=100)
    assert resumed.summary() == {"attempts": 2, "observed_tokens": 10, "failed_attempts": 1}
    with pytest.raises(BudgetStopped), resumed.measure("answerer", "fake"):
        pytest.fail("over-budget request sent")


def test_uncertain_attempt_is_not_automatically_resent(tmp_path):
    path = tmp_path / "requests.jsonl"
    append_json(path, {"event": "start", "attempt": 1})
    with pytest.raises(BudgetStopped, match="unresolved"):
        Meter(path, max_requests=10, token_limit=100)


def test_completed_in_row_call_replays_without_another_request(tmp_path):
    meter = Meter(tmp_path / "requests.jsonl", max_requests=3, token_limit=100)
    c = {"cell_id": "count0.r1.control", "kind": "count", "arm": "control"}
    cache = tmp_path / "completions.jsonl"
    client = DurableClient(None, cache, meter)
    client.begin(c, SimpleNamespace())
    first = client.generate(role="answerer", model="fake", prompt="fixed")
    resumed = DurableClient(None, cache, meter)
    resumed.begin(c, SimpleNamespace())
    assert resumed.generate(role="answerer", model="fake", prompt="fixed") == first
    assert meter.attempts == 1
    resumed.begin(c, SimpleNamespace())
    with pytest.raises(ValueError, match="cached request"):
        resumed.generate(role="answerer", model="fake", prompt="changed")


def test_frozen_source_tampering_is_detected_before_runtime_setup(tmp_path, monkeypatch):
    import run_v42_comparison as tool

    monkeypatch.setattr(tool, "REPO", tmp_path)
    (tmp_path / "code.py").write_text("changed", encoding="utf-8")
    (tmp_path / "freeze.json").write_text(
        json.dumps({"sources": [{"path": "code.py", "sha256": "old"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen source changed"):
        verify(tmp_path)


def test_read_only_store_cannot_modify_frozen_data(tmp_path):
    import sqlite3

    from llm_long_term_memory.store import SQLiteMemoryStore

    path = tmp_path / "store.db"
    store = SQLiteMemoryStore(path)
    store.initialize()
    store.close()
    reader = SQLiteMemoryStore(path, read_only=True)
    reader.initialize()
    with pytest.raises(sqlite3.OperationalError):
        reader._conn.execute("delete from memories")
    reader.close()


def test_observed_token_threshold_stops_before_next_request(tmp_path):
    meter = Meter(tmp_path / "usage.jsonl", max_requests=10, token_limit=5)
    with meter.measure("answerer", "fake") as box:
        box["input_tokens"] = 6
    with pytest.raises(BudgetStopped), meter.measure("answerer", "fake"):
        pytest.fail("token threshold was ignored")
    assert meter.attempts == 1


def test_frozen_database_drift_is_detected(tmp_path, monkeypatch):
    import run_v42_comparison as tool
    from v42_protocol import sha

    monkeypatch.setattr(tool, "REPO", tmp_path)
    (tmp_path / "source.tar.gz").write_bytes(b"fixture source archive")
    (tmp_path / "store.db").write_bytes(b"changed data")
    (tmp_path / "freeze.json").write_text(
        json.dumps(
            {
                "sources": [],
                "source_archive_sha256": sha(tmp_path / "source.tar.gz"),
                "private_data": str(tmp_path),
                "data": [{"path": "store.db", "sha256": "old"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen data changed"):
        verify(tmp_path)


# ------------------------------------------------- the higher-resolution reading


def _memories(n: int) -> dict[str, dict[str, str]]:
    return {
        f"g{i}": {"content": f"user acquired thing{i}", "object": f"thing{i}"} for i in range(n)
    }


def _count_probe(pid: str, namespace: str, gold: int) -> dict:
    return {
        "probe_id": pid,
        "kind": "count",
        "namespace": namespace,
        "question": "how many",
        "answer": gold,
        "evidence_memory_ids": [f"g{i}" for i in range(gold)],
    }


def _count_row(pid: str, arm: str, repeat: int, named: int, gold: int) -> dict:
    return {
        "cell_id": f"{pid}.r{repeat}.{arm}",
        "probe_id": pid,
        "kind": "count",
        "arm": arm,
        "repeat": repeat,
        "retrieved_ids": [f"g{i}" for i in range(gold)],
        "synthesis_computation": {"items": [f"thing{i}" for i in range(named)]},
    }


def test_the_candidate_naming_more_gold_facts_reads_as_a_negative_difference():
    """Fewer unnamed facts is the direction the mechanism predicts, so the registered
    difference is negative when the candidate wins."""
    probes = [_count_probe("count_0", "ns0", 5)]
    rows = [
        *(_count_row("count_0", "control", r, 2, 5) for r in (1, 2, 3)),
        *(_count_row("count_0", "candidate", r, 4, 5) for r in (1, 2, 3)),
    ]
    out = uncited_gold_facts(rows, probes, _memories(5))

    assert out["control_unnamed"] == 3.0
    assert out["candidate_unnamed"] == 1.0
    assert out["difference"] == -2.0


def test_the_reading_averages_repeats_rather_than_collapsing_them_to_a_binary():
    """The quantity is a count. Majority-voting it would discard the resolution that
    makes this ruler finer than the binary gate."""
    probes = [_count_probe("count_0", "ns0", 4)]
    rows = [
        _count_row("count_0", "control", 1, 0, 4),
        _count_row("count_0", "control", 2, 4, 4),
        _count_row("count_0", "control", 3, 2, 4),
        *(_count_row("count_0", "candidate", r, 2, 4) for r in (1, 2, 3)),
    ]
    out = uncited_gold_facts(rows, probes, _memories(4))

    # control unnamed per repeat: 4, 0, 2 -> mean 2. Not a majority vote of 3 outcomes.
    assert out["control_unnamed"] == 2.0
    assert out["difference"] == 0.0


def test_a_fact_absent_from_the_context_is_not_counted_as_unnamed():
    """It is a retrieval gap, not a reading failure, and conflating them would credit the
    candidate for something the answerer never saw."""
    probes = [_count_probe("count_0", "ns0", 4)]
    rows = [
        *(
            {**_count_row("count_0", arm, r, 1, 4), "retrieved_ids": ["g0", "g1"]}
            for arm in ("control", "candidate")
            for r in (1, 2, 3)
        )
    ]
    out = uncited_gold_facts(rows, probes, _memories(4))

    # Only g0 and g1 are in context; one of them is named, so one is unnamed.
    assert out["control_unnamed"] == 1.0


def test_the_registered_threshold_is_two_sigma_on_the_measured_pair():
    """The pair's paired per-probe spread is sd 0.83 over 30 probes, so the standard error
    of the total is about 4.6 facts and the registered bar is 10."""
    assert UNCITED_TWO_SIGMA_FACTS == 10


def test_clearing_the_reading_is_not_a_promotion():
    """A candidate that demonstrates its mechanism has not earned promotion. Keeping these
    separate is the point of adding the reading instead of relaxing the gate."""
    probes = [_count_probe(f"count_{i}", f"ns{i}", 5) for i in range(30)]
    rows = [
        *(_count_row(f"count_{i}", "control", r, 0, 5) for i in range(30) for r in (1, 2, 3)),
        *(_count_row(f"count_{i}", "candidate", r, 5, 5) for i in range(30) for r in (1, 2, 3)),
    ]
    out = uncited_gold_facts(rows, probes, _memories(5))

    assert out["cleared"] is True
    assert "decides nothing on its own" not in out  # it is stated in `limitations`
    assert out["difference"] == -150.0
