"""Re-deriving the budget from what the run actually cost, and refusing to invent one."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

budget = pytest.importorskip("beam_budget")

CAPS = {"extractor": 500, "answerer": 500, "judge": 1500}


def test_quota_days_are_counted_per_pool_not_per_request():
    """Extraction, answering and judging run on separate daily caps, and the judge's is
    three times the others'. A plan costed in requests hides that."""
    plan = budget.plan(extraction=512, adjudication=500, pass2_adjudication=300, caps=CAPS)
    ingest = plan["stages"][0]
    assert ingest["requests"] == {"extractor": 1012}
    assert ingest["quota_days"] == 3
    judging = next(row for row in plan["stages"] if row["stage"].startswith("noise pass B"))
    assert judging["quota_days"] == 1


def test_a_stage_costs_the_worst_of_its_pools():
    """Answering and judging run together, so the stage takes as long as the slower pool."""
    plan = budget.plan(512, 0, 0, CAPS)
    stage = next(row for row in plan["stages"] if row["stage"].startswith("3.3"))
    assert stage["requests"]["answerer"] == 590
    assert stage["requests"]["judge"] == 440
    assert stage["quota_days"] == 2


def test_the_daily_caps_come_from_what_the_provider_actually_allowed():
    """The config says 500 for every model; the quota manager learned that the judge's is
    1,500. Costing the plan against the config would overstate the judge by three times."""
    source = (REPO / "tools/beam_budget.py").read_text(encoding="utf-8")
    assert "load_learned()" in source


def test_a_handful_of_neighbour_pairs_is_not_a_measurement():
    """Early in a run the store holds a few namespaces and almost no above-threshold
    neighbours. Projecting pass 2 from four pairs against two would be arithmetic wearing
    a number."""
    assert budget.MIN_NEIGHBOUR_PAIRS >= 30


def test_days_round_up_because_a_partial_day_still_ends_the_run():
    assert budget.days(1, 500) == 1
    assert budget.days(500, 500) == 1
    assert budget.days(501, 500) == 2
    assert budget.days(0, 500) == 0
