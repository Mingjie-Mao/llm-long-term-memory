"""The BEAM power table, against a figure that no longer describes its own inputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from llm_long_term_memory.evaluation.clustered import minimum_detectable_effect

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

power = pytest.importorskip("beam_power")


def split(dev=22, test=33, test_100k=12):
    return {
        "dev": {"conversations": [f"c{i}" for i in range(dev)]},
        "test": {
            "conversations": [f"t{i}" for i in range(test)],
            "per_scale": {"100K": test_100k, "500K": test - test_100k},
        },
    }


DECLARATION = {"primary": {"abilities": ["a", "b", "c", "d", "e", "f", "g"]}}


def test_strata_follow_the_split_and_the_declaration():
    rows = power.strata(split(), DECLARATION)
    assert [(r["half"], r["questions"], r["cluster_size"]) for r in rows] == [
        ("dev", 440, 20),
        ("dev", 308, 14),
        ("dev", 44, 2),
        ("test", 660, 20),
        ("test", 462, 14),
        ("test", 66, 2),
        ("test", 240, 20),
    ]


def test_each_cell_is_the_clustered_null():
    row = power.table(split(), DECLARATION)[3]
    expected = 100 * minimum_detectable_effect(660, 0.19, 20, 0.15)
    assert row["mde_points"]["v4 probes run-to-run, icc 0.15"] == round(expected, 1)


def test_the_committed_table_still_describes_its_inputs():
    if not power.OUT.is_file():
        pytest.skip("no BEAM power table has been written in this checkout")
    committed = json.loads(power.OUT.read_text(encoding="utf-8"))
    assert committed == power.build()
