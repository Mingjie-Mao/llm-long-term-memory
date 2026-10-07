"""The registered rules of the raw-primary gate, on synthetic runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import raw_primary_gate as gate

TYPES = ["single-session-user"] * 50 + ["temporal-reasoning"] * 50


def _write(raw: Path, name: str, correct: list[bool], context: int = 1000) -> None:
    rows = [
        {
            "question_id": f"q{i:03d}",
            "question_type": TYPES[i],
            "correct": ok,
            "context_tokens": context,
            "notes": {"raw_primary_turns": 7},
        }
        for i, ok in enumerate(correct)
    ]
    (raw / f"{name}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "RAW", tmp_path)
    return tmp_path


def _controls(raw, score=70):
    base = [i < score for i in range(100)]
    for name in (*gate.CONTROLS, gate.FRESH):
        _write(raw, name, base)


def test_a_clear_gain_passes(raw):
    _controls(raw)
    _write(raw, gate.R, [i < 78 for i in range(100)], context=5000)
    _write(raw, gate.RAW_ONLY, [i < 74 for i in range(100)])
    result = gate.evaluate()
    assert result["primary_R_vs_control"] == {"wins": 8, "losses": 0, "net": 8}
    assert result["pass"] is True
    assert result["secondary_R_vs_O"]["net"] == 4


def test_a_net_below_the_floor_stops(raw):
    _controls(raw)
    _write(raw, gate.R, [i < 74 for i in range(100)], context=5000)
    _write(raw, gate.RAW_ONLY, [i < 70 for i in range(100)])
    assert gate.evaluate()["rules"]["net_at_least_5"] is False


def test_a_type_that_loses_three_stops_even_with_a_big_net(raw):
    # Control: user questions 0-29 and temporal questions 50-79 right.
    control = [i < 30 or 50 <= i < 80 for i in range(100)]
    for name in (*gate.CONTROLS, gate.FRESH):
        _write(raw, name, control)
    # R gains every user question and loses temporal 50-52.
    _write(raw, gate.R, [i < 50 or 53 <= i < 80 for i in range(100)])
    _write(raw, gate.RAW_ONLY, [False] * 100)
    result = gate.evaluate()
    assert result["primary_R_vs_control"]["net"] == 17
    assert result["by_type_R_vs_control"]["temporal-reasoning"]["net"] == -3
    assert result["rules"]["no_type_loses_more_than_2"] is False
    assert result["pass"] is False


def test_too_much_context_stops(raw):
    _controls(raw)
    _write(raw, gate.R, [i < 80 for i in range(100)], context=6000)
    _write(raw, gate.RAW_ONLY, [False] * 100)
    assert gate.evaluate()["rules"]["median_context_at_most_5500"] is False


def test_a_tie_counts_as_wrong_for_the_control(raw):
    """Two of four control runs right: the control is not credited with it."""
    for index, name in enumerate((*gate.CONTROLS, gate.FRESH)):
        _write(raw, name, [i < 70 or (i == 99 and index < 2) for i in range(100)])
    _write(raw, gate.R, [i < 70 for i in range(100)])
    _write(raw, gate.RAW_ONLY, [False] * 100)
    result = gate.evaluate()
    assert result["control_ties"] == ["q099"]
    assert result["control_majority_score"] == 70


def test_drift_switches_to_the_fresh_control_alone(raw):
    for name in gate.CONTROLS:
        _write(raw, name, [i < 70 for i in range(100)])
    _write(raw, gate.FRESH, [i < 60 for i in range(100)])
    _write(raw, gate.R, [i < 66 for i in range(100)])
    _write(raw, gate.RAW_ONLY, [False] * 100)
    result = gate.evaluate()
    assert result["drift"]["declared"] is True
    assert result["control_runs_used"] == [gate.FRESH]
    assert result["primary_R_vs_control"]["net"] == 6


def test_an_incomplete_run_is_not_evaluated(raw):
    _controls(raw)
    _write(raw, gate.R, [True] * 99)
    with pytest.raises(SystemExit, match="incomplete"):
        gate.evaluate()
