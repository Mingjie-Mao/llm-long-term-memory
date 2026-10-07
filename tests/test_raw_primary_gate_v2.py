"""The registered rules of the train150 raw-primary gate, on synthetic runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import raw_primary_gate_v2 as gate

TYPES = ["single-session-user"] * 75 + ["temporal-reasoning"] * 75


def _write(raw: Path, name: str, correct: list[bool], context: int = 1500, fallback=()):
    rows = [
        {
            "question_id": f"q{i:03d}",
            "question_type": TYPES[i],
            "correct": ok,
            "context_tokens": context,
            "notes": {"fallback_turns": [1] if i in fallback else []},
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


def test_a_net_of_eight_passes(raw):
    _write(raw, gate.CONTROL, [i < 100 for i in range(150)])
    _write(raw, gate.CANDIDATE, [i < 108 for i in range(150)], context=5700)
    result = gate.evaluate()
    assert result["primary"]["net"] == 8
    assert result["pass"] is True


def test_seven_does_not(raw):
    _write(raw, gate.CONTROL, [i < 100 for i in range(150)])
    _write(raw, gate.CANDIDATE, [i < 107 for i in range(150)], context=5700)
    assert gate.evaluate()["rules"]["net_at_least_8"] is False


def test_the_total_context_limit_is_6000(raw):
    _write(raw, gate.CONTROL, [i < 100 for i in range(150)])
    _write(raw, gate.CANDIDATE, [i < 120 for i in range(150)], context=6001)
    result = gate.evaluate()
    assert result["rules"]["median_total_context_at_most_6000"] is False
    assert result["pass"] is False


def test_a_type_losing_three_stops(raw):
    control = [i < 40 or 75 <= i < 115 for i in range(150)]
    _write(raw, gate.CONTROL, control)
    _write(raw, gate.CANDIDATE, [i < 75 or 78 <= i < 115 for i in range(150)], context=5700)
    result = gate.evaluate()
    assert result["by_type"]["temporal-reasoning"]["net"] == -3
    assert result["pass"] is False


def test_fallback_second_calls_are_counted(raw):
    _write(raw, gate.CONTROL, [True] * 150, fallback={1, 2, 3})
    _write(raw, gate.CANDIDATE, [True] * 150, fallback={4})
    assert gate.evaluate()["fallback_second_calls"] == {gate.CONTROL: 3, gate.CANDIDATE: 1}


def test_an_incomplete_run_is_not_evaluated(raw):
    _write(raw, gate.CONTROL, [True] * 150)
    with pytest.raises(SystemExit, match="incomplete"):
        gate.evaluate()
