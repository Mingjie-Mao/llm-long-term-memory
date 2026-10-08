"""The registered rules of the dev100 v1-against-v4 gate, on synthetic runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import raw_primary_gate_v4 as gate

TYPES = ["multi-session"] * 50 + ["temporal-reasoning"] * 50


def _write(raw, name, correct, context=5700):
    rows = [
        {
            "question_id": f"q{i:03d}",
            "question_type": TYPES[i],
            "correct": ok,
            "context_tokens": context,
            "output_tokens": 50,
            "notes": {"fallback_turns": []},
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


def _arms(raw, v1_scores, v2_scores, context=5700):
    for name, score in zip(gate.V1, v1_scores, strict=True):
        _write(raw, name, [i < score for i in range(100)])
    for name, score in zip(gate.V2, v2_scores, strict=True):
        _write(raw, name, [i < score for i in range(100)], context)


def test_a_mean_gain_of_four_passes(raw):
    _arms(raw, (80, 80, 80), (84, 84, 84))
    result = gate.evaluate()
    assert result["mean_net"] == pytest.approx(4.0)
    assert result["pass"] is True


def test_run_to_run_noise_is_averaged_not_voted(raw):
    """One lucky run of three moves the mean by a third of its size."""
    _arms(raw, (80, 80, 80), (80, 80, 92))
    assert gate.evaluate()["mean_net"] == pytest.approx(4.0)


def test_a_mean_gain_below_four_stops(raw):
    _arms(raw, (80, 80, 80), (83, 83, 83))
    assert gate.evaluate()["rules"]["mean_net_at_least_4"] is False


def test_too_much_context_stops(raw):
    _arms(raw, (80, 80, 80), (90, 90, 90), context=6100)
    assert gate.evaluate()["rules"]["median_total_context_at_most_6000"] is False


def test_an_incomplete_run_is_not_evaluated(raw):
    _arms(raw, (80, 80, 80), (84, 84, 84))
    (raw / f"{gate.V2[2]}.jsonl").unlink()
    with pytest.raises(SystemExit, match="incomplete"):
        gate.evaluate()
