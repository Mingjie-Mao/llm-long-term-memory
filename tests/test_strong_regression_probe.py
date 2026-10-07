"""The registered reading of the strong-answerer regression probe, on synthetic runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import strong_regression_probe as probe

IDS = [f"q{i:02d}" for i in range(10)]


def _write(raw, name, correct, qtype="multi-session"):
    rows = [
        {"question_id": q, "question_type": qtype, "correct": ok}
        for q, ok in zip(IDS, correct, strict=True)
    ]
    (raw / f"{name}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


@pytest.fixture
def raw(tmp_path, monkeypatch):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"question_ids": IDS}), encoding="utf-8")
    monkeypatch.setattr(probe, "RAW", tmp_path)
    monkeypatch.setattr(probe, "MANIFEST", manifest)
    monkeypatch.setattr(probe, "flip_rates", lambda: {"multi-session": 0.05})
    _write(tmp_path, probe.LITE, [True] * 10)
    return tmp_path


@pytest.mark.parametrize(
    ("wrong", "word"),
    [(0, "low"), (1, "low"), (2, "confirm"), (3, "do not replace")],
)
def test_the_reading_follows_the_registered_thresholds(raw, wrong, word):
    _write(raw, probe.STRONG, [i >= wrong for i in range(10)])
    result = probe.evaluate()
    assert result["R"] == wrong
    assert result["expected_noise_flips"] == pytest.approx(0.5)
    assert word in result["reading"]


def test_projected_nets_follow_the_registered_formulas(raw):
    _write(raw, probe.STRONG, [i >= 2 for i in range(10)])
    result = probe.evaluate()
    assert result["projected_net_conservative"] == pytest.approx(5 - 127 * 2 / 10)
    assert result["projected_net_noise_adjusted"] == pytest.approx(5 - 127 * 1.5 / 10)


def test_a_sample_holding_a_lite_error_is_refused(raw):
    _write(raw, probe.LITE, [i > 0 for i in range(10)])
    _write(raw, probe.STRONG, [True] * 10)
    with pytest.raises(SystemExit, match="only flash-lite's correct"):
        probe.evaluate()


def test_the_draw_covers_every_type_and_only_lite_correct_answers():
    lite = probe.load(probe.LITE)
    ids = probe.draw(lite)
    assert len(ids) == 10
    assert all(lite[q]["correct"] for q in ids)
    assert len({lite[q]["question_type"] for q in ids}) == 6
