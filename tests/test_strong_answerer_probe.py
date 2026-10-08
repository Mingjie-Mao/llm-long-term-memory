"""The registered reading of the minimal strong-answerer probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import strong_answerer_probe as probe


@pytest.fixture
def raw(tmp_path, monkeypatch):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"question_ids": [f"q{i}" for i in range(9)]}), encoding="utf-8")
    monkeypatch.setattr(probe, "RAW", tmp_path)
    monkeypatch.setattr(probe, "MANIFEST", manifest)
    return tmp_path


def _write(raw, name, correct):
    rows = [{"question_id": f"q{i}", "correct": ok} for i, ok in enumerate(correct)]
    (raw / f"{name}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("strong", "lite", "effect", "word"),
    [(5, 1, 4, "real lever"), (2, 1, 1, "stop"), (4, 1, 3, "undecided")],
)
def test_the_reading_follows_the_registered_thresholds(raw, strong, lite, effect, word):
    _write(raw, probe.STRONG, [i < strong for i in range(9)])
    _write(raw, probe.LITE, [i < lite for i in range(9)])
    result = probe.evaluate()
    assert result["model_effect"] == effect
    assert word in result["reading"]


def test_an_incomplete_run_gives_no_reading(raw):
    _write(raw, probe.STRONG, [True] * 8)
    _write(raw, probe.LITE, [False] * 9)
    with pytest.raises(SystemExit, match="incomplete"):
        probe.evaluate()
