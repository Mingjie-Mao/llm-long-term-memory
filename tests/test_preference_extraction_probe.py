"""The registered counting of the preference extraction probe, on a synthetic archive."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import preference_extraction_probe as probe


class _Turn:
    def __init__(self, role, content, has_answer=False):
        self.role, self.content, self.has_answer = role, content, has_answer


class _Session:
    def __init__(self, i):
        self.session_id = f"s{i}"
        self.turns = [_Turn("user", "Hello."), _Turn("user", "In bed by 9:30.", True)]


def _memory(i, turn, kind, n=0):
    return {
        "source_session_id": f"s{i}",
        "source_turn_index": turn,
        "type": kind,
        "scope": None,
        "content": f"fact {i}-{n}",
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "cohort", lambda: [(f"q{i}", _Session(i)) for i in range(24)])
    archive = tmp_path / "a.json"
    monkeypatch.setattr(probe, "ARCHIVE", archive)
    return archive


def _write(archive, plain, observer):
    archive.write_text(
        json.dumps(
            {
                "arms": {
                    "plain": {"version": "a", "memories": plain},
                    "observer": {"version": "b", "memories": observer},
                }
            }
        ),
        encoding="utf-8",
    )


def test_only_a_preference_memory_on_a_gold_turn_counts(setup):
    plain = [_memory(i, 0, "preference") for i in range(24)]  # anchored to a non-gold turn
    observer = [_memory(i, 1, "preference") for i in range(8)] + [
        _memory(i, 0, "semantic") for i in range(8, 24)
    ]
    _write(setup, plain, observer)
    result = probe.analyse()
    assert result["arms"]["plain"]["preference_covered"] == 0
    assert result["arms"]["observer"]["preference_covered"] == 8
    assert result["pursue"] is True


def test_a_lift_bought_with_volume_is_not_pursued(setup):
    plain = [_memory(i, 0, "semantic") for i in range(24)]
    observer = [_memory(i, 1, "preference") for i in range(24)] + [
        _memory(i, 0, "semantic", n) for i in range(24) for n in range(1, 2)
    ]
    _write(setup, plain, observer)
    result = probe.analyse()
    assert result["rules"]["preference_covered_up_by_6"] is True
    assert result["rules"]["volume_up_at_most_30pct"] is False
    assert result["pursue"] is False
