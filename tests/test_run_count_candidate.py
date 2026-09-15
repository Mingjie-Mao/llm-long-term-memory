"""The candidate runner, against the two ways a comparison can start unearned.

It must refuse to run without a frozen human review, and it must refuse to spend quota
without a registered plan. The dry run exists so the price is known before either.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

runner = pytest.importorskip("run_count_candidate")


GOLD = {
    "gold_sha256": "a" * 64,
    "questions": [
        {
            "id": "q1",
            "question": "How many distinct plants does the user grow?",
            "answer": 2,
            "proposed_entities": ["basil", "Basil", "mint"],
            "sources": [
                {
                    "id": "s1",
                    "role": "user",
                    "text": "I planted basil and mint.",
                    "recorded_at": None,
                }
            ],
        }
    ],
}


def test_a_dry_run_prices_every_question_without_calling_a_provider():
    report = runner.dry_run(GOLD)
    assert report["provider_calls"] == 0
    assert report["questions"] == 1
    assert report["estimated_input_tokens"] > 0
    assert report["rows"][0]["gold_answer"] == 2


def test_the_control_arm_counts_distinct_generator_members_case_insensitively():
    assert runner.control_answer(GOLD["questions"][0]) == 2


def test_a_run_without_a_frozen_human_review_stops(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run", "--gold", str(tmp_path / "missing.json")])
    assert runner.main() == 2
    assert "freeze" in capsys.readouterr().out


def test_spending_quota_requires_a_registered_plan(tmp_path, capsys, monkeypatch):
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps(GOLD), encoding="utf-8")
    monkeypatch.delenv("COUNT_PREREG", raising=False)
    monkeypatch.setattr(sys, "argv", ["run", "--gold", str(gold), "--execute"])
    assert runner.main() == 2
    assert "COUNT_PREREG" in capsys.readouterr().out
