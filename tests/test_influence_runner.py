"""Checkpoint integrity for the expensive P6 ablation runner."""

from __future__ import annotations

import json

import numpy as np

from llm_long_term_memory.evaluation.datasets.longmemeval import Instance
from llm_long_term_memory.influence import FEATURE_NAMES, InfluenceDataset, measure_influence
from llm_long_term_memory.store import Memory


class AlwaysCorrectJudge:
    def grade(self, **kwargs):
        return type("Verdict", (), {"correct": True})()


def _instance() -> Instance:
    return Instance(
        question_id="q1",
        question_type="single-session-user",
        question="What does the user prefer?",
        answer="tea",
        question_date="2026-08-14",
        sessions=[],
        answer_session_ids=[],
    )


def _memory(memory_id: str) -> Memory:
    return Memory(
        id=memory_id,
        user_id="q1",
        type="semantic",
        content=f"The user prefers tea ({memory_id}).",
        token_count=8,
    )


def test_interrupted_question_is_not_checkpointed_partially(tmp_path):
    """A retry must remeasure every probe, rather than fit mixed question rows."""
    path = tmp_path / "influence.jsonl"
    calls = 0

    def answer(instance, memories):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated transport failure")
        return "tea"

    outcome = measure_influence(
        [_instance()],
        lambda instance: [(_memory("m1"), 0.8), (_memory("m2"), 0.4)],
        answer,
        AlwaysCorrectJudge(),
        path,
    )

    assert not outcome.completed
    assert outcome.questions_done == 0
    assert outcome.dataset.rows == []
    assert InfluenceDataset.load(path).rows == []
    assert np.load(path.with_suffix(".features.npy")).shape == (0, len(FEATURE_NAMES))
    assert json.loads(path.with_suffix(".features.json").read_text(encoding="utf-8")) == {
        "features": list(FEATURE_NAMES),
        "completed_questions": [],
    }
