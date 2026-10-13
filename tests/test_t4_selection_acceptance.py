import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_long_term_memory.evaluation.datasets.longmemeval import (
    HaystackSession,
    HaystackTurn,
    Instance,
)
from llm_long_term_memory.llm.usage import UsageTracker
from llm_long_term_memory.retrieve.time_notes import annotate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from t4_selection_acceptance import COHORT, GAP_GOLD, CapturingClient, report, source_coverage


def test_auditing_preserves_request_and_does_not_repeat_generation():
    received = []

    class Client:
        quota = object()

        def generate(self, **kwargs):
            received.append(kwargs)
            return SimpleNamespace(text="one original completion")

    client = CapturingClient(Client())
    request = {"role": "answerer", "model": "same-model", "prompt": "my question", "temperature": 0}
    result = client.generate(**request)
    assert received == [request] and result.text == "one original completion"
    assert client.calls[0]["request"]["prompt"] == "my question"
    assert client.calls[0]["response"] == result.text
    assert client.quota is client.delegate.quota


def test_reference_coverage_handles_annotation_and_empty_abstention():
    text = "I started this class two weeks ago."
    instance = Instance(
        "q",
        "temporal-reasoning",
        "when",
        "gold",
        "2023-06-01",
        [HaystackSession("s", "2023-05-10", [HaystackTurn("user", text, True)])],
        ["s"],
    )
    annotated = annotate(text, datetime(2023, 5, 10))[0]
    reader = {"answer": {"notes": {"actual_answer_calls": [{"request": {"prompt": annotated}}]}}}
    assert source_coverage(instance, reader)["complete"]
    reader["answer"]["notes"]["actual_answer_calls"][0]["request"]["prompt"] = "unrelated"
    assert not source_coverage(instance, reader)["complete"]
    assert source_coverage(replace(instance, sessions=[]), reader)["reference_turns"] == 0


def sample_rows(q, identity="identity"):
    reader = {
        "question_id": q,
        "cohort": COHORT,
        "inventory_sha256": identity,
        "answer_prompt_version": "memory-aware-v2-clt",
        "answer": {
            "text": "answer",
            "context_tokens": 6000,
            "prompt_tokens": 100,
            "output_tokens": 10,
            "latency_ms": 1,
            "notes": {
                "execution_identity": {
                    "inventory_sha256": identity,
                    "arm": COHORT,
                    "answerer": "gemini-3.5-flash-lite",
                },
                "actual_answer_calls": [],
            },
        },
    }
    import hashlib

    grade = {
        "question_id": q,
        "cohort": COHORT,
        "inventory_sha256": identity,
        "judge_prompt_version": "judge",
        "reader_row_sha256": hashlib.sha256(
            json.dumps(reader, sort_keys=True).encode()
        ).hexdigest(),
        "verdict": {"correct": True, "reason": "original grade retained"},
    }
    return reader, grade


def test_original_gold_success_cannot_hide_failed_faithful_gap(tmp_path, monkeypatch):
    import t4_selection_acceptance as tool

    monkeypatch.setattr(tool, "REPO", tmp_path)
    base = tmp_path / "results/raw/test"
    base.parent.mkdir(parents=True)
    (tmp_path / "results/analysis").mkdir(parents=True)
    monkeypatch.setattr(tool, "BASE", base)
    ids = [*GAP_GOLD, *(f"supported-{n}" for n in range(18))]
    instances = [
        Instance(q, "multi-session", "question", "original gold", "2023-06-01", [], []) for q in ids
    ]
    readers, grades = zip(*(sample_rows(q) for q in ids), strict=True)
    gaps = [
        dict(g, verdict={"correct": q != "f9e8c073", "reason": "separate gap rubric"})
        for q, g in zip(ids[:3], grades[:3], strict=True)
    ]
    for suffix, rows in [
        ("readers.jsonl", readers),
        ("grades.jsonl", grades),
        ("gap-grades.jsonl", gaps),
    ]:
        base.with_suffix("." + suffix).write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    result = report(instances, "identity", UsageTracker())
    assert result["original_correct"] == 21 and result["supported_correct"] == 18
    assert result["faithful_gaps_passed"] == 2 and not result["pass"]
    assert all(i.answer == "original gold" for i in instances)


def test_missing_supplementary_grades_cannot_complete_acceptance(tmp_path, monkeypatch):
    import t4_selection_acceptance as tool

    monkeypatch.setattr(tool, "BASE", tmp_path / "missing")
    with pytest.raises(ValueError, match="all three"):
        report([], "identity", UsageTracker())
