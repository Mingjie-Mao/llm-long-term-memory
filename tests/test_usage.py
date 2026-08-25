from __future__ import annotations

import json

import pytest

from llm_long_term_memory.llm.usage import CallRecord, UsageTracker


def _record(role: str) -> CallRecord:
    return CallRecord(
        role=role,
        model="test-model",
        input_tokens=10,
        output_tokens=2,
        latency_ms=3.0,
    )


def test_resumed_save_merges_prior_calls(tmp_path):
    path = tmp_path / "usage.json"
    UsageTracker(records=[_record("extractor")]).save(path)

    UsageTracker(records=[_record("judge")]).save(path, merge=True)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["summary"]["total_requests"] == 2
    assert [call["role"] for call in saved["calls"]] == ["extractor", "judge"]


def test_fresh_save_replaces_prior_calls(tmp_path):
    path = tmp_path / "usage.json"
    UsageTracker(records=[_record("extractor")]).save(path)

    UsageTracker(records=[_record("judge")]).save(path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["summary"]["total_requests"] == 1
    assert [call["role"] for call in saved["calls"]] == ["judge"]


def test_merge_refuses_to_silently_overwrite_a_corrupt_usage_artifact(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        UsageTracker(records=[_record("judge")]).save(path, merge=True)

    assert path.read_text(encoding="utf-8") == "not-json"


def test_usage_loader_rejects_a_non_object_payload(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        UsageTracker._load_records(path)
