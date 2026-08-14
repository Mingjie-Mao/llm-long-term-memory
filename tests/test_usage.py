from __future__ import annotations

import json

from chronomem.llm.usage import CallRecord, UsageTracker


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

    saved = json.loads(path.read_text())
    assert saved["summary"]["total_requests"] == 2
    assert [call["role"] for call in saved["calls"]] == ["extractor", "judge"]


def test_fresh_save_replaces_prior_calls(tmp_path):
    path = tmp_path / "usage.json"
    UsageTracker(records=[_record("extractor")]).save(path)

    UsageTracker(records=[_record("judge")]).save(path)

    saved = json.loads(path.read_text())
    assert saved["summary"]["total_requests"] == 1
    assert [call["role"] for call in saved["calls"]] == ["judge"]
