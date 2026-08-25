from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_long_term_memory.ingest.zero_yield_report import (
    render_zero_yield_markdown,
    summarize_batch_pilot,
)

REPO = Path(__file__).resolve().parent.parent


def _real_pilot():
    payload = json.loads(
        (REPO / "results/raw/batch-position-pilot.json").read_text(encoding="utf-8")
    )
    return summarize_batch_pilot(payload)


def _audit(*, complete: bool = True) -> dict:
    return {
        "complete": complete,
        "total_sessions": 100,
        "all_zero_yield_sessions": 12,
        "short_zero_yield_sessions_below_min_turns": 2,
        "substantive_zero_yield_sessions": 10,
        "categories": {
            "content_policy": 1,
            "source_format": 1,
            "source_id_collision": 1,
            "inconsistent_repeat": 1,
            "evidence_extraction_miss": 1,
            "date_variant_outcome_difference": 1,
            "unclassified_single_zero": 4,
        },
        "all_categories": {
            "content_policy": 1,
            "source_format": 1,
            "source_id_collision": 1,
            "inconsistent_repeat": 1,
            "evidence_extraction_miss": 1,
            "date_variant_outcome_difference": 1,
            "unclassified_single_zero": 4,
            "short_no_durable_fact_candidate": 2,
        },
        "position_bands": {
            "front_0_3": {"total": 40, "zero": 2, "rate": 0.05},
            "later_4_end": {"total": 60, "zero": 8, "rate": 8 / 60},
        },
        "batch_position_effect": {"later_vs_front_risk_ratio": (8 / 60) / 0.05},
    }


def test_real_batch_pilot_is_complete_and_reproduces_published_causal_counts():
    summary = _real_pilot()

    assert summary.rows == 420
    assert summary.cohort_size == 60
    assert summary.duplicate_keys == 0
    assert summary.position_paired_sessions == 51
    assert summary.yielded_front_zero_back == 8
    assert summary.zero_front_yielded_back == 0
    assert summary.position_mcnemar_p == pytest.approx(0.0078125)
    assert summary.front_mean_memories == pytest.approx(4.5, abs=0.05)
    assert summary.back_mean_memories == pytest.approx(1.6, abs=0.05)
    assert {size: row["zero"] for size, row in summary.batch_sizes.items()} == {
        "15": 7,
        "5": 5,
        "1": 0,
    }


def test_report_partitions_unknown_cases_without_calling_them_random():
    text = render_zero_yield_markdown(
        _audit(),
        _real_pilot(),
        audit_source="audit.json",
        pilot_source="pilot.json",
    )

    assert "完全相同输入出现不同结果 | 1" in text
    assert "短会话, 可能没有长期信息 | 2" in text
    assert "仍无法确定, 或原聊天没有持久信息 | 4" in text
    assert "位置相关的遗漏" in text
    assert "audit.json" in text
    assert "pilot.json" in text


def test_report_refuses_an_incomplete_audit():
    with pytest.raises(ValueError, match="incomplete"):
        render_zero_yield_markdown(
            _audit(complete=False),
            _real_pilot(),
            audit_source="audit.json",
            pilot_source="pilot.json",
        )


def test_pilot_summary_refuses_duplicate_records():
    payload = json.loads(
        (REPO / "results/raw/batch-position-pilot.json").read_text(encoding="utf-8")
    )
    payload["records"].append(dict(payload["records"][0]))

    with pytest.raises(ValueError, match="duplicate"):
        summarize_batch_pilot(payload)
