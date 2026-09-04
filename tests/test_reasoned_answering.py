from __future__ import annotations

import json
from pathlib import Path

import yaml

from llm_long_term_memory.evaluation.runners.reasoning import (
    REASONED_ANSWER_PROMPT_VERSION,
    REASONED_ANSWER_SYSTEM,
    ReasonedAnswerVerdict,
    reasoning_kind,
    render_reasoned_prompt,
)


def test_real_development_regressions_route_without_benchmark_labels():
    path = Path("results/raw-retrieval-regressions.json")
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    expected = {
        "temporal": "temporal",
        "multi_session_aggregation": "multi_session_aggregation",
        "preference_application": "preference_application",
    }

    for case in cases:
        if not case.get("reasoning_required"):
            continue
        assert reasoning_kind(case["query"]) == expected[case["reasoning_kind"]]


def test_current_update_and_direct_lookup_are_distinct():
    assert reasoning_kind("Which framework do I use now?") == "current_state"
    assert reasoning_kind("What is my dog's name?") == "direct"


def test_temporal_routing_wins_over_generic_how_many():
    assert reasoning_kind("How many weeks ago did I receive it?") == "temporal"


def test_distinguishes_elapsed_time_from_counting_distinct_days():
    assert reasoning_kind("How many days did it take to arrive after I bought it?") == "temporal"
    assert reasoning_kind("How many days did I spend volunteering in May?") == (
        "multi_session_aggregation"
    )


def test_routes_event_order_duration_and_average_operations():
    assert reasoning_kind("Which event happened first, the trip or the delivery?") == "temporal"
    assert reasoning_kind("How long had I been using it when I moved?") == "temporal"
    assert reasoning_kind("What is the average age of the group?") == ("multi_session_aggregation")


def test_reasoned_prompt_names_the_operation_and_safety_boundary():
    prompt = render_reasoned_prompt(
        "- The user visited two galleries.",
        "2026-09-03",
        "How many galleries did I visit?",
    )

    assert "Required answer operation: multi_session_aggregation" in prompt
    assert "merge duplicates" in prompt
    assert "Use only supplied evidence" in prompt


def test_reasoned_schema_keeps_a_short_auditable_check():
    verdict = ReasonedAnswerVerdict.model_validate(
        {
            "status": "answer",
            "answer": "Four weeks ago.",
            "evidence_summary": ["Received on 2026-08-06", "Today is 2026-09-03"],
            "calculation": "28 days / 7 = 4 weeks",
            "confidence": "high",
        }
    )

    assert verdict.answer == "Four weeks ago."
    assert len(verdict.evidence_summary) == 2
    assert verdict.confidence == "high"


def test_v3_prompt_has_a_distinct_version_and_does_not_invite_guessing():
    assert REASONED_ANSWER_PROMPT_VERSION == "memory-reasoned-v3.1"
    assert "Never turn an unstated assumption" in REASONED_ANSWER_SYSTEM


def test_phase3_config_keeps_pilot_retrieval_context_and_models():
    pilot = yaml.safe_load(Path("configs/v3-answer.yaml").read_text(encoding="utf-8"))
    phase3 = yaml.safe_load(Path("configs/v3-phase3.yaml").read_text(encoding="utf-8"))

    for metadata_key in ("name", "description"):
        pilot.pop(metadata_key)
        phase3.pop(metadata_key)

    assert phase3 == pilot


def test_phase4_config_changes_only_metadata_from_phase3():
    phase3 = yaml.safe_load(Path("configs/v3-phase3.yaml").read_text(encoding="utf-8"))
    phase4 = yaml.safe_load(Path("configs/v3-phase4-adaptive.yaml").read_text(encoding="utf-8"))

    for metadata_key in ("name", "description"):
        phase3.pop(metadata_key)
        phase4.pop(metadata_key)

    assert phase4 == phase3


def test_phase5_config_changes_only_the_registered_hydration_budget_and_allocation():
    phase4 = yaml.safe_load(Path("configs/v3-phase4-adaptive.yaml").read_text(encoding="utf-8"))
    phase5 = yaml.safe_load(Path("configs/v3-phase5-compact.yaml").read_text(encoding="utf-8"))

    for metadata_key in ("name", "description"):
        phase4.pop(metadata_key)
        phase5.pop(metadata_key)
    phase4["hydration"] = phase5["hydration"]

    assert phase5 == phase4
    assert phase5["hydration"] == {
        "neighbouring_sentences": 0,
        "max_tokens": 425,
        "allocation": "session_fair",
    }
