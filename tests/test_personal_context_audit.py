import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from personal_context_audit import audit


def instance():
    return NS(
        question_id="train",
        question_type="single-session-preference",
        sessions=[
            NS(
                session_id="s",
                turns=[
                    NS(role="user", content="My commute is 40 minutes each way.", has_answer=True),
                    NS(
                        role="assistant",
                        content="Your 40-minute commute each way.",
                        has_answer=False,
                    ),
                ],
            )
        ],
    )


def archive(*memories):
    return {"arms": {"plain": {"memories": list(memories)}}}


def memory(**changes):
    return {
        "source_session_id": "s",
        "source_turn_index": 0,
        "content": "The user has a 40-minute commute each way.",
        "type": "semantic",
        "scope": "profile",
    } | changes


def test_profile_scope_counts_as_context_without_relabelling_as_preference():
    result = audit([instance()], archive(memory()), {"train"})
    assert result["proxy_counts"]["plain"] == {
        "old_preference_proxy": 0,
        "user_personal_context_proxy": 1,
    }


def test_assistant_echo_and_foreign_session_do_not_count_as_user_context():
    result = audit(
        [instance()],
        archive(memory(source_turn_index=1), memory(source_session_id="other-tenant")),
        {"train"},
    )
    arm = result["rows"][0]["arms"]["plain"]
    assert not arm["user_personal_context_proxy"]
    assert len(arm["memories"]) == 1
    assert arm["memories"][0]["user_turn_candidates_for_numeric_anchor_review"] == [0]
    assert arm["memories"][0]["turn_index"] == 1  # Alert does not mutate the anchor.


def test_foreign_split_is_filtered_before_reading_any_reference_labels():
    class Forbidden:
        question_id = "dev100"
        question_type = "single-session-preference"

        @property
        def sessions(self):
            raise AssertionError("must not inspect dev100 reference turns")

    result = audit([Forbidden(), instance()], archive(memory()), {"train"})
    assert result["questions"] == 1


def test_invalid_turn_indices_do_not_create_coverage():
    result = audit(
        [instance()],
        archive(memory(source_turn_index=True), memory(source_turn_index=99)),
        {"train"},
    )
    assert not result["rows"][0]["arms"]["plain"]["user_personal_context_proxy"]


def test_assistant_recommendation_type_does_not_establish_user_preference():
    result = audit(
        [instance()],
        archive(memory(source_turn_index=1, type="preference", scope="recommendation")),
        {"train"},
    )
    assert not result["rows"][0]["arms"]["plain"]["user_personal_context_proxy"]
