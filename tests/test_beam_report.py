"""The registered BEAM numbers, and the two ways an aggregation can lie about them."""

from __future__ import annotations

import pytest

from llm_long_term_memory.evaluation import beam_report

DECLARED = {
    "judge": {"correct_at": 0.5},
    "primary": {"abilities": ["information_extraction", "preference_following"]},
    "reported_separately": {"abilities": ["summarization"]},
}


def row(question_id, ability, scores, recalled=True):
    return {
        "question_id": question_id,
        "question_type": ability,
        "source_session_recalled": recalled,
        "notes": {
            "judge": {
                "version": "beam-rubric-v1",
                "grades": [
                    {"item": n, "score": s, "reason": "", "position": n}
                    for n, s in enumerate(scores, start=1)
                ],
            }
        },
    }


def test_a_conversation_is_read_off_the_question_id():
    assert (
        beam_report.conversation_of("beam-100K-10-event_ordering-1", "event_ordering")
        == "beam-100K-10"
    )


def test_a_row_without_grades_is_refused_rather_than_scored():
    """A judge call that never landed must not read as a zero. Zero is a grade."""
    with pytest.raises(beam_report.MissingGrades):
        beam_report.scored([{"question_id": "q", "question_type": "abstention", "notes": {}}])


def test_a_twelve_item_question_does_not_outweigh_a_one_item_question():
    """Summarization carries 234 of the development half's 1,165 rubric items over 44 of
    its 440 questions. Averaging items rather than questions would hand it five times the
    weight its questions earn."""
    rows = [
        row("beam-100K-1-information_extraction-0", "information_extraction", [0.0]),
        row("beam-100K-1-summarization-0", "summarization", [1.0] * 12),
    ]
    everything = beam_report.summarise(beam_report.scored(rows))
    assert everything["mean_score"] == pytest.approx(0.5)


def test_conversations_weigh_equally_however_many_questions_they_hold():
    rows = [
        row("beam-100K-1-information_extraction-0", "information_extraction", [1.0]),
        row("beam-100K-1-information_extraction-1", "information_extraction", [1.0]),
        row("beam-100K-1-information_extraction-2", "information_extraction", [1.0]),
        row("beam-500K-2-information_extraction-0", "information_extraction", [0.0]),
    ]
    summary = beam_report.summarise(beam_report.scored(rows))
    assert summary["conversations"] == 2
    assert summary["mean_score"] == pytest.approx(0.5)


def test_the_primary_covers_exactly_the_declared_abilities():
    rows = [
        row("beam-100K-1-information_extraction-0", "information_extraction", [1.0]),
        row("beam-100K-1-preference_following-0", "preference_following", [1.0]),
        row("beam-100K-1-summarization-0", "summarization", [0.0]),
    ]
    report = beam_report.report(rows, DECLARED)
    assert report["primary"]["questions"] == 2
    assert report["primary"]["mean_score"] == pytest.approx(1.0)
    assert report["reported_separately"]["summarization"]["mean_score"] == pytest.approx(0.0)
    assert report["all_abilities"]["questions"] == 3


def test_an_arm_compared_with_itself_differs_by_nothing():
    rows = [
        row("beam-100K-1-information_extraction-0", "information_extraction", [1.0, 0.0]),
        row("beam-100K-1-preference_following-0", "preference_following", [0.5]),
    ]
    result = beam_report.compare(rows, rows, DECLARED)
    assert result["mean_difference"] == 0.0
    assert result["conversations_compared"] == 1


def test_a_conversation_only_one_arm_finished_is_dropped_whole():
    """Half a conversation is a mean over a different question set, not a paired
    difference — and on a quota-interrupted run half a conversation is the normal state."""
    left = [
        row("beam-100K-1-information_extraction-0", "information_extraction", [1.0]),
        row("beam-100K-1-preference_following-0", "preference_following", [1.0]),
        row("beam-500K-2-information_extraction-0", "information_extraction", [0.0]),
    ]
    right = [
        row("beam-100K-1-information_extraction-0", "information_extraction", [0.0]),
        row("beam-500K-2-information_extraction-0", "information_extraction", [1.0]),
    ]
    result = beam_report.compare(left, right, DECLARED)
    assert result["conversations_dropped_incomplete"] == ["beam-100K-1"]
    assert result["conversations_compared"] == 1
    assert result["mean_difference"] == pytest.approx(1.0)
