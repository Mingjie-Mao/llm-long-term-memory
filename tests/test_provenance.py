"""Verbatim source anchors for fidelity-preserving memory."""

from chronomem.evaluation.datasets.longmemeval import HaystackSession, HaystackTurn
from chronomem.ingest.provenance import attach_source_span, source_span_for
from chronomem.store import Memory


def session() -> HaystackSession:
    return HaystackSession(
        session_id="s1",
        date="2026-01-05",
        turns=[
            HaystackTurn(
                role="user",
                content=(
                    "I started collecting vintage cameras three months ago. "
                    + "I now own 17 cameras."
                ),
            )
        ],
    )


def test_anchor_points_to_a_verbatim_source_sentence():
    span = source_span_for("The user owns 17 vintage cameras.", session())

    assert span is not None
    turn_index, start, end = span
    assert turn_index == 0
    assert session().turns[turn_index].content[start:end] == "I now own 17 cameras."


def test_attaching_a_source_span_preserves_the_original_coordinates():
    memory = Memory(
        id="m1",
        user_id="u",
        type="semantic",
        content="The user owns 17 vintage cameras.",
        token_count=8,
        source_session_id="s1",
    )

    attach_source_span(memory, session())

    assert memory.source_turn_index == 0
    assert memory.source_char_start is not None
    assert memory.source_char_end is not None
