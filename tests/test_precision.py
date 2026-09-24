"""The mirror of the fidelity ruler: is what was written actually in the conversation?

`fidelity` measures recall and is blind to invention, which is why a model that writes
forty per cent more memories scores higher on it almost mechanically. This asks the
other direction with the same detector and the same normalisation, so the two numbers
are commensurable.
"""

from __future__ import annotations

from llm_long_term_memory.conversation import ConversationSession, ConversationTurn
from llm_long_term_memory.ingest.precision import score_support, session_text
from llm_long_term_memory.store import Memory


def _session(*turns: tuple[str, str]) -> ConversationSession:
    return ConversationSession(
        session_id="s1",
        date="2023/05/20",
        turns=[ConversationTurn(role=role, content=content) for role, content in turns],
    )


def _memory(content: str, obj: str | None = None) -> Memory:
    return Memory(id="m", user_id="u", type="semantic", content=content, token_count=1, object=obj)


def test_a_specific_that_is_in_the_conversation_is_supported():
    session = _session(("user", "I walked 416 steps to the museum."))

    report = score_support([(session, [_memory("The user walked 416 steps.")])])

    assert report.memories_with_an_unsupported_specific == 0
    assert report.specific_support_rate == 1.0


def test_a_specific_that_is_nowhere_in_the_conversation_is_not():
    session = _session(("user", "I walked to the museum."))

    report = score_support([(session, [_memory("The user walked 416 steps.")])])

    assert report.memories_with_an_unsupported_specific == 1
    assert report.examples[0][1] == "416"


def test_an_assistant_turn_is_a_legitimate_source():
    """Scoring against the user's words alone would invent unsupported memories.

    A memory about what the assistant recommended is grounded in an assistant turn, and
    the recall ruler's narrower `user_assertions` view is right for recall and wrong here.
    """
    session = _session(
        ("user", "Any headphone suggestions?"),
        ("assistant", "I'd recommend the Sony WH-1000XM6."),
    )

    report = score_support([(session, [_memory("The assistant recommended the Sony WH-1000XM6.")])])

    assert report.memories_with_an_unsupported_specific == 0
    assert "sony" in session_text(session).lower()


def test_a_memory_with_no_detectable_specific_is_set_aside_not_scored_clean():
    """Counting it as supported would make a vague extractor look precise."""
    session = _session(("user", "I do not enjoy running."))

    report = score_support([(session, [_memory("The user dislikes running.")])])

    assert report.memories == 1
    assert report.memories_with_a_checkable_specific == 0
    assert report.uncheckable_memories == 1
    assert report.unsupported_memory_rate == 0.0


def test_one_memory_with_several_unsupported_specifics_counts_once():
    session = _session(("user", "I went out."))

    report = score_support(
        [(session, [_memory("The user walked 416 steps for 2 hours to the Tate Modern.")])]
    )

    assert report.memories_with_an_unsupported_specific == 1
    assert report.asserted > report.supported


def test_the_rate_is_over_checkable_memories_not_all_of_them():
    session = _session(("user", "I walked 416 steps."))
    memories = [
        _memory("The user walked 416 steps."),
        _memory("The user walked 999 steps."),
        _memory("The user likes walking."),
    ]

    report = score_support([(session, memories)])

    assert report.memories == 3
    assert report.memories_with_a_checkable_specific == 2
    assert report.unsupported_memory_rate == 0.5


def test_the_object_column_is_read_as_well_as_the_sentence():
    """Specifics live in the typed fields too, and the recall ruler reads both."""
    session = _session(("user", "I went out."))

    report = score_support([(session, [_memory("The user walked.", obj="416 steps")])])

    assert report.memories_with_an_unsupported_specific == 1


def test_per_facet_support_is_reported():
    session = _session(("user", "I walked 416 steps."))

    report = score_support([(session, [_memory("The user walked 416 steps for 2 hours.")])])

    assert report.per_facet["quantity"].supported >= 1
    assert report.per_facet["duration"].asserted == 1
    assert report.per_facet["duration"].supported == 0
