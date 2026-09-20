"""The product owns its conversation types; the benchmark extends them.

Before this, ingestion, extraction, provenance and the API were all typed against
`HaystackSession`, so the product could not be read without the benchmark — and those
types carry the benchmark's own labels. A turn knows whether it holds the gold answer,
which is exactly what an extractor must never be able to see.
"""

from __future__ import annotations

import pytest

from llm_long_term_memory.conversation import (
    AnswerRequest,
    ConversationSession,
    ConversationSource,
    ConversationTurn,
)
from llm_long_term_memory.evaluation.datasets.longmemeval import (
    HaystackSession,
    HaystackTurn,
    Instance,
)


def test_a_benchmark_session_is_a_conversation_so_ingestion_never_needs_the_benchmark():
    session = HaystackSession("s1", "2026/01/05", [HaystackTurn("user", "hello", True)])

    assert isinstance(session, ConversationSession)
    assert isinstance(session.turns[0], ConversationTurn)
    assert session.char_count == 5


def test_the_gold_label_exists_only_on_the_benchmark_subclass():
    assert "has_answer" in {f.name for f in __import__("dataclasses").fields(HaystackTurn)}
    assert "has_answer" not in {f.name for f in __import__("dataclasses").fields(ConversationTurn)}
    assert not hasattr(ConversationSession("s", "d", []), "is_evidence")


def test_a_benchmark_instance_is_a_conversation_source_without_being_imported_as_one():
    instance = Instance(
        question_id="q1",
        question_type="single-session-user",
        question="where do I live?",
        answer="Berlin",
        question_date="2026/01/06",
        sessions=[HaystackSession("s1", "2026/01/05", [])],
        answer_session_ids=["s1"],
    )

    assert isinstance(instance, ConversationSource)
    assert instance.store_namespace == "q1"


def test_a_request_carries_what_answering_needs_and_nothing_about_the_answer():
    request = AnswerRequest(question="where do I live?", asked_on="2026/01/06", user_id="alice")

    assert {f.name for f in __import__("dataclasses").fields(request)} == {
        "question",
        "asked_on",
        "user_id",
    }


def test_a_request_without_a_tenant_is_refused_because_retrieval_is_tenant_scoped():
    with pytest.raises(ValueError, match="tenant-scoped"):
        AnswerRequest(question="where do I live?", asked_on="2026/01/06", user_id="")


def test_a_request_without_a_question_is_refused():
    with pytest.raises(ValueError, match="needs a question"):
        AnswerRequest(question="   ", asked_on="2026/01/06", user_id="alice")
