"""The query-time count candidate, against the ways a count can look grounded and not be.

`compute` is the deterministic half: the model proposes members and quotes, code decides
whether that proposal is checkable. These tests pin the refusals — an unreviewed source,
a quote that is not in the text, a member asserted by the assistant rather than the user,
an empty list offered as zero. They do not show the candidate answers questions better;
only a human-reviewed development set can show that.
"""

from __future__ import annotations

import json

import pytest

from llm_long_term_memory.runtime.evidence_count import (
    CountEvidence,
    Member,
    Source,
    SourceReview,
    answer,
    compute,
    make_prompt,
)

SOURCES = [
    Source("s1", "I finally planted the basil and the mint this weekend.", "user"),
    Source("s2", "Basil and mint both like a sunny sill.", "assistant"),
]


def evidence(reviews, **over) -> CountEvidence:
    payload = {"sources": reviews, "scope_complete": True}
    payload.update(over)
    return CountEvidence(**payload)


def reviewed(source_id: str, members: list[tuple[str, str]]) -> SourceReview:
    return SourceReview(
        source_id=source_id,
        status="reviewed",
        members=[Member(entity=e, quote=q) for e, q in members],
    )


def test_a_reviewed_pool_with_quoted_user_members_answers_with_its_citations():
    result = compute(
        evidence(
            [
                reviewed("s1", [("basil", "planted the basil"), ("mint", "the mint")]),
                reviewed("s2", []),
            ]
        ),
        SOURCES,
    )
    assert result.status == "answered"
    assert result.count == 2
    assert result.members == ("basil", "mint")
    assert {c["source_id"] for c in result.citations} == {"s1"}


def test_the_same_entity_in_two_casings_is_one_member():
    result = compute(
        evidence(
            [
                reviewed("s1", [("basil", "planted the basil"), ("Basil", "the basil")]),
                reviewed("s2", []),
            ]
        ),
        SOURCES,
    )
    assert result.count == 1
    # Both quotes stay in the citations: the reader can see why it was merged.
    assert len(result.citations) == 2


def test_a_source_left_unreviewed_refuses_rather_than_counting_what_was_seen():
    result = compute(evidence([reviewed("s1", [("basil", "planted the basil")])]), SOURCES)
    assert result.status == "insufficient"
    assert "coverage" in result.reason


def test_reviewing_the_same_source_twice_is_refused():
    result = compute(
        evidence([reviewed("s1", []), reviewed("s1", []), reviewed("s2", [])]),
        SOURCES,
    )
    assert result.status == "insufficient"
    assert "duplicated" in result.reason


def test_a_source_id_that_was_never_supplied_is_refused():
    result = compute(
        evidence([reviewed("s1", []), reviewed("s3", [])]),
        SOURCES,
    )
    assert result.status == "insufficient"


@pytest.mark.parametrize(
    "over",
    [
        {"scope_complete": False},
    ],
)
def test_an_incomplete_scope_refuses_instead_of_answering_from_what_it_has(over):
    result = compute(
        evidence(
            [reviewed("s1", [("basil", "planted the basil")]), reviewed("s2", [])],
            **over,
        ),
        SOURCES,
    )
    assert result.status == "insufficient"
    assert "uncertain" in result.reason


def test_one_uncertain_source_stops_the_whole_count():
    ev = evidence(
        [
            reviewed("s1", [("basil", "planted the basil")]),
            SourceReview(source_id="s2", status="uncertain", members=[]),
        ]
    )
    assert compute(ev, SOURCES).status == "insufficient"


def test_a_quote_that_is_not_in_the_source_text_is_refused():
    result = compute(
        evidence([reviewed("s1", [("rosemary", "planted the rosemary")]), reviewed("s2", [])]),
        SOURCES,
    )
    assert result.status == "insufficient"
    assert "verbatim" in result.reason


def test_a_member_established_only_by_the_assistant_is_refused():
    result = compute(
        evidence([reviewed("s1", []), reviewed("s2", [("mint", "Basil and mint")])]),
        SOURCES,
    )
    assert result.status == "insufficient"
    assert "user-source quote" in result.reason


def test_an_empty_list_is_not_zero_without_a_quote_saying_so():
    result = compute(evidence([reviewed("s1", []), reviewed("s2", [])]), SOURCES)
    assert result.status == "insufficient"
    assert "zero" in result.reason


def test_zero_is_answered_when_a_user_source_states_it():
    result = compute(
        evidence(
            [reviewed("s1", []), reviewed("s2", [])],
            zero_source_id="s1",
            zero_quote="planted the basil",
        ),
        SOURCES,
    )
    assert result.status == "answered"
    assert result.count == 0
    assert result.members == ()


def test_zero_evidence_quoted_from_the_assistant_is_refused():
    result = compute(
        evidence(
            [reviewed("s1", []), reviewed("s2", [])],
            zero_source_id="s2",
            zero_quote="sunny sill",
        ),
        SOURCES,
    )
    assert result.status == "insufficient"


def test_the_prompt_carries_every_source_with_its_role_and_time():
    payload = json.loads(make_prompt("How many plants?", SOURCES))
    assert payload["question"] == "How many plants?"
    assert [s["id"] for s in payload["sources"]] == ["s1", "s2"]
    assert [s["role"] for s in payload["sources"]] == ["user", "assistant"]


@pytest.mark.parametrize(
    ("question", "sources"),
    [
        ("  ", SOURCES),
        ("How many plants?", []),
        ("How many plants?", [Source("s1", "a"), Source("s1", "b")]),
    ],
)
def test_a_prompt_without_a_question_or_uniquely_identified_sources_is_refused(question, sources):
    with pytest.raises(ValueError):
        make_prompt(question, sources)


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return type("Completion", (), {"text": self.text})()


def test_an_unparseable_model_response_refuses_rather_than_guessing():
    client = FakeClient("not json")
    result, _ = answer("How many plants?", SOURCES, client, "model-x")
    assert result.status == "insufficient"
    assert result.reason == "invalid structured response"
    assert client.calls[0]["temperature"] == 0.0


def test_a_well_formed_model_response_is_recounted_by_code_not_trusted():
    payload = CountEvidence(
        sources=[
            reviewed("s1", [("basil", "planted the basil"), ("mint", "the mint")]),
            reviewed("s2", []),
        ],
        scope_complete=True,
    ).model_dump_json()
    result, _ = answer("How many plants?", SOURCES, FakeClient(payload), "model-x")
    assert (result.status, result.count) == ("answered", 2)
