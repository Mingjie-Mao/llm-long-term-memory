"""Recalling a memory is not the same as using it.

Three preference questions scored 0/3 for every system, `full_context` included. A
hand audit (results/preference-audit.md) split that into two distinct defects, and
these tests pin both fixes:

* the judge graded a *rubric* as if it were a reference answer, failing a reply that
  had in fact done what the rubric asked;
* the answerer treated memories as a lookup table, so it declined on advice-shaped
  questions instead of personalizing from what it had retrieved.
"""

from __future__ import annotations

from datetime import datetime

from llm_long_term_memory.evaluation.judge import (
    _ABSTENTION_PROMPT,
    _PREFERENCE_PROMPT,
    _QA_PROMPT,
    Judge,
)
from llm_long_term_memory.evaluation.runners.base import ANSWER_SYSTEM
from llm_long_term_memory.evaluation.runners.memory import render_grouped
from llm_long_term_memory.store import Memory

NOW = datetime(2026, 8, 14)


def memory(content: str, scope: str | None = None, **kw) -> Memory:
    return Memory(
        id=kw.pop("id", content[:12]),
        user_id="u",
        type="semantic",
        content=content,
        token_count=8,
        ingested_at=NOW,
        valid_from=NOW,
        scope=scope,
        **kw,
    )


# --------------------------------------------------------------- judge routing


def _prompt(question_type: str | None, is_abstention: bool = False) -> str:
    return Judge._prompt_for(
        question="Any advice?",
        gold="The user would prefer responses that build on their turbinado sugar use.",
        hypothesis="Try turbinado sugar again.",
        is_abstention=is_abstention,
        question_type=question_type,
    )


def test_preference_questions_are_graded_against_a_rubric():
    """The defect: a rubric graded as a reference answer fails a correct reply for
    not restating the rubric."""
    prompt = _prompt("single-session-preference")

    assert prompt.startswith(_PREFERENCE_PROMPT.split("{")[0])
    assert "rubric" in prompt.lower()
    assert "NOT a reference answer" in prompt
    assert "does NOT need to cover every point" in prompt


def test_ordinary_questions_still_use_the_reference_judge():
    prompt = _prompt("temporal-reasoning")
    assert prompt.startswith(_QA_PROMPT.split("{")[0])
    assert "Reference answer" in prompt


def test_an_unknown_question_type_falls_back_to_the_reference_judge():
    """A new LongMemEval category must not silently become rubric-graded."""
    assert _prompt(None).startswith(_QA_PROMPT.split("{")[0])
    assert _prompt("some-future-category").startswith(_QA_PROMPT.split("{")[0])


def test_abstention_outranks_the_type_router():
    """An `_abs` question has nothing to personalize from, so declining is correct
    whatever category it belongs to."""
    prompt = _prompt("single-session-preference", is_abstention=True)
    assert prompt.startswith(_ABSTENTION_PROMPT.split("{")[0])


# ------------------------------------------------------- answerer disposition


def test_the_answer_prompt_asks_for_memories_to_be_applied_not_quoted():
    lowered = ANSWER_SYSTEM.lower()
    assert "context about the user" in lowered
    assert "personalize" in lowered
    # The specific failure being prevented: refusing because no memory is a verbatim
    # answer.
    assert "word for word" in lowered


def test_the_answer_prompt_still_requires_declining_on_a_missing_fact():
    """100% abstention is a measured strength (full_context manages 50%). Teaching
    the model to personalize must not teach it to guess."""
    lowered = ANSWER_SYSTEM.lower()
    assert "say you do not know rather than guessing" in lowered
    assert "specific fact" in lowered


# ------------------------------------------------------ scope-aware rendering


def test_memories_are_grouped_by_scope():
    rendered = render_grouped(
        [
            memory("The user dislikes raw fish.", scope="preference"),
            memory("The user is travelling to Tokyo in September.", scope="plan"),
            memory("The assistant recommended Mod Podge.", scope="recommendation"),
        ],
        temporal=False,
    )

    assert "User preferences:" in rendered
    assert "Current plans:" in rendered
    assert "Previously recommended by the assistant:" in rendered
    # Semantic ordering: what the user wants precedes what was suggested to them.
    assert rendered.index("User preferences:") < rendered.index("Current plans:")
    assert rendered.index("Current plans:") < rendered.index("Previously recommended")


def test_an_unscoped_store_renders_exactly_as_before():
    """Pre-P10 memories carry no scope, so a store built before this change must not
    suddenly acquire headings."""
    rendered = render_grouped(
        [memory("The user owns a fern."), memory("The user moved to Sydney.")],
        temporal=False,
    )

    assert rendered == "- The user owns a fern.\n- The user moved to Sydney."


def test_unscoped_memories_are_grouped_honestly_not_guessed_at():
    rendered = render_grouped(
        [
            memory("The user dislikes raw fish.", scope="preference"),
            memory("The user owns a fern."),
        ],
        temporal=False,
    )

    assert "User preferences:" in rendered
    assert "Other things known about the user:" in rendered
    assert "The user owns a fern." in rendered


def test_grouping_keeps_the_validity_window():
    rendered = render_grouped([memory("The user lives in Sydney.", scope="profile")], temporal=True)
    assert "since 2026-08-14" in rendered


def test_every_scope_is_dated_the_same_way():
    """`since` on a one-off event reads oddly, and changing it to `on` was tried and
    reverted (`results/render-preposition.md`).

    The hypothesis was that framing an occurrence as the start of an ongoing state
    was why two dev50 questions failed: asked how many days passed between two
    events, the answerer took a date out of the prose and reported zero. Rendering
    `event` scopes as `(on <date>)` left both of those questions answered exactly
    as before, word for word. The answerer is not misreading the annotation, it is
    ignoring it in favour of the sentence — so the fix has to reach the prose,
    where an unresolved "today" still sits next to a resolved date in the same row.
    """
    for scope in ("event", "profile", "preference"):
        rendered = render_grouped([memory("The user did a thing.", scope=scope)], temporal=True)
        assert "since 2026-08-14" in rendered, scope


def test_a_store_written_before_scope_existed_renders_as_it_did():
    rendered = render_grouped([memory("The user owns a fern.")], temporal=True)
    assert "since 2026-08-14" in rendered


def test_a_superseded_memory_shows_both_ends_of_its_window():
    """A fact that stopped being true needs both dates, not just its start."""
    from datetime import datetime

    m = memory("The user lived in Canberra.", scope="event")
    m.valid_to = datetime(2026, 8, 20)
    rendered = render_grouped([m], temporal=True)

    assert "2026-08-14 to 2026-08-20, no longer current" in rendered
