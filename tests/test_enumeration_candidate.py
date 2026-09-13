"""v4.2 must make under-enumeration visible, and must not buy that with a worse count.

Every failure here is one that was measured on saved rows before this candidate existed
(`results/analysis/v4-probe-diagnosis.md`): counts too low with complete evidence in
context, spurious members, the same fact named twice, and a citation to something that
was never shown. All of it runs on fixtures, so the candidate is falsified or supported
before a single provider call is bought.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from llm_long_term_memory.evaluation.runners.memory import (
    label_memories,
    render_grouped,
    render_memory,
)
from llm_long_term_memory.evaluation.runners.synthesis import (
    EnumeratingSynthesisVerdict,
    SynthesisVerdict,
    check_citations,
    compute,
    normalise_label,
)
from llm_long_term_memory.store import Memory


def memory(mid, content, *, subject="user", predicate="acquired", scope="", when=None):
    return Memory(
        mid,
        "alice",
        "episodic",
        content,
        5,
        subject=subject,
        predicate=predicate,
        scope=scope,
        event_time=when,
    )


def verdict(labels, items=None, **kwargs):
    return EnumeratingSynthesisVerdict(
        status="answer",
        answer="",
        operation="count",
        member_labels=labels,
        items=items if items is not None else [],
        **kwargs,
    )


# --------------------------------------------------------------- the four fixtures


def test_under_enumeration_is_reported_instead_of_passing_silently():
    """The measured failure: nine members in context, six named, and nothing in the row
    saying three were skipped. The number is still six — the candidate does not invent
    members — but `labels_uncited` now says so."""
    context = {f"M{i}" for i in range(1, 10)}
    result = compute(verdict([f"M{i}" for i in range(1, 7)]), context)

    assert result.computed
    assert result.detail["count"] == 6
    assert result.detail["labels_in_context"] == 9
    assert result.detail["labels_uncited"] == 3
    assert result.detail["counted_from"] == "cited_labels"


def test_over_enumeration_by_citing_something_never_shown_is_dropped_not_counted():
    """A cited label that was not in the context cannot be a member of a set drawn from
    the context. Counting it would let an invented citation inflate an auditable total."""
    result = compute(verdict(["M1", "M2", "M99"]), {"M1", "M2"})

    assert result.detail["count"] == 2
    assert result.detail["labels_not_in_context"] == ["M99"]


def test_the_same_member_cited_twice_is_counted_once_and_recorded():
    result = compute(verdict(["M1", "M2", "M1"]), {"M1", "M2"})

    assert result.detail["count"] == 2
    assert result.detail["labels_cited_twice"] == 1


def test_a_missing_date_still_blocks_a_duration_rather_than_being_computed():
    """The enumerate arm changes `count` and must leave every other operation alone."""
    result = compute(
        EnumeratingSynthesisVerdict(
            status="answer",
            answer="about a week",
            operation="duration",
            start_date="2026-01-01",
            end_date="",
        ),
        {"M1"},
    )

    assert not result.computed
    assert result.detail["reason"] == "a date operand is missing"


def test_subject_mixing_is_not_this_layer_s_job_and_is_not_silently_absorbed():
    """An assistant-subject memory reaching the context is a retrieval defect, and the
    scanner already refuses it. What must not happen is this layer quietly counting it:
    if it is labelled, it is citable, and the count includes it. That is the honest
    behaviour — the contamination stays visible upstream instead of being hidden here."""
    memories = [
        memory("a", "user bought a bike"),
        memory("b", "assistant suggested a bike", subject="assistant"),
    ]
    labels = label_memories(memories)
    result = compute(verdict(list(labels.values())), set(labels.values()))

    assert result.detail["count"] == 2, "the count reflects the context it was given"
    assert result.detail["labels_uncited"] == 0


# --------------------------------------------------------------- citation mechanics


@pytest.mark.parametrize("raw", ["M3", "[M3]", "m3", "M 3", "M03", " [m03] "])
def test_a_label_survives_the_ways_a_model_transcribes_it(raw):
    """The model copies these out of rendered text, so the failure to expect is a
    bracket, not an invention. Discarding a correctly chosen member over punctuation
    would be the candidate causing the exact loss it exists to fix."""
    assert normalise_label(raw) == "M3"


def test_something_that_is_not_a_label_resolves_to_nothing():
    assert normalise_label("the rose bush") == ""
    assert normalise_label("") == ""


def test_citing_nothing_that_exists_refuses_to_compute():
    """Returning 0 would be a confident count with a derivation attached to nothing."""
    result = compute(verdict(["M42"]), {"M1"})

    assert not result.computed
    assert result.detail["reason"] == "no cited label was present in the context"


def test_check_citations_counts_uncited_members_against_the_whole_context():
    outcome = check_citations(["M1", "M1", "M9"], {"M1", "M2", "M3"})

    assert outcome.named == ["M1"]
    assert outcome.duplicated == ["M1"]
    assert outcome.unknown == ["M9"]
    assert outcome.in_context == 3
    assert outcome.uncited == 2


# --------------------------------------------------------------- single-variable


def test_the_control_arm_renders_exactly_as_before():
    """v4.0-flat is the registered control. If labelling leaked into it, the comparison
    would differ by a changed context as well as by the mechanism — the v4.0 design
    fault, one layer down."""
    memories = [memory("a", "user bought a bike"), memory("b", "user bought a kayak")]

    assert render_grouped(memories, temporal=False, labels=None) == (
        "- user bought a bike\n- user bought a kayak"
    )
    assert render_memory(memories[0], temporal=False) == "- user bought a bike"


def test_the_candidate_arm_labels_every_memory_in_reading_order():
    memories = [memory("a", "user bought a bike"), memory("b", "user bought a kayak")]
    labels = label_memories(memories)

    assert labels == {"a": "M1", "b": "M2"}
    assert render_grouped(memories, temporal=False, labels=labels) == (
        "- [M1] user bought a bike\n- [M2] user bought a kayak"
    )


def test_a_memory_inside_a_supersession_chain_is_still_citable():
    """A count member can also be part of a timeline. An unlabelled chain member would be
    a member the answerer has no way to name, which is under-enumeration built into the
    rendering rather than caused by the model."""
    memories = [
        memory("a", "user owns a road bike", when=datetime(2026, 1, 1), predicate="owns_bike"),
        memory("b", "user owns a gravel bike", when=datetime(2026, 2, 1), predicate="owns_bike"),
    ]
    labels = label_memories(memories)
    rendered = render_grouped(memories, temporal=True, labels=labels)

    assert "[M1]" in rendered and "[M2]" in rendered


def test_a_verdict_without_citations_falls_back_to_the_free_text_count():
    """Additive, so a model that ignores the new field still produces a usable count and
    the arm does not collapse to abstention on its first bad reply."""
    result = compute(verdict([], items=["bike", "kayak", "bike"]), {"M1", "M2"})

    assert result.computed
    assert result.detail["count"] == 2
    assert result.detail["counted_from"] == "free_text"


def test_the_control_schema_has_no_citation_field_to_read():
    """Sharing one schema across both arms is how `answer_confidence` came to be read as
    a measurement on v3 rows that never carried it."""
    assert "member_labels" not in SynthesisVerdict.model_fields
    assert "member_labels" in EnumeratingSynthesisVerdict.model_fields


def test_a_named_missing_operand_still_wins_over_citations():
    """The self-contradiction guard must not be bypassed by the new path: a verdict that
    says something is missing and then cites members anyway is not computed."""
    result = compute(verdict(["M1"], missing_field="the full list"), {"M1"})

    assert not result.computed
    assert result.detail["reason"] == "the verdict named a missing operand"


# --------------------------------------------------------------- end to end, no provider


class _ScriptedClient:
    """Replies with a fixed verdict, and records what it was asked."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.systems: list[str] = []

    def generate(self, *, role, model, prompt, system=None, **kw):
        self.prompts.append(prompt)
        self.systems.append(system or "")

        class _Completion:
            text = self.reply
            input_tokens = 100
            output_tokens = 5
            thinking_tokens = 0
            api_latency_ms = 1.0

        return _Completion()


class _StubEncoder:
    def encode(self, texts, show_progress=False):
        import numpy as np

        return np.ones((len(texts), 2), dtype="float32")

    def encode_one(self, text):
        import numpy as np

        return np.ones(2, dtype="float32")


@pytest.fixture
def wired(tmp_path):
    import json

    import numpy as np

    from llm_long_term_memory.evaluation.datasets.longmemeval import Instance
    from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
    from llm_long_term_memory.store import NumpyFlatIndex, Session, SQLiteMemoryStore

    store = SQLiteMemoryStore(tmp_path / "m.db")
    store.initialize()
    store.add_session(
        Session(id="s1", user_id="q1", started_at=datetime(2026, 1, 1), source="test")
    )
    memories = [
        Memory(
            f"m{i}",
            "q1",
            "episodic",
            f"The user bought item {i}",
            6,
            subject="user",
            predicate="acquired",
            object=f"item {i}",
            event_time=datetime(2026, 1, i + 1),
            source_session_id="s1",
        )
        for i in range(1, 4)
    ]
    store.add_memories(memories)
    index = NumpyFlatIndex(tmp_path / "idx", dim=2)
    index.add([m.id for m in memories], np.ones((3, 2), dtype=np.float32))

    def build(policy: str, reply: dict):
        return MemoryRunner(
            _ScriptedClient(json.dumps(reply)),
            model="m",
            encoder=_StubEncoder(),
            store=store,
            index=index,
            temporal=False,
            answer_policy=policy,
            timeline_rendering=False,
            # The structured verdict — and so the computation — is only reached on the
            # fallback path, which is the path both v4 arms actually run on.
            raw_fallback=True,
        )

    def ask(runner):
        return runner.answer(
            Instance(
                question_id="q1",
                question_type="synthesis",
                question="How many items did I buy?",
                answer="3",
                question_date="2026-02-01",
                sessions=[],
                answer_session_ids=["s1"],
            )
        )

    yield build, ask
    store.close()


def test_end_to_end_the_candidate_counts_from_citations_and_reports_what_it_skipped(wired):
    build, ask = wired
    runner = build(
        "synthesis_v4_enumerate",
        {
            "status": "answer",
            "answer": "two items",
            "operation": "count",
            "member_labels": ["M1", "M2"],
            "items": ["item 1", "item 2"],
        },
    )
    answer = ask(runner)

    assert "[M1]" in runner.client.prompts[0], "the context must carry citable labels"
    assert answer.text == "2 distinct: item 1, item 2"
    computation = answer.notes["synthesis_computation"]
    assert computation["counted_from"] == "cited_labels"
    # Three memories were in context and two were cited: the skipped one is on record.
    assert computation["labels_in_context"] == 3
    assert computation["labels_uncited"] == 1


def test_end_to_end_the_control_arm_sees_no_labels_at_all(wired):
    build, ask = wired
    runner = build(
        "synthesis_v4",
        {
            "status": "answer",
            "answer": "two items",
            "operation": "count",
            "items": ["item 1", "item 2"],
        },
    )
    answer = ask(runner)

    assert "[M1]" not in runner.client.prompts[0]
    assert answer.notes["synthesis_computation"]["counted_from"] == "free_text"


# ------------------------------------------------- the arms must differ in one thing only


def test_both_prompts_still_demand_a_prose_answer():
    """Caught on the live run's second row, at a cost of three requests. The candidate
    returned `status`, `operation`, `items` and `member_labels` and no `answer`, where the
    control supplied one. `compute` rescued the count from the citations, so the reply read
    correctly — but on `current_state` and `lookup` nothing rescues it, and the candidate
    would have abstained where the control answered. That is a second difference between
    the arms, which is the confound the whole single-variable rule exists to prevent."""
    from llm_long_term_memory.evaluation.runners.synthesis import (
        SYNTHESIS_ANSWER_SYSTEM,
        SYNTHESIS_ENUMERATE_ANSWER_SYSTEM,
    )

    requirement = "Always write your reply in prose in `answer`"
    assert requirement in SYNTHESIS_ANSWER_SYSTEM
    assert requirement in SYNTHESIS_ENUMERATE_ANSWER_SYSTEM
    # The original count sentence must survive byte-identically: the candidate prompt is
    # that sentence plus an appended instruction, not a rewrite of it. The rewrite is what
    # cost the other operations their prose.
    original = (
        "phrase each. List them even if you are unsure of the total; the total is not "
        "your job. Do not put a number in `items`."
    )
    assert original in SYNTHESIS_ANSWER_SYSTEM
    assert original in SYNTHESIS_ENUMERATE_ANSWER_SYSTEM
    assert "member_labels" in SYNTHESIS_ENUMERATE_ANSWER_SYSTEM


def test_the_two_prompts_differ_only_inside_the_count_instruction():
    """If the edit had leaked into another operation's instructions, the comparison would
    differ by more than count-member citation."""
    from llm_long_term_memory.evaluation.runners.synthesis import (
        SYNTHESIS_ANSWER_SYSTEM,
        SYNTHESIS_ENUMERATE_ANSWER_SYSTEM,
    )

    marker = "   For duration and comparison:"
    before_control, _, after_control = SYNTHESIS_ANSWER_SYSTEM.partition(marker)
    before_candidate, _, after_candidate = SYNTHESIS_ENUMERATE_ANSWER_SYSTEM.partition(marker)

    assert after_control == after_candidate, "everything after the count block is identical"
    assert before_control != before_candidate, "the count block is what changed"
