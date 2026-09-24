"""The conditional specificity repair, as an ingestion stage.

It passed its registered gate on the extraction ruler — 44.8% to 59.3%, 21 specifics
gained and 0 lost — and was then never wired into anything. These tests pin the three
properties that made it promotable: it is additive, it only pays for sessions that lost
something, and it accepts a fact only when the recovered specific is in both the quoted
span and the fact's own content.

They also pin that turning it on changes the ingest fingerprint, because a store built
without it is not the same system as one built with it.
"""

from __future__ import annotations

from types import SimpleNamespace

from llm_long_term_memory.conversation import ConversationSession, ConversationTurn
from llm_long_term_memory.ingest.grounded import Anchored, GroundedFact, GroundingReport
from llm_long_term_memory.ingest.repair import SpecificityRepair, missing_specifics
from llm_long_term_memory.store import Memory

SPOKEN = "I walked 416 steps to the Museum of Contemporary Art."


def _session(*contents: str) -> ConversationSession:
    return ConversationSession(
        session_id="s1",
        date="2023/05/20",
        turns=[ConversationTurn(role="user", content=c) for c in contents],
    )


def _memory(content: str) -> Memory:
    return Memory(id="m", user_id="u", type="semantic", content=content, token_count=1)


class _Extractor:
    """Returns a scripted report and records whether it was asked."""

    def __init__(self, report: GroundingReport) -> None:
        self.report = report
        self.calls = 0

    @staticmethod
    def prompt_texts() -> tuple[str, ...]:
        return ("grounded-prompt",)

    def extract(self, turns, session_date):
        self.calls += 1
        return self.report


def _anchored(content: str, span: str, turn_index: int = 0) -> Anchored:
    return Anchored(
        GroundedFact(content=content, verbatim_span=span, turn_index=turn_index),
        turn_index,
        0,
        len(span),
    )


def test_a_session_that_lost_nothing_costs_nothing():
    """The trigger is the same detector the ruler scores with, so what is paid for is
    exactly what the ruler would have counted as lost."""
    extractor = _Extractor(GroundingReport())
    repair = SpecificityRepair(extractor)

    outcome = repair.repair(
        _session(SPOKEN),
        [_memory("The user walked 416 steps to the Museum of Contemporary Art.")],
        "u",
    )

    assert outcome.called is False
    assert extractor.calls == 0
    assert outcome.memories == []


def test_a_lost_specific_buys_one_call_and_comes_back():
    report = GroundingReport(anchored=[_anchored("The user walked 416 steps.", SPOKEN)])
    extractor = _Extractor(report)

    outcome = SpecificityRepair(extractor).repair(
        _session(SPOKEN), [_memory("The user went to a museum.")], "u"
    )

    assert outcome.called is True
    assert extractor.calls == 1
    assert [m.content for m in outcome.memories] == ["The user walked 416 steps."]
    assert outcome.memories[0].user_id == "u"
    assert outcome.memories[0].source_char_end == len(SPOKEN)


def test_a_fact_whose_span_lacks_the_recovered_specific_is_refused():
    """The span has to be the evidence, not a citation sitting next to it."""
    report = GroundingReport(
        anchored=[_anchored("The user walked 416 steps.", "I went to a museum.")]
    )

    outcome = SpecificityRepair(_Extractor(report)).repair(
        _session(SPOKEN), [_memory("The user went out.")], "u"
    )

    assert outcome.memories == []
    assert outcome.called is True, "the call was still spent, and is still counted"


def test_a_fact_whose_content_drops_the_specific_is_refused():
    report = GroundingReport(anchored=[_anchored("The user walked to a museum.", SPOKEN)])

    outcome = SpecificityRepair(_Extractor(report)).repair(
        _session(SPOKEN), [_memory("The user went out.")], "u"
    )

    assert outcome.memories == []


def test_an_assistant_turn_is_not_a_source_for_this_repair():
    """What was measured as lost is what the *user* said."""
    session = ConversationSession(
        session_id="s1",
        date="2023/05/20",
        turns=[
            ConversationTurn(role="user", content=SPOKEN),
            ConversationTurn(role="assistant", content="That is 416 steps exactly."),
        ],
    )
    report = GroundingReport(
        anchored=[_anchored("The assistant said 416 steps.", "That is 416 steps exactly.", 1)]
    )

    outcome = SpecificityRepair(_Extractor(report)).repair(
        session, [_memory("The user went out.")], "u"
    )

    assert outcome.memories == []


def test_the_repair_never_touches_what_was_already_extracted():
    """Additive by construction, which is why its measured losses are zero."""
    report = GroundingReport(anchored=[_anchored("The user walked 416 steps.", SPOKEN)])
    baseline = [_memory("The user went to a museum.")]

    outcome = SpecificityRepair(_Extractor(report)).repair(_session(SPOKEN), baseline, "u")

    assert [m.content for m in baseline] == ["The user went to a museum."]
    assert outcome.memories[0] is not baseline[0]


def test_missing_specifics_reads_the_object_column_as_the_ruler_does():
    session = _session(SPOKEN)

    assert "416" not in missing_specifics(session, [_memory("The user walked 416 steps.")])
    assert "416" in missing_specifics(session, [_memory("The user walked.")])


def test_turning_it_on_changes_the_ingest_fingerprint():
    """A store built without it must refuse to resume with it."""
    from llm_long_term_memory.config import ExperimentConfig
    from llm_long_term_memory.ingest import fingerprint

    cfg = ExperimentConfig.from_yaml("configs/v2b-batch8.yaml")
    off = fingerprint.from_config(cfg, sessions_per_request=8).as_dict()
    cfg.ingest.specificity_repair = True
    on = fingerprint.from_config(cfg, sessions_per_request=8).as_dict()

    assert "specificity_repair" not in off, "an existing store's fingerprint must still match"
    assert fingerprint.differences(off, on) == [
        f"specificity_repair: (absent) -> {on['specificity_repair']}"
    ]


def test_the_pipeline_hands_repaired_memories_through_deduplication():
    """A repaired memory that skipped dedup would be a second copy of a stored fact."""
    import inspect

    from llm_long_term_memory.ingest.pipeline import IngestionPipeline

    source = inspect.getsource(IngestionPipeline._ingest_batch)
    repair_at = source.index("self._repair_batch")
    dedup_at = source.index("self.deduplicator.process")
    assert repair_at < dedup_at, "the repair must run before deduplication, not after"


def test_the_pipeline_scopes_the_session_id_before_matching():
    """`_ingest_batch` rewrites ids before the repair runs; matching on the raw one
    would compare every session against an empty set and call for all of them."""
    import inspect

    from llm_long_term_memory.ingest.pipeline import IngestionPipeline

    source = inspect.getsource(IngestionPipeline._repair_batch)
    assert "scoped_session_id(namespace, session.session_id)" in source


def test_it_is_off_unless_the_configuration_asks():
    from llm_long_term_memory.config import ExperimentConfig

    assert ExperimentConfig.from_yaml("configs/v2b-batch8.yaml").ingest.specificity_repair is False


def test_a_repair_object_with_no_prompts_still_fingerprints_as_on():
    from llm_long_term_memory.ingest import fingerprint

    spec = fingerprint.from_runtime(
        SimpleNamespace(version="x", model="m", prompt_texts=lambda: ("p",)),
        sessions_per_request=8,
        repair=SimpleNamespace(version="repair-v9"),
    )

    assert spec.specificity_repair == "repair-v9"


def test_the_same_repair_under_two_tenants_gets_two_ids():
    """One haystack session can belong to several tenants, and the store writes with
    `INSERT OR REPLACE`: a shared id would let one tenant's repair overwrite another's."""
    from llm_long_term_memory.ingest.repair import grounded_memory

    anchored = _anchored("The user walked 416 steps.", SPOKEN)

    a = grounded_memory(anchored, "s1", "tenant-a")
    b = grounded_memory(anchored, "s1", "tenant-b")

    assert a.id != b.id
    assert (a.user_id, b.user_id) == ("tenant-a", "tenant-b")


def test_two_facts_from_one_span_do_not_collide():
    from llm_long_term_memory.ingest.repair import grounded_memory

    first = grounded_memory(_anchored("The user walked 416 steps.", SPOKEN), "s1", "u")
    second = grounded_memory(_anchored("The user visited the MCA.", SPOKEN), "s1", "u")

    assert first.id != second.id


def test_a_repair_id_is_the_same_in_another_process():
    """`hash()` is salted per process; a resumed ingest must reproduce the id."""
    import subprocess
    import sys

    from llm_long_term_memory.ingest.repair import grounded_memory

    here = grounded_memory(_anchored("The user walked 416 steps.", SPOKEN), "s1", "u").id
    script = (
        "from llm_long_term_memory.ingest.grounded import Anchored, GroundedFact\n"
        "from llm_long_term_memory.ingest.repair import grounded_memory\n"
        f"span = {SPOKEN!r}\n"
        "fact = GroundedFact(content='The user walked 416 steps.', verbatim_span=span,"
        " turn_index=0)\n"
        "print(grounded_memory(Anchored(fact, 0, 0, len(span)), 's1', 'u').id)\n"
    )
    there = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    ).stdout.strip()

    assert here == there
