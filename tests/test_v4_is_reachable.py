"""An answer policy that no configuration can select is not implemented.

Both defects this pins were live at the same time: `synthesis_v4` existed in the
runner's policy table but no variant name mapped to it, so the whole of v4.0 was
unreachable code; and the derivation the code computed was written to an attribute
nothing ever read, so a v4 row could not be audited. Each looked finished from the
inside.
"""

from __future__ import annotations

import inspect

from llm_long_term_memory import cli
from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
from llm_long_term_memory.evaluation.runners.synthesis import SynthesisVerdict


def test_a_variant_name_selects_the_v4_policy():
    """Without this mapping the policy is dead code that still passes its own tests."""
    source = inspect.getsource(cli._build)
    assert '"two_stage_synthesis"' in source, "no variant selects synthesis_v4"
    assert '"two_stage_synthesis": "synthesis_v4"' in source


def test_the_v4_variant_is_in_the_recognised_list():
    """Recognition and dispatch are separate lists; a name in one and not the other
    fails only at run time, on a machine that has already spent quota getting there."""
    source = inspect.getsource(cli._build)
    recognised, _, dispatched = source.partition("answer_policy")
    assert '"two_stage_synthesis",' in recognised, "variant not recognised"
    assert "two_stage_synthesis" in dispatched, "variant recognised but not dispatched"


def test_v4_keeps_the_v3_evidence_hydration():
    """The registered comparison is v3.3 retrieval against a v4 answerer. If v4 also
    turned hydration off, a difference would have two causes."""
    source = inspect.getsource(cli._build)
    # Asserted as membership rather than as a literal set: the set gained a third
    # variant and a formatter split it across lines, which broke a string match on a
    # property that had not changed.
    hydration_block = source.split("adaptive_reasoning_hydration=variant", 1)[1][:220]
    for variant in ("two_stage_reasoned_evidence", "two_stage_synthesis"):
        assert f'"{variant}"' in hydration_block, f"{variant} lost its hydration"


def test_the_runner_accepts_the_policy_and_pairs_it_with_its_own_schema():
    runner = MemoryRunner.__new__(MemoryRunner)
    table = {
        "v2": "AnswerVerdict",
        "reasoned_v3": "ReasonedAnswerVerdict",
        "synthesis_v4": "SynthesisVerdict",
    }
    for policy in table:
        assert policy in inspect.getsource(MemoryRunner.__init__)
    assert runner is not None
    # The pairing itself is asserted in test_verdict_schema_is_sent.py; here it is
    # enough that the policy name is known to the constructor at all.
    assert "synthesis_v4" in inspect.getsource(MemoryRunner.__init__)


def test_the_derivation_reaches_the_row_notes():
    """`compute` doing the arithmetic is only half of it. A number nobody can trace
    back to its operands is not more trustworthy than the model's own."""
    source = inspect.getsource(MemoryRunner)
    assert '"synthesis_computation": self._computation' in source
    assert '"synthesis_operation"' in source


def test_the_derivation_is_reset_per_question():
    """Carrying the previous question's derivation into this row would be worse than
    recording nothing: it would look like evidence."""
    assert "self._computation: dict | None = None" in inspect.getsource(MemoryRunner.answer)


def test_the_v4_schema_carries_the_operands_the_code_needs():
    fields = set(SynthesisVerdict.model_fields)
    assert {"operation", "items", "start_date", "end_date", "missing_field"} <= fields
