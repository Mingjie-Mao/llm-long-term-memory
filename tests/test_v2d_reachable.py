from __future__ import annotations

import inspect

from llm_long_term_memory import cli
from llm_long_term_memory.evaluation.runners.memory import MemoryRunner
from llm_long_term_memory.evaluation.runners.v2d import V2DVerdict


def test_v2d_variants_are_recognised_and_dispatch_to_the_v2d_policy():
    source = inspect.getsource(cli._build)
    for variant in ("two_stage_v2d_memory_only", "two_stage_v2d"):
        assert f'"{variant}"' in source
        assert f'"{variant}": "v2d"' in source


def test_v2d_runner_pairs_its_prompt_with_its_schema():
    source = inspect.getsource(MemoryRunner.__init__)
    assert '"v2d": (V2D_ANSWER_SYSTEM, V2D_ANSWER_PROMPT_VERSION, V2DVerdict)' in source
    fields = set(V2DVerdict.model_fields)
    assert {
        "premise_status",
        "operation",
        "operand_values",
        "operand_labels",
        "operand_qualifiers",
        "result_unit",
        "calculation_mode",
        "missing_field",
    } <= fields


def test_v2d_keeps_v2c_evidence_planning_and_labels_context():
    source = inspect.getsource(MemoryRunner)
    assert 'self.answer_policy in {"v2c", "v2d"}' in source
    assert 'answer_policy in {"synthesis_v4_enumerate", "v2d"}' in source
