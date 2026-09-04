from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_v3_phase3_tune.py"
    spec = importlib.util.spec_from_file_location("run_v3_phase3_tune_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tune = _load_script()


def test_tune_comparison_is_one_repeat_with_only_answer_policy_changed():
    assert tune.RUNS == 1
    assert tune.ARMS == (
        ("v2-control", "two_stage_fallback"),
        ("v3.1-reasoned", "two_stage_reasoned"),
    )


def test_tune_allocation_is_exactly_42_questions():
    assert sum(tune.EXPECTED_TYPES.values()) == 42
    assert tune.EXPECTED_TYPES["temporal-reasoning"] == 10
    assert tune.EXPECTED_TYPES["multi-session"] == 10
