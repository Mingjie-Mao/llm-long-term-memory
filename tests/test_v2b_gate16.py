"""The v2b gate manifest is reproducible from the pilot rows — where those rows exist.

The verifier re-derives the 16 question ids from the three sealed v3 answer-pilot runs
and the local dataset copy. Both are deliberately absent from the published repository:
`results/sealed/**/*.jsonl` is ignored so the rows are not exposed in diffs and code
search while a decision is open, and `data/` is downloaded on demand. So this asserts
the selector where the inputs are present, and skips — loudly, naming what is missing —
where they are not, rather than failing CI for keeping the seal.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _module():
    path = REPO / "tools/verify_v2b_gate16.py"
    spec = importlib.util.spec_from_file_location("verify_v2b_gate16_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_gate_matches_deterministic_selector():
    module = _module()
    missing = [path for path in module.BASELINES if not path.is_file()]
    if missing:
        pytest.skip(f"the v3 answer-pilot rows are kept local: {missing[0].name} is absent")
    if not module.DATA.is_dir():
        pytest.skip("the LongMemEval corpus is downloaded on demand and is not in this checkout")

    result = module.verify()
    assert result["questions"] == 16
    assert result["majority_right_controls"] == 9
    assert result["majority_wrong_targets"] == 7
