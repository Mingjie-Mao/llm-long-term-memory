from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_long_term_memory.evaluation.validation import ValidationArtifactError


def _load_validation_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_validation.py"
    spec = importlib.util.spec_from_file_location("validation_protocol_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_validation = _load_validation_script()


def _args(**changes):
    values = {
        "runs": 3,
        "arm": list(run_validation.REGISTERED_DEV_ARMS),
        "manifest": Path("results/manifests/dev100.json"),
        "store_name": "dev100",
        "freeze_name": "v2-candidate",
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_exact_preregistered_dev_protocol_is_accepted():
    run_validation._validate_protocol(_args())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"runs": 5}, "exactly 3"),
        ({"arm": [("flat20", "two_stage_fallback")]}, "must exactly match"),
        ({"store_name": "other"}, "store name 'dev100'"),
        ({"freeze_name": "other"}, "freeze name 'v2-candidate'"),
    ],
)
def test_protocol_drift_is_rejected(change, message):
    with pytest.raises(ValidationArtifactError, match=message):
        run_validation._validate_protocol(_args(**change))
