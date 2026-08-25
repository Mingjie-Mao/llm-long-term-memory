from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from llm_long_term_memory.config import ExperimentConfig


def _load_finalizer():
    path = Path(__file__).resolve().parent.parent / "scripts" / "finalize_train150.py"
    spec = importlib.util.spec_from_file_location("train_finalizer_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


finalizer = _load_finalizer()


def test_candidate_config_is_not_visible_until_explicit_publish(tmp_path):
    output = tmp_path / "v2.yaml"
    config = ExperimentConfig(name="v2")

    staged = finalizer.stage_config(config, output)

    assert staged.candidate_path.exists()
    assert staged.candidate_path != output
    assert not output.exists()

    staged.cleanup()

    assert not staged.candidate_path.exists()
    assert not output.exists()


def test_verified_candidate_is_atomically_published_and_reusable(tmp_path):
    output = tmp_path / "v2.yaml"
    config = ExperimentConfig(name="v2")
    staged = finalizer.stage_config(config, output)

    staged.publish()

    assert output.exists()
    assert ExperimentConfig.from_yaml(output).model_dump() == config.model_dump()
    existing = finalizer.stage_config(config, output)
    assert existing.candidate_path == output
    assert not existing.needs_publish


def test_existing_different_candidate_cannot_be_overwritten(tmp_path):
    output = tmp_path / "v2.yaml"
    ExperimentConfig(name="other").to_yaml(output)

    with pytest.raises(ValueError, match="automatic refreezing is forbidden"):
        finalizer.stage_config(ExperimentConfig(name="v2"), output)
