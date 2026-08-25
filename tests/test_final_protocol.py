from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_long_term_memory.evaluation.manifest import Manifest
from llm_long_term_memory.evaluation.reproducibility import FreezeError


def _load_final_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_final_test.py"
    spec = importlib.util.spec_from_file_location("run_final_test_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_final_test = _load_final_script()

COHERENT_ARMS = [
    ("v2", "two_stage_coherent"),
    ("full_context", "full_context"),
    ("naive_rag", "naive_rag"),
    ("flat_memory_fallback", "two_stage_fallback"),
]
FALLBACK_ARMS = [
    ("v2", "two_stage_fallback"),
    ("full_context", "full_context"),
    ("naive_rag", "naive_rag"),
]


def args():
    return SimpleNamespace(arm=COHERENT_ARMS)


def test_one_shot_ledger_is_bound_to_freeze_and_exact_arm_list(tmp_path):
    freeze = tmp_path / "freeze.json"
    freeze.write_text("frozen", encoding="utf-8")
    ledger = run_final_test._new_ledger(args(), freeze)

    run_final_test._validate_ledger(ledger, args(), freeze)

    freeze.write_text("moved", encoding="utf-8")
    with pytest.raises(FreezeError, match="does not match"):
        run_final_test._validate_ledger(ledger, args(), freeze)


def _wire_preflight(
    monkeypatch,
    tmp_path,
    *,
    completed=False,
    existing_raw=False,
    existing_usage=False,
    race_complete=False,
):
    results = tmp_path / "results"
    freeze = tmp_path / "v2-final" / "freeze.json"
    freeze.parent.mkdir(parents=True)
    freeze.write_text("freeze", encoding="utf-8")
    manifest = Manifest(
        name="test100",
        variant="s",
        seed=0,
        question_ids=("sealed",),
    )
    monkeypatch.setattr(
        run_final_test,
        "Settings",
        lambda: SimpleNamespace(results_dir=results, data_dir=tmp_path, store_dir=tmp_path),
    )
    monkeypatch.setattr(run_final_test, "_instances", lambda *unused: (manifest, [object()]))
    monkeypatch.setattr(run_final_test, "_verify_freeze", lambda *unused: ({}, freeze))
    monkeypatch.setattr(
        run_final_test,
        "load_dev_decision",
        lambda *unused: {"selected_variant": "two_stage_coherent"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_final_test.py",
            "--config",
            "configs/v2.yaml",
            "--arm",
            "v2=two_stage_coherent",
            "--arm",
            "full_context=full_context",
            "--arm",
            "naive_rag=naive_rag",
            "--arm",
            "flat_memory_fallback=two_stage_fallback",
            *(["--run"] if completed or race_complete else []),
        ],
    )
    if completed:
        ledger = run_final_test._new_ledger(args(), freeze)
        ledger["status"] = "complete"
        (freeze.parent / "one-shot.json").write_text(json.dumps(ledger), encoding="utf-8")
    if existing_raw:
        raw = results / "sealed" / "test100" / "v2.jsonl"
        raw.parent.mkdir(parents=True)
        raw.write_text("untracked final row", encoding="utf-8")
    if existing_usage:
        usage = results / "sealed" / "test100" / "v2.usage.json"
        usage.parent.mkdir(parents=True, exist_ok=True)
        usage.write_text("{}", encoding="utf-8")
    if race_complete:

        @contextmanager
        def complete_before_lock_yields(*unused, **unused_kwargs):
            ledger = run_final_test._new_ledger(args(), freeze)
            ledger["status"] = "complete"
            (freeze.parent / "one-shot.json").write_text(json.dumps(ledger), encoding="utf-8")
            yield

        monkeypatch.setattr(run_final_test, "exclusive", complete_before_lock_yields)
    return freeze, results


def test_preflight_does_not_create_the_one_shot_ledger(monkeypatch, tmp_path):
    freeze, _ = _wire_preflight(monkeypatch, tmp_path)

    assert run_final_test.main() == 0

    assert not (freeze.parent / "one-shot.json").exists()


def test_completed_protocol_refuses_a_second_run(monkeypatch, tmp_path):
    _wire_preflight(monkeypatch, tmp_path, completed=True)

    assert run_final_test.main() == 2


def test_rows_without_a_ledger_are_not_adopted(monkeypatch, tmp_path):
    _wire_preflight(monkeypatch, tmp_path, existing_raw=True)

    assert run_final_test.main() == 2


def test_usage_without_a_ledger_is_not_adopted(monkeypatch, tmp_path):
    _wire_preflight(monkeypatch, tmp_path, existing_usage=True)

    assert run_final_test.main() == 2


def test_ledger_is_rechecked_after_lock_to_close_the_two_process_race(monkeypatch, tmp_path):
    _wire_preflight(monkeypatch, tmp_path, race_complete=True)

    assert run_final_test.main() == 2


def test_final_instance_loader_rejects_a_non_test_manifest_before_loading_data(tmp_path):
    manifest = tmp_path / "dev100.json"
    manifest.write_text(
        json.dumps({"name": "dev100", "variant": "s", "question_ids": ["q"]}),
        encoding="utf-8",
    )
    settings = SimpleNamespace(data_dir=tmp_path)

    with pytest.raises(ValueError, match="only the sealed test100"):
        run_final_test._instances(manifest, settings)


def test_final_instance_loader_rejects_a_truncated_test_manifest_before_loading_data(tmp_path):
    manifest = tmp_path / "test100.json"
    manifest.write_text(
        json.dumps({"name": "test100", "variant": "s", "question_ids": ["q"]}),
        encoding="utf-8",
    )
    settings = SimpleNamespace(data_dir=tmp_path)

    with pytest.raises(ValueError, match="exactly 100"):
        run_final_test._instances(manifest, settings)


def test_final_runner_requires_the_product_candidate_first(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_final_test.py",
            "--config",
            "configs/v2.yaml",
            "--arm",
            "full_context=full_context",
        ],
    )

    assert run_final_test.main() == 2


def test_final_runner_refuses_to_omit_preregistered_baselines(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_final_test.py",
            "--config",
            "configs/v2.yaml",
            "--arm",
            "v2=two_stage_coherent",
        ],
    )

    assert run_final_test.main() == 2


@pytest.mark.parametrize("arms", [COHERENT_ARMS, FALLBACK_ARMS])
def test_exact_preregistered_arm_lists_are_accepted(arms):
    run_final_test._validate_final_arms(arms)


def test_unregistered_candidate_variant_is_rejected():
    with pytest.raises(FreezeError, match="selected on dev100"):
        run_final_test._validate_final_arms(
            [
                ("v2", "experimental"),
                ("full_context", "full_context"),
                ("naive_rag", "naive_rag"),
            ]
        )
