from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_long_term_memory.evaluation.manifest import Manifest
from llm_long_term_memory.evaluation.reproducibility import FreezeError


def _load_script(filename: str):
    path = Path(__file__).resolve().parent.parent / "scripts" / filename
    name = f"{path.stem}_for_tests"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_frozen_ingest = _load_script("run_frozen_ingest.py")
freeze_v2 = _load_script("freeze_v2.py")


def _documented_dev100_commands(text: str) -> list[str]:
    commands = re.findall(
        r"python scripts/(?:freeze_v2|run_frozen_ingest)\.py.*?(?=\n\s*\n|\Z)",
        text,
        flags=re.DOTALL,
    )
    return [command for command in commands if "results/manifests/dev100.json" in command]


def _formal_args(name: str, variants: list[str], freeze_name: str):
    return SimpleNamespace(
        manifest=Path(f"results/manifests/{name}.json"),
        store_name=name,
        freeze_name=freeze_name,
        variant=variants,
    )


def _freeze_args(
    name: str,
    variants: list[str],
    freeze_name: str,
    *,
    pre: bool,
    replace: bool = False,
):
    return SimpleNamespace(
        manifest=Path(f"results/manifests/{name}.json"),
        store_name=None if pre else name,
        pre_ingest_store_name=name if pre else None,
        name=freeze_name,
        variant=variants,
        replace=replace,
    )


def test_formal_dev_ingest_requires_exact_registered_variants():
    manifest = Manifest("dev100", "s", 0, tuple(f"q{i}" for i in range(100)))
    valid = _formal_args(
        "dev100",
        list(run_frozen_ingest._DEV_VARIANTS),
        "v2-candidate-preingest",
    )

    run_frozen_ingest._validate_formal_target(valid, manifest)
    valid.variant = ["two_stage_coherent"]
    with pytest.raises(FreezeError, match="pre-registration"):
        run_frozen_ingest._validate_formal_target(valid, manifest)


def test_dev100_documentation_matches_registered_variants():
    expected = list(freeze_v2._DEV_VARIANTS)
    assert list(run_frozen_ingest._DEV_VARIANTS) == expected

    repo = Path(__file__).resolve().parent.parent
    documents = {
        "results/v2-runbook.md": (repo / "results" / "v2-runbook.md").read_text(encoding="utf-8"),
    }
    expected_command_counts = {"results/v2-runbook.md": 3}
    for name, text in documents.items():
        commands = _documented_dev100_commands(text)
        assert len(commands) == expected_command_counts[name], name
        for command in commands:
            assert re.findall(r"--variant\s+([\w-]+)", command) == expected, name


@pytest.mark.parametrize(
    ("selected", "variants"),
    [
        (
            "two_stage_coherent",
            ["two_stage_coherent", "full_context", "naive_rag", "two_stage_fallback"],
        ),
        ("two_stage_fallback", ["two_stage_fallback", "full_context", "naive_rag"]),
    ],
)
def test_formal_test_ingest_is_bound_to_dev_selection(selected, variants):
    manifest = Manifest("test100", "s", 0, tuple(f"q{i}" for i in range(100)))
    args = _formal_args("test100", variants, "v2-final-preingest")

    run_frozen_ingest._validate_formal_target(args, manifest, selected_variant=selected)


def test_formal_dev_freeze_requires_registered_name_and_cannot_be_replaced():
    manifest = Manifest("dev100", "s", 0, tuple(f"q{i}" for i in range(100)))
    args = _freeze_args(
        "dev100",
        list(freeze_v2._DEV_VARIANTS),
        "v2-candidate-preingest",
        pre=True,
    )

    freeze_v2._validate_formal_freeze(args, manifest)
    args.replace = True
    with pytest.raises(FreezeError, match="never be replaced"):
        freeze_v2._validate_formal_freeze(args, manifest)


def test_formal_test_freeze_is_bound_to_selected_variant():
    manifest = Manifest("test100", "s", 0, tuple(f"q{i}" for i in range(100)))
    args = _freeze_args(
        "test100",
        ["two_stage_fallback", "full_context", "naive_rag"],
        "v2-final",
        pre=False,
    )

    freeze_v2._validate_formal_freeze(
        args,
        manifest,
        selected_variant="two_stage_fallback",
    )


def test_post_ingest_freeze_must_extend_the_pre_ingest_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze_v2, "REPO", tmp_path)
    pre = {
        "schema_version": 5,
        "pre_ingest_store_name": "dev100",
        "store": None,
        "source": {"tree_sha256": "before"},
    }
    path = tmp_path / "results" / "frozen" / "v2-candidate-preingest" / "freeze.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"system": pre}), encoding="utf-8")
    args = _freeze_args("dev100", list(freeze_v2._DEV_VARIANTS), "v2-candidate", pre=False)
    manifest = Manifest("dev100", "s", 0, tuple(f"q{i}" for i in range(100)))
    current = copy.deepcopy(pre)
    current["pre_ingest_store_name"] = None
    current["store"] = {"terminal": {"expected": 100}}

    freeze_v2._validate_pre_to_post_continuity(args, manifest, current)
    current["source"]["tree_sha256"] = "after"
    with pytest.raises(FreezeError, match="between pre- and post-ingest"):
        freeze_v2._validate_pre_to_post_continuity(args, manifest, current)


def test_pre_ingest_verifier_requires_matching_planned_store(tmp_path, monkeypatch):
    monkeypatch.setattr(run_frozen_ingest, "REPO", tmp_path)
    destination = tmp_path / "results" / "frozen" / "pre" / "freeze.json"
    destination.parent.mkdir(parents=True)
    frozen = {
        "store": None,
        "pre_ingest_store_name": "dev100",
        "variants": ["two_stage_coherent"],
    }
    destination.write_text(json.dumps({"system": frozen}), encoding="utf-8")
    monkeypatch.setattr(run_frozen_ingest, "capture_system", lambda request: frozen)
    args = SimpleNamespace(
        freeze_name="pre",
        config=tmp_path / "config.yaml",
        manifest=tmp_path / "manifest.json",
        store_name="dev100",
        variant=["two_stage_coherent"],
    )
    settings = SimpleNamespace(data_dir=tmp_path, store_dir=tmp_path, results_dir=tmp_path)

    assert run_frozen_ingest._verify_pre_ingest(args, settings) == destination

    args.store_name = "test100"
    with pytest.raises(FreezeError, match="different planned store"):
        run_frozen_ingest._verify_pre_ingest(args, settings)


def test_post_ingest_freeze_cannot_be_used_as_pre_ingest_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(run_frozen_ingest, "REPO", tmp_path)
    destination = tmp_path / "results" / "frozen" / "pre" / "freeze.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(
        json.dumps({"system": {"store": {"terminal": {}}, "pre_ingest_store_name": None}}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        freeze_name="pre",
        config=tmp_path / "config.yaml",
        manifest=tmp_path / "manifest.json",
        store_name="dev100",
        variant=["two_stage_coherent"],
    )
    settings = SimpleNamespace(data_dir=tmp_path, store_dir=tmp_path, results_dir=tmp_path)

    with pytest.raises(FreezeError, match="post-ingest"):
        run_frozen_ingest._verify_pre_ingest(args, settings)


def test_pre_ingest_capture_refuses_a_reused_store_name(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze_v2, "REPO", tmp_path)
    stores = tmp_path / "stores"
    stores.mkdir()
    (stores / "dev100.db").touch()
    monkeypatch.setattr(
        freeze_v2,
        "Settings",
        lambda: SimpleNamespace(
            store_dir=stores,
            data_dir=tmp_path / "data",
            results_dir=tmp_path / "results",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_v2.py",
            "--capture",
            "--name",
            "pre",
            "--config",
            "config.yaml",
            "--manifest",
            "manifest.json",
            "--pre-ingest-store-name",
            "dev100",
            "--variant",
            "two_stage_coherent",
        ],
    )

    assert freeze_v2.main() == 2
