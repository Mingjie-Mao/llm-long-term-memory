from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest


def _load_tool():
    path = Path(__file__).resolve().parent.parent / "tools" / "verify_v2_conclusion.py"
    spec = importlib.util.spec_from_file_location("verify_v2_conclusion_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _source_record(content: bytes) -> dict[str, int | str]:
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _fixture(tmp_path: Path) -> tuple[Path, dict]:
    source = b"VALUE = 1\n"
    freeze = {
        "system": {
            "source": {
                "tree_sha256": "a" * 64,
                "files": {"src/example.py": _source_record(source)},
            },
            "data": {
                "manifest": {"sha256": "b" * 64},
                "dataset": {"sha256": "c" * 64},
            },
            "config": {"path": "configs/v2.yaml", "sha256": "d" * 64},
            "store": {"database": {"logical_sha256": "e" * 64}},
        }
    }
    archive = tmp_path / tool.SOURCE_SNAPSHOT
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("src/example.py")
        info.size = len(source)
        handle.addfile(info, io.BytesIO(source))
    return tmp_path, freeze


def test_source_snapshot_matches_frozen_inventory(tmp_path):
    repo, freeze = _fixture(tmp_path)

    tool._verify_source_snapshot(repo, freeze)


def test_source_snapshot_rejects_changed_content(tmp_path):
    repo, freeze = _fixture(tmp_path)
    freeze["system"]["source"]["files"]["src/example.py"]["sha256"] = "f" * 64

    with pytest.raises(tool.ConclusionError, match="hash changed"):
        tool._verify_source_snapshot(repo, freeze)


def test_manifest_verifier_rejects_a_changed_artifact(tmp_path):
    repo, freeze = _fixture(tmp_path)
    artifact = repo / "result.json"
    artifact.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "experiment_id": "v2-final-test100",
        "created_at_utc": "2026-09-03T00:00:00+00:00",
        "producer": {"command": "capture", "source_commit": "1" * 40},
        "fingerprints": {
            "source_sha256": "a" * 64,
            "config_sha256": "d" * 64,
            "inputs_sha256": tool._combined_input_sha256(freeze),
            "store_sha256": "e" * 64,
        },
        "files": [
            {
                "path": "result.json",
                "sha256": tool._sha256_file(artifact),
                "bytes": artifact.stat().st_size,
                "class": "summary",
            }
        ],
        "summary_file": "result.json",
        "outcome": "inconclusive",
        "outcome_reason": "mixed result",
        "evidence": ["result.json"],
    }
    conclusion = repo / tool.CONCLUSION
    conclusion.parent.mkdir(parents=True, exist_ok=True)
    conclusion.write_text(json.dumps(manifest), encoding="utf-8")
    artifact.write_text("changed\n", encoding="utf-8")

    with pytest.raises(tool.ConclusionError, match="file changed"):
        tool._verify_manifest(repo, freeze)
