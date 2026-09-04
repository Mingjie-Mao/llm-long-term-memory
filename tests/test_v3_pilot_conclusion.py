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
    path = Path(__file__).resolve().parent.parent / "tools" / "verify_v3_answer_pilot.py"
    spec = importlib.util.spec_from_file_location("verify_v3_answer_pilot_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _record(content: bytes) -> dict[str, int | str]:
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def test_snapshot_verifier_rejects_content_different_from_freeze(tmp_path):
    content = b"frozen\n"
    freeze = {
        "system": {
            "source": {
                "files": {"src/example.py": _record(content)},
            }
        }
    }
    snapshot = tmp_path / tool.SNAPSHOT
    snapshot.parent.mkdir(parents=True)
    with tarfile.open(snapshot, "w:gz") as handle:
        info = tarfile.TarInfo("src/example.py")
        changed = b"changed\n"
        info.size = len(changed)
        handle.addfile(info, io.BytesIO(changed))

    with pytest.raises(tool.VerificationError, match="size changed"):
        tool._verify_snapshot(tmp_path, freeze)


def test_conclusion_verifier_rejects_changed_evidence(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    freeze = {"system": {"source": {"tree_sha256": "a" * 64}}}
    payload = {
        "schema_version": 1,
        "experiment_id": "v3-answer-pilot-train48",
        "source_sha256": "a" * 64,
        "files": [
            {
                "path": "evidence.json",
                "bytes": evidence.stat().st_size,
                "sha256": tool._sha256(evidence),
            }
        ],
    }
    conclusion = tmp_path / tool.CONCLUSION
    conclusion.parent.mkdir(parents=True)
    conclusion.write_text(json.dumps(payload), encoding="utf-8")
    evidence.write_text("changed\n", encoding="utf-8")

    with pytest.raises(tool.VerificationError, match="size changed"):
        tool._verify_conclusion(tmp_path, freeze)
