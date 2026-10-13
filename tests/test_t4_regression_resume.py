import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import t4_regression_resume as tool


def test_saved_prefix_may_grow_but_never_change_and_source_is_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "REPO", tmp_path)
    source, rows = tmp_path / "source.py", tmp_path / "rows.jsonl"
    source.write_bytes(b"unchanged source")
    rows.write_bytes(b'{"question_id":"a"}\n')
    prefix = rows.read_bytes()
    snapshot = {
        "files": {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()},
        "starting_artifacts": {
            "rows.jsonl": {
                "bytes": len(prefix),
                "sha256": hashlib.sha256(prefix).hexdigest(),
            }
        },
    }
    rows.write_bytes(prefix + b'{"question_id":"b"}\n')
    tool.verify(snapshot)
    rows.write_bytes(prefix.replace(b'"a"', b'"x"'))
    with pytest.raises(ValueError, match="saved prefix"):
        tool.verify(snapshot)
    rows.write_bytes(prefix)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen resume source"):
        tool.verify(snapshot)


def test_second_service_exhaustion_stops_without_another_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "ANALYSIS", tmp_path)
    monkeypatch.setattr(tool, "STATE", tmp_path / "resume-progress.json")
    monkeypatch.setattr(tool, "INVENTORY", tmp_path / "inventory.json")
    tool.INVENTORY.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(tool, "freeze", lambda: {})
    monkeypatch.setattr(tool, "verify", lambda s: None)
    (tmp_path / f"{tool.STEM}.progress.json").write_text(
        json.dumps({"status": "RECOVERY_EXHAUSTED"}),
        encoding="utf-8",
    )
    calls = []

    def stopped(command, **kwargs):
        calls.append(Path(command[1]).name)
        return 2

    monkeypatch.setattr(tool.subprocess, "call", stopped)
    monkeypatch.setattr(sys, "argv", ["t4_regression_resume.py", "--execute"])
    assert tool.main() == 2
    assert calls == ["t4_judge_amendment.py"]
    assert json.loads(tool.STATE.read_text(encoding="utf-8"))["status"] == "STOPPED"
