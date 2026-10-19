import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "public-demo"))
tool = pytest.importorskip("update_regression_figures")


@pytest.fixture
def publication(tmp_path):
    gate_rel = f"results/analysis/{tool.STEM}.gate.json"
    audit_rel = f"results/analysis/{tool.STEM}.completion-audit-v1.json"
    gate = json.loads((REPO / gate_rel).read_text(encoding="utf-8"))
    audit = json.loads((REPO / audit_rel).read_text(encoding="utf-8"))
    (tmp_path / gate_rel).parent.mkdir(parents=True)
    (tmp_path / gate_rel).write_text(json.dumps(gate), encoding="utf-8")
    audit["inputs"] = {gate_rel: tool.sha256(tmp_path / gate_rel)}
    (tmp_path / audit_rel).write_text(json.dumps(audit), encoding="utf-8")
    manifest = tmp_path / "public-demo/site/release.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes((REPO / "public-demo/site/release.json").read_bytes())
    return tmp_path, gate_rel, audit_rel, manifest


def test_publication_keeps_historical_comparison_and_binds_new_evidence(publication):
    root, gate, audit, manifest = publication
    before = json.loads(manifest.read_text(encoding="utf-8"))
    tool.update(root)
    tool.update(root, check=True)
    after = json.loads(manifest.read_text(encoding="utf-8"))
    assert {k: v for k, v in after.items() if k != "regression"} == {
        k: v for k, v in before.items() if k != "regression"
    }
    assert after["regression"]["gate_sha256"] == tool.sha256(root / gate)
    assert after["regression"]["audit_sha256"] == tool.sha256(root / audit)


def test_gate_modified_after_audit_cannot_be_published(publication):
    root, gate, _, manifest = publication
    before = manifest.read_bytes()
    payload = json.loads((root / gate).read_text(encoding="utf-8"))
    payload["accuracy_percent"]["candidate"] += 1
    (root / gate).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        tool.update(root)
    assert manifest.read_bytes() == before


def test_partial_run_cannot_replace_public_result(publication):
    root, _, audit, manifest = publication
    before = manifest.read_bytes()
    payload = json.loads((root / audit).read_text(encoding="utf-8"))
    payload["grade_rows"] -= 1
    (root / audit).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="complete"):
        tool.update(root)
    assert manifest.read_bytes() == before


def test_hand_edited_public_mean_is_detected(publication):
    root, _, _, manifest = publication
    tool.update(root)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["regression"]["candidate_accuracy"] += 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="differ"):
        tool.update(root, check=True)
