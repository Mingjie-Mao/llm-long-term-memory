"""Failures must close release/experiment gates before any provider call."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from check_arm_invariant import main as check_arms  # noqa: E402
from probe_safety import (  # noqa: E402
    claim_holdout,
    complete_holdout,
    resume_ids,
    select_probes,
)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def split_spec(tmp_path):
    spec = {
        "store_fingerprint": "store",
        "probes": [
            {"probe_id": "dev", "question": "Development"},
            {"probe_id": "held", "question": "Held out"},
        ],
    }
    raw = json.dumps(spec).encode()
    path = tmp_path / "split.json"
    split = {
        "store_fingerprint": "store",
        "probe_set_sha256": hashlib.sha256(raw).hexdigest(),
        "development": ["dev"],
        "held_out": ["held"],
    }
    write_json(path, split)
    return spec, raw, path, split


def test_missing_split_never_opens_the_holdout(split_spec):
    spec, raw, path, _ = split_spec
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        select_probes(spec, raw, path, "development")


@pytest.mark.parametrize("change", ["hash", "overlap", "missing_id", "duplicate"])
def test_corrupt_splits_are_rejected(split_spec, change):
    spec, raw, path, split = split_spec
    if change == "hash":
        split["probe_set_sha256"] = "tampered"
    elif change == "overlap":
        split["development"].append("held")
    elif change == "missing_id":
        split["held_out"] = []
    else:
        split["development"].append("dev")
    write_json(path, split)
    with pytest.raises(ValueError):
        select_probes(spec, raw, path, "development")


def test_valid_development_selection_excludes_holdout(split_spec):
    spec, raw, path, _ = split_spec
    assert [p["probe_id"] for p in select_probes(spec, raw, path, "development")] == ["dev"]


def test_resume_rejects_different_variant_and_duplicate_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    identity = {"variant": "v4", "source_sha256": "same-source"}
    row = {"probe_id": "x", "question": "count", "variant": "v4", "run_identity": identity}
    probes = [{"probe_id": "x", "question": "count"}]
    write_json(path, row)
    assert resume_ids(path, probes, identity) == {"x"}
    with pytest.raises(ValueError, match="provenance"):
        resume_ids(path, probes, {**identity, "variant": "v3"})
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        resume_ids(path, probes, identity)


def test_holdout_ledger_survives_labels_and_allows_only_exact_incomplete_resume(tmp_path):
    ledger_path, output = tmp_path / "ledger.json", tmp_path / "first.jsonl"
    identity = {"variant": "v4"}
    ledger = claim_holdout(ledger_path, output, identity)
    output.touch()
    assert claim_holdout(ledger_path, output, identity) == ledger
    with pytest.raises(ValueError, match="different"):
        claim_holdout(ledger_path, tmp_path / "new-label.jsonl", identity)
    complete_holdout(ledger_path, ledger)
    with pytest.raises(ValueError, match="spent"):
        claim_holdout(ledger_path, output, identity)


def run_gate(tmp_path, monkeypatch, left, right, *flags):
    paths = [tmp_path / name for name in ("left.jsonl", "right.jsonl")]
    for path, rows in zip(paths, (left, right), strict=True):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["gate", *map(str, paths), *flags])
    return check_arms()


def row():
    return {
        "probe_id": "x",
        "retrieved_ids": ["a", "b"],
        "evidence_found": 2,
        "evidence_needed": 2,
        "context_complete": True,
        "question": "count",
        "kind": "count",
        "namespace": "alice",
        "gold": 2,
        "store_fingerprint": "same",
    }


@pytest.mark.parametrize("case", ["empty", "missing", "duplicate", "order", "identity"])
def test_gate_rejects_false_positive_comparisons(tmp_path, monkeypatch, case):
    left, right = [row()], [row()]
    if case == "empty":
        left = right = []
    elif case == "missing":
        left = right = [{"probe_id": "x"}]
    elif case == "duplicate":
        left = right = [row(), row()]
    elif case == "order":
        right[0]["retrieved_ids"].reverse()
    else:
        right[0]["question"] = "a different question"
    assert run_gate(tmp_path, monkeypatch, left, right) == 2


def test_scan_gate_permits_only_observed_routed_changes(tmp_path, monkeypatch):
    left, right = row(), row()
    right["retrieved_ids"].append("c")
    right["scan_route"] = {"routed": False}
    assert run_gate(tmp_path, monkeypatch, [left], [right], "--routed-scan") == 2
    right["scan_route"]["routed"] = True
    assert run_gate(tmp_path, monkeypatch, [left], [right], "--routed-scan") == 0


def test_release_check_refuses_partial_test_success(monkeypatch):
    path = REPO / "public-demo/check_release_figures.py"
    spec = importlib.util.spec_from_file_location("release_check_for_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1, stdout="TOTAL 100 17 83%\n710 passed, 1 error", stderr=""
        ),
    )
    with pytest.raises(SystemExit):
        module.measure()
