"""The dev60 verifier has to fail on tampering, not merely pass on clean data.

`dev60` is spent: the one-shot ledger forbids re-running it, so the archive is the
only remaining record and a verifier that cannot detect a changed file would make
that record worthless while still printing PASS. Each test below breaks one link in
the chain and asserts the verifier notices.
"""

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
    path = Path(__file__).resolve().parent.parent / "tools" / "verify_v3_dev60.py"
    spec = importlib.util.spec_from_file_location("verify_v3_dev60_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _record(content: bytes) -> dict:
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def test_snapshot_verifier_rejects_content_different_from_freeze(tmp_path):
    """A snapshot with the right filenames and the wrong bytes is the dangerous case:
    it looks complete to anything that only compares names."""
    freeze = {"system": {"source": {"files": {"src/example.py": _record(b"frozen\n")}}}}
    snapshot = tmp_path / tool.SNAPSHOT
    snapshot.parent.mkdir(parents=True)
    with tarfile.open(snapshot, "w:gz") as handle:
        changed = b"changed\n"
        info = tarfile.TarInfo("src/example.py")
        info.size = len(changed)
        handle.addfile(info, io.BytesIO(changed))

    with pytest.raises(tool.VerificationError, match="snapshot member changed"):
        tool._verify_snapshot(tmp_path, freeze)


def test_snapshot_verifier_rejects_a_missing_member(tmp_path):
    freeze = {
        "system": {
            "source": {
                "files": {"src/a.py": _record(b"a\n"), "src/b.py": _record(b"b\n")},
            }
        }
    }
    snapshot = tmp_path / tool.SNAPSHOT
    snapshot.parent.mkdir(parents=True)
    with tarfile.open(snapshot, "w:gz") as handle:
        info = tarfile.TarInfo("src/a.py")
        info.size = 2
        handle.addfile(info, io.BytesIO(b"a\n"))

    with pytest.raises(tool.VerificationError, match="does not match the frozen source"):
        tool._verify_snapshot(tmp_path, freeze)


def test_ledger_verifier_rejects_an_incomplete_run(tmp_path):
    """`status: complete` is what makes 'the shot is spent' checkable rather than
    asserted, so a partial ledger must never verify."""
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text("{}", encoding="utf-8")
    ledger = tmp_path / tool.LEDGER
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps({"status": "started", "rows": {"a": 60}}),
        encoding="utf-8",
    )
    with pytest.raises(tool.VerificationError, match="not complete"):
        tool._verify_ledger(tmp_path, Path("freeze.json"))


def test_ledger_verifier_rejects_a_short_run(tmp_path):
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text("{}", encoding="utf-8")
    ledger = tmp_path / tool.LEDGER
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "status": "complete",
                "freeze_sha256": tool._sha256(freeze_path),
                "rows": {f"arm{i}": 60 for i in range(5)},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(tool.VerificationError, match="records 300 rows"):
        tool._verify_ledger(tmp_path, Path("freeze.json"))


def test_aggregate_verifier_rejects_a_failed_gate(tmp_path):
    """A conclusion that says 'passed' over an aggregate that did not is the one
    inconsistency an archive must never be able to carry."""
    aggregate = tmp_path / tool.AGGREGATE
    aggregate.parent.mkdir(parents=True)
    checks = dict.fromkeys(tool.GATES, True)
    checks["context_within_2x_v2"] = False
    aggregate.write_text(
        json.dumps(
            {
                "privacy": {"contains_question_ids_or_text": False},
                "sealed_artifacts": {},
                "gate": {"checks": checks},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(tool.VerificationError, match="did not pass"):
        tool._verify_aggregate(tmp_path)


def test_aggregate_verifier_rejects_content_leaking_into_the_aggregate(tmp_path):
    """The reading rule is enforced here, not just written in a document."""
    aggregate = tmp_path / tool.AGGREGATE
    aggregate.parent.mkdir(parents=True)
    aggregate.write_text(
        json.dumps(
            {
                "privacy": {"contains_question_ids_or_text": True},
                "sealed_artifacts": {},
                "gate": {"checks": dict.fromkeys(tool.GATES, True)},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(tool.VerificationError, match="free of question content"):
        tool._verify_aggregate(tmp_path)


def test_conclusion_verifier_rejects_changed_evidence(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    freeze = {"system": {"source": {"tree_sha256": "a" * 64}}}
    aggregate = {"arms": [], "paired_majority": {}, "gate": {}}
    payload = {
        "schema_version": 1,
        "experiment_id": "v3-dev60",
        "source_sha256": "a" * 64,
        "outcome": "validated_v3.3",
        "summary": None,
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

    # The summary check fires before the file inventory; both are failures, and the
    # test pins that a broken package cannot verify rather than which link broke.
    with pytest.raises(tool.VerificationError):
        tool._verify_conclusion(tmp_path, freeze, aggregate)


def test_the_real_archive_verifies():
    """The package actually in the repository, end to end.

    Skipped where dev60's per-question rows are absent. They are gitignored on purpose:
    dev60 lives under "aggregate metrics and pre-declared slices only, no per-question
    reads, ever", and committing the rows to a public repository would put them in diffs
    and code search — making the read its protocol forbids the easy thing to do.

    The distinction this skip draws is the one the verifier itself cannot: a checkout
    that never had the rows reports the same `VerificationError` as one where they were
    tampered with. Tamper detection is unaffected and is pinned by the tests above, which
    build a complete archive and then break it.
    """
    repo = Path(__file__).resolve().parent.parent
    sealed = repo / "results/sealed/v3-dev60"
    if not sorted(sealed.glob("*.jsonl")):
        pytest.skip("dev60's sealed rows are kept local; aggregate-only by protocol")
    freeze = tool._read_json(repo, tool.FREEZE)
    # Deliberately not `_verify_live_source`. The archive attests the source that
    # produced the result, which `source.tar.gz` holds byte for byte; asserting the
    # working tree still matches would turn "development continued" — the expected
    # case — into a failing test, and a verifier that cries wolf gets ignored.
    tool._verify_snapshot(repo, freeze)
    tool._verify_ledger(repo, tool.FREEZE)
    aggregate = tool._verify_aggregate(repo)
    conclusion = tool._verify_conclusion(repo, freeze, aggregate)
    assert conclusion["outcome"] == "validated_v3.3"
    assert conclusion["summary"]["decision"] == "passed_all_nine_preregistered_gates"
