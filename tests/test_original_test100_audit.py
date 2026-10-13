import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import audit_original_test100 as audit


@pytest.fixture
def empty_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(audit.experiment, "ANALYSIS", tmp_path)
    monkeypatch.setattr(audit.experiment, "INVENTORY", tmp_path / "execution.json")
    monkeypatch.setattr(
        audit.experiment, "instances", lambda: pytest.fail("partial outcomes inspected")
    )
    return tmp_path


def test_no_gate_means_no_partial_accuracy_audit(empty_protocol):
    with pytest.raises(ValueError, match="complete original-test100"):
        audit.audit()


@pytest.mark.parametrize("flag", ["true", 1, None])
def test_untyped_gate_cannot_authorize_accuracy_publication(empty_protocol, flag):
    stem = audit.experiment.STEM
    (empty_protocol / f"{stem}.progress.json").write_text(
        json.dumps({"status": "COMPLETE", "pass": True}), encoding="utf-8"
    )
    (empty_protocol / f"{stem}.gate.json").write_text(json.dumps({"pass": flag}), encoding="utf-8")
    with pytest.raises(ValueError, match="complete typed"):
        audit.audit()


def test_edited_execution_identity_is_rejected_before_outcomes_are_loaded(empty_protocol):
    stem = audit.experiment.STEM
    (empty_protocol / "execution.json").write_text("{}", encoding="utf-8")
    (empty_protocol / f"{stem}.progress.json").write_text(
        json.dumps({"status": "COMPLETE", "pass": True}), encoding="utf-8"
    )
    (empty_protocol / f"{stem}.gate.json").write_text(
        json.dumps({"pass": True, "inventory_sha256": "other-experiment"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="inventory binding changed"):
        audit.audit()


@pytest.fixture
def complete_protocol(empty_protocol, monkeypatch):
    from llm_long_term_memory.llm.usage import CallRecord, UsageTracker
    from llm_long_term_memory.prompt_versions import JUDGE_PROMPT_VERSION

    root = empty_protocol
    monkeypatch.setattr(audit, "REPO", root)
    monkeypatch.setattr(audit.experiment, "REPO", root)
    selected = [
        SimpleNamespace(
            question_id=f"q{n}", question_type="user", answer="gold", is_abstention=False
        )
        for n in range(100)
    ]
    monkeypatch.setattr(audit.experiment, "instances", lambda: selected)
    snapshot = {
        "files": {},
        "question_ids": [i.question_id for i in selected],
        "arms": {"control": "memory-aware-v2", "candidate": audit.experiment.VERSION},
    }
    audit.experiment.INVENTORY.write_text(json.dumps(snapshot), encoding="utf-8")
    identity = audit.sha256_file(audit.experiment.INVENTORY)
    raw = root / "results/raw"
    raw.mkdir(parents=True)
    runs = {a: [] for a in snapshot["arms"]}
    workload = {a: [] for a in snapshot["arms"]}
    for rep in (1, 2, 3):
        for arm, version in snapshot["arms"].items():
            base = raw / f"{audit.experiment.STEM}-{arm}-rep{rep}"
            readers, grades, joined = [], [], []
            for n, i in enumerate(selected):
                reader = {
                    "question_id": i.question_id,
                    "cohort": arm,
                    "inventory_sha256": identity,
                    "answer_prompt_version": version,
                    "answer": {
                        "text": "synthetic fixture",
                        "context_tokens": 6400,
                        "prompt_tokens": 20,
                        "output_tokens": 5,
                        "latency_ms": 1000,
                        "notes": {
                            "execution_identity": {
                                "inventory_sha256": identity,
                                "arm": arm,
                                "answerer": "gemini-3.5-flash-lite",
                            },
                            "actual_answer_calls": [
                                {"request": {"model": "gemini-3.5-flash-lite"}}
                            ],
                            "end_to_end_wall_ms": 1000,
                        },
                    },
                }
                grade = {
                    "question_id": i.question_id,
                    "cohort": arm,
                    "inventory_sha256": identity,
                    "judge_prompt_version": JUDGE_PROMPT_VERSION,
                    "reader_row_sha256": hashlib.sha256(
                        json.dumps(reader, sort_keys=True).encode()
                    ).hexdigest(),
                    "verdict": {"correct": n < 85, "reason": "synthetic fixture"},
                }
                readers.append(reader)
                grades.append(grade)
                joined.append(asdict(audit.combine(i, reader, grade, identity, arm)))
            for suffix, rows in [
                ("readers.jsonl", readers),
                ("grades.jsonl", grades),
                ("jsonl", joined),
            ]:
                base.with_suffix("." + suffix).write_text(
                    "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
                )
            records = [CallRecord("answerer", "gemini-3.5-flash-lite", 20, 5, 1000)] * 100
            records += [CallRecord("judge", "gemma-4-31b-it", 20, 5, 1000)] * 100
            UsageTracker(records).save(base.with_suffix(".usage.json"))
            workload[arm].extend(records)
            runs[arm].append({r["question_id"]: r for r in joined})
    result = audit.experiment.report(runs, workload, identity)
    (root / f"{audit.experiment.STEM}.progress.json").write_text(
        json.dumps({"status": "COMPLETE", "pass": result["pass"]}), encoding="utf-8"
    )
    return root


def test_complete_six_blocks_reproduce_and_audit_without_provider_calls(complete_protocol):
    result = audit.audit()
    assert result["provider_calls"] == 0
    assert result["reader_rows"] == result["grade_rows"] == 600
    assert result["accuracy_percent"] == {"control": 85, "candidate": 85}


def test_modified_bound_grade_is_refused(complete_protocol):
    base = complete_protocol / f"results/raw/{audit.experiment.STEM}-candidate-rep1"
    path = base.with_suffix(".grades.jsonl")
    rows = [json.loads(r) for r in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["verdict"]["correct"] = False
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="combined outcome differs"):
        audit.audit()


def test_edited_mean_gate_is_refused(complete_protocol):
    path = complete_protocol / f"{audit.experiment.STEM}.gate.json"
    gate = json.loads(path.read_text(encoding="utf-8"))
    gate["accuracy_percent"]["candidate"] = 99
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        audit.audit()


def test_missing_actual_request_audit_cannot_publish_accuracy(complete_protocol):
    path = complete_protocol / f"results/raw/{audit.experiment.STEM}-candidate-rep1.readers.jsonl"
    rows = [json.loads(r) for r in path.read_text(encoding="utf-8").splitlines()]
    del rows[0]["answer"]["notes"]["actual_answer_calls"]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="missing actual answer"):
        audit.audit()
