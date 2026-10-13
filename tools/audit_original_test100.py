"""Zero-call completion audit before publishing the original-test100 regression."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
import run_original_test100 as experiment  # noqa: E402
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from run_grounded_paired import combine  # noqa: E402
from t4_savings_regression import immutable_json  # noqa: E402


def audit():
    from llm_long_term_memory.llm.usage import UsageTracker
    from llm_long_term_memory.prompt_versions import JUDGE_PROMPT_VERSION

    root, stem = experiment.ANALYSIS, experiment.STEM
    state_path, gate_path = root / f"{stem}.progress.json", root / f"{stem}.gate.json"
    if not state_path.exists() or not gate_path.exists():
        raise ValueError("complete original-test100 results required before audit")
    state, gate = [json.loads(p.read_text(encoding="utf-8")) for p in (state_path, gate_path)]
    if (
        state.get("status") != "COMPLETE"
        or type(gate.get("pass")) is not bool
        or state.get("pass") is not gate["pass"]
    ):
        raise ValueError("complete typed original-test100 gate required")
    identity = sha256_file(experiment.INVENTORY)
    if gate["inventory_sha256"] != identity:
        raise ValueError("gate inventory binding changed")
    snapshot = json.loads(experiment.INVENTORY.read_text(encoding="utf-8"))
    experiment.verify(snapshot)
    selected = experiment.instances()
    ids = {i.question_id for i in selected}
    if len(ids) != 100 or ids != set(snapshot["question_ids"]):
        raise ValueError("original100 manifest/execution mismatch")
    runs, workload = (
        {a: [] for a in ("control", "candidate")},
        {a: [] for a in ("control", "candidate")},
    )
    inputs = {}
    for rep in (1, 2, 3):
        for arm in ("control", "candidate"):
            base = REPO / f"results/raw/{stem}-{arm}-rep{rep}"
            rp, gp, up, cp = [
                base.with_suffix("." + s)
                for s in ("readers.jsonl", "grades.jsonl", "usage.json", "jsonl")
            ]
            readers, grades, joined = [rows_by_question(p) for p in (rp, gp, cp)]
            if any(set(rows) != ids for rows in (readers, grades, joined)):
                raise ValueError("exact100 readers/grades/combined rows required in every block")
            for i in selected:
                reader, grade = readers[i.question_id], grades[i.question_id]
                if (
                    reader["answer_prompt_version"] != snapshot["arms"][arm]
                    or grade["judge_prompt_version"] != JUDGE_PROMPT_VERSION
                ):
                    raise ValueError("registered prompt identity changed")
                if "actual_answer_calls" not in reader["answer"]["notes"]:
                    raise ValueError("missing actual answer request audit")
                calls = reader["answer"]["notes"]["actual_answer_calls"]
                if any(c["request"]["model"] != "gemini-3.5-flash-lite" for c in calls):
                    raise ValueError("wrong-model actual answer requests")
                expected = asdict(combine(i, reader, grade, identity, arm))
                if joined[i.question_id] != expected:
                    raise ValueError("saved combined outcome differs from bound reader/grade")
            records = UsageTracker._load_records(up)
            if any(
                (r.role, r.model)
                not in {("answerer", "gemini-3.5-flash-lite"), ("judge", "gemma-4-31b-it")}
                for r in records
            ):
                raise ValueError("unexpected role/model in actual usage")
            answer_requests = sum(
                len(r["answer"]["notes"].get("actual_answer_calls", [])) for r in readers.values()
            )
            # The original judge grades an empty hypothesis False locally,
            # without a provider call. Scores and API requests are distinct.
            judge_requests = sum(bool(r["answer"]["text"].strip()) for r in readers.values())
            if (
                sum(r.role == "answerer" and r.ok for r in records) < answer_requests
                or sum(r.role == "judge" and r.ok for r in records) < judge_requests
            ):
                raise ValueError("fresh reader/judge workload does not support full block")
            workload[arm].extend(records)
            runs[arm].append(joined)
            inputs.update({str(p.relative_to(REPO)): sha256_file(p) for p in (rp, gp, up, cp)})
    # The registered report generator refuses a different existing JSON/Markdown;
    # reproducing the gate is a CPU-only calculation, never a regrade.
    recomputed = experiment.report(runs, workload, identity)
    if recomputed != gate:
        raise ValueError("registered gate does not reproduce")
    paths = (state_path, gate_path, root / f"{stem}.gate.md", experiment.INVENTORY)
    inputs.update({str(p.relative_to(REPO)): sha256_file(p) for p in paths})
    return {
        "provider_calls": 0,
        "experiment_class": "exposed-original-test100-regression-completion-audit",
        "questions": 100,
        "reader_rows": 600,
        "grade_rows": 600,
        "gate_reproduces": True,
        "accuracy_percent": gate["accuracy_percent"],
        "historical_v2_one_shot_accuracy": 72.0,
        "inventory_sha256": identity,
        "inputs": inputs,
        "generator_sha256": sha256_file(Path(__file__)),
    }


if __name__ == "__main__":
    result = audit()
    immutable_json(experiment.ANALYSIS / f"{experiment.STEM}.completion-audit-v1.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "provider_calls",
                    "questions",
                    "reader_rows",
                    "grade_rows",
                    "gate_reproduces",
                    "accuracy_percent",
                )
            }
        )
    )
