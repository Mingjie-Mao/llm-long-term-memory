"""Zero-call, complete paired request audit of the failed scoped dev100 repair."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from run_grounded_paired import combine  # noqa: E402
from run_recursive_summary_cost import instances_for  # noqa: E402
from t4_regression_resume import freeze, verify  # noqa: E402
from t4_savings_regression import STEM as SOURCE  # noqa: E402
from t4_savings_regression import evaluate, immutable_json  # noqa: E402

STEM = "t4-scoped-dev100-postmortem-v1"
ANALYSIS = REPO / "results/analysis"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def request_audit(runs, readers, types):
    if set(runs) != {"control", "candidate"} or any(len(v) != 3 for v in runs.values()):
        raise ValueError("all three repeats per arm required")
    if set(readers) != set(runs) or any(len(v) != 3 for v in readers.values()):
        raise ValueError("paired readers required")
    rows = []
    for arm in runs:
        for repeat in range(3):
            if set(runs[arm][repeat]) != set(types) or set(readers[arm][repeat]) != set(types):
                raise ValueError("complete exact question inventory required")
            if any(type(row["correct"]) is not bool for row in runs[arm][repeat].values()):
                raise ValueError("typed original verdicts required")
    for qid, kind in types.items():
        counts, first, calls, mechanisms = {}, {}, {}, {}
        for arm in runs:
            counts[arm] = sum(r[qid]["correct"] for r in runs[arm])
            first[arm], calls[arm], mechanisms[arm] = [], [], []
            for repeat in readers[arm]:
                notes = repeat[qid]["answer"]["notes"]
                if "actual_answer_calls" not in notes:
                    raise ValueError("missing actual request capture")
                captured = notes["actual_answer_calls"]
                first[arm].append(digest(captured[0]["request"]) if captured else None)
                calls[arm].append(len(captured))
                selection = notes.get("selection_repair") or {}
                calculation = selection.get("calculation") or {}
                mechanisms[arm].append(calculation.get("mechanism"))
        rows.append(
            {
                "question_id": qid,
                "question_type": kind,
                "correct_out_of_three": counts,
                "mean_delta": (counts["candidate"] - counts["control"]) / 3,
                "majority_regression": counts["control"] >= 2 and counts["candidate"] <= 1,
                "first_request_sha256": first,
                "unique_nonempty_first_requests": {
                    a: len({h for h in hashes if h is not None}) for a, hashes in first.items()
                },
                "same_first_request_across_arms": [
                    c == t if c is not None and t is not None else None
                    for c, t in zip(first["control"], first["candidate"], strict=True)
                ],
                "actual_generation_counts": calls,
                "deterministic_mechanism": mechanisms,
                "first_loss": "not assigned by request/score audit alone",
            }
        )
    return rows


def analyze():
    from recursive_summary_report import model_workload

    from llm_long_term_memory.llm.usage import UsageTracker

    verify(freeze())
    ip = ANALYSIS / f"{SOURCE}.execution.json"
    inventory, identity = json.loads(ip.read_text(encoding="utf-8")), sha256_file(ip)
    selected = instances_for("dev100")
    types = {i.question_id: i.question_type for i in selected}
    if len(types) != 100 or set(types) != set(inventory["question_ids"]):
        raise ValueError("exact original dev100 inventory required")
    runs = {a: [] for a in ("control", "candidate")}
    readers = {a: [] for a in runs}
    inputs, workload = {}, []
    for rep in (1, 2, 3):
        for arm in runs:
            base = REPO / f"results/raw/{SOURCE}-{arm}-rep{rep}"
            paths = [
                base.with_suffix("." + s)
                for s in ("readers.jsonl", "grades.jsonl", "jsonl", "usage.json")
            ]
            saved, grades, joined = [rows_by_question(p) for p in paths[:3]]
            if any(set(rows) != set(types) for rows in (saved, grades, joined)):
                raise ValueError("all600 paired artifacts required")
            for i in selected:
                if saved[i.question_id]["answer_prompt_version"] != inventory["arms"][arm]:
                    raise ValueError("prompt identity changed")
                if joined[i.question_id] != asdict(
                    combine(i, saved[i.question_id], grades[i.question_id], identity, arm)
                ):
                    raise ValueError("bound joined result changed")
            readers[arm].append(saved)
            runs[arm].append(joined)
            workload.extend(UsageTracker._load_records(paths[3]))
            inputs.update({str(p.relative_to(REPO)): sha256_file(p) for p in paths})
    gp = ANALYSIS / f"{SOURCE}.gate.json"
    gate = json.loads(gp.read_text(encoding="utf-8"))
    recomputed = evaluate(runs)
    recomputed.update(inventory_sha256=identity, resources_by_model=model_workload(workload))
    if recomputed != gate:
        raise ValueError("frozen gate does not reproduce")
    rows = request_audit(runs, readers, types)
    detail_ids = {
        r["question_id"]
        for r in rows
        if r["mean_delta"] < 0 or any(r["deterministic_mechanism"]["candidate"])
    }
    inputs.update(
        {
            str(p.relative_to(REPO)): sha256_file(p)
            for p in (
                ip,
                gp,
                REPO / "results/manifests/dev100.json",
                REPO / f"results/prereg-{STEM}.md",
                Path(__file__),
            )
        }
    )
    return {
        "provider_calls": 0,
        "experiment_class": "exposed-dev100-postmortem-no-new-grading",
        "source_experiment": SOURCE,
        "source_gate_reproduces": True,
        "questions": 100,
        "reader_rows": 600,
        "grade_rows": 600,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "python": platform.python_version(),
        "source_models": {
            "answerer": "gemini-3.5-flash-lite",
            "judge": "gemma-4-31b-it",
            "extractor_not_called": "gemini-3.1-flash-lite",
            "embedder": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "source_prompt_versions": inventory["arms"],
        "rows": rows,
        "summary": {
            "majority_regressions": [r["question_id"] for r in rows if r["majority_regression"]],
            "negative_questions": [r["question_id"] for r in rows if r["mean_delta"] < 0],
            "positive_questions": [r["question_id"] for r in rows if r["mean_delta"] > 0],
            "all_captured_first_requests_changed": all(
                all(v is False for v in r["same_first_request_across_arms"]) for r in rows
            ),
            "deterministic_override_questions": [
                r["question_id"] for r in rows if any(r["deterministic_mechanism"]["candidate"])
            ],
            "detail_inspection_ids": sorted(detail_ids),
        },
        "inputs": inputs,
        "limitations": "Scores and request differences alone do not establish causation or "
        "first-loss. All original outcomes are retained; this is not fresh accuracy.",
    }


if __name__ == "__main__":
    result = analyze()
    immutable_json(ANALYSIS / f"{STEM}.json", result)
    print(json.dumps({"provider_calls": 0, **result["summary"]}, ensure_ascii=False))
