"""Bounded judge transport resumption around the frozen t4 continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from run_paged_paired import save_json  # noqa: E402
from run_recursive_summary_cost import instances_for  # noqa: E402
from t4_regression_resume import freeze as parent_freeze  # noqa: E402
from t4_regression_resume import verify as verify_parent  # noqa: E402
from t4_savings_regression import STEM, immutable_json  # noqa: E402

ANALYSIS = REPO / "results/analysis"
INVENTORY = ANALYSIS / f"{STEM}.transport-resume-v1.execution.json"
STATE = ANALYSIS / f"{STEM}.transport-resume-v1.progress.json"
LEDGER = ANALYSIS / f"{STEM}.transport-resume-v1.json"
AMENDMENT = REPO / f"results/amendment-{STEM}-judge-transport-v1.md"
MAX_RESUMES = 2
TRANSPORT = (
    "ConnectError:",
    "ConnectTimeout:",
    "ReadTimeout:",
    "ReadError:",
    "WriteTimeout:",
    "RemoteProtocolError:",
)


def transport_failure(progress, calls):
    return (
        progress.get("status") == "RUNNING"
        and bool(calls)
        and calls[-1].get("role") == "judge"
        and calls[-1].get("ok") is False
        and str(calls[-1].get("error", "")).startswith(TRANSPORT)
    )


def receipt(progress):
    arm, rep = progress.get("arm"), progress.get("rep")
    if arm not in ("control", "candidate") or rep not in (1, 2, 3):
        raise ValueError("foreign grading block")
    base = REPO / f"results/raw/{STEM}-{arm}-rep{rep}"
    readers = rows_by_question(base.with_suffix(".readers.jsonl"))
    grades_path = base.with_suffix(".grades.jsonl")
    grades = rows_by_question(grades_path) if grades_path.exists() else {}
    instances = instances_for("dev100")
    ids = {i.question_id for i in instances}
    if set(readers) != ids or not set(grades) < ids:
        raise ValueError("complete saved answers and missing grade required")
    qid = next(i.question_id for i in instances if i.question_id not in grades)
    calls = json.loads(base.with_suffix(".usage.json").read_text(encoding="utf-8"))["calls"]
    if not transport_failure(progress, calls):
        raise ValueError("not a judge transport interruption")
    return {
        "identity": f"{arm}:rep{rep}:{qid}",
        "arm": arm,
        "rep": rep,
        "question_id": qid,
        "reader_sha256": hashlib.sha256(
            json.dumps(readers[qid], sort_keys=True).encode()
        ).hexdigest(),
        "failure_sha256": hashlib.sha256(
            json.dumps(calls[-1], sort_keys=True).encode()
        ).hexdigest(),
        "error_type": str(calls[-1]["error"]).split(":", 1)[0],
    }


def reserve(ledger, entry):
    previous = [r for r in ledger if r["identity"] == entry["identity"]]
    if len(previous) >= MAX_RESUMES:
        raise ValueError("judge transport resumption budget exhausted")
    if any(r["reader_sha256"] != entry["reader_sha256"] for r in previous):
        raise ValueError("saved grading answer changed")
    return [*ledger, {**entry, "at": datetime.now(UTC).isoformat(), "attempt": len(previous) + 1}]


def main():
    from llm_long_term_memory.locking import exclusive

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    with exclusive(STATE, what="t4 judge transport continuation"):
        parent = parent_freeze()
        files = {str(p.relative_to(REPO)): sha256_file(p) for p in (Path(__file__), AMENDMENT)}
        immutable_json(
            INVENTORY,
            {
                "parent_files": parent["files"],
                "files": files,
                "max_resumes_per_grading_identity": MAX_RESUMES,
            },
        )
        if not args.execute:
            print(
                json.dumps({"provider_calls": 0, "max_resumes_per_grading_identity": MAX_RESUMES})
            )
            return 0
        while True:
            verify_parent(parent)
            if any(sha256_file(REPO / rel) != h for rel, h in files.items()):
                raise ValueError("transport continuation source changed")
            progress = json.loads((ANALYSIS / f"{STEM}.progress.json").read_text(encoding="utf-8"))
            if progress.get("status") == "COMPLETE":
                save_json(STATE, {"status": "COMPLETE", "parent": progress})
                return 0 if progress["pass"] else 1
            if progress.get("status") == "RUNNING":
                entry = receipt(progress)
                ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else []
                save_json(LEDGER, reserve(ledger, entry))
            save_json(STATE, {"status": "RUNNING", "inventory_sha256": sha256_file(INVENTORY)})
            rc = subprocess.call(
                [
                    str(REPO / ".venv/bin/python"),
                    str(REPO / "tools/t4_regression_resume.py"),
                    "--execute",
                ],
                cwd=REPO,
                env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"},
            )
            if rc == 0:
                continue
            after = json.loads((ANALYSIS / f"{STEM}.progress.json").read_text(encoding="utf-8"))
            if after.get("status") == "RUNNING":
                try:
                    receipt(after)
                except ValueError:
                    save_json(STATE, {"status": "STOPPED", "returncode": rc, "parent": after})
                    return rc
                continue
            save_json(STATE, {"status": "STOPPED", "returncode": rc, "parent": after})
            return rc


if __name__ == "__main__":
    raise SystemExit(main())
