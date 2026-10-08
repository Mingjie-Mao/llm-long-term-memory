"""One bounded continuation job; daily quota waits, immutable checkpoints, no cron."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import sha256_file  # noqa: E402
from run_paged_paired import STEM as PRIMARY  # noqa: E402
from run_paged_paired import save_json  # noqa: E402
from run_recursive_summary_cost import STEM as SUMMARY  # noqa: E402

JOB = "acceptance-and-summary-continuation-v1"
ANALYSIS = REPO / "results/analysis"
STATE = ANALYSIS / f"{JOB}.progress.json"
INVENTORY = ANALYSIS / f"{JOB}.execution.json"
LOG = ANALYSIS / f"{JOB}.log"
MAX_RESETS = 21


def next_reset(now):
    local = now.astimezone(ZoneInfo("America/Los_Angeles"))
    tomorrow = local.date() + timedelta(days=1)
    return datetime.combine(tomorrow, datetime.min.time(), tzinfo=local.tzinfo).astimezone(UTC)


def retryable_primary(rc, progress):
    if rc == 0:
        return True
    if rc != 2:
        return False
    return progress.get("status") in {"QUOTA_WAIT", "WAIT_FOR_READERS", "RUNNING"} or (
        progress.get("status") == "INCOMPLETE"
        and progress.get("reason", "").startswith("daily quota exhausted")
    )


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def freeze():
    primary = read(ANALYSIS / f"{PRIMARY}.execution.json")
    files = dict(primary["files"])
    for name in (
        "continue_acceptance_job.py",
        "run_recursive_summary_cost.py",
        "recursive_summary_report.py",
        "recursive_summary_nav.py",
        "recursive_summary_preflight.py",
        "paged_pair_report.py",
    ):
        p = REPO / "tools" / name
        files[str(p.relative_to(REPO))] = sha256_file(p)
    protocol = REPO / "results/prereg-recursive-summary-cost-v1.md"
    files[str(protocol.relative_to(REPO))] = sha256_file(protocol)
    preflight = ANALYSIS / f"{SUMMARY}.preflight-r2.json"
    files[str(preflight.relative_to(REPO))] = sha256_file(preflight)
    value = {
        "files": files,
        "max_quota_reset_waits": MAX_RESETS,
        "primary_namespace": PRIMARY,
        "summary_namespace": SUMMARY,
        "no_commits": True,
        "no_model_change": True,
        "one_fixed_job_not_recurring": True,
    }
    if INVENTORY.exists() and read(INVENTORY) != value:
        raise ValueError("continuation job source changed")
    if not INVENTORY.exists():
        save_json(INVENTORY, value)
    return value


def verify(identity):
    if any(sha256_file(REPO / p) != sha for p, sha in identity["files"].items()):
        raise ValueError("frozen experiment or continuation source changed")


def run(tool, stage, log):
    command = [
        str(REPO / ".venv/bin/python"),
        str(REPO / "tools" / tool),
        "--execute",
        "--stage",
        stage,
    ]
    log.write(f"\n{datetime.now(UTC).isoformat()} {tool} {stage}\n")
    log.flush()
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONWARNDEFAULTENCODING": "1",
    }
    return subprocess.call(command, cwd=REPO, env=environment, stdout=log, stderr=log)


def wait_for_existing_writer(identity, job):
    from llm_long_term_memory.locking import lock_holder

    while any(
        lock_holder(p) for p in (REPO / "results/raw").glob(f"{PRIMARY}-*.readers.jsonl.lock")
    ):
        verify(identity)
        job.update(status="WAIT_FOR_EXISTING_WRITER", updated_at=datetime.now(UTC).isoformat())
        save_json(STATE, job)
        time.sleep(30)


def quota_wait(identity, job):
    target = datetime.fromisoformat(job["waiting_until"]) if job.get("waiting_until") else None
    if target is None:
        if job["quota_reset_waits"] >= MAX_RESETS:
            job.update(status="QUOTA_WAIT_LIMIT_REACHED")
            save_json(STATE, job)
            return False
        job["quota_reset_waits"] += 1
        target = next_reset(datetime.now(UTC)) + timedelta(seconds=60)
        job["waiting_until"] = target.isoformat()
    job["status"] = "QUOTA_WAIT"
    save_json(STATE, job)
    while datetime.now(UTC) < target:
        verify(identity)
        time.sleep(min(30, (target - datetime.now(UTC)).total_seconds()))
    job.pop("waiting_until", None)
    job["status"] = "RESUMING"
    save_json(STATE, job)
    return True


def main():
    from llm_long_term_memory.locking import exclusive

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "provider_calls": 0,
                    "max_quota_reset_waits": MAX_RESETS,
                    "primary": PRIMARY,
                    "summary_conditional_on_primary_pass": SUMMARY,
                }
            )
        )
        return 0
    with exclusive(STATE, what="acceptance and summary continuation"):
        identity = freeze()
        job = read(STATE) or {"quota_reset_waits": 0, "started_at": datetime.now(UTC).isoformat()}
        if job.get("status") in {
            "COMPLETE",
            "PRIMARY_FAILED",
            "SUMMARY_FAILED",
            "STOPPED",
            "QUOTA_WAIT_LIMIT_REACHED",
        }:
            print(json.dumps(job))
            return 0 if job["status"] == "COMPLETE" else 1
        job["pid"] = os.getpid()
        try:
            with LOG.open("a", encoding="utf-8") as log:
                wait_for_existing_writer(identity, job)
                if job.get("waiting_until") and not quota_wait(identity, job):
                    return 1
                while True:
                    verify(identity)
                    gate = read(ANALYSIS / f"{PRIMARY}.gate.json")
                    if gate:
                        from paged_pair_report import report

                        report()
                        if not gate["pass"]:
                            job.update(
                                status="PRIMARY_FAILED", reason="paired progression gate failed"
                            )
                            save_json(STATE, job)
                            return 1
                        rc = run("run_recursive_summary_cost.py", "all", log)
                        result = read(ANALYSIS / f"{SUMMARY}.progress.json")
                        job["summary_progress"] = result
                        if rc == 0 and result.get("status") == "COMPLETE":
                            job.update(status="COMPLETE", finished_at=datetime.now(UTC).isoformat())
                            save_json(STATE, job)
                            return 0
                        if rc != 2 or result.get("status") != "QUOTA_WAIT":
                            job.update(
                                status="SUMMARY_FAILED",
                                reason="summary phase stopped",
                                returncode=rc,
                            )
                            save_json(STATE, job)
                            return 1
                    else:
                        rc = run("run_paged_paired.py", "readers", log)
                        readers = read(ANALYSIS / f"{PRIMARY}.progress.json")
                        job["primary_readers_progress"] = readers
                        if not retryable_primary(rc, readers):
                            job.update(
                                status="STOPPED", reason="primary reader failure", returncode=rc
                            )
                            save_json(STATE, job)
                            return 1
                        if (ANALYSIS / f"{PRIMARY}.gate.json").exists():
                            continue
                        # Grade completed arms with the independent judge pool while
                        # the answerer is quota paused. No duplicate readers.
                        rc = run("run_paged_paired.py", "grades", log)
                        grades = read(ANALYSIS / f"{PRIMARY}.progress.json")
                        job["primary_grading_progress"] = grades
                        if (ANALYSIS / f"{PRIMARY}.gate.json").exists():
                            continue
                        if not retryable_primary(rc, grades):
                            job.update(
                                status="STOPPED", reason="primary grading failure", returncode=rc
                            )
                            save_json(STATE, job)
                            return 1
                    if not quota_wait(identity, job):
                        return 1
        except Exception as exc:
            job.update(status="STOPPED", reason=f"{type(exc).__name__}: {exc}")
            save_json(STATE, job)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
