"""Finish paged-paired-dev100-v18-v1 under its amendment 1; stops at the run's gate.

Replaces the stopped `acceptance-and-summary-continuation-v1` job for the rest of this
run, whose files stay as they are. Answers and grades come from the unchanged
`run_paged_paired.py`; an exhausted judge recovery goes to `v18_judge_amendment.py`
once per identity. The conditional summary-cost phase is not started here.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from continue_acceptance_job import next_reset, retryable_primary  # noqa: E402
from run_paged_paired import STEM, save_json  # noqa: E402
from v18_judge_amendment import EXHAUSTED  # noqa: E402

JOB = "v18-amended-continuation-v1"
ANALYSIS = REPO / "results/analysis"
STATE = ANALYSIS / f"{JOB}.progress.json"
LOG = ANALYSIS / f"{JOB}.log"
PRIMARY = ANALYSIS / f"{STEM}.progress.json"
GATE = ANALYSIS / f"{STEM}.gate.json"
MAX_RESETS = 6
# Amendment 2: an unsaved answer whose call failed after the client's registered
# transport retries is resumed whole, as after a quota stop; at most this many times.
MAX_TRANSPORT_RESTARTS = 3
TRANSPORT_ERRORS = (
    "httpx.ReadTimeout",
    "httpx.ConnectTimeout",
    "httpx.ConnectError",
    "httpx.RemoteProtocolError",
    "httpx.ReadError",
    "httpx.WriteTimeout",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run(log, tool, *extra):
    command = [str(REPO / ".venv/bin/python"), str(REPO / "tools" / tool), "--execute", *extra]
    log.write(f"\n{datetime.now(UTC).isoformat()} {tool} {' '.join(extra)}\n")
    log.flush()
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONWARNDEFAULTENCODING": "1",
    }
    return subprocess.call(command, cwd=REPO, env=environment, stdout=log, stderr=log)


def judge_exhausted(progress):
    return progress.get("status") == "INCOMPLETE" and progress.get("reason", "").startswith(
        EXHAUSTED
    )


def main():
    from llm_long_term_memory.locking import exclusive

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"provider_calls": 0, "max_quota_reset_waits": MAX_RESETS}))
        return 0
    with exclusive(STATE, what="v18 amended continuation"):
        job = read(STATE) or {"quota_reset_waits": 0, "amended": []}
        if job.get("status") not in (None, "RUNNING", "QUOTA_WAIT"):
            ended = {
                k: job.pop(k) for k in ("status", "reason", "returncode", "finished_at") if k in job
            }
            job.setdefault("previous_stops", []).append(ended)
        job.setdefault("transport_restarts", [])
        job.update(pid=os.getpid(), status="RUNNING", started_at=datetime.now(UTC).isoformat())

        def stop(status, **why):
            job.update(status=status, finished_at=datetime.now(UTC).isoformat(), **why)
            save_json(STATE, job)
            return 0 if status == "GATE_WRITTEN" else 1

        with LOG.open("a", encoding="utf-8") as log:
            while True:
                save_json(STATE, job)
                if GATE.exists():
                    return stop("GATE_WRITTEN")
                start = LOG.stat().st_size
                rc = run(log, "run_paged_paired.py", "--stage", "readers")
                job["readers_progress"] = read(PRIMARY)
                if GATE.exists():
                    continue
                if not retryable_primary(rc, job["readers_progress"]):
                    with LOG.open(encoding="utf-8", errors="replace") as tail:
                        tail.seek(start)
                        output = tail.read()
                    error = next((e for e in TRANSPORT_ERRORS if e + ":" in output), None)
                    if error and len(job["transport_restarts"]) < MAX_TRANSPORT_RESTARTS:
                        job["transport_restarts"].append(
                            {"at": datetime.now(UTC).isoformat(), "error": error, "returncode": rc}
                        )
                        continue
                    return stop("STOPPED", reason="reader stage failure", returncode=rc)
                while True:
                    rc = run(log, "run_paged_paired.py", "--stage", "grades")
                    grades = job["grading_progress"] = read(PRIMARY)
                    if GATE.exists() or not judge_exhausted(grades):
                        break
                    identity = f"{grades['arm']}-rep{grades['repeat']}"
                    rc_amend = run(log, "v18_judge_amendment.py")
                    job["amended"].append({"stage": identity, "returncode": rc_amend})
                    save_json(STATE, job)
                    if rc_amend != 0:
                        return stop("STOPPED", reason="amended judge recovery exhausted again")
                if GATE.exists():
                    continue
                if not retryable_primary(rc, grades):
                    return stop("STOPPED", reason="grade stage failure", returncode=rc)
                if job["quota_reset_waits"] >= MAX_RESETS:
                    return stop("QUOTA_WAIT_LIMIT_REACHED")
                job["quota_reset_waits"] += 1
                target = next_reset(datetime.now(UTC)) + timedelta(seconds=60)
                job.update(status="QUOTA_WAIT", waiting_until=target.isoformat())
                save_json(STATE, job)
                while datetime.now(UTC) < target:
                    time.sleep(min(60, max(1, (target - datetime.now(UTC)).total_seconds())))
                job.pop("waiting_until", None)
                job["status"] = "RUNNING"


if __name__ == "__main__":
    raise SystemExit(main())
