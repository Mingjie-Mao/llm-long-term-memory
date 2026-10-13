"""One bounded execution chain: await active dev acceptance, then run test100."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import sha256_file  # noqa: E402
from run_paged_paired import save_json  # noqa: E402
from t4_savings_regression import immutable_json  # noqa: E402

STEM = "original-test100-v2-t4-regression-v1"
ANALYSIS = REPO / "results/analysis"
STATE = ANALYSIS / f"{STEM}.chain-v1.progress.json"
INVENTORY = ANALYSIS / f"{STEM}.chain-v1.execution.json"
DEV = "t4-scoped-regression-dev100-v3"
MAX_WAIT_SECONDS = 12 * 3600


def prerequisite(progress, gate, active):
    if progress.get("status") == "COMPLETE":
        if progress.get("pass") is not True or gate is None or gate.get("pass") is not True:
            return "FAILED"
        return "READY"
    if progress.get("status") == "QUOTA_WAIT":
        return "QUOTA_WAIT"
    # The active, previously registered supervisor may be recovering an
    # exhausted transient error. A stale RUNNING file is not a running job.
    return "WAIT" if active else "STOPPED"


def verify(snapshot):
    for rel, digest in snapshot["files"].items():
        if sha256_file(REPO / rel) != digest:
            raise ValueError(f"one-shot execution chain source changed: {rel}")


def freeze():
    paths = [
        Path(__file__),
        REPO / "tools/run_original_test100.py",
        REPO / "tools/test100_v2_worker.py",
        REPO / "tests/test_original_test100_after_dev.py",
        REPO / f"results/prereg-{STEM}.md",
        REPO / f"results/amendment-{STEM}-execution-chain-v1.md",
        ANALYSIS / f"{STEM}.candidate-preflight-r2.json",
        ANALYSIS / f"{STEM}.baseline-preflight.json",
        ANALYSIS / f"{STEM}.resolution.json",
        ANALYSIS / f"{DEV}.execution.json",
        ANALYSIS / f"{DEV}.resume-v1.execution.json",
        ANALYSIS / f"{DEV}.transport-resume-v1.execution.json",
    ]
    snapshot = {
        "files": {str(p.relative_to(REPO)): sha256_file(p) for p in paths},
        "max_wait_seconds": MAX_WAIT_SECONDS,
        "attempts_to_start_original_test100": 1,
        "automatic_quota_or_failure_restarts": 0,
    }
    immutable_json(INVENTORY, snapshot)
    verify(snapshot)
    return snapshot


def main():
    from llm_long_term_memory.locking import exclusive, lock_holder

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    snapshot = freeze()
    if not args.execute:
        print(json.dumps({"provider_calls": 0, "max_wait_seconds": MAX_WAIT_SECONDS}))
        return 0
    with exclusive(STATE, what="one-shot dev-to-test100 execution chain"):
        started = time.monotonic()
        while True:
            verify(snapshot)
            progress = json.loads((ANALYSIS / f"{DEV}.progress.json").read_text(encoding="utf-8"))
            gp = ANALYSIS / f"{DEV}.gate.json"
            gate = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else None
            locks = [
                ANALYSIS / f"{DEV}.{label}.progress.json.lock"
                for label in ("resume-v1", "transport-resume-v1")
            ]
            action = prerequisite(progress, gate, any(lock_holder(p) for p in locks))
            if action == "READY":
                inventory = ANALYSIS / f"{DEV}.execution.json"
                if gate["inventory_sha256"] != sha256_file(inventory):
                    raise ValueError("dev gate inventory binding changed")
                save_json(
                    STATE, {"status": "EXECUTING_TEST100", "dev_gate_sha256": sha256_file(gp)}
                )
                rc = subprocess.call(
                    [
                        str(REPO / ".venv/bin/python"),
                        str(REPO / "tools/run_original_test100.py"),
                        "--execute",
                    ],
                    cwd=REPO,
                    env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"},
                )
                result_path = ANALYSIS / f"{STEM}.progress.json"
                result = (
                    json.loads(result_path.read_text(encoding="utf-8"))
                    if result_path.exists()
                    else {}
                )
                save_json(
                    STATE,
                    {
                        "status": "COMPLETE" if result.get("status") == "COMPLETE" else "STOPPED",
                        "returncode": rc,
                        "test100": result,
                    },
                )
                return rc
            if action != "WAIT":
                save_json(STATE, {"status": "PREREQUISITE_" + action, "dev": progress})
                print(json.dumps({"status": "PREREQUISITE_" + action}), flush=True)
                return 2
            if time.monotonic() - started >= MAX_WAIT_SECONDS:
                save_json(STATE, {"status": "WAIT_LIMIT_REACHED", "dev": progress})
                return 2
            save_json(STATE, {"status": "WAITING_FOR_ACTIVE_DEV", "dev": progress})
            time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
