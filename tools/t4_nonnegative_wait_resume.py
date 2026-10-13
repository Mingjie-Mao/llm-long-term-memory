"""Operational sleep-boundary adapter for the frozen dev/test100 execution chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
import continue_dev_and_original_test100 as chain  # noqa: E402
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from run_paged_paired import save_json  # noqa: E402
from run_recursive_summary_cost import instances_for  # noqa: E402
from t4_savings_regression import immutable_json  # noqa: E402

ANALYSIS = REPO / "results/analysis"
INVENTORY = ANALYSIS / f"{chain.DEV}.wait-boundary-v1.execution.json"
STATE = ANALYSIS / f"{chain.DEV}.wait-boundary-v1.progress.json"
LEDGER = ANALYSIS / f"{chain.DEV}.wait-boundary-v1.bootstrap.json"
AMENDMENT = REPO / f"results/amendment-{chain.DEV}-wait-boundary-v1.md"
DRIVERS = frozenset(
    {
        "t4_regression_resume.py",
        "t4_savings_regression.py",
        "t4_judge_amendment.py",
        "run_original_test100.py",
    }
)


def safe_sleep(sleeper):
    def sleep(seconds):
        if seconds > 0:
            sleeper(seconds)

    return sleep


def bootstrap_receipt(progress):
    if progress != {"status": "RUNNING", "arm": "control", "rep": 2, "readers": 100, "grades": 95}:
        raise ValueError("not the registered wait-boundary stop")
    base = REPO / f"results/raw/{chain.DEV}-control-rep2"
    readers = rows_by_question(base.with_suffix(".readers.jsonl"))
    grades = rows_by_question(base.with_suffix(".grades.jsonl"))
    selected = instances_for("dev100")
    ids = {i.question_id for i in selected}
    if set(readers) != ids or not set(grades) < ids or len(grades) != 95:
        raise ValueError("wait-boundary saved block mismatch")
    qid = next(i.question_id for i in selected if i.question_id not in grades)
    if qid != "gpt4_c27434e8_abs":
        raise ValueError("wait-boundary first missing grade changed")
    state = base.with_suffix(f".{qid}.recovery.json")
    calls = chain.read(base.with_suffix(".usage.json"))["calls"]
    if not calls or calls[-1].get("ok") is not False or calls[-1].get("role") != "judge":
        raise ValueError("missing wait-boundary failure usage")
    return {
        "wait_boundary_bootstrap": True,
        "identity": f"control:rep2:{qid}",
        "reader_sha256": hashlib.sha256(
            json.dumps(readers[qid], sort_keys=True).encode()
        ).hexdigest(),
        "recovery_sha256": sha256_file(state),
        "failure_sha256": hashlib.sha256(
            json.dumps(calls[-1], sort_keys=True).encode()
        ).hexdigest(),
    }


def verify(snapshot):
    chain.verify(snapshot["parent"])
    for rel, digest in snapshot["files"].items():
        if sha256_file(REPO / rel) != digest:
            raise ValueError(f"wait-boundary adapter source changed: {rel}")
    for rel, proof in snapshot["saved_prefixes"].items():
        with (REPO / rel).open("rb") as f:
            data = f.read(proof["bytes"])
        if len(data) != proof["bytes"] or hashlib.sha256(data).hexdigest() != proof["sha256"]:
            raise ValueError("wait-boundary saved prefix changed")


def freeze():
    if INVENTORY.exists():
        snapshot = chain.read(INVENTORY)
        verify(snapshot)
        return snapshot
    parent = chain.freeze()
    stopped = chain.read(chain.STATE)
    if stopped.get("status") != "STOPPED" or stopped.get("returncode") != 1:
        raise ValueError("registered stopped execution chain required")
    entry = bootstrap_receipt(stopped["dev"])
    paths = [Path(__file__), AMENDMENT, REPO / "tests/test_t4_wait_boundary.py", chain.INVENTORY]
    prefixes = {
        str(p.relative_to(REPO)): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted((REPO / "results/raw").glob(f"{chain.DEV}-*.jsonl"))
    }
    snapshot = {
        "parent": parent,
        "files": {str(p.relative_to(REPO)): sha256_file(p) for p in paths},
        "saved_prefixes": prefixes,
        "bootstrap_receipt": entry,
        "stopped_chain": stopped,
        "max_timer_bug_bootstraps": 1,
        "additional_provider_retry_budget": 0,
    }
    immutable_json(INVENTORY, snapshot)
    verify(snapshot)
    return snapshot


@contextmanager
def runtime_adapter():
    from llm_long_term_memory.evaluation.recovery import RecoveringJudge

    original_init, original_call = RecoveringJudge.__init__, subprocess.call

    def init(self, *args, **kwargs):
        kwargs["sleep"] = safe_sleep(kwargs.get("sleep", time.sleep))
        original_init(self, *args, **kwargs)

    def call(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and len(command) == 3 and command[2] == "--execute":
            path = Path(command[1])
            if path.name in DRIVERS and path.resolve() == REPO / "tools" / path.name:
                command = [
                    command[0],
                    str(Path(__file__).resolve()),
                    "--driver",
                    path.name,
                    "--execute",
                ]
        return original_call(command, *args, **kwargs)

    RecoveringJudge.__init__, subprocess.call = init, call
    try:
        yield
    finally:
        RecoveringJudge.__init__, subprocess.call = original_init, original_call


def bootstrap_adapters(snapshot):
    original_receipt, original_charge = chain.receipt, chain.charge

    def receipt(progress):
        try:
            return original_receipt(progress)
        except ValueError:
            if LEDGER.exists():
                raise
            entry = bootstrap_receipt(progress)
            if entry != snapshot["bootstrap_receipt"]:
                raise ValueError("registered timer interruption changed") from None
            return entry

    def charge(entry, *, ready):
        if not entry.get("wait_boundary_bootstrap"):
            return original_charge(entry, ready=ready)
        if not ready:
            return False
        if LEDGER.exists():
            raise ValueError("timer bootstrap already consumed")
        immutable_json(LEDGER, {**entry, "started_at": datetime.now(UTC).isoformat()})
        return True

    return receipt, charge


def main():
    from llm_long_term_memory.locking import exclusive

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--driver", choices=sorted(DRIVERS))
    args = parser.parse_args()
    snapshot = freeze()
    if not args.execute:
        print(json.dumps({"provider_calls": 0, "additional_provider_retry_budget": 0}))
        return 0
    if args.driver:
        sys.argv = [str(REPO / "tools" / args.driver), "--execute"]
        with runtime_adapter():
            runpy.run_path(sys.argv[0], run_name="__main__")
        return 0
    with exclusive(STATE, what="frozen wait-boundary execution adapter"):
        old_receipt, old_charge = chain.receipt, chain.charge
        chain.receipt, chain.charge = bootstrap_adapters(snapshot)
        sys.argv = [str(REPO / "tools/continue_dev_and_original_test100.py"), "--execute"]
        save_json(STATE, {"status": "RUNNING", "inventory_sha256": sha256_file(INVENTORY)})
        try:
            with runtime_adapter():
                rc = chain.main()
            save_json(
                STATE,
                {
                    "status": "COMPLETE" if rc == 0 else "STOPPED",
                    "returncode": rc,
                    "chain": chain.read(chain.STATE),
                },
            )
            return rc
        finally:
            chain.receipt, chain.charge = old_receipt, old_charge


if __name__ == "__main__":
    raise SystemExit(main())
