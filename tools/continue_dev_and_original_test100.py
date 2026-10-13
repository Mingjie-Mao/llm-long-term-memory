"""Finite, human-triggered network recovery, then the registered original test100."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import sha256_file  # noqa: E402
from run_original_test100_after_dev import freeze as chain_freeze  # noqa: E402
from run_original_test100_after_dev import prerequisite  # noqa: E402
from run_original_test100_after_dev import verify as verify_chain  # noqa: E402
from run_paged_paired import save_json  # noqa: E402
from t4_regression_resume import freeze as parent_freeze  # noqa: E402
from t4_regression_resume import verify as verify_parent  # noqa: E402
from t4_savings_regression import STEM as DEV  # noqa: E402
from t4_savings_regression import immutable_json  # noqa: E402
from t4_transport_resume import LEDGER as OLD_LEDGER  # noqa: E402
from t4_transport_resume import STATE as OLD_STATE  # noqa: E402
from t4_transport_resume import receipt, reserve  # noqa: E402

ANALYSIS = REPO / "results/analysis"
STEM = "original-test100-v2-t4-regression-v1"
STATE = ANALYSIS / f"{STEM}.network-chain-v2.progress.json"
INVENTORY = ANALYSIS / f"{STEM}.network-chain-v2.execution.json"
LEDGER = ANALYSIS / f"{DEV}.transport-resume-v2.json"
AMENDMENT = REPO / f"results/amendment-{DEV}-network-restored-v2.md"
MAX_EXTRA = 2


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def extra_reserve(old, extra, entry):
    previous = [r for r in old if r["identity"] == entry["identity"]]
    if len(previous) != 2:
        raise ValueError("original two transport allowances must be consumed first")
    used = [r for r in extra if r["identity"] == entry["identity"]]
    if any(r["reader_sha256"] != entry["reader_sha256"] for r in [*previous, *used]):
        raise ValueError("saved grading answer changed")
    if len(used) >= MAX_EXTRA:
        raise ValueError("additional transport budget exhausted")
    return [*extra, {**entry, "attempt": len(used) + 1, "at": datetime.now(UTC).isoformat()}]


def network_ready():
    # A resolver hang must not spend a process-resumption allowance.
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import socket; socket.getaddrinfo('generativelanguage.googleapis.com',443)",
            ],
            capture_output=True,
            timeout=15,
            check=False,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def charge(entry, *, ready):
    if not ready:
        return False
    old = read(OLD_LEDGER, [])
    if sum(r["identity"] == entry["identity"] for r in old) < 2:
        save_json(OLD_LEDGER, reserve(old, entry))
    else:
        save_json(LEDGER, extra_reserve(old, read(LEDGER, []), entry))
    return True


def freeze():
    parent, chain = parent_freeze(), chain_freeze()
    transport = ANALYSIS / f"{DEV}.transport-resume-v1.execution.json"
    paths = [Path(__file__), AMENDMENT, REPO / "tests/test_network_test100_chain.py", transport]
    files = {str(p.relative_to(REPO)): sha256_file(p) for p in paths}
    files.update(read(transport)["files"])
    snapshot = {"parent": parent, "chain": chain, "files": files, "max_extra": MAX_EXTRA}
    immutable_json(INVENTORY, snapshot)
    return snapshot


def verify(snapshot):
    verify_parent(snapshot["parent"])
    verify_chain(snapshot["chain"])
    for rel, digest in snapshot["files"].items():
        if sha256_file(REPO / rel) != digest:
            raise ValueError(f"network continuation source changed: {rel}")


def call(tool):
    return subprocess.call(
        [str(REPO / ".venv/bin/python"), str(REPO / "tools" / tool), "--execute"],
        cwd=REPO,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"},
    )


def main():
    from llm_long_term_memory.locking import exclusive

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    snapshot = freeze()
    verify(snapshot)
    if not args.execute:
        print(
            json.dumps({"provider_calls": 0, "max_additional_resumptions_per_identity": MAX_EXTRA})
        )
        return 0
    # Hold the earlier supervisors' locks so another continuation cannot race us.
    with (
        exclusive(STATE, what="one-shot network dev-to-test100 chain"),
        exclusive(OLD_STATE, what="t4 transport continuation"),
        exclusive(ANALYSIS / f"{STEM}.chain-v1.progress.json", what="dev-to-test100 chain"),
    ):
        while True:
            verify(snapshot)
            progress = read(ANALYSIS / f"{DEV}.progress.json")
            if progress.get("status") == "COMPLETE":
                gate_path = ANALYSIS / f"{DEV}.gate.json"
                gate = read(gate_path)
                if prerequisite(progress, gate, False) != "READY":
                    save_json(STATE, {"status": "PREREQUISITE_FAILED", "dev": progress})
                    return 2
                if gate["inventory_sha256"] != sha256_file(ANALYSIS / f"{DEV}.execution.json"):
                    raise ValueError("dev gate inventory binding changed")
                save_json(
                    STATE,
                    {"status": "EXECUTING_TEST100", "dev_gate_sha256": sha256_file(gate_path)},
                )
                rc = call("run_original_test100.py")
                after = read(ANALYSIS / f"{STEM}.progress.json", {})
                save_json(
                    STATE,
                    {
                        "status": "COMPLETE" if after.get("status") == "COMPLETE" else "STOPPED",
                        "returncode": rc,
                        "test100": after,
                    },
                )
                return rc
            if not network_ready():
                save_json(STATE, {"status": "NETWORK_WAIT", "provider_calls_in_this_launch": 0})
                return 2
            if progress.get("status") == "RUNNING":
                try:
                    entry = receipt(progress)
                    charge(entry, ready=True)
                except ValueError as exc:
                    save_json(STATE, {"status": "STOPPED", "reason": str(exc), "dev": progress})
                    return 2
            save_json(
                STATE, {"status": "EXECUTING_DEV", "inventory_sha256": sha256_file(INVENTORY)}
            )
            rc = call("t4_regression_resume.py")
            verify(snapshot)
            after = read(ANALYSIS / f"{DEV}.progress.json")
            if rc == 0 or after.get("status") == "COMPLETE":
                continue
            if after.get("status") == "RUNNING":
                try:
                    receipt(after)
                except ValueError:
                    pass
                else:
                    continue
            save_json(STATE, {"status": "STOPPED", "returncode": rc, "dev": after})
            return rc


if __name__ == "__main__":
    raise SystemExit(main())
