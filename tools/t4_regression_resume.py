"""Resume frozen t4 comparison under two visible operational amendments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from run_paged_paired import save_json  # noqa: E402
from t4_savings_regression import STEM, immutable_json  # noqa: E402

ANALYSIS = REPO / "results/analysis"
INVENTORY = ANALYSIS / f"{STEM}.resume-v1.execution.json"
STATE = ANALYSIS / f"{STEM}.resume-v1.progress.json"


def verify(snapshot):
    for rel, digest in snapshot["files"].items():
        if sha256_file(REPO / rel) != digest:
            raise ValueError(f"frozen resume source changed: {rel}")
    for rel, receipt in snapshot["starting_artifacts"].items():
        with (REPO / rel).open("rb") as f:
            prefix = f.read(receipt["bytes"])
        if (
            len(prefix) != receipt["bytes"]
            or hashlib.sha256(prefix).hexdigest() != receipt["sha256"]
        ):
            raise ValueError(f"saved prefix changed: {rel}")


def freeze():
    parent_path = ANALYSIS / f"{STEM}.execution.json"
    if INVENTORY.exists():
        snapshot = json.loads(INVENTORY.read_text(encoding="utf-8"))
        verify(snapshot)
        return snapshot
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    files = dict(parent["files"])
    for p in (
        parent_path,
        Path(__file__),
        REPO / "tools/analysis_io.py",
        REPO / "tools/t4_judge_amendment.py",
        REPO / "tools/v18_judge_amendment.py",
        REPO / f"results/amendment-{STEM}-jsonl-resume-v1.md",
        REPO / f"results/amendment-{STEM}-judge-recovery-v1.md",
    ):
        files[str(p.relative_to(REPO))] = sha256_file(p)
    starting = {}
    for p in sorted((REPO / "results/raw").glob(f"{STEM}-*.jsonl")):
        rows_by_question(p)  # Strict JSONL/unique-ID validation; no verdict report.
        starting[str(p.relative_to(REPO))] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    snapshot = {
        "parent_identity": sha256_file(parent_path),
        "files": files,
        "starting_artifacts": starting,
        "models_and_gates_unchanged": True,
        "summary_calls": 0,
    }
    verify(snapshot)
    immutable_json(INVENTORY, snapshot)
    return snapshot


def main():
    from llm_long_term_memory.locking import exclusive

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    with exclusive(STATE, what="t4 amended regression continuation"):
        snapshot = freeze()
        if not args.execute:
            print(
                json.dumps(
                    {
                        "provider_calls": 0,
                        "frozen_files": len(snapshot["files"]),
                        "saved_prefixes": len(snapshot["starting_artifacts"]),
                    }
                )
            )
            return 0
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
        while True:
            verify(snapshot)
            progress = json.loads((ANALYSIS / f"{STEM}.progress.json").read_text(encoding="utf-8"))
            if progress.get("status") == "COMPLETE":
                save_json(
                    STATE,
                    {
                        "status": "COMPLETE",
                        "parent": progress,
                        "resume_inventory_sha256": sha256_file(INVENTORY),
                    },
                )
                return 0 if progress["pass"] else 1
            amendment = progress.get("status") == "RECOVERY_EXHAUSTED"
            tool = "t4_judge_amendment.py" if amendment else "t4_savings_regression.py"
            save_json(
                STATE,
                {
                    "status": "RUNNING",
                    "tool": tool,
                    "resume_inventory_sha256": sha256_file(INVENTORY),
                },
            )
            rc = subprocess.call(
                [str(REPO / ".venv/bin/python"), str(REPO / "tools" / tool), "--execute"],
                cwd=REPO,
                env=environment,
            )
            verify(snapshot)
            after = json.loads((ANALYSIS / f"{STEM}.progress.json").read_text(encoding="utf-8"))
            if (amendment and rc != 0) or (
                not amendment and rc != 0 and after.get("status") != "RECOVERY_EXHAUSTED"
            ):
                save_json(
                    STATE, {"status": "STOPPED", "tool": tool, "returncode": rc, "parent": after}
                )
                return rc
            # An amended saved grade lets the original runner move past the old
            # exhausted state. Do not invoke the amendment a second time first.
            if amendment:
                verify(snapshot)
                rc = subprocess.call(
                    [
                        str(REPO / ".venv/bin/python"),
                        str(REPO / "tools/t4_savings_regression.py"),
                        "--execute",
                    ],
                    cwd=REPO,
                    env=environment,
                )
                verify(snapshot)
                after = json.loads((ANALYSIS / f"{STEM}.progress.json").read_text(encoding="utf-8"))
                if rc != 0 and after.get("status") != "RECOVERY_EXHAUSTED":
                    save_json(STATE, {"status": "STOPPED", "returncode": rc, "parent": after})
                    return rc


if __name__ == "__main__":
    raise SystemExit(main())
