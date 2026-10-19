"""Separately registered second service recovery; never resets earlier budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import resume_original_test100_server as previous

REPO, RUNTIME, STEM = previous.REPO, previous.RUNTIME, previous.STEM
PHASE = "server-recovery-v2"
AMENDMENT = REPO / f"results/amendment-{STEM}-judge-recovery-v2.md"
REGISTRY = REPO / f"results/analysis/{STEM}.{PHASE}.execution.json"


def phase_factory(factory, *args, **kwargs):
    args = list(args)
    if len(args) != 5 or not str(args[3]).endswith(".recovery-amendment-v1.json"):
        raise ValueError("only the separately registered saved-grade extension is eligible")
    args[3] = Path(str(args[3]).removesuffix("-v1.json") + "-v2.json")
    args[4] = hashlib.sha256(f"{args[4]}:{PHASE}".encode()).hexdigest()
    return factory(*args, **kwargs)


@contextmanager
def isolated_phase(amendment, profile, cli, recovery):
    factory = recovery.RecoveringJudge
    with previous.amendment_profile(amendment, profile, cli):
        amendment.AMENDMENT = profile.ANALYSIS.parent / AMENDMENT.name
        amendment.INVENTORY = profile.ANALYSIS / f"{STEM}.judge-amendment-v2.execution.json"
        amendment.LEDGER = profile.ANALYSIS / f"{STEM}.judge-amendment-v2.json"
        recovery.RecoveringJudge = lambda *a, **kw: phase_factory(factory, *a, **kw)
        try:
            yield
        finally:
            recovery.RecoveringJudge = factory


def prior_binding(profile, amendment):
    progress = json.loads(profile.STATE.read_text(encoding="utf-8"))
    base = REPO / f"results/raw/{STEM}-{progress.get('arm')}-rep{progress.get('rep')}"
    readers = amendment.rows_by_question(base.with_suffix(".readers.jsonl"))
    grades = amendment.rows_by_question(base.with_suffix(".grades.jsonl"))
    arm, rep, item = amendment.target(progress, profile.experiment.instances(), readers, grades)
    state_path = base.with_suffix(f".{item.question_id}.recovery-amendment-v1.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not amendment.eligible(state):
        raise ValueError("previous phase must be exhausted by transient errors only")
    ledger_path = REPO / f"results/analysis/{STEM}.judge-amendment-v1.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    matching = [
        entry
        for entry in ledger
        if (entry["arm"], entry["repeat"], entry["question_id"]) == (arm, rep, item.question_id)
    ]
    original = base.with_suffix(f".{item.question_id}.recovery.json")
    reader_hash = hashlib.sha256(
        json.dumps(readers[item.question_id], sort_keys=True).encode()
    ).hexdigest()
    if not matching or any(
        entry["outcome"] != "EXHAUSTED_AGAIN"
        or entry["original_recovery_sha256"] != amendment.sha256_file(original)
        or entry["reader_sha256"] != reader_hash
        or entry["amendment_recovery"] != str(state_path.relative_to(REPO))
        for entry in matching
    ):
        raise ValueError("previous phase no longer binds the same saved answer and recovery")
    return state_path, ledger_path, original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    sys.path[:0] = [str(RUNTIME / "src"), str(RUNTIME / "tools")]
    import run_original_test100_dispatch as profile
    import t4_judge_amendment as amendment

    from llm_long_term_memory import cli
    from llm_long_term_memory.evaluation import recovery

    for module in (profile, amendment):
        if Path(module.__file__).resolve().parent != RUNTIME / "tools":
            raise ValueError("must use the exact frozen runtime")
    parent = json.loads(profile.INVENTORY.read_text(encoding="utf-8"))
    profile.experiment.verify(parent)
    profile.passing_prerequisite()
    prior_files = prior_binding(profile, amendment)
    files = (
        Path(__file__),
        Path(previous.__file__),
        AMENDMENT,
        REPO / "tests/test_original_test100_server_resume_v2.py",
        REPO / f"results/analysis/{STEM}.execution.json",
        REPO / f"results/analysis/{STEM}.server-recovery-v1.execution.json",
        *prior_files,
    )
    registration = {
        "namespace": f"{STEM}.{PHASE}",
        "parent_namespace": STEM,
        "extra_recoveries": 3,
        "extra_wait_cap_seconds": 180,
        "scope": "first missing saved grade with exhausted prior phase; then original driver",
        "files": {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }
    profile.immutable_json(REGISTRY, registration)
    with profile.runtime_adapter(), isolated_phase(amendment, profile, cli, recovery):
        previous.verify_restart(amendment, profile)
        sys.argv = [str(Path(__file__)), *(["--execute"] if args.execute else [])]
        rc = amendment.main()
    if rc or not args.execute:
        return rc
    return profile.main()


if __name__ == "__main__":
    raise SystemExit(main())
