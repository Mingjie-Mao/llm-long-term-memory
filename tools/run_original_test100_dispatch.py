"""New frozen original-test100 profile, reusing the unchanged full reader protocol."""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
import run_original_test100 as experiment  # noqa: E402
import t4_dispatch_replay as prerequisite  # noqa: E402
from analysis_io import sha256_file  # noqa: E402
from t4_nonnegative_wait_resume import runtime_adapter  # noqa: E402
from t4_savings_regression import immutable_json  # noqa: E402
from t4_scoped_dispatch import VERSION, ScopedSavingsRepairRunner  # noqa: E402

STEM = "original-test100-v2-t4-dispatch-v1"
ANALYSIS = REPO / "results/analysis"
INVENTORY = ANALYSIS / f"{STEM}.execution.json"
STATE = ANALYSIS / f"{STEM}.progress.json"
PREREQ = prerequisite.GATE
PROFILE_PREFLIGHT = ANALYSIS / f"{STEM}.profile-preflight.json"


def passing_prerequisite():
    gate = json.loads(PREREQ.read_text(encoding="utf-8"))
    snapshot = json.loads(prerequisite.INVENTORY.read_text(encoding="utf-8"))
    prerequisite.verify(snapshot)
    if (
        gate.get("pass") is not True
        or gate.get("decision", {}).get("pass") is not True
        or gate.get("candidate_prompt_version") != VERSION
        or gate.get("inventory_sha256") != sha256_file(prerequisite.INVENTORY)
        or gate.get("fresh_reader_rows") != 0
        or gate.get("fresh_grade_rows") != 0
        or gate.get("dev100_replayed_rows") != 300
        or gate.get("train21_replayed_rows") != 21
        or prerequisite.decide(gate["dev"], gate["train"]) != gate["decision"]
    ):
        raise ValueError("complete typed, source-bound dispatcher replay acceptance required")
    return snapshot


def profile_preflight():
    passing_prerequisite()
    baseline = ANALYSIS / f"{STEM}.baseline-preflight.json"
    candidate = ANALYSIS / f"{STEM}.candidate-preflight-r2.json"
    worker = REPO / "tools/test100_v2_dispatch_worker.py"
    rc = subprocess.call([str(REPO / ".venv/bin/python"), str(worker)], cwd=REPO)
    if rc:
        raise ValueError("historical baseline preflight failed")
    experiment.candidate_preflight()
    b, c = (
        json.loads(baseline.read_text(encoding="utf-8")),
        json.loads(candidate.read_text(encoding="utf-8")),
    )
    if (
        b["provider_calls"] != 0
        or c["provider_calls"] != 0
        or b["questions"] != 100
        or c["questions"] != 100
    ):
        raise ValueError("nonzero/incomplete composition preflight")
    result = {
        "provider_calls": 0,
        "questions": 100,
        "files": {
            str(p.relative_to(REPO)): sha256_file(p)
            for p in (
                baseline,
                candidate,
                Path(__file__),
                worker,
                REPO / "tools/t4_scoped_dispatch.py",
            )
        },
        "median_context": {
            "control": statistics.median(b["context_tokens"]),
            "candidate": statistics.median(c["context_tokens"]),
        },
        "estimated_six_run_context_tokens": 3
        * (sum(b["context_tokens"]) + sum(c["context_tokens"])),
        "limitations": "Offline composition only; excludes system/output/fallback/grading "
        "and failed-call tokens.",
    }
    immutable_json(PROFILE_PREFLIGHT, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "provider_calls",
                    "questions",
                    "median_context",
                    "estimated_six_run_context_tokens",
                )
            }
        )
    )


def freeze():
    accepted = passing_prerequisite()
    proof = json.loads(PROFILE_PREFLIGHT.read_text(encoding="utf-8"))
    for rel, digest in proof["files"].items():
        if sha256_file(REPO / rel) != digest:
            raise ValueError("dispatch profile preflight source changed")
    files = dict(accepted["files"])
    restored_path = ANALYSIS / "test100-v2-source-restore-v1.json"
    restored = json.loads(restored_path.read_text(encoding="utf-8"))
    restored_root = REPO / restored["root"]
    paths = list((REPO / "src").rglob("*.py"))
    paths += [restored_root / rel for rel in restored["restored_files"]]
    paths += [
        restored_root / "captured-v2.yaml",
        restored_path,
        Path(__file__),
        REPO / "tools/run_original_test100.py",
        REPO / "tools/test100_v2_worker.py",
        REPO / "tools/test100_v2_dispatch_worker.py",
        REPO / "tools/audit_original_test100.py",
        REPO / "tools/audit_original_test100_dispatch.py",
        REPO / "tools/t4_nonnegative_wait_resume.py",
        REPO / "tests/test_original_test100_dispatch.py",
        REPO / "results/manifests/test100.json",
        REPO / "stores/test100.db",
        REPO / "stores/test100-index.npy",
        REPO / "stores/test100-index.ids.json",
        REPO / f"results/prereg-{STEM}.md",
        PREREQ,
        prerequisite.INVENTORY,
        PROFILE_PREFLIGHT,
        ANALYSIS / f"{STEM}.baseline-preflight.json",
        ANALYSIS / f"{STEM}.candidate-preflight-r2.json",
        ANALYSIS / "original-test100-v2-t4-regression-v1.resolution.json",
    ]
    wal = REPO / "stores/test100.db-wal"
    if wal.exists() and wal.stat().st_size:
        paths.append(wal)
    files.update({str(p.relative_to(REPO)): sha256_file(p) for p in paths})
    old = json.loads(
        (ANALYSIS / "t4-scoped-regression-dev100-v3.execution.json").read_text(encoding="utf-8")
    )
    snapshot = {
        "files": files,
        "question_ids": [i.question_id for i in experiment.instances()],
        "models": old["models"],
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "python": platform.python_version(),
        "experiment_class": "exposed-original-test100-regression",
        "arms": {"control": "memory-aware-v2", "candidate": VERSION},
        "repeats": 3,
        "judge_recoveries": 7,
        "judge_cooldown_cap": 480,
        "reader_client_api_budget": 2,
        "reader_transport_budget": 2,
        "prerequisite_kind": "strict-request-output-replay-not-fresh-accuracy",
        "prerequisite_sha256": sha256_file(PREREQ),
        "profile_preflight": proof,
        "operational_sleep_adapter": "RecoveringJudge nonpositive sleep skipped; "
        "no retry budget change",
    }
    experiment.verify(snapshot)
    immutable_json(INVENTORY, snapshot)
    immutable_json(
        ANALYSIS / f"{STEM}.source-snapshot.json",
        {
            rel: {"sha256": h, "text": (REPO / rel).read_text(encoding="utf-8")}
            for rel, h in files.items()
            if Path(rel).suffix in (".py", ".md", ".yaml")
        },
    )
    return snapshot, sha256_file(INVENTORY)


def namespace_worker_command(command):
    if (
        isinstance(command, (list, tuple))
        and len(command) >= 2
        and Path(command[1]).resolve() == REPO / "tools/test100_v2_worker.py"
    ):
        return [command[0], str(REPO / "tools/test100_v2_dispatch_worker.py"), *command[2:]]
    return command


@contextmanager
def configured():
    values = {
        "STEM": STEM,
        "INVENTORY": INVENTORY,
        "STATE": STATE,
        "PREREQ": PREREQ,
        "VERSION": VERSION,
        "SavingsRepairRunner": ScopedSavingsRepairRunner,
        "freeze": freeze,
    }
    originals = {key: getattr(experiment, key) for key in values}
    call = subprocess.call

    def dispatch(command, *args, **kwargs):
        return call(namespace_worker_command(command), *args, **kwargs)

    for key, value in values.items():
        setattr(experiment, key, value)
    subprocess.call = dispatch
    try:
        yield
    finally:
        for key, value in originals.items():
            setattr(experiment, key, value)
        subprocess.call = call


def main():
    if "--preflight" in sys.argv:
        with configured():
            profile_preflight()
        return 0
    with configured(), runtime_adapter():
        return experiment.main()


if __name__ == "__main__":
    raise SystemExit(main())
