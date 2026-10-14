"""Human-triggered, separately registered recovery of the frozen original test100."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNTIME = REPO / "stores/experiments/original-test100-frozen-runtime-v1"
STEM = "original-test100-v2-t4-dispatch-v1"
AMENDMENT = REPO / f"results/amendment-{STEM}-judge-recovery-v1.md"
REGISTRY = REPO / f"results/analysis/{STEM}.server-recovery-v1.execution.json"


def verify_previous_binding(entries, expected):
    identity = ("arm", "repeat", "question_id")
    binding = ("original_recovery_sha256", "reader_sha256", "amendment_inventory_sha256")
    for entry in entries:
        if all(entry[k] == expected[k] for k in identity) and any(
            entry[k] != expected[k] for k in binding
        ):
            raise ValueError("saved answer or original recovery changed across extension")


def verify_restart(amendment, profile):
    amendment.freeze()
    progress = json.loads(profile.STATE.read_text(encoding="utf-8"))
    base = profile.REPO / (
        f"results/raw/{profile.STEM}-{progress.get('arm')}-rep{progress.get('rep')}"
    )
    readers = amendment.rows_by_question(base.with_suffix(".readers.jsonl"))
    grades = amendment.rows_by_question(base.with_suffix(".grades.jsonl"))
    arm, rep, item = amendment.target(progress, profile.experiment.instances(), readers, grades)
    recovery = base.with_suffix(f".{item.question_id}.recovery.json")
    expected = {
        "arm": arm,
        "repeat": rep,
        "question_id": item.question_id,
        "original_recovery_sha256": amendment.sha256_file(recovery),
        "reader_sha256": hashlib.sha256(
            json.dumps(readers[item.question_id], sort_keys=True).encode()
        ).hexdigest(),
        "amendment_inventory_sha256": amendment.sha256_file(amendment.INVENTORY),
    }
    entries = (
        json.loads(amendment.LEDGER.read_text(encoding="utf-8"))
        if amendment.LEDGER.exists()
        else []
    )
    verify_previous_binding(entries, expected)


@contextmanager
def amendment_profile(amendment, profile, cli):
    names = ("STEM", "ANALYSIS", "AMENDMENT", "INVENTORY", "LEDGER", "instances_for")
    saved = {name: getattr(amendment, name) for name in names}
    build, argv = cli._build, sys.argv
    amendment.STEM = profile.STEM
    amendment.ANALYSIS = profile.ANALYSIS
    amendment.AMENDMENT = profile.ANALYSIS.parent / AMENDMENT.name
    amendment.INVENTORY = profile.ANALYSIS / f"{profile.STEM}.judge-amendment-v1.execution.json"
    amendment.LEDGER = profile.ANALYSIS / f"{profile.STEM}.judge-amendment-v1.json"
    amendment.instances_for = lambda split: profile.experiment.instances()

    def original_store(variant, config, store, *args, **kwargs):
        return build(variant, config, "test100" if store == "dev100" else store, *args, **kwargs)

    cli._build = original_store
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(amendment, name, value)
        cli._build, sys.argv = build, argv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    sys.path[:0] = [str(RUNTIME / "src"), str(RUNTIME / "tools")]
    import run_original_test100_dispatch as profile
    import t4_judge_amendment as amendment

    from llm_long_term_memory import cli

    for module in (profile, amendment):
        if Path(module.__file__).resolve().parent != RUNTIME / "tools":
            raise ValueError("must import the exact frozen runtime, not current main")
    parent = json.loads(profile.INVENTORY.read_text(encoding="utf-8"))
    profile.experiment.verify(parent)
    profile.passing_prerequisite()
    files = (Path(__file__), AMENDMENT, REPO / f"results/analysis/{STEM}.execution.json")
    registration = {
        "namespace": f"{STEM}.server-recovery-v1",
        "parent_namespace": STEM,
        "extra_recoveries_per_missing_grade": 3,
        "extra_wait_cap_seconds": 180,
        "files": {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }
    if REGISTRY.exists():
        if json.loads(REGISTRY.read_text(encoding="utf-8")) != registration:
            raise ValueError("server recovery registration changed")
    else:
        REGISTRY.write_text(json.dumps(registration, indent=2) + "\n", encoding="utf-8")
    with amendment_profile(amendment, profile, cli), profile.runtime_adapter():
        verify_restart(amendment, profile)
        sys.argv = [str(Path(__file__)), *(["--execute"] if args.execute else [])]
        rc = amendment.main()
        if rc or not args.execute:
            return rc
        return profile.main()


if __name__ == "__main__":
    raise SystemExit(main())
