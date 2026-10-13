"""One uniform, audited service recovery extension for the frozen t4 regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from grounded_reader_diagnostic import grade_saved  # noqa: E402
from run_grounded_paired import combine  # noqa: E402
from run_paged_paired import save_json  # noqa: E402
from run_recursive_summary_cost import instances_for  # noqa: E402
from t4_savings_regression import STEM, VARIANT, immutable_json  # noqa: E402
from v18_judge_amendment import eligible  # noqa: E402

ANALYSIS = REPO / "results/analysis"
AMENDMENT = REPO / f"results/amendment-{STEM}-judge-recovery-v1.md"
INVENTORY = ANALYSIS / f"{STEM}.judge-amendment-v1.execution.json"
LEDGER = ANALYSIS / f"{STEM}.judge-amendment-v1.json"


def target(progress, instances, readers, grades):
    if progress.get("status") != "RECOVERY_EXHAUSTED":
        raise ValueError("no exhausted judge recovery")
    arm, rep = progress.get("arm"), progress.get("rep")
    if arm not in ("control", "candidate") or rep not in (1, 2, 3):
        raise ValueError("foreign arm/repeat")
    ids = {i.question_id for i in instances}
    if set(readers) != ids or not set(grades) < ids:
        raise ValueError("complete saved readers and missing grade required")
    return arm, rep, next(i for i in instances if i.question_id not in grades)


def freeze():
    parent = ANALYSIS / f"{STEM}.execution.json"
    original = json.loads(parent.read_text(encoding="utf-8"))
    for rel, digest in original["files"].items():
        if sha256_file(REPO / rel) != digest:
            raise ValueError(f"frozen regression input changed: {rel}")
    identity = sha256_file(parent)
    files = {
        str(p.relative_to(REPO)): sha256_file(p)
        for p in (Path(__file__), AMENDMENT, REPO / "tools/v18_judge_amendment.py", parent)
    }
    immutable_json(
        INVENTORY,
        {"parent_identity": identity, "files": files, "recoveries": 3, "cooldown_cap": 180},
    )
    return original, identity


def main():
    from llm_long_term_memory import cli
    from llm_long_term_memory.evaluation.harness import _exclusive
    from llm_long_term_memory.evaluation.recovery import RecoveringJudge, RecoveryExhausted
    from llm_long_term_memory.llm.usage import UsageTracker

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    original, identity = freeze()
    progress = json.loads((ANALYSIS / f"{STEM}.progress.json").read_text(encoding="utf-8"))
    base = REPO / f"results/raw/{STEM}-{progress.get('arm')}-rep{progress.get('rep')}"
    rp, gp, up = [
        base.with_suffix("." + s) for s in ("readers.jsonl", "grades.jsonl", "usage.json")
    ]
    with _exclusive(rp):
        readers, grades = rows_by_question(rp), rows_by_question(gp)
        arm, rep, instance = target(progress, instances_for("dev100"), readers, grades)
        for saved in readers.values():
            if (
                saved["inventory_sha256"] != identity
                or saved["answer_prompt_version"] != original["arms"][arm]
                or saved["cohort"] != arm
                or saved["answer"]["notes"].get("execution_identity")
                != {"inventory_sha256": identity, "arm": arm, "answerer": "gemini-3.5-flash-lite"}
            ):
                raise ValueError("saved reader source/prompt changed")
        selected = {i.question_id: i for i in instances_for("dev100")}
        for qid, grade in grades.items():
            combine(selected[qid], readers[qid], grade, identity, arm)
        row = readers[instance.question_id]
        state_path = base.with_suffix(f".{instance.question_id}.recovery.json")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected = hashlib.sha256(
            f"{identity}:{arm}:{rep}:{instance.question_id}".encode()
        ).hexdigest()
        if state["identity"] != expected or not eligible(state):
            raise ValueError("recovery must be exhausted by transient errors only")
        extended = base.with_suffix(f".{instance.question_id}.recovery-amendment-v1.json")
        entry = {
            "arm": arm,
            "repeat": rep,
            "question_id": instance.question_id,
            "original_recovery": str(state_path.relative_to(REPO)),
            "original_recovery_sha256": sha256_file(state_path),
            "reader_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
            "amendment_inventory_sha256": sha256_file(INVENTORY),
            "amendment_recovery": str(extended.relative_to(REPO)),
        }
        if not args.execute:
            print(json.dumps({"provider_calls": 0, **entry}))
            return 0
        _, _, runner, judge, usage = cli._build(
            VARIANT, "configs/fallback.yaml", "dev100", read_only_store=True
        )
        try:
            usage.records[:] = UsageTracker._load_records(up)
            judge.client.max_retries = judge.client.max_transport_retries = 2
            scoped = RecoveringJudge(
                judge,
                usage,
                up,
                extended,
                hashlib.sha256(f"{expected}:judge-amendment-v1".encode()).hexdigest(),
                max_recoveries=3,
                max_wait_seconds=180,
            )
            ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else []
            entry.update(started_at=datetime.now(UTC).isoformat(), outcome="STARTED")
            # Record invocation before the provider call; crash/restart keeps the
            # same durable extended state, never grants another recovery budget.
            ledger.append(entry)
            save_json(LEDGER, ledger)
            try:
                with gp.open("a", encoding="utf-8") as sink:
                    grade = grade_saved(scoped, instance, row, sink, identity)
                combine(instance, row, grade, identity, arm)
                entry["outcome"] = "GRADED"
            except RecoveryExhausted:
                entry["outcome"] = "EXHAUSTED_AGAIN"
                return 2
            finally:
                entry["finished_at"] = datetime.now(UTC).isoformat()
                save_json(LEDGER, ledger)
            print(json.dumps({"status": "GRADED", "arm": arm, "repeat": rep}))
            return 0
        finally:
            usage.save(up)
            runner.store.close()


if __name__ == "__main__":
    raise SystemExit(main())
