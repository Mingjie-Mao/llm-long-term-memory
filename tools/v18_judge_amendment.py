"""Amendment 1 of paged-paired-dev100-v18-v1: one visible judge-recovery extension.

See `results/amendment-paged-paired-dev100-v18-v1-judge-recovery.md`. Grades the one
saved answer whose registered recovery the run's progress file reports exhausted, with
the registered request and a fresh, separately recorded budget. Never runs a reader and
never prints a verdict.
"""

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
from judge_request_cache import ExactRequestJudge  # noqa: E402
from paired_manifest_scope import select_manifest  # noqa: E402
from run_grounded_paired import combine  # noqa: E402
from run_paged_paired import (  # noqa: E402
    BASELINE,
    STEM,
    ScopedRecoveringCache,
    save_json,
    seed_cache,
)

ANALYSIS = REPO / "results/analysis"
LEDGER = ANALYSIS / f"{STEM}.judge-amendment.json"
AMENDMENT = REPO / f"results/amendment-{STEM}-judge-recovery.md"
TRANSIENT_CODES = {500, 502, 503, 504}
TRANSIENT_TYPES = {"ConnectionError", "TimeoutError"}
# RecoveringJudge's two messages: exhausted by this attempt, or found exhausted on restart.
EXHAUSTED = ("judge recovery stopped", "durable judge recovery budget exhausted")


def eligible(state):
    """The registered recovery is spent, and only by transient provider failures."""
    events = state.get("events") or []
    return (
        state.get("failures", 0) > state.get("max_recoveries", 3)
        and bool(events)
        and events[-1].get("exhausted") is True
        and all(
            e.get("code") in TRANSIENT_CODES or e.get("error_type") in TRANSIENT_TYPES
            for e in events
        )
    )


def target(progress, instances, grades):
    if not (
        progress.get("status") == "INCOMPLETE"
        and progress.get("stage") == "grades"
        and progress.get("reason", "").startswith(EXHAUSTED)
    ):
        raise ValueError("run is not stopped by an exhausted judge recovery")
    # The runner grades in manifest order and stops at the first failure.
    qid = next(i.question_id for i in instances if i.question_id not in grades)
    return progress["arm"], progress["repeat"], qid


def main():
    from llm_long_term_memory import cli
    from llm_long_term_memory.commands.common import manifest_instances
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.harness import _exclusive
    from llm_long_term_memory.evaluation.manifest import load_manifest, require_claim
    from llm_long_term_memory.evaluation.recovery import RecoveryExhausted
    from llm_long_term_memory.llm.usage import UsageTracker

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not AMENDMENT.exists():
        raise ValueError("amendment must be written before it is applied")
    inventory_path = ANALYSIS / f"{STEM}.execution.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if any(sha256_file(REPO / p) != sha for p, sha in inventory["files"].items()):
        raise ValueError("frozen source/config/data changed")
    digest = sha256_file(inventory_path)
    manifest = load_manifest(REPO / "results/manifests/dev100.json")
    require_claim(manifest, "regression")
    instances = select_manifest(manifest_instances(manifest, Settings()), manifest.question_ids)
    selected = {i.question_id: i for i in instances}
    progress = json.loads((ANALYSIS / f"{STEM}.progress.json").read_text(encoding="utf-8"))
    arm, rep = progress.get("arm"), progress.get("repeat")
    base = REPO / f"results/raw/{STEM}-{arm}-rep{rep}"
    rp, gp, up = (
        base.with_suffix(f".{s}") for s in ("readers.jsonl", "grades.jsonl", "usage.json")
    )
    with _exclusive(rp):
        readers, grades = rows_by_question(rp), rows_by_question(gp)
        arm, rep, qid = target(progress, instances, grades)
        if qid not in readers:
            raise ValueError("amendment grades saved answers only")
        original = base.with_suffix(f".{qid}.recovery.json")
        state = json.loads(original.read_text(encoding="utf-8"))
        identity = hashlib.sha256(f"{digest}:{arm}:{rep}:{qid}".encode()).hexdigest()
        if state["identity"] != identity or not eligible(state):
            raise ValueError("not an exhausted, transient-only registered recovery")
        extended = base.with_suffix(f".{qid}.recovery-amendment-1.json")
        ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else []
        entry = {
            "arm": arm,
            "repeat": rep,
            "question_id": qid,
            "original_recovery": str(original.relative_to(REPO)),
            "original_recovery_sha256": sha256_file(original),
            "original_failures": len(state["events"]),
            "amendment_recovery": str(extended.relative_to(REPO)),
            "amendment": str(AMENDMENT.relative_to(REPO)),
            "amendment_sha256": sha256_file(AMENDMENT),
        }
        if not args.execute:
            print(json.dumps({"provider_calls": 0, **entry}))
            return 0
        variant = BASELINE if arm == "baseline" else "two_stage_raw_primary_grounded_v18"
        _, _, runner, judge, usage = cli._build(
            variant, "configs/fallback.yaml", "dev100", read_only_store=True
        )
        try:
            if up.exists():
                usage.records[:] = UsageTracker._load_records(up)
            cache = ExactRequestJudge(judge)
            seed_cache(cache, selected, inventory, digest)
            scoped = ScopedRecoveringCache(
                cache,
                usage,
                up,
                extended,
                hashlib.sha256(
                    f"{digest}:{arm}:{rep}:{qid}:judge-amendment-1".encode()
                ).hexdigest(),
            )
            entry["started_at"] = datetime.now(UTC).isoformat()
            try:
                with gp.open("a", encoding="utf-8") as sink:
                    grade = grade_saved(scoped, selected[qid], readers[qid], sink, digest)
            except RecoveryExhausted:
                entry.update(outcome="EXHAUSTED_AGAIN", finished_at=datetime.now(UTC).isoformat())
                save_json(LEDGER, [*ledger, entry])
                print(json.dumps({"status": "EXHAUSTED_AGAIN", "question_id": qid}))
                return 1
            combine(selected[qid], readers[qid], grade, digest, arm)
            entry.update(outcome="GRADED", finished_at=datetime.now(UTC).isoformat())
            save_json(LEDGER, [*ledger, entry])
            print(json.dumps({"status": "GRADED", "arm": arm, "repeat": rep, "question_id": qid}))
            return 0
        finally:
            usage.save(up)
            runner.store.close()


if __name__ == "__main__":
    raise SystemExit(main())
