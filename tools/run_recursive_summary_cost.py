"""Conditional, resumable recursive-navigation pilot and fresh paired cost regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from grounded_reader_diagnostic import grade_saved, save_reader  # noqa: E402
from judge_request_cache import ExactRequestJudge  # noqa: E402
from paired_manifest_scope import select_manifest  # noqa: E402
from recursive_summary_nav import (  # noqa: E402
    SummaryNavigationAnswerer,
    build_tree,
    digest,
    validate_tree,
)
from recursive_summary_report import (  # noqa: E402
    amortization,
    latency_summary,
    model_workload,
    paired_accuracy,
    pilot_gate,
    raw_reference_coverage,
    resource_total,
)
from run_grounded_experiment import frozen_identity  # noqa: E402
from run_grounded_paired import combine  # noqa: E402
from run_paged_paired import (  # noqa: E402
    STEM as PRIMARY,
)
from run_paged_paired import (  # noqa: E402
    CostBoundRunner,
    ScopedRecoveringCache,
    expected_calls,
    save_json,
)

STEM = "recursive-summary-cost-dev100-v1"
PILOT = (
    "09ba9854",
    "0ddfec37_abs",
    "1568498a",
    "51a45a95",
    "6456829e",
    "6e984301",
    "95228167",
    "9d25d4e0",
    "a89d7624",
    "af8d2e46",
    "b46e15ed",
    "eaca4986",
    "ed4ddc30",
    "f685340e",
    "f9e8c073",
    "gpt4_2f8be40d",
    "gpt4_731e37d7",
    "gpt4_7a0daae1",
    "gpt4_7ddcf75f",
    "gpt4_a1b77f9c",
    "gpt4_a2d1d1f6",
)
GAPS = {"f9e8c073", "gpt4_2f8be40d", "gpt4_731e37d7"}
ANALYSIS = REPO / "results/analysis"
ROOT = REPO / "stores/experiments" / STEM
VARIANT = "two_stage_raw_primary_grounded_v18"


class TimedRunner(CostBoundRunner):
    def answer_request(self, request):
        before = len(self.usage.records)
        started = time.perf_counter()
        answer = super().answer_request(request)
        answer.notes["end_to_end_wall_ms"] = (time.perf_counter() - started) * 1000
        answer.notes["whole_question_usage"]["by_model"] = model_workload(
            self.usage.records[before:]
        )
        return answer


def instances_for(split):
    from llm_long_term_memory.commands.common import manifest_instances
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.manifest import load_manifest, require_claim

    manifest = load_manifest(REPO / f"results/manifests/{split}.json")
    require_claim(manifest, "regression")
    ids = PILOT if split == "train150" else manifest.question_ids
    if not set(ids) <= set(manifest.question_ids):
        raise ValueError("pilot outside exposed train150")
    return select_manifest(manifest_instances(manifest, Settings()), ids)


def freeze():
    gate_path = ANALYSIS / f"{PRIMARY}.gate.json"
    if not gate_path.exists():
        raise ValueError("complete primary regression required before any summary API calls")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate["pass"]:
        raise ValueError("primary regression failed; summary API calls forbidden")
    primary_inventory = ANALYSIS / f"{PRIMARY}.execution.json"
    primary = json.loads(primary_inventory.read_text(encoding="utf-8"))
    if any(sha256_file(REPO / p) != sha for p, sha in primary["files"].items()):
        raise ValueError("primary accepted source changed")
    if gate["execution_inventory"] != str(primary_inventory.relative_to(REPO)):
        raise ValueError("primary gate inventory mismatch")
    snapshot = frozen_identity(
        "recursive-summary-cost", REPO / "results/manifests/dev100.json", "dev100"
    )
    train = frozen_identity(
        "recursive-summary-pilot", REPO / "results/manifests/train150.json", "train150"
    )
    snapshot["files"].update(train["files"])
    extra = [
        Path(__file__),
        REPO / "tools/recursive_summary_nav.py",
        REPO / "tools/recursive_summary_report.py",
        REPO / "tools/recursive_summary_preflight.py",
        ANALYSIS / f"{STEM}.preflight-r2.json",
        REPO / "tools/run_paged_paired.py",
        REPO / "tools/run_grounded_paired.py",
        REPO / "tools/judge_request_cache.py",
        REPO / "tools/paired_manifest_scope.py",
        REPO / "tools/grounded_reader_diagnostic.py",
        REPO / "tools/grade_saved_recovering.py",
        gate_path,
        primary_inventory,
        REPO / "results/prereg-recursive-summary-cost-v1.md",
    ]
    snapshot["files"].update({str(p.relative_to(REPO)): sha256_file(p) for p in extra})
    preflight = json.loads((ANALYSIS / f"{STEM}.preflight-r2.json").read_text(encoding="utf-8"))
    if preflight["by_split"]["dev100"]["questions"] != 100:
        raise ValueError("summary preflight did not select dev100")
    for p, sha in preflight["inputs"].items():
        if sha256_file(REPO / p) != sha:
            raise ValueError("summary preflight input changed")
    snapshot.update(
        {
            "sample_count": 100,
            "pilot_ids": PILOT,
            "pilot_faithful_gaps": sorted(GAPS),
            "fresh_dev_answers": 600,
            "repeats_per_arm": 3,
            "control_prompt": "memory-grounded-v18",
            "candidate_prompt": SummaryNavigationAnswerer.prompt_version,
            "client_attempts": 2,
            "transport_attempts": 2,
            "node_invocations": 2,
            "judge_recoveries": 3,
            "judge_wait_seconds": 180,
            "timeout_seconds": 180,
            "max_quota_reset_waits": 21,
            "preflight": preflight,
        }
    )
    inventory_path = ANALYSIS / f"{STEM}.execution.json"
    source_path = ANALYSIS / f"{STEM}.source-snapshot.json"
    # JSON tuples become lists, so compare the serialized canonical object.
    snapshot = json.loads(json.dumps(snapshot))
    if inventory_path.exists():
        existing = json.loads(inventory_path.read_text(encoding="utf-8"))
        snapshot["source_snapshot_sha256"] = sha256_file(source_path)
        if existing != snapshot:
            raise ValueError("summary frozen execution changed; new namespace required")
    else:
        save_json(
            source_path,
            {
                p: {"sha256": sha, "text": (REPO / p).read_text(encoding="utf-8")}
                for p, sha in snapshot["files"].items()
                if Path(p).suffix in {".py", ".yaml", ".md"}
            },
        )
        snapshot["source_snapshot_sha256"] = sha256_file(source_path)
        save_json(inventory_path, snapshot)
    return snapshot, sha256_file(inventory_path)


def build_runner(split):
    from llm_long_term_memory import cli

    cfg, _, runner, judge, usage = cli._build(
        VARIANT, "configs/fallback.yaml", split, read_only_store=True
    )
    runner.client.max_retries = runner.client.max_transport_retries = 2
    return cfg, runner, judge, usage


def tree_manifest(split, identity=None):
    path = ROOT / split / "trees.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if identity is not None and manifest["inventory_sha256"] != identity:
        raise ValueError("tree construction execution identity changed")
    for user, sha in manifest["hashes"].items():
        if sha256_file(ROOT / split / (digest(user) + ".tree.json")) != sha:
            raise ValueError("frozen tree manifest changed")
    return manifest


def use_summary(runner, split, identity):
    g = runner.grounded
    runner.grounded = SummaryNavigationAnswerer(
        g.client,
        model=g.model,
        encoder=g.encoder,
        store=g.store,
        turn_index=g.turn_index,
        fallback=g.fallback,
        chars_per_token=g.chars_per_token,
        max_output_tokens=g.max_output_tokens,
        tree_dir=ROOT / split,
        tree_hashes=tree_manifest(split, identity)["hashes"],
    )
    runner.answer_prompt_version = runner.grounded.prompt_version


def paths_for(split, arm, rep):
    base = REPO / f"results/raw/{STEM}-{split}-{arm}-rep{rep}"
    return base, *(
        base.with_suffix("." + suffix) for suffix in ("readers.jsonl", "grades.jsonl", "usage.json")
    )


def state(value):
    save_json(ANALYSIS / f"{STEM}.progress.json", value)


def build_summaries(split, instances, identity, updates=False):
    from llm_long_term_memory.llm.usage import UsageTracker
    from llm_long_term_memory.runtime.grounded_answering import (
        EvidenceSource,
        personal_archive_review,
    )

    cfg, runner, _, usage = build_runner(split)
    destination = ROOT / split / ("updates" if updates else "")
    destination.mkdir(parents=True, exist_ok=True)
    hashes, selected = {}, [i for i in instances if personal_archive_review(i.question)]
    originals = tree_manifest(split, identity)["hashes"] if updates else None
    try:
        for index, instance in enumerate(selected):
            user = instance.store_namespace
            path = destination / (digest(user) + ".tree.json")
            up = destination / (digest(user) + ".build-usage.json")
            usage.records[:] = UsageTracker._load_records(up) if up.exists() else []
            inherited, extra = None, None
            if updates:
                original_path = ROOT / split / (digest(user) + ".tree.json")
                if sha256_file(original_path) != originals[user]:
                    raise ValueError("update changed original tree")
                inherited = json.loads(original_path.read_text(encoding="utf-8"))
                extra = EvidenceSource(
                    "E999999",
                    "raw",
                    "synthetic-navigation-cost-update",
                    "My project name changed from CostStudyA to CostStudyB today.",
                    "user",
                    "2026-10-04",
                    session_id="synthetic-cost-only",
                    turn_index=0,
                )
            tree = build_tree(
                runner.store,
                user,
                runner.client,
                cfg.models.extractor,
                path,
                usage,
                up,
                inherited=inherited,
                extra_source=extra,
                build_identity=identity,
            )
            validate_tree(tree, user)
            hashes[user] = sha256_file(path)
            state(
                {
                    "status": "RUNNING",
                    "stage": "updates" if updates else "build",
                    "split": split,
                    "trees": index + 1,
                    "total": len(selected),
                    "inventory_sha256": identity,
                }
            )
            print(
                f"{split}: {'update' if updates else 'tree'} {index + 1}/{len(selected)}",
                flush=True,
            )
        manifest = {"inventory_sha256": identity, "hashes": hashes, "cost_only": updates}
        out = destination / "trees.json"
        if out.exists() and json.loads(out.read_text(encoding="utf-8")) != manifest:
            raise ValueError("tree manifest identity changed")
        if not out.exists():
            save_json(out, manifest)
    finally:
        runner.store.close()


def seed_own_cache(cache, instances, split, identity):
    from run_paged_paired import seed_cache

    if split == "dev100":
        primary_path = ANALYSIS / f"{PRIMARY}.execution.json"
        seed_cache(
            cache,
            instances,
            json.loads(primary_path.read_text(encoding="utf-8")),
            sha256_file(primary_path),
        )
    for rep in range(1, 4):
        for arm in ("control", "candidate"):
            _, rp, gp, _ = paths_for(split, arm, rep)
            if not rp.exists() or not gp.exists():
                continue
            readers, grades = rows_by_question(rp), rows_by_question(gp)
            if not set(grades) <= set(readers) <= set(instances):
                raise ValueError("foreign grading cache rows")
            for q, grade in grades.items():
                combine(instances[q], readers[q], grade, identity, arm)
                if not (grade["verdict"].get("details") or {}).get("judge_cache"):
                    cache.add(instances[q], readers[q], grade, rp, gp)


def qa(split, instances, identity, stage):
    from llm_long_term_memory.evaluation.harness import _exclusive
    from llm_long_term_memory.llm.client import DailyQuotaExhausted
    from llm_long_term_memory.llm.rate_limiter import Wait
    from llm_long_term_memory.llm.usage import UsageTracker
    from llm_long_term_memory.runtime.grounded_answering import personal_archive_review

    selected = {i.question_id: i for i in instances}
    iterations = (
        [(1, "candidate")]
        if split == "train150"
        else [(rep, arm) for rep in (1, 2, 3) for arm in ("control", "candidate")]
    )
    for rep, arm in iterations:
        base, rp, gp, up = paths_for(split, arm, rep)
        with _exclusive(rp):
            readers, grades = [rows_by_question(p) if p.exists() else {} for p in (rp, gp)]
            if not set(grades) <= set(readers) <= set(selected):
                raise ValueError("foreign or orphan rows")
            if (readers or grades) and not up.exists():
                raise ValueError("missing usage checkpoint")
            _cfg, runner, judge, usage = build_runner(split)
            try:
                if up.exists():
                    usage.records[:] = UsageTracker._load_records(up)
                if arm == "candidate":
                    use_summary(runner, split, identity)
                for q, reader in readers.items():
                    if reader["inventory_sha256"] != identity or reader["cohort"] != arm:
                        raise ValueError("saved reader identity changed")
                    if reader["answer_prompt_version"] != runner.answer_prompt_version:
                        raise ValueError("saved reader prompt changed")
                    if q in grades:
                        combine(selected[q], reader, grades[q], identity, arm)
                if stage == "grades" and len(readers) != len(selected):
                    state(
                        {
                            "status": "WAIT_FOR_READERS",
                            "split": split,
                            "arm": arm,
                            "repeat": rep,
                            "readers": len(readers),
                        }
                    )
                    return 2
                cache = ExactRequestJudge(judge)
                if stage == "grades":
                    seed_own_cache(cache, selected, split, identity)
                with (rp if stage == "readers" else gp).open("a", encoding="utf-8") as sink:
                    for i in instances:
                        q = i.question_id
                        if stage == "readers":
                            if q in readers:
                                continue
                            needed = (
                                42
                                if arm == "candidate" and personal_archive_review(i.question)
                                else 3 * (expected_calls(runner.store, i, arm == "control") // 2)
                            )
                            if runner.client.quota.for_model(runner.model).remaining_today < needed:
                                raise DailyQuotaExhausted(runner.model, Wait(0, "rpd"))
                            readers[q] = save_reader(
                                TimedRunner(runner, identity, arm, usage), i, sink, identity, arm
                            )
                        else:
                            if q in grades:
                                continue
                            scoped = ScopedRecoveringCache(
                                cache,
                                usage,
                                up,
                                base.with_suffix(f".{q}.recovery.json"),
                                hashlib.sha256(
                                    f"{identity}:{split}:{arm}:{rep}:{q}".encode()
                                ).hexdigest(),
                            )
                            grades[q] = grade_saved(scoped, i, readers[q], sink, identity)
                        usage.save(up)
                        completed = len(readers) if stage == "readers" else len(grades)
                        print(
                            f"{split} {arm} rep{rep}: {stage} {completed}/{len(selected)}",
                            flush=True,
                        )
                state(
                    {
                        "status": "RUNNING",
                        "stage": stage,
                        "split": split,
                        "arm": arm,
                        "repeat": rep,
                        "readers": len(readers),
                        "grades": len(grades),
                    }
                )
            finally:
                usage.save(up)
                runner.store.close()
    return 0


def report_pilot(instances, identity):
    from llm_long_term_memory.llm.usage import UsageTracker

    _base, rp, gp, up = paths_for("train150", "candidate", 1)
    readers, grades = [rows_by_question(p) if p.exists() else {} for p in (rp, gp)]
    for q in grades:
        combine(
            next(i for i in instances if i.question_id == q),
            readers[q],
            grades[q],
            identity,
            "candidate",
        )
    result = pilot_gate({i.question_id: i for i in instances}, readers, grades, GAPS)
    result["inventory_sha256"] = identity
    result["inputs"] = {
        str(p.relative_to(REPO)): sha256_file(p) for p in (rp, gp, up) if p.exists()
    }
    records = UsageTracker._load_records(up) if up.exists() else []
    for p in (ROOT / "train150").glob("*.build-usage.json"):
        records.extend(UsageTracker._load_records(p))
        result["inputs"][str(p.relative_to(REPO))] = sha256_file(p)
    for p in (ROOT / "train150").glob("*.tree.json"):
        result["inputs"][str(p.relative_to(REPO))] = sha256_file(p)
    result["pilot_resources_by_model"] = model_workload(records)
    out = ANALYSIS / f"{STEM}.pilot-gate.json"
    if out.exists() and json.loads(out.read_text(encoding="utf-8")) != result:
        raise ValueError("pilot gate changed")
    if not out.exists():
        save_json(out, result)
    return result


def require_pilot(identity):
    result = json.loads((ANALYSIS / f"{STEM}.pilot-gate.json").read_text(encoding="utf-8"))
    if not result["pass"] or result["inventory_sha256"] != identity:
        raise ValueError("pilot failed; dev summary API calls forbidden")
    for p, sha in result["inputs"].items():
        if sha256_file(REPO / p) != sha:
            raise ValueError("pilot acceptance evidence changed")


def report_dev(instances, identity):
    from llm_long_term_memory.llm.usage import UsageTracker
    from llm_long_term_memory.runtime.grounded_answering import personal_archive_review

    runs, latency, workloads, query_resources, coverage, inputs = {}, {}, {}, {}, {}, {}
    all_usage, coverage_pairs = [], {}
    for arm in ("control", "candidate"):
        coverage_pairs[arm] = {}
        runs[arm], arm_readers, records, per_user, covered = [], {}, [], {}, []
        for rep in (1, 2, 3):
            _, rp, gp, up = paths_for("dev100", arm, rep)
            readers, grades = rows_by_question(rp), rows_by_question(gp)
            if set(readers) != set(grades) or set(readers) != {i.question_id for i in instances}:
                raise ValueError("all 600 fresh answers and grades required")
            runs[arm].append(
                {
                    i.question_id: {
                        "correct": combine(
                            i, readers[i.question_id], grades[i.question_id], identity, arm
                        ).correct
                    }
                    for i in instances
                }
            )
            for i in instances:
                r = readers[i.question_id]
                arm_readers[f"{rep}:{i.question_id}"] = r
                per_user.setdefault(i.store_namespace, []).append(
                    resource_total(r["answer"]["notes"]["whole_question_usage"]["by_model"])
                )
                present = raw_reference_coverage(i, r)
                covered.append(present)
                coverage_pairs[arm][(rep, i.question_id)] = present
            records.extend(UsageTracker._load_records(up))
            inputs.update({str(p.relative_to(REPO)): sha256_file(p) for p in (rp, gp, up)})
        latency[arm] = latency_summary(arm_readers)
        workloads[arm] = {
            "answering": model_workload(records, "answerer"),
            "grading": model_workload(records, "judge"),
        }
        query_resources[arm] = {u: sum(v) / len(v) for u, v in per_user.items()}
        coverage[arm] = {"covered_question_runs": sum(covered), "question_runs": len(covered)}
        all_usage.extend(records)
    users = [i.store_namespace for i in instances if personal_archive_review(i.question)]
    build, updates, local, disk = {}, {}, {}, {}
    for label, directory, costs in [
        ("construction", ROOT / "dev100", build),
        ("update", ROOT / "dev100/updates", updates),
    ]:
        records = []
        local[label], disk[label] = 0, 0
        for user in users:
            path = directory / (digest(user) + ".tree.json")
            up = directory / (digest(user) + ".build-usage.json")
            tree = json.loads(path.read_text(encoding="utf-8"))
            usage = UsageTracker._load_records(up)
            records.extend(usage)
            costs[user] = resource_total(model_workload(usage))
            local[label] += tree["active_build_ms"]
            disk[label] += path.stat().st_size
            inputs.update({str(p.relative_to(REPO)): sha256_file(p) for p in (path, up)})
        workloads[label] = model_workload(records)
        all_usage.extend(records)
    accuracy = paired_accuracy(runs, {i.question_id: i.question_type for i in instances})
    lost_sources = sum(
        present and not coverage_pairs["candidate"][key]
        for key, present in coverage_pairs["control"].items()
    )
    source_no_loss = lost_sources == 0
    coverage["paired_reference_losses"] = lost_sources
    pilot_records = []
    for p in (ROOT / "train150").glob("*.build-usage.json"):
        pilot_records.extend(UsageTracker._load_records(p))
        inputs[str(p.relative_to(REPO))] = sha256_file(p)
    _, _, _, pilot_usage = paths_for("train150", "candidate", 1)
    pilot_records.extend(UsageTracker._load_records(pilot_usage))
    workloads["pilot_research"] = model_workload(pilot_records)
    result = {
        "experiment_class": "exposed-dev100-regression-not-unseen-final",
        "inventory_sha256": identity,
        "accuracy": accuracy,
        "source_coverage": coverage,
        "strict_no_loss_supported": accuracy["statistical_no_loss_supported"] and source_no_loss,
        "resources_by_model": workloads,
        "end_to_end_latency": latency,
        "summary_build_active_ms": local,
        "extra_tree_disk_bytes": disk,
        "amortization": amortization(
            build,
            updates,
            {u: query_resources["control"][u] for u in users},
            {u: query_resources["candidate"][u] for u in users},
        ),
        "new_dev_and_update_workload": model_workload(all_usage),
        "shared_historical_fact_ingestion_cost": "not rebuilt; unknown, not zero",
        "financial_cost": "unavailable; tokens across models are resource workload only",
        "failed_request_token_cost": "provider may not report failed tokens; unknown, not free",
        "pilot_research_cost_separate": True,
        "inputs": inputs,
        "default_adoption": False,
    }
    out = ANALYSIS / f"{STEM}.report.json"
    if out.exists() and json.loads(out.read_text(encoding="utf-8")) != result:
        raise ValueError("summary final report changed")
    if not out.exists():
        save_json(out, result)
    return result


def main():
    from llm_long_term_memory.evaluation.recovery import RecoveryExhausted
    from llm_long_term_memory.llm.client import DailyQuotaExhausted

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "pilot-build",
            "pilot-readers",
            "pilot-grades",
            "pilot-report",
            "dev-build",
            "dev-readers",
            "dev-grades",
            "updates",
            "report",
        ),
        default="all",
    )
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "provider_calls": 0,
                    "status": "CONDITIONAL_ON_PRIMARY_PASS",
                    "pilot_questions": len(PILOT),
                    "fresh_dev_answers": 600,
                    "max_candidate_calls_per_query_before_client_retries": 14,
                    "models_unchanged": True,
                    "prereg": "results/prereg-recursive-summary-cost-v1.md",
                }
            )
        )
        return 0
    snapshot, identity = freeze()
    train, dev = instances_for("train150"), instances_for("dev100")
    stages = (
        (
            "pilot-build",
            "pilot-readers",
            "pilot-grades",
            "pilot-report",
            "dev-build",
            "dev-readers",
            "dev-grades",
            "updates",
            "report",
        )
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        if any(sha256_file(REPO / p) != sha for p, sha in snapshot["files"].items()):
            raise ValueError("frozen summary source changed")
        try:
            if stage.startswith("dev") or stage in ("updates", "report"):
                require_pilot(identity)
            if stage in ("pilot-build", "dev-build", "updates"):
                split = "train150" if stage == "pilot-build" else "dev100"
                build_summaries(
                    split,
                    train if split == "train150" else dev,
                    identity,
                    updates=stage == "updates",
                )
            elif stage.endswith(("readers", "grades")):
                split = "train150" if stage.startswith("pilot") else "dev100"
                rc = qa(split, train if split == "train150" else dev, identity, stage.split("-")[1])
                if rc:
                    return rc
            elif stage == "pilot-report":
                result = report_pilot(train, identity)
                if not result["pass"]:
                    state({"status": "PILOT_FAILED", "gate": result})
                    print(json.dumps({"status": "PILOT_FAILED", "pass": False}), flush=True)
                    return 1
            else:
                result = report_dev(dev, identity)
                state(
                    {
                        "status": "COMPLETE",
                        "strict_no_loss_supported": result["strict_no_loss_supported"],
                    }
                )
                print(
                    json.dumps(
                        {
                            "status": "COMPLETE",
                            "accuracy": result["accuracy"],
                            "strict_no_loss_supported": result["strict_no_loss_supported"],
                        }
                    ),
                    flush=True,
                )
        except DailyQuotaExhausted as exc:
            state({"status": "QUOTA_WAIT", "stage": stage, "model": exc.model, "reason": str(exc)})
            print(
                json.dumps({"status": "QUOTA_WAIT", "stage": stage, "model": exc.model}), flush=True
            )
            return 2
        except RecoveryExhausted as exc:
            state({"status": "RECOVERY_EXHAUSTED", "stage": stage, "reason": str(exc)})
            print(json.dumps({"status": "RECOVERY_EXHAUSTED", "reason": str(exc)}), flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
