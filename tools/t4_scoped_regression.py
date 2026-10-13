"""Fresh t4 versus scoped-selection repair, three exposed dev100 repeats per arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from grounded_reader_diagnostic import grade_saved, save_reader  # noqa: E402
from recursive_summary_report import model_workload, paired_accuracy  # noqa: E402
from run_grounded_experiment import frozen_identity  # noqa: E402
from run_grounded_paired import combine  # noqa: E402
from run_paged_paired import save_json  # noqa: E402
from run_recursive_summary_cost import instances_for  # noqa: E402
from t4_scoped_selection_acceptance import AuditedRunner, CapturingClient  # noqa: E402
from t4_selection_preflight import OfflineClient  # noqa: E402
from t4_selection_repair import VERSION, SelectionRepairRunner  # noqa: E402

STEM = "t4-scoped-regression-dev100-v1"
ARMS = ("control", "candidate")
REPS = (1, 2, 3)
VARIANT = "two_stage_raw_primary_t4"
ANALYSIS = REPO / "results/analysis"
PREREQ = ANALYSIS / "t4-scoped-selection-train21-v2.gate.json"
PREFLIGHT = ANALYSIS / f"{STEM}.preflight-r2.json"


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"immutable artifact changed: {path}")
    else:
        save_json(path, value)


def acceptance_check():
    if not PREREQ.exists() or not json.loads(PREREQ.read_text(encoding="utf-8"))["pass"]:
        raise ValueError("passing fixed21 acceptance required before regression calls")
    execution = ANALYSIS / "t4-scoped-selection-train21-v2.execution.json"
    gate = json.loads(PREREQ.read_text(encoding="utf-8"))
    if gate["inventory_sha256"] != sha256_file(execution):
        raise ValueError("acceptance inventory binding changed")
    snapshot = json.loads(execution.read_text(encoding="utf-8"))
    for rel, digest in snapshot["files"].items():
        if sha256_file(REPO / rel) != digest:
            raise ValueError(f"accepted input changed: {rel}")
    for rel, digest in gate["inputs"].items():
        if sha256_file(REPO / rel) != digest:
            raise ValueError(f"acceptance reader/grade changed: {rel}")


def preflight():
    from llm_long_term_memory import cli
    from llm_long_term_memory.conversation import AnswerRequest
    from llm_long_term_memory.store import external_session_id

    _, _, runner, _, _ = cli._build(
        VARIANT,
        "configs/fallback.yaml",
        "dev100",
        client_override=CapturingClient(OfflineClient()),
        read_only_store=True,
    )
    candidate = SelectionRepairRunner.from_t4(runner)
    contexts = {a: [] for a in ARMS}
    turns_checked = 0
    try:
        for instance in instances_for("dev100"):
            # Operational source audit only. No correctness/first-loss diagnosis.
            originals = {s.session_id: s for s in instance.sessions}
            seen = set()
            for session_id in runner.store.session_ids_for_user(instance.store_namespace):
                session = runner.store.get_session(session_id)
                if session is None or session.user_id != instance.store_namespace:
                    raise ValueError("stored session ownership mismatch")
                sid = external_session_id(session.id)
                if sid not in originals:
                    raise ValueError("stored session absent from public source")
                actual = runner.store.turns_for_session(session.id)
                expected = originals[sid].turns
                if len(actual) != len(expected) or any(
                    t.role != e.role or t.content != e.content or t.turn_index != n
                    for n, (t, e) in enumerate(zip(actual, expected, strict=True))
                ):
                    raise ValueError("public raw source mismatch")
                seen.add(sid)
                turns_checked += len(actual)
            if seen != set(originals):
                raise ValueError("missing original sessions")
            for memory in runner.store.iter_all(instance.store_namespace):
                if memory.user_id != instance.store_namespace:
                    raise ValueError("memory owner mismatch")
                if memory.source_session_id:
                    origin = runner.store.get_session(memory.source_session_id)
                    if origin is None or origin.user_id != instance.store_namespace:
                        raise ValueError("memory provenance owner mismatch")
            request = AnswerRequest(
                instance.question, instance.question_date, instance.store_namespace
            )
            for arm, system in (("control", runner), ("candidate", candidate)):
                contexts[arm].append(system.answer_request(request).context_tokens)
    finally:
        runner.store.close()
    result = {
        "provider_calls": 0,
        "experiment_class": "exposed-dev100-offline-operational-preflight",
        "questions": 100,
        "raw_turns_checked": turns_checked,
        "median_context": {a: statistics.median(v) for a, v in contexts.items()},
        "max_context": {a: max(v) for a, v in contexts.items()},
        "estimated_reader_input_tokens_six_runs": 3 * sum(sum(v) for v in contexts.values()),
        "inputs": {
            str(p.relative_to(REPO)): sha256_file(p)
            for p in (
                Path(__file__),
                REPO / "tools/t4_selection_repair.py",
                REPO / "stores/dev100.db",
                REPO / "results/manifests/dev100.json",
            )
        },
        "limitations": "Stub composition only; no accuracy claim or dev-item outcome tuning.",
    }
    immutable_json(PREFLIGHT, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("provider_calls", "questions", "raw_turns_checked", "median_context")
            }
        )
    )


def evaluate(runs):
    instances = instances_for("dev100")
    types = {i.question_id: i.question_type for i in instances}
    if len(types) != 100:
        raise ValueError("exact registered 100-question manifest required")
    result = paired_accuracy(runs, types)
    median = statistics.median(
        r["context_tokens"] for rep in runs["candidate"] for r in rep.values()
    )
    rules = {
        "mean_net_nonnegative": result["mean_net"] >= 0,
        "no_type_loss_below_minus2": all(d >= -2 for d in result["by_type_mean_net"].values()),
        "median_context_at_most6500": median <= 6500,
    }
    return {
        "experiment_class": "exposed-dev100-development-regression",
        "accuracy": result,
        "median_candidate_context": median,
        "rules": rules,
        "pass": all(rules.values()),
        "limitations": "Operational regression gate, not a statistical proof of no accuracy loss. "
        "Any summary experiment requires its own fresh paired cost/accuracy measurements.",
    }


def freeze():
    acceptance_check()
    proof = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    for rel, digest in proof["inputs"].items():
        if sha256_file(REPO / rel) != digest:
            raise ValueError(f"operational preflight input changed: {rel}")
    if proof["provider_calls"] != 0 or proof["questions"] != 100:
        raise ValueError("invalid operational preflight")
    snapshot = frozen_identity(STEM, REPO / "results/manifests/dev100.json", "dev100")
    extra = [
        Path(__file__),
        PREREQ,
        REPO / "tools/t4_selection_repair.py",
        REPO / "tools/t4_selection_preflight.py",
        REPO / "tools/t4_scoped_selection_acceptance.py",
        REPO / "tools/t4_selection_acceptance.py",
        REPO / "tools/recursive_summary_report.py",
        REPO / "tools/run_recursive_summary_cost.py",
        REPO / "tools/run_grounded_paired.py",
        REPO / "tools/run_paged_paired.py",
        REPO / "tools/grounded_reader_diagnostic.py",
        REPO / f"results/prereg-{STEM}.md",
        PREFLIGHT,
        ANALYSIS / f"{STEM}.resolution.json",
    ]
    snapshot["files"].update({str(p.relative_to(REPO)): sha256_file(p) for p in extra})
    snapshot.update(
        {
            "question_ids": [i.question_id for i in instances_for("dev100")],
            "arms": {"control": "memory-aware-v2-clt", "candidate": VERSION},
            "repeats": 3,
            "client_retries": 2,
            "transport_retries": 2,
            "judge_recoveries": 3,
            "judge_cooldown_cap": 180,
        }
    )
    inventory = ANALYSIS / f"{STEM}.execution.json"
    source_path = ANALYSIS / f"{STEM}.source-snapshot.json"
    if not inventory.exists():
        immutable_json(
            source_path,
            {
                rel: {"sha256": digest, "text": (REPO / rel).read_text(encoding="utf-8")}
                for rel, digest in snapshot["files"].items()
                if Path(rel).suffix in {".py", ".md", ".yaml"}
            },
        )
    snapshot["source_snapshot_sha256"] = sha256_file(source_path)
    immutable_json(inventory, snapshot)
    return snapshot, sha256_file(inventory)


def main():
    from llm_long_term_memory import cli
    from llm_long_term_memory.evaluation.harness import _exclusive
    from llm_long_term_memory.evaluation.recovery import RecoveringJudge, RecoveryExhausted
    from llm_long_term_memory.llm.client import DailyQuotaExhausted
    from llm_long_term_memory.llm.rate_limiter import Wait
    from llm_long_term_memory.llm.usage import UsageTracker

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
        return 0
    if not args.execute:
        print(
            json.dumps(
                {
                    "provider_calls": 0,
                    "questions": 100,
                    "repeats_per_arm": 3,
                    "fresh_readers": 600,
                    "fresh_grades": 600,
                    "summary_calls": 0,
                }
            )
        )
        return 0
    snapshot, identity = freeze()
    instances = instances_for("dev100")
    ids = {i.question_id for i in instances}
    runs = {a: [] for a in ARMS}
    workload = []
    state = ANALYSIS / f"{STEM}.progress.json"
    for rep in REPS:
        for arm in ARMS:
            if any(sha256_file(REPO / rel) != h for rel, h in snapshot["files"].items()):
                raise ValueError("frozen regression inputs changed")
            base = REPO / f"results/raw/{STEM}-{arm}-rep{rep}"
            rp, gp, up, cp = [
                base.with_suffix("." + suffix)
                for suffix in ("readers.jsonl", "grades.jsonl", "usage.json", "jsonl")
            ]
            with _exclusive(rp):
                readers, grades = [rows_by_question(p) if p.exists() else {} for p in (rp, gp)]
                if not set(grades) <= set(readers) <= ids or (
                    (readers or grades) and not up.exists()
                ):
                    raise ValueError("foreign/orphan rows or missing saved usage")
                for row in readers.values():
                    if (
                        row["inventory_sha256"] != identity
                        or row["cohort"] != arm
                        or row["answer_prompt_version"] != snapshot["arms"][arm]
                    ):
                        raise ValueError("reader identity/prompt changed")
                _, _, runner, judge, usage = cli._build(
                    VARIANT, "configs/fallback.yaml", "dev100", read_only_store=True
                )
                if arm == "candidate":
                    runner = SelectionRepairRunner.from_t4(runner)
                runner.client.max_retries = runner.client.max_transport_retries = 2
                runner.client = CapturingClient(runner.client)
                if up.exists():
                    usage.records[:] = UsageTracker._load_records(up)
                try:
                    with rp.open("a", encoding="utf-8") as sink:
                        for i in instances:
                            if i.question_id in readers:
                                continue
                            if runner.client.quota.for_model(runner.model).remaining_today < 6:
                                raise DailyQuotaExhausted(runner.model, Wait(0, "rpd"))
                            readers[i.question_id] = save_reader(
                                AuditedRunner(runner, identity, arm, usage), i, sink, identity, arm
                            )
                            usage.save(up)
                            save_json(
                                state,
                                {
                                    "status": "RUNNING",
                                    "arm": arm,
                                    "rep": rep,
                                    "readers": len(readers),
                                    "grades": len(grades),
                                },
                            )
                            print(f"{arm} rep{rep} readers {len(readers)}/100", flush=True)
                    # All answers in a block precede its grades; no dev item outcomes
                    # are inspected to change the frozen configuration.
                    with gp.open("a", encoding="utf-8") as sink:
                        for i in instances:
                            if i.question_id in grades:
                                combine(
                                    i, readers[i.question_id], grades[i.question_id], identity, arm
                                )
                                continue
                            scoped = RecoveringJudge(
                                judge,
                                usage,
                                up,
                                base.with_suffix(f".{i.question_id}.recovery.json"),
                                hashlib.sha256(
                                    f"{identity}:{arm}:{rep}:{i.question_id}".encode()
                                ).hexdigest(),
                                max_recoveries=3,
                                max_wait_seconds=180,
                            )
                            grades[i.question_id] = grade_saved(
                                scoped, i, readers[i.question_id], sink, identity
                            )
                            usage.save(up)
                            save_json(
                                state,
                                {
                                    "status": "RUNNING",
                                    "arm": arm,
                                    "rep": rep,
                                    "readers": len(readers),
                                    "grades": len(grades),
                                },
                            )
                            print(f"{arm} rep{rep} grades {len(grades)}/100", flush=True)
                    joined = [
                        asdict(
                            combine(i, readers[i.question_id], grades[i.question_id], identity, arm)
                        )
                        for i in instances
                    ]
                    if cp.exists():
                        if list(rows_by_question(cp).values()) != joined:
                            raise ValueError("combined rows changed")
                    else:
                        cp.write_text(
                            "".join(json.dumps(r) + "\n" for r in joined), encoding="utf-8"
                        )
                    runs[arm].append({r["question_id"]: r for r in joined})
                    workload.extend(usage.records)
                except (DailyQuotaExhausted, RecoveryExhausted) as exc:
                    save_json(
                        state,
                        {
                            "status": "QUOTA_WAIT"
                            if isinstance(exc, DailyQuotaExhausted)
                            else "RECOVERY_EXHAUSTED",
                            "reason": str(exc),
                            "arm": arm,
                            "rep": rep,
                            "readers": len(readers),
                            "grades": len(grades),
                        },
                    )
                    print(json.dumps({"status": "INCOMPLETE", "reason": str(exc)}), flush=True)
                    return 2
                finally:
                    usage.save(up)
                    runner.store.close()
    result = evaluate(runs)
    result.update({"inventory_sha256": identity, "resources_by_model": model_workload(workload)})
    immutable_json(ANALYSIS / f"{STEM}.gate.json", result)
    save_json(state, {"status": "COMPLETE", "pass": result["pass"]})
    print(json.dumps(result), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
