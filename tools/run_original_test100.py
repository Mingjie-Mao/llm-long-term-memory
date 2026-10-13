"""Fresh same-question v2/t4 comparison on the original exposed test100."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from grounded_reader_diagnostic import grade_saved, save_reader  # noqa: E402
from recursive_summary_report import model_workload, paired_accuracy  # noqa: E402
from run_grounded_paired import combine  # noqa: E402
from run_paged_paired import save_json  # noqa: E402
from t4_savings_regression import immutable_json  # noqa: E402
from t4_savings_repair import VERSION, SavingsRepairRunner  # noqa: E402
from t4_scoped_selection_acceptance import AuditedRunner, CapturingClient  # noqa: E402
from t4_selection_preflight import OfflineClient  # noqa: E402

STEM = "original-test100-v2-t4-regression-v1"
ANALYSIS = REPO / "results/analysis"
INVENTORY = ANALYSIS / f"{STEM}.execution.json"
STATE = ANALYSIS / f"{STEM}.progress.json"
PREREQ = ANALYSIS / "t4-scoped-regression-dev100-v3.gate.json"


def instances():
    from llm_long_term_memory.commands.common import manifest_instances
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.manifest import load_manifest, require_claim

    manifest = load_manifest(REPO / "results/manifests/test100.json")
    require_claim(manifest, "regression")
    loaded = {i.question_id: i for i in manifest_instances(manifest, Settings())}
    result = [loaded[q] for q in manifest.question_ids]
    if len(result) != 100 or len({i.question_id for i in result}) != 100:
        raise ValueError("exact original test100 required")
    return result


class TransportAwareJudge:
    def __init__(self, judge):
        self.judge, self.prompt_version = judge, judge.prompt_version

    def grade(self, **kwargs):
        try:
            return self.judge.grade(**kwargs)
        except httpx.TransportError as exc:
            # Only changes bounded recovery classification; request and verdict
            # are the same. Actual transport attempts are already in usage.
            raise ConnectionError(str(exc)) from exc


def candidate_preflight():
    from llm_long_term_memory import cli
    from llm_long_term_memory.conversation import AnswerRequest
    from llm_long_term_memory.evaluation.reproducibility import logical_sqlite_fingerprint
    from llm_long_term_memory.store import external_session_id

    _, _, runner, _, _ = cli._build(
        "two_stage_raw_primary_t4",
        "configs/fallback.yaml",
        "test100",
        client_override=CapturingClient(OfflineClient()),
        read_only_store=True,
    )
    candidate = SavingsRepairRunner.from_t4(runner)
    contexts, turns = [], 0
    try:
        for i in instances():
            originals = {s.session_id: s for s in i.sessions}
            seen = set()
            for sid in runner.store.session_ids_for_user(i.store_namespace):
                session = runner.store.get_session(sid)
                if session is None or session.user_id != i.store_namespace:
                    raise ValueError("source ownership mismatch")
                external = external_session_id(sid)
                expected, actual = originals[external].turns, runner.store.turns_for_session(sid)
                if len(actual) != len(expected) or any(
                    t.role != e.role or t.content != e.content or t.turn_index != n
                    for n, (t, e) in enumerate(zip(actual, expected, strict=True))
                ):
                    raise ValueError("stored original differs from public benchmark")
                seen.add(external)
                turns += len(actual)
            if seen != set(originals):
                raise ValueError("missing original sessions")
            for memory in runner.store.iter_all(i.store_namespace):
                if memory.user_id != i.store_namespace:
                    raise ValueError("memory owner mismatch")
                if memory.source_session_id:
                    owner = runner.store.get_session(memory.source_session_id)
                    if owner is None or owner.user_id != i.store_namespace:
                        raise ValueError("memory provenance mismatch")
            contexts.append(
                candidate.answer_request(
                    AnswerRequest(i.question, i.question_date, i.store_namespace)
                ).context_tokens
            )
    finally:
        runner.store.close()
    historical = json.loads(
        (REPO / "results/frozen/v2-final/freeze.json").read_text(encoding="utf-8")
    )["system"]
    current = logical_sqlite_fingerprint(REPO / "stores/test100.db")
    if current["logical_sha256"] != historical["store"]["database"]["logical_sha256"]:
        raise ValueError("original memory store changed from historical v2")
    result = {
        "provider_calls": 0,
        "questions": 100,
        "raw_turns_checked": turns,
        "context_tokens": contexts,
        "store_logical_sha256": current["logical_sha256"],
        "worker_sha256": sha256_file(Path(__file__)),
        "candidate_sha256": sha256_file(REPO / "tools/t4_savings_repair.py"),
    }
    immutable_json(ANALYSIS / f"{STEM}.candidate-preflight-r2.json", result)
    print(
        json.dumps(
            {
                "provider_calls": 0,
                "questions": 100,
                "raw_turns_checked": turns,
                "median_context": statistics.median(contexts),
            }
        )
    )


def freeze():
    import platform

    if not PREREQ.exists() or not json.loads(PREREQ.read_text(encoding="utf-8"))["pass"]:
        raise ValueError("complete passing dev100 repair regression required")
    parent_path = ANALYSIS / "t4-scoped-regression-dev100-v3.execution.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    gate = json.loads(PREREQ.read_text(encoding="utf-8"))
    if gate["inventory_sha256"] != sha256_file(parent_path):
        raise ValueError("prerequisite inventory changed")
    files = dict(parent["files"])
    for rel, digest in files.items():
        if sha256_file(REPO / rel) != digest:
            raise ValueError(f"accepted source changed: {rel}")
    roots = list((REPO / "src").rglob("*.py"))
    restored = json.loads(
        (ANALYSIS / "test100-v2-source-restore-v1.json").read_text(encoding="utf-8")
    )
    restored_root = REPO / restored["root"]
    roots += [restored_root / rel for rel in restored["restored_files"]]
    roots += [
        restored_root / "captured-v2.yaml",
        Path(__file__),
        PREREQ,
        parent_path,
        REPO / "tools/test100_v2_worker.py",
        REPO / "tools/analysis_io.py",
        REPO / "tools/recursive_summary_report.py",
        REPO / "tools/t4_scoped_selection_acceptance.py",
        REPO / "results/manifests/test100.json",
        REPO / "stores/test100.db",
        REPO / "stores/test100-index.npy",
        REPO / "stores/test100-index.ids.json",
        ANALYSIS / "test100-v2-source-restore-v1.json",
        ANALYSIS / f"{STEM}.baseline-preflight.json",
        ANALYSIS / f"{STEM}.candidate-preflight-r2.json",
        REPO / "tests/test_original_test100_protocol.py",
        ANALYSIS / f"{STEM}.resolution.json",
        REPO / f"results/prereg-{STEM}.md",
    ]
    wal = REPO / "stores/test100.db-wal"
    if wal.exists() and wal.stat().st_size:
        roots.append(wal)
    files.update({str(p.relative_to(REPO)): sha256_file(p) for p in roots})
    baseline = json.loads(
        (ANALYSIS / f"{STEM}.baseline-preflight.json").read_text(encoding="utf-8")
    )
    candidate = json.loads(
        (ANALYSIS / f"{STEM}.candidate-preflight-r2.json").read_text(encoding="utf-8")
    )
    if baseline["worker_sha256"] != sha256_file(REPO / "tools/test100_v2_worker.py") or candidate[
        "worker_sha256"
    ] != sha256_file(Path(__file__)):
        raise ValueError("preflight implementation changed")
    snapshot = {
        "files": files,
        "question_ids": [i.question_id for i in instances()],
        "models": parent["models"],
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
    }
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


def verify(snapshot):
    for rel, h in snapshot["files"].items():
        if sha256_file(REPO / rel) != h:
            raise ValueError(f"frozen original-test100 input changed: {rel}")


def report(runs, workload, identity):
    types = {i.question_id: i.question_type for i in instances()}
    accuracy = paired_accuracy(runs, types)
    mean = accuracy["mean_correct"]
    contexts = {
        arm: statistics.median(r["context_tokens"] for rep in rows for r in rep.values())
        for arm, rows in runs.items()
    }
    latency = {}
    for arm, repeats in runs.items():
        values = sorted(r["notes"]["end_to_end_wall_ms"] for rep in repeats for r in rep.values())
        latency[arm] = {
            "p50_ms": statistics.median(values),
            "p95_ms": values[round(0.95 * (len(values) - 1))],
            "max_ms": values[-1],
        }
    result = {
        "experiment_class": "exposed-original-test100-regression",
        "inventory_sha256": identity,
        "questions": 100,
        "reader_rows": 600,
        "grade_rows": 600,
        "accuracy": accuracy,
        "accuracy_percent": mean,
        "historical_v2_one_shot_accuracy": 72.0,
        "median_context": contexts,
        "answer_end_to_end_latency": latency,
        "resources_by_model": model_workload([r for records in workload.values() for r in records]),
        "resources_by_arm_and_model": {
            arm: model_workload(records) for arm, records in workload.items()
        },
        "failed_call_tokens_may_be_unreported": True,
        "pass": accuracy["mean_net"] >= 0
        and all(v >= -2 for v in accuracy["by_type_mean_net"].values())
        and contexts["candidate"] <= 6500,
        "limitations": "Exposed same-question regression, not unseen final. Historical72 and "
        "contemporaneous control are separate. Three repeats are100 unique questions, "
        "not300 independent samples; hosted weights cannot be hashed.",
    }
    immutable_json(ANALYSIS / f"{STEM}.gate.json", result)
    lines = [
        "# Original test100 same-question regression",
        "",
        "Exposed regression; models unchanged. Historical frozen72% is retained.",
        "",
        "| Arm | Three runs | Mean accuracy |",
        "|---|---|---:|",
    ]
    for arm in ("control", "candidate"):
        lines.append(
            f"| {arm} | {' / '.join(str(n) for n in accuracy['per_run_correct'][arm])} "
            f"| {mean[arm]:.2f}% |"
        )
    lines += [
        "",
        f"Mean paired difference: {accuracy['mean_net']:+.2f} percentage points.",
        "",
        "See the bound JSON for paired uncertainty and complete usage. "
        "This is not an unseen final result.",
    ]
    path = ANALYSIS / f"{STEM}.gate.md"
    text = "\n".join(lines) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError("immutable report changed")
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    return result


def main():
    from llm_long_term_memory.evaluation.harness import _exclusive

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        candidate_preflight()
        return 0
    if not args.execute:
        print(
            json.dumps(
                {"provider_calls": 0, "fresh_answers": 600, "fresh_grades": 600, "questions": 100}
            )
        )
        return 0
    with _exclusive(STATE):
        return execute()


def execute():
    from llm_long_term_memory import cli
    from llm_long_term_memory.evaluation.harness import _exclusive
    from llm_long_term_memory.evaluation.recovery import RecoveringJudge, RecoveryExhausted
    from llm_long_term_memory.llm.client import DailyQuotaExhausted
    from llm_long_term_memory.llm.usage import UsageTracker
    from llm_long_term_memory.prompt_versions import JUDGE_PROMPT_VERSION

    snapshot, identity = freeze()
    selected = instances()
    ids = {i.question_id for i in selected}
    runs = {"control": [], "candidate": []}
    workload = {"control": [], "candidate": []}
    for rep in (1, 2, 3):
        for arm in ("control", "candidate"):
            verify(snapshot)
            base = REPO / f"results/raw/{STEM}-{arm}-rep{rep}"
            rp, gp, up, cp = [
                base.with_suffix("." + s)
                for s in ("readers.jsonl", "grades.jsonl", "usage.json", "jsonl")
            ]
            if arm == "control":
                rc = subprocess.call(
                    [
                        str(REPO / ".venv/bin/python"),
                        str(REPO / "tools/test100_v2_worker.py"),
                        "--execute",
                        "--rep",
                        str(rep),
                    ],
                    cwd=REPO,
                    env={**os.environ, "HF_HUB_OFFLINE": "1"},
                )
                if rc:
                    save_json(
                        STATE,
                        {
                            "status": "QUOTA_WAIT" if rc == 2 else "STOPPED",
                            "arm": arm,
                            "rep": rep,
                            "returncode": rc,
                        },
                    )
                    return rc
            with _exclusive(rp):
                readers = rows_by_question(rp) if rp.exists() else {}
                grades = rows_by_question(gp) if gp.exists() else {}
                if not set(grades) <= set(readers) <= ids or (
                    (readers or grades) and not up.exists()
                ):
                    raise ValueError("foreign/orphan results or missing usage")
                if any(
                    r["inventory_sha256"] != identity
                    or r["cohort"] != arm
                    or r["answer_prompt_version"] != snapshot["arms"][arm]
                    for r in readers.values()
                ):
                    raise ValueError("reader identity/prompt mismatch")
                if any(r["judge_prompt_version"] != JUDGE_PROMPT_VERSION for r in grades.values()):
                    raise ValueError("judge prompt mismatch")
                _, _, runner, judge, usage = cli._build(
                    "two_stage_raw_primary_t4",
                    "configs/fallback.yaml",
                    "test100",
                    read_only_store=True,
                )
                runner = SavingsRepairRunner.from_t4(runner)
                runner.client.max_retries = runner.client.max_transport_retries = 2
                runner.client = CapturingClient(runner.client)
                if up.exists():
                    usage.records[:] = UsageTracker._load_records(up)
                try:
                    if arm == "candidate":
                        with rp.open("a", encoding="utf-8") as sink:
                            for i in selected:
                                if i.question_id in readers:
                                    continue
                                if runner.client.quota.for_model(runner.model).remaining_today < 6:
                                    save_json(
                                        STATE,
                                        {
                                            "status": "QUOTA_WAIT",
                                            "arm": arm,
                                            "rep": rep,
                                            "readers": len(readers),
                                            "grades": len(grades),
                                        },
                                    )
                                    return 2
                                readers[i.question_id] = save_reader(
                                    AuditedRunner(runner, identity, arm, usage),
                                    i,
                                    sink,
                                    identity,
                                    arm,
                                )
                                usage.save(up)
                                save_json(
                                    STATE,
                                    {
                                        "status": "RUNNING",
                                        "arm": arm,
                                        "rep": rep,
                                        "readers": len(readers),
                                        "grades": len(grades),
                                    },
                                )
                                print(
                                    f"test100 {arm} rep{rep} readers {len(readers)}/100", flush=True
                                )
                    if set(readers) != ids:
                        raise ValueError("all100 readers required before grading")
                    with gp.open("a", encoding="utf-8") as sink:
                        for i in selected:
                            if i.question_id in grades:
                                combine(
                                    i, readers[i.question_id], grades[i.question_id], identity, arm
                                )
                                continue
                            scoped = RecoveringJudge(
                                TransportAwareJudge(judge),
                                usage,
                                up,
                                base.with_suffix(f".{i.question_id}.recovery.json"),
                                hashlib.sha256(
                                    f"{identity}:{arm}:{rep}:{i.question_id}".encode()
                                ).hexdigest(),
                                max_recoveries=7,
                                max_wait_seconds=480,
                            )
                            grades[i.question_id] = grade_saved(
                                scoped, i, readers[i.question_id], sink, identity
                            )
                            usage.save(up)
                            save_json(
                                STATE,
                                {
                                    "status": "RUNNING",
                                    "arm": arm,
                                    "rep": rep,
                                    "readers": len(readers),
                                    "grades": len(grades),
                                },
                            )
                            print(f"test100 {arm} rep{rep} grades {len(grades)}/100", flush=True)
                    joined = [
                        asdict(
                            combine(i, readers[i.question_id], grades[i.question_id], identity, arm)
                        )
                        for i in selected
                    ]
                    if cp.exists() and list(rows_by_question(cp).values()) != joined:
                        raise ValueError("combined rows changed")
                    if not cp.exists():
                        cp.write_text(
                            "".join(json.dumps(r) + "\n" for r in joined), encoding="utf-8"
                        )
                    runs[arm].append({r["question_id"]: r for r in joined})
                    workload[arm].extend(usage.records)
                except (DailyQuotaExhausted, RecoveryExhausted) as exc:
                    save_json(
                        STATE,
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
                    return 2
                except Exception as exc:
                    save_json(
                        STATE,
                        {
                            "status": "STOPPED",
                            "reason": f"{type(exc).__name__}: {exc}"[:300],
                            "arm": arm,
                            "rep": rep,
                            "readers": len(readers),
                            "grades": len(grades),
                        },
                    )
                    raise
                finally:
                    usage.save(up)
                    runner.store.close()
    result = report(runs, workload, identity)
    save_json(STATE, {"status": "COMPLETE", "pass": result["pass"]})
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
