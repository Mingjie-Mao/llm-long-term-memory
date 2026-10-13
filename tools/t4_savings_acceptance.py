"""Fresh scoped-t4 acceptance; actual train21 requests saved before grading."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from audit_t3t4_completion import audit  # noqa: E402
from grounded_reader_diagnostic import grade_saved, save_reader  # noqa: E402
from run_grounded_experiment import frozen_identity  # noqa: E402
from run_grounded_paired import combine  # noqa: E402
from run_paged_paired import CostBoundRunner, save_json  # noqa: E402
from run_recursive_summary_cost import PILOT, instances_for  # noqa: E402
from t4_savings_replay import SavedReplyClient, cached_grade, row_sha  # noqa: E402
from t4_savings_replay_preflight import historical  # noqa: E402

STEM = "t4-scoped-selection-train21-v4"
VARIANT = "two_stage_raw_primary_t4"
COHORT = "t4-scoped-selection-v3"
GAP_GOLD = {
    "f9e8c073": "The covered periods of the quantity reports and whether they overlap are missing. "
    "An exact combined total is not justified; clarify each period and overlap.",
    "gpt4_2f8be40d": "The exact number of weddings attended this year cannot be established "
    "reliably from all records. Besides three named events, an unnamed cousin's wedding "
    "last month and an undated sister's participation report appear. Clarify event "
    "identity/dates, which mentions refer to the same event, attendance and reporting period; "
    "do not force either 3 or 5 as an exact total.",
    "gpt4_731e37d7": "The explicit payments are $20, $200 and $500, totaling $720. "
    "This is the sum of cited payments, not a confirmed total for the requested four-month period. "
    "Event date, event year and reporting period are unresolved; ask for clarification "
    "and do not fabricate a corrected year.",
}
BASE = REPO / f"results/raw/{STEM}"
STATE = REPO / f"results/analysis/{STEM}.progress.json"


class CapturingClient:
    def __init__(self, client):
        self.delegate = client
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def generate(self, **kwargs):
        schema = kwargs.get("schema")
        saved = {k: v for k, v in kwargs.items() if k != "schema"}
        saved["schema"] = schema.model_json_schema() if schema else None
        response = self.delegate.generate(**kwargs)
        self.calls.append({"request": saved, "response": response.text})
        return response


class AuditedRunner(CostBoundRunner):
    def answer_request(self, request):
        self.runner.client.calls.clear()
        started = time.perf_counter()
        answer = super().answer_request(request)
        answer.notes["actual_answer_calls"] = list(self.runner.client.calls)
        answer.notes["end_to_end_wall_ms"] = (time.perf_counter() - started) * 1000
        return answer


class ReplayAuditedRunner(AuditedRunner):
    def __init__(self, runner, identity, arm, usage, original, proof):
        super().__init__(runner, identity, arm, usage)
        self.original, self.proof = original, proof

    def answer_request(self, request):
        answer = super().answer_request(request)
        self.runner.client.delegate.verify_output(answer, self.original)
        if row_sha(answer.notes["selection_repair"]) != row_sha(
            self.original["answer"]["notes"]["selection_repair"]
        ):
            raise ValueError("unchanged source/calculation proof changed")
        answer.notes["exact_request_replay"] = self.proof
        answer.notes["execution_kind"] = "historical_response_cpu_replay"
        return answer


def source_coverage(instance, reader):
    from llm_long_term_memory.ingest.extract import _parse_date
    from llm_long_term_memory.retrieve.time_notes import annotate

    prompts = [c["request"]["prompt"] for c in reader["answer"]["notes"]["actual_answer_calls"]]
    gold = [(s, t) for s in instance.sessions for t in s.turns if t.has_answer]
    found = sum(
        any(
            text in p
            for p in prompts
            for text in {
                t.content,
                annotate(t.content, _parse_date(s.date))[0]
                if t.role == "user" and _parse_date(s.date)
                else t.content,
            }
        )
        for s, t in gold
    )
    return {
        "reference_turns": len(gold),
        "full_reference_turns_present": found,
        "complete": found == len(gold),
        "scope": "whole reference text in actual prompts; conservative diagnostic, not a gate",
    }


def report(instances, identity, usage):
    from recursive_summary_report import model_workload

    rp, gp, xp = [
        BASE.with_suffix("." + s) for s in ("readers.jsonl", "grades.jsonl", "gap-grades.jsonl")
    ]
    readers, grades, gaps = [rows_by_question(p) if p.exists() else {} for p in (rp, gp, xp)]
    if set(readers) != set(grades) or set(readers) != {i.question_id for i in instances}:
        raise ValueError("all21 readers and original grades required")
    if set(gaps) != set(GAP_GOLD):
        raise ValueError("all three preregistered faithful-gap grades required")
    rows = []
    for i in instances:
        q = i.question_id
        original = combine(i, readers[q], grades[q], identity, COHORT)
        faithful = None
        if q in gaps:
            faithful = combine(
                replace(i, answer=GAP_GOLD[q]), readers[q], gaps[q], identity, COHORT
            ).correct
        rows.append(
            {
                "question_id": q,
                "question_type": i.question_type,
                "original_judged_correct": original.correct,
                "faithful_gap": faithful,
                "supported_case": q not in GAP_GOLD,
                "source_coverage": source_coverage(i, readers[q]),
                "context_tokens": original.context_tokens,
                "pass": original.correct if q not in GAP_GOLD else faithful,
            }
        )
    context = statistics.median(r["context_tokens"] for r in rows)
    result = {
        "experiment_class": "exposed-train150-component-acceptance-with-exact-historical-replay",
        "inventory_sha256": identity,
        "questions": len(rows),
        "original_correct": sum(r["original_judged_correct"] for r in rows),
        "supported_correct": sum(r["original_judged_correct"] for r in rows if r["supported_case"]),
        "supported_total": 18,
        "faithful_gaps_passed": sum(r["faithful_gap"] is True for r in rows),
        "median_context": context,
        "pass": all(r["pass"] for r in rows) and context <= 6500,
        "rows": rows,
        "resources_by_model": model_workload(usage.records),
        "execution_kinds": {
            "historical_reader_replays": sum(
                bool(r["answer"]["notes"].get("exact_request_replay")) for r in readers.values()
            ),
            "fresh_reader_questions": sum(
                not r["answer"]["notes"].get("exact_request_replay") for r in readers.values()
            ),
            "historical_grade_replays": sum(
                bool(g.get("exact_grade_replay")) for g in [*grades.values(), *gaps.values()]
            ),
            "fresh_grades": sum(
                not g.get("exact_grade_replay") for g in [*grades.values(), *gaps.values()]
            ),
        },
        "inputs": {str(p.relative_to(REPO)): sha256_file(p) for p in (rp, gp, xp)},
        "limitations": "Historical replays are not fresh model samples or latency measurements. "
        "Not population accuracy. A failed acceptance blocks summary calls; "
        "do not tune heldout100 failures or change original grades.",
    }
    output = REPO / f"results/analysis/{STEM}.gate.json"
    if output.exists() and json.loads(output.read_text(encoding="utf-8")) != result:
        raise ValueError("acceptance gate changed")
    if not output.exists():
        save_json(output, result)
    return result


def main():
    from llm_long_term_memory import cli
    from llm_long_term_memory.evaluation.harness import _exclusive
    from llm_long_term_memory.evaluation.recovery import RecoveringJudge, RecoveryExhausted
    from llm_long_term_memory.llm.client import DailyQuotaExhausted
    from llm_long_term_memory.llm.usage import UsageTracker

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stage", choices=("all", "readers", "grades", "report"), default="all")
    args = parser.parse_args()
    plan_path = REPO / "results/analysis/t4-scoped-selection-train21-v4.replay-preflight-r2.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if any(sha256_file(REPO / p) != h for p, h in plan["inputs"].items()):
        raise ValueError("frozen zero-call replay proof changed")
    old_readers, old_grades, old_gaps, _ = historical()
    replay_proofs = {r["question_id"]: r["proof"] for r in plan["rows"] if r["replay_eligible"]}
    evidence = audit()
    if not evidence["gate"]["pass"]:
        raise ValueError("t4 complete comparison did not pass")
    if not args.execute:
        print(
            json.dumps(
                {
                    "provider_calls": 0,
                    "questions": len(PILOT),
                    "fresh_answer_upper": 2 * len(plan["fresh_ids"]),
                    "historical_replays": len(replay_proofs),
                    "original_grades": 21,
                    "faithful_gap_grades": 3,
                    "summary_calls": 0,
                    "t4_verified_pass": True,
                }
            )
        )
        return 0
    snapshot = frozen_identity(
        "t4-scoped-selection-v3", REPO / "results/manifests/train150.json", "train150"
    )
    extras = [
        Path(__file__),
        REPO / "tools/t4_selection_repair.py",
        REPO / "tools/t4_selection_preflight.py",
        REPO / "tools/t4_event_count_repair.py",
        REPO / "tools/t4_savings_repair.py",
        REPO / "tools/t4_savings_replay.py",
        REPO / "tools/t4_savings_replay_preflight.py",
        REPO / "tools/t4_event_count_preflight.py",
        plan_path,
        REPO / "results/analysis/t4-selection-train21-v1.gate.json",
        REPO / "results/analysis/t4-scoped-selection-train21-v2.gate.json",
        REPO / "results/analysis/t4-scoped-selection-train21-v3.gate.json",
        REPO / "tools/audit_t3t4_completion.py",
        REPO / "tools/run_paged_paired.py",
        REPO / "tools/run_grounded_paired.py",
        REPO / "tools/grounded_reader_diagnostic.py",
        REPO / "tools/run_recursive_summary_cost.py",
        REPO / "tools/recursive_summary_report.py",
        REPO / f"results/prereg-{STEM}.md",
    ]
    snapshot["files"].update({str(p.relative_to(REPO)): sha256_file(p) for p in extras})
    snapshot.update(
        {
            "question_ids": list(PILOT),
            "faithful_gap_targets": GAP_GOLD,
            "prompt_version": "memory-t4-scoped-selection-v3",
            "variant": VARIANT,
            "source_audit": evidence["inputs"],
            "client_attempts": 2,
            "transport_attempts": 2,
            "judge_recoveries": 3,
            "judge_wait_seconds": 180,
        }
    )
    ip = REPO / f"results/analysis/{STEM}.execution.json"
    sp = REPO / f"results/analysis/{STEM}.source-snapshot.json"
    if not ip.exists():
        save_json(
            sp,
            {
                p: {"sha256": h, "text": (REPO / p).read_text(encoding="utf-8")}
                for p, h in snapshot["files"].items()
                if Path(p).suffix in {".py", ".md", ".yaml"}
            },
        )
        snapshot["source_snapshot_sha256"] = sha256_file(sp)
        save_json(ip, snapshot)
    else:
        snapshot["source_snapshot_sha256"] = sha256_file(sp)
        if json.loads(ip.read_text(encoding="utf-8")) != snapshot:
            raise ValueError("frozen selection source/config/data changed")
    identity = sha256_file(ip)
    instances = instances_for("train150")
    rp, gp, xp, up = [
        BASE.with_suffix("." + s)
        for s in ("readers.jsonl", "grades.jsonl", "gap-grades.jsonl", "usage.json")
    ]
    with _exclusive(rp):
        readers, grades, gaps = [rows_by_question(p) if p.exists() else {} for p in (rp, gp, xp)]
        ids = {i.question_id for i in instances}
        if not set(grades) <= set(readers) <= ids or not set(gaps) <= set(readers) & set(GAP_GOLD):
            raise ValueError("foreign/orphan rows")
        if (readers or grades or gaps) and not up.exists():
            raise ValueError("missing resumed usage")
        _, _, runner, judge, usage = cli._build(
            VARIANT, "configs/fallback.yaml", "train150", read_only_store=True
        )
        from t4_savings_repair import SavingsRepairRunner

        runner = SavingsRepairRunner.from_t4(runner)
        runner.client.max_retries = runner.client.max_transport_retries = 2
        live_client = runner.client
        runner.client = CapturingClient(live_client)
        if up.exists():
            usage.records[:] = UsageTracker._load_records(up)
        for row in readers.values():
            if row["inventory_sha256"] != identity or row["cohort"] != COHORT:
                raise ValueError("reader identity changed")
        try:
            stages = ("readers", "grades", "report") if args.stage == "all" else (args.stage,)
            for stage in stages:
                if any(sha256_file(REPO / p) != h for p, h in snapshot["files"].items()):
                    raise ValueError("frozen inputs changed during execution")
                if stage == "readers":
                    with rp.open("a", encoding="utf-8") as sink:
                        for i in instances:
                            if i.question_id in readers:
                                continue
                            if (
                                i.question_id not in replay_proofs
                                and live_client.quota.for_model(runner.model).remaining_today < 6
                            ):
                                save_json(
                                    STATE,
                                    {
                                        "status": "QUOTA_WAIT",
                                        "stage": stage,
                                        "readers": len(readers),
                                        "grades": len(grades),
                                    },
                                )
                                return 2
                            if i.question_id in replay_proofs:
                                runner.client = CapturingClient(
                                    SavedReplyClient(live_client, old_readers[i.question_id])
                                )
                                bound = ReplayAuditedRunner(
                                    runner,
                                    identity,
                                    COHORT,
                                    usage,
                                    old_readers[i.question_id],
                                    replay_proofs[i.question_id],
                                )
                            else:
                                runner.client = CapturingClient(live_client)
                                bound = AuditedRunner(runner, identity, COHORT, usage)
                            readers[i.question_id] = save_reader(
                                bound,
                                i,
                                sink,
                                identity,
                                COHORT,
                            )
                            usage.save(up)
                            print(f"readers {len(readers)}/21", flush=True)
                elif stage == "grades":
                    if set(readers) != ids:
                        raise ValueError("save all21 answers before grading")
                    for gap_mode, path, saved in ((False, gp, grades), (True, xp, gaps)):
                        with path.open("a", encoding="utf-8") as sink:
                            for i in instances:
                                q = i.question_id
                                if q in saved or (gap_mode and q not in GAP_GOLD):
                                    continue
                                kind = "gap" if gap_mode else "original"
                                scoped = RecoveringJudge(
                                    judge,
                                    usage,
                                    up,
                                    BASE.with_suffix(f".{q}.{kind}.recovery.json"),
                                    hashlib.sha256(f"{identity}:{q}:{kind}".encode()).hexdigest(),
                                    max_recoveries=3,
                                    max_wait_seconds=180,
                                )
                                target = replace(i, answer=GAP_GOLD[q]) if gap_mode else i
                                if q in replay_proofs:
                                    old_grade = old_gaps[q] if gap_mode else old_grades[q]
                                    if judge.prompt_version != old_grade["judge_prompt_version"]:
                                        raise ValueError("replayed judge version changed")
                                    saved[q] = cached_grade(
                                        target,
                                        readers[q],
                                        old_readers[q],
                                        old_grade,
                                        identity,
                                        "gap" if gap_mode else "original",
                                    )
                                    sink.write(json.dumps(saved[q], ensure_ascii=False) + "\n")
                                    sink.flush()
                                    print(f"{kind} historical grade replay {q}", flush=True)
                                    continue
                                saved[q] = grade_saved(scoped, target, readers[q], sink, identity)
                                usage.save(up)
                                print(
                                    f"{kind} grades {len(saved)}/{3 if gap_mode else 21}",
                                    flush=True,
                                )
                else:
                    result = report(instances, identity, usage)
                    save_json(
                        STATE,
                        {
                            "status": "COMPLETE",
                            "pass": result["pass"],
                            "supported_correct": result["supported_correct"],
                            "faithful_gaps_passed": result["faithful_gaps_passed"],
                        },
                    )
                    print(
                        json.dumps(
                            {
                                k: result[k]
                                for k in (
                                    "pass",
                                    "original_correct",
                                    "supported_correct",
                                    "faithful_gaps_passed",
                                    "median_context",
                                )
                            }
                        ),
                        flush=True,
                    )
                    return 0 if result["pass"] else 1
        except (DailyQuotaExhausted, RecoveryExhausted) as exc:
            save_json(
                STATE,
                {
                    "status": "QUOTA_WAIT"
                    if isinstance(exc, DailyQuotaExhausted)
                    else "RECOVERY_EXHAUSTED",
                    "reason": str(exc),
                    "readers": len(readers),
                    "grades": len(grades),
                    "gap_grades": len(gaps),
                },
            )
            print(json.dumps({"status": "INCOMPLETE", "reason": str(exc)}), flush=True)
            return 2
        finally:
            usage.save(up)
            runner.store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
