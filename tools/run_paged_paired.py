"""New three-repeat dev100 regression; immutable readers and exact grading cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from grounded_gate import evaluate  # noqa: E402
from grounded_reader_diagnostic import grade_saved, save_reader  # noqa: E402
from judge_request_cache import ExactRequestJudge, request_key  # noqa: E402
from paired_manifest_scope import select_manifest  # noqa: E402
from run_grounded_experiment import BASELINE, frozen_identity  # noqa: E402
from run_grounded_paired import BoundRunner, combine  # noqa: E402

STEM = "paged-paired-dev100-v18-v1"


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class CostBoundRunner(BoundRunner):
    def __init__(self, runner, identity, arm, usage):
        super().__init__(runner, identity, arm)
        self.usage = usage

    def answer_request(self, request):
        before = len(self.usage.records)
        answer = super().answer_request(request)
        calls = self.usage.records[before:]
        answer.notes["whole_question_usage"] = {
            "requests": len(calls),
            "failures": sum(not c.ok for c in calls),
            "input_tokens": sum(c.input_tokens for c in calls),
            "output_tokens": sum(c.output_tokens for c in calls),
        }
        return answer


def expected_calls(store, instance, candidate):
    from llm_long_term_memory.runtime.evidence_pages import archive_pages
    from llm_long_term_memory.runtime.grounded_answering import personal_archive_review

    pages = (
        archive_pages(store, instance.store_namespace, roles=("user",), split_oversized=True)
        if candidate and personal_archive_review(instance.question)
        else None
    )
    if pages and not pages.complete:
        raise ValueError("candidate archive no longer matches complete preflight")
    return 2 * (2 + (len(pages.pages) if pages else 0))


def seed_cache(cache, instances, identity, current_digest):
    # Immutable stopped evidence is used ONLY for exact grading requests, never
    # as new reader outputs or for question-level dev error diagnosis.
    stopped = json.loads(
        (
            REPO
            / "results/analysis/grounded-paired-dev100-v15.stopped-for-prerequisite-repair.json"
        ).read_text(encoding="utf-8")
    )
    old_inventory = REPO / "results/analysis/grounded-paired-dev100-v15.execution.json"
    if sha256_file(old_inventory) != stopped["execution_inventory_sha256"]:
        raise ValueError("historical grading inventory changed")
    old = json.loads(old_inventory.read_text(encoding="utf-8"))
    for path in ("src/llm_long_term_memory/evaluation/judge.py", "configs/fallback.yaml"):
        if old["files"][path] != identity["files"][path]:
            raise ValueError("historical judge/config incompatible with cache")
    sources = []
    for arm in ("baseline", "candidate"):
        base = REPO / f"results/raw/grounded-paired-dev100-v15-{arm}-rep1"
        rp, gp = base.with_suffix(".readers.jsonl"), base.with_suffix(".grades.jsonl")
        for path in (rp, gp):
            if sha256_file(path) != stopped["artifacts"][str(path.relative_to(REPO))]["sha256"]:
                raise ValueError("stopped grading source changed")
        sources.append((rp, gp, stopped["execution_inventory_sha256"]))
    for rep in (1, 2, 3):
        for arm in ("baseline", "candidate"):
            base = REPO / f"results/raw/{STEM}-{arm}-rep{rep}"
            rp, gp = base.with_suffix(".readers.jsonl"), base.with_suffix(".grades.jsonl")
            if rp.exists() and gp.exists():
                sources.append((rp, gp, current_digest))
    for rp, gp, digest in sources:
        readers, grades = rows_by_question(rp), rows_by_question(gp)
        if not set(grades) <= set(readers) <= set(instances):
            raise ValueError("foreign cache question")
        for qid, grade in grades.items():
            reader = readers[qid]
            if reader["inventory_sha256"] != digest:
                raise ValueError("cache reader execution identity changed")
            # Retain primary verdicts only; do not grow recursive cache chains.
            if (grade["verdict"].get("details") or {}).get("judge_cache"):
                continue
            cache.add(instances[qid], reader, grade, rp, gp)


class ScopedRecoveringCache:
    def __init__(self, cache, usage, usage_path, state_path, identity):
        self.cache, self.usage, self.usage_path = cache, usage, usage_path
        self.state_path, self.identity = state_path, identity
        self.prompt_version = cache.prompt_version

    def grade(self, **kwargs):
        from llm_long_term_memory.evaluation.recovery import RecoveringJudge

        key = request_key(self.cache.judge, kwargs)
        witnesses = self.cache.entries.get(key, [])
        if witnesses and len({w["result"]["correct"] for w in witnesses}) == 1:
            return self.cache.grade(**kwargs)  # No provider-budget reset on a cache hit.
        judge = RecoveringJudge(
            self.cache,
            self.usage,
            self.usage_path,
            self.state_path,
            self.identity,
            max_recoveries=3,
            max_wait_seconds=180,
        )
        return judge.grade(**kwargs)


def main():
    from llm_long_term_memory import cli
    from llm_long_term_memory.commands.common import manifest_instances
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.harness import _exclusive
    from llm_long_term_memory.evaluation.manifest import load_manifest, require_claim
    from llm_long_term_memory.evaluation.recovery import RecoveryExhausted
    from llm_long_term_memory.llm.client import DailyQuotaExhausted
    from llm_long_term_memory.llm.usage import UsageTracker

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stage", choices=("readers", "grades", "all", "report"), default="all")
    args = parser.parse_args()
    acceptance_path = REPO / "results/analysis/prerequisite-cache-acceptance-v1.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if not acceptance["formal_mechanism_acceptance_pass"]:
        raise ValueError("formal mechanism acceptance did not pass")
    for path, expected in {**acceptance["inputs"], **acceptance["addendum_inputs"]}.items():
        if sha256_file(REPO / path) != expected:
            raise ValueError("acceptance evidence changed")
    proof_path = REPO / "results/analysis/grounded-paired-dev100-v15.public-source-audit.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof["status"] != "PASS" or proof["store_sha256"] != sha256_file(REPO / "stores/dev100.db"):
        raise ValueError("public same-user source audit changed")
    manifest_path = REPO / "results/manifests/dev100.json"
    manifest = load_manifest(manifest_path)
    require_claim(manifest, "regression")
    instances = select_manifest(manifest_instances(manifest, Settings()), manifest.question_ids)
    selected = {i.question_id: i for i in instances}
    if len(instances) != 100:
        raise ValueError("exactly100 exposed dev questions required")
    preflight_path = REPO / f"results/analysis/{STEM}.preflight-r2.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight["complete_archives"] != 100:
        raise ValueError("page preflight incomplete")
    snapshot = frozen_identity("paired-dev100-v18", manifest_path, "dev100", 1)
    snapshot["version"] = 18
    extra = [
        Path(__file__),
        REPO / "tools/run_grounded_paired.py",
        REPO / "tools/judge_request_cache.py",
        REPO / "tools/paired_manifest_scope.py",
        REPO / "tools/grounded_reader_diagnostic.py",
        REPO / "tools/grade_saved_recovering.py",
        acceptance_path,
        proof_path,
        preflight_path,
        REPO / f"results/prereg-{STEM}.md",
        REPO / "results/analysis/resolution-paired-dev100-v18-v1.json",
        REPO / "results/analysis/grounded-paired-dev100-v15.stopped-for-prerequisite-repair.json",
    ]
    snapshot["files"].update({str(p.relative_to(REPO)): sha256_file(p) for p in extra})
    snapshot.update(
        {
            "sample_count": 100,
            "repeats_per_arm": 3,
            "answers": 600,
            "grading": "exact-request-cache-or-original-judge",
            "preflight": preflight,
        }
    )
    inventory_path = REPO / f"results/analysis/{STEM}.execution.json"
    snapshot_path = REPO / f"results/analysis/{STEM}.source-snapshot.json"
    if not args.execute:
        if inventory_path.exists():
            raise ValueError("new preview namespace required")
        save_json(
            snapshot_path,
            {
                p: {"sha256": sha, "text": (REPO / p).read_text(encoding="utf-8")}
                for p, sha in snapshot["files"].items()
                if Path(p).suffix in {".py", ".yaml", ".md"}
            },
        )
        snapshot["source_snapshot_sha256"] = sha256_file(snapshot_path)
        save_json(inventory_path, snapshot)
        print(
            json.dumps(
                {"provider_calls": 0, "preflight": preflight, "answers": 600, "grading_upper": 600}
            )
        )
        return 0
    snapshot["source_snapshot_sha256"] = sha256_file(snapshot_path)
    if json.loads(inventory_path.read_text(encoding="utf-8")) != snapshot:
        raise ValueError("paired frozen execution changed")
    digest = sha256_file(inventory_path)
    paths = {"baseline": [], "candidate": []}
    state_path = REPO / f"results/analysis/{STEM}.progress.json"
    for stage in ("readers", "grades") if args.stage == "all" else (args.stage,):
        if stage == "report":
            continue
        for rep in (1, 2, 3):
            for arm, variant in (
                ("baseline", BASELINE),
                ("candidate", "two_stage_raw_primary_grounded_v18"),
            ):
                if any(sha256_file(REPO / p) != sha for p, sha in snapshot["files"].items()):
                    raise ValueError("frozen source/config/data changed")
                base = REPO / f"results/raw/{STEM}-{arm}-rep{rep}"
                rp, gp, up = (
                    base.with_suffix(f".{s}")
                    for s in ("readers.jsonl", "grades.jsonl", "usage.json")
                )
                with _exclusive(rp):
                    readers, grades = (rows_by_question(p) if p.exists() else {} for p in (rp, gp))
                    if not set(grades) <= set(readers) <= set(selected):
                        raise ValueError("foreign or orphan result rows")
                    if (readers or grades) and not up.exists():
                        raise ValueError("missing resumed usage")
                    for row in readers.values():
                        if row["inventory_sha256"] != digest or row["cohort"] != arm:
                            raise ValueError("reader identity changed")
                    for qid, row in grades.items():
                        combine(selected[qid], readers[qid], row, digest, arm)
                    _, _, runner, judge, usage = cli._build(
                        variant, "configs/fallback.yaml", "dev100", read_only_store=True
                    )
                    runner.client.max_retries = runner.client.max_transport_retries = 2
                    if up.exists():
                        usage.records[:] = UsageTracker._load_records(up)
                    cache = ExactRequestJudge(judge)
                    if stage == "grades":
                        seed_cache(cache, selected, snapshot, digest)
                    try:
                        if stage == "readers":
                            with rp.open("a", encoding="utf-8") as sink:
                                for instance in instances:
                                    if instance.question_id in readers:
                                        continue
                                    needed = expected_calls(
                                        runner.store, instance, arm == "candidate"
                                    )
                                    limiter = runner.client.quota.for_model(runner.model)
                                    if limiter.remaining_today < needed:
                                        save_json(
                                            state_path,
                                            {
                                                "status": "QUOTA_WAIT",
                                                "stage": stage,
                                                "arm": arm,
                                                "repeat": rep,
                                                "readers": len(readers),
                                                "available": limiter.remaining_today,
                                                "required_headroom": needed,
                                            },
                                        )
                                        print(
                                            json.dumps(
                                                {
                                                    "status": "QUOTA_WAIT",
                                                    "arm": arm,
                                                    "repeat": rep,
                                                    "readers": len(readers),
                                                }
                                            ),
                                            flush=True,
                                        )
                                        return 2
                                    readers[instance.question_id] = save_reader(
                                        CostBoundRunner(runner, digest, arm, usage),
                                        instance,
                                        sink,
                                        digest,
                                        arm,
                                    )
                                    usage.save(up)
                                    print(f"{arm} rep{rep}: readers {len(readers)}/100", flush=True)
                        else:
                            if len(readers) != 100:
                                save_json(
                                    state_path,
                                    {
                                        "status": "WAIT_FOR_READERS",
                                        "stage": stage,
                                        "arm": arm,
                                        "repeat": rep,
                                        "readers": len(readers),
                                    },
                                )
                                return 2
                            with gp.open("a", encoding="utf-8") as sink:
                                for instance in instances:
                                    qid = instance.question_id
                                    if qid in grades:
                                        continue
                                    scoped = ScopedRecoveringCache(
                                        cache,
                                        usage,
                                        up,
                                        base.with_suffix(f".{qid}.recovery.json"),
                                        hashlib.sha256(
                                            f"{digest}:{arm}:{rep}:{qid}".encode()
                                        ).hexdigest(),
                                    )
                                    grades[qid] = grade_saved(
                                        scoped, instance, readers[qid], sink, digest
                                    )
                                    usage.save(up)
                                    print(f"{arm} rep{rep}: grades {len(grades)}/100", flush=True)
                            save_json(base.with_suffix(".cache-stats.json"), cache.stats())
                        save_json(
                            state_path,
                            {
                                "status": "RUNNING",
                                "stage": stage,
                                "arm": arm,
                                "repeat": rep,
                                "readers": len(readers),
                                "grades": len(grades),
                            },
                        )
                    except (DailyQuotaExhausted, RecoveryExhausted) as exc:
                        save_json(
                            state_path,
                            {
                                "status": "INCOMPLETE",
                                "reason": str(exc),
                                "stage": stage,
                                "arm": arm,
                                "repeat": rep,
                                "readers": len(readers),
                                "grades": len(grades),
                            },
                        )
                        print(json.dumps({"status": "INCOMPLETE", "reason": str(exc)}), flush=True)
                        return 2
                    finally:
                        usage.save(up)
                        runner.store.close()
    for rep in (1, 2, 3):
        for arm in ("baseline", "candidate"):
            base = REPO / f"results/raw/{STEM}-{arm}-rep{rep}"
            rp, gp = base.with_suffix(".readers.jsonl"), base.with_suffix(".grades.jsonl")
            readers, grades = (rows_by_question(p) if p.exists() else {} for p in (rp, gp))
            if len(readers) != 100 or len(grades) != 100:
                return 2
            combined = [
                asdict(combine(i, readers[i.question_id], grades[i.question_id], digest, arm))
                for i in instances
            ]
            out = base.with_suffix(".jsonl")
            text = "".join(json.dumps(r) + "\n" for r in combined)
            if out.exists() and out.read_text(encoding="utf-8") != text:
                raise ValueError("combined result changed")
            if not out.exists():
                out.write_text(text, encoding="utf-8")
            paths[arm].append(out)
    result = evaluate(
        paths,
        {i.question_id: i.question_type for i in instances},
        digest,
        candidate_prompt="memory-grounded-v18",
    )
    result["outcome_scope"] = (
        "PASS_TO_SUMMARY_COST_COMPARISON only; no default promotion or unseen-final claim"
    )
    result["execution_inventory"] = str(inventory_path.relative_to(REPO))
    out = REPO / f"results/analysis/{STEM}.gate.json"
    if out.exists():
        raise ValueError("paired gate exists")
    save_json(out, result)
    save_json(state_path, {"status": "COMPLETE", "pass": result["pass"]})
    print(json.dumps(result), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
