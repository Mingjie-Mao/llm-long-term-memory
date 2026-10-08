"""Persist exposed-train reader outputs before any grading, with resumable identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import rows_by_question, sha256_file  # noqa: E402
from run_grounded_experiment import frozen_identity  # noqa: E402


def save_reader(runner, instance, sink, identity, cohort):
    from llm_long_term_memory.conversation import AnswerRequest

    answer = runner.answer_request(
        AnswerRequest(instance.question, instance.question_date, instance.store_namespace)
    )
    row = {
        "question_id": instance.question_id,
        "cohort": cohort,
        "grading_status": "UNGRADED",
        "inventory_sha256": identity,
        "answer_prompt_version": runner.answer_prompt_version,
        "answer": asdict(answer),
    }
    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    sink.flush()
    return row


def grade_saved(judge, instance, row, sink, identity):
    verdict = judge.grade(
        question=instance.question,
        gold=instance.answer,
        hypothesis=row["answer"]["text"],
        is_abstention=instance.is_abstention,
        question_type=instance.question_type,
    )
    grade = {
        "question_id": instance.question_id,
        "cohort": row["cohort"],
        "inventory_sha256": identity,
        "judge_prompt_version": judge.prompt_version,
        "reader_row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
        "verdict": asdict(verdict),
    }
    sink.write(json.dumps(grade, ensure_ascii=False) + "\n")
    sink.flush()
    return grade


def main():
    from llm_long_term_memory import cli
    from llm_long_term_memory.commands.common import manifest_instances
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.harness import _exclusive
    from llm_long_term_memory.evaluation.manifest import load_manifest, require_claim
    from llm_long_term_memory.llm.usage import UsageTracker

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=15)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--offline-replay", type=Path)
    parser.add_argument(
        "--grade", action="store_true", help="Grade saved answers; never rerun readers"
    )
    args = parser.parse_args()
    if args.version < 6:
        raise SystemExit("this separate reader protocol was registered starting with v6")
    manifests = {
        "reasoning-errors": REPO / "results/manifests/train150-raw-v1-reasoning-errors.json",
        "originally-correct": REPO / "results/manifests/train150-raw-v1-correct-sample10.json",
    }
    instances, cohorts = {}, {}
    for cohort, path in manifests.items():
        manifest = load_manifest(path)
        require_claim(manifest, "regression")
        selected = {i.question_id: i for i in manifest_instances(manifest, Settings())}
        for qid in manifest.question_ids:
            if qid in instances:
                raise ValueError("overlapping registered diagnostic cohorts")
            instances[qid], cohorts[qid] = selected[qid], cohort
    if not args.execute:
        print(
            json.dumps(
                {
                    "provider_calls": 0,
                    "questions": len(instances),
                    "reader_upper_bound": 0 if args.grade else 2 * len(instances),
                    "judge_calls": len(instances) if args.grade else 0,
                }
            )
        )
        return 0
    offline_path = (
        args.offline_replay
        or REPO / f"results/analysis/grounded-context-v{args.version}.train150.json"
    ).resolve()
    offline = json.loads(offline_path.read_text(encoding="utf-8"))
    previous = json.loads(
        (REPO / f"results/analysis/grounded-context-v{args.version - 1}.train150.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        not all(offline["offline_gate"].values())
        or offline["all_gold_turn_coverage"]["candidate"]
        < previous["all_gold_turn_coverage"]["candidate"]
    ):
        raise SystemExit("STOP: registered offline gates failed")
    if offline["candidate_source_sha256"] != sha256_file(
        REPO / "src/llm_long_term_memory/runtime/grounded_answering.py"
    ):
        raise SystemExit("STOP: source differs from offline replay")
    snapshot = frozen_identity(
        "reader-diagnostic", manifests["reasoning-errors"], "train150", args.version
    )
    for path in [
        Path(__file__),
        REPO / "tools/grounded_context_replay.py",
        manifests["originally-correct"],
        offline_path,
    ]:
        snapshot["files"][str(path.relative_to(REPO))] = sha256_file(path)
    stem = f"grounded-reader-v{args.version}"
    inventory = REPO / f"results/analysis/{stem}.execution.json"
    output = REPO / f"results/raw/{stem}.answers.jsonl"
    grades_path = REPO / f"results/raw/{stem}.grades.jsonl"
    usage_path = REPO / f"results/raw/{stem}.usage.json"
    with _exclusive(output):
        if inventory.exists():
            if json.loads(inventory.read_text(encoding="utf-8")) != snapshot:
                raise SystemExit("STOP: execution identity changed; register a new namespace")
        else:
            inventory.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        identity = sha256_file(inventory)
        rows = rows_by_question(output) if output.exists() else {}
        grades = rows_by_question(grades_path) if grades_path.exists() else {}
        for saved in [rows, grades]:
            if not set(saved) <= set(instances) or any(
                r["inventory_sha256"] != identity for r in saved.values()
            ):
                raise SystemExit("STOP: foreign rows or identity on resume")
        if args.grade and set(rows) != set(instances):
            raise SystemExit("STOP: complete reader outputs before grading")
        if (rows or grades) and not usage_path.exists():
            raise SystemExit("STOP: missing usage provenance")
        _, _, runner, judge, usage = cli._build(
            f"two_stage_raw_primary_grounded_v{args.version}",
            "configs/fallback.yaml",
            "train150",
            read_only_store=True,
        )
        if usage_path.exists():
            usage.records[:] = UsageTracker._load_records(usage_path)
        try:
            with (grades_path if args.grade else output).open("a", encoding="utf-8") as sink:
                for qid, instance in instances.items():
                    if args.grade:
                        if qid in grades:
                            continue
                        grades[qid] = grade_saved(judge, instance, rows[qid], sink, identity)
                    else:
                        if qid in rows:
                            continue
                        rows[qid] = save_reader(runner, instance, sink, identity, cohorts[qid])
                    usage.save(usage_path)
                    print(
                        json.dumps(
                            {
                                "question_id": qid,
                                "stage": "grade" if args.grade else "reader",
                                "completed": len(grades) if args.grade else len(rows),
                                "total": len(instances),
                            }
                        ),
                        flush=True,
                    )
        finally:
            usage.save(usage_path)
            runner.store.close()
        for name, sha in snapshot["files"].items():
            if sha256_file(REPO / name) != sha:
                raise SystemExit("STOP: execution files changed during run")
        print(
            json.dumps(
                {
                    "status": "GRADED" if args.grade else "UNGRADED",
                    "reader_questions": len(rows),
                    "graded_questions": len(grades),
                    "provider_usage": usage.summary(),
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
