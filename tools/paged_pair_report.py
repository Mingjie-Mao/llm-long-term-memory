"""Aggregate completed paged regression and actual per-model usage; zero calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from recursive_summary_report import model_workload  # noqa: E402
from run_grounded_paired import combine  # noqa: E402
from run_paged_paired import STEM, save_json  # noqa: E402


def report():
    from paired_manifest_scope import select_manifest

    from llm_long_term_memory.commands.common import manifest_instances
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.manifest import load_manifest
    from llm_long_term_memory.llm.usage import UsageTracker

    analysis = REPO / "results/analysis"
    inventory = analysis / f"{STEM}.execution.json"
    digest = sha256_file(inventory)
    gate_path = analysis / f"{STEM}.gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    manifest = load_manifest(REPO / "results/manifests/dev100.json")
    instances = select_manifest(manifest_instances(manifest, Settings()), manifest.question_ids)
    arms, inputs = {}, {str(p.relative_to(REPO)): sha256_file(p) for p in (inventory, gate_path)}
    for arm in ("baseline", "candidate"):
        runs, all_records = [], []
        for rep in (1, 2, 3):
            base = REPO / f"results/raw/{STEM}-{arm}-rep{rep}"
            rp, gp, up = [
                base.with_suffix("." + s) for s in ("readers.jsonl", "grades.jsonl", "usage.json")
            ]
            readers, grades = rows_by_question(rp), rows_by_question(gp)
            if set(readers) != set(grades) or set(readers) != set(manifest.question_ids):
                raise ValueError("600 fresh readers and corresponding grades required")
            correct = sum(
                combine(i, readers[i.question_id], grades[i.question_id], digest, arm).correct
                for i in instances
            )
            reviews = [
                r["answer"]["notes"]["archive_review"]
                for r in readers.values()
                if r["answer"]["notes"].get("archive_review", {}).get("role_scope") == ["user"]
            ]
            contexts = [r["answer"]["context_tokens"] for r in readers.values()]
            records = UsageTracker._load_records(up)
            all_records.extend(records)
            cached = sum(
                bool((g["verdict"].get("details") or {}).get("judge_cache"))
                for g in grades.values()
            )
            runs.append(
                {
                    "repeat": rep,
                    "fresh_answers": len(readers),
                    "correct": correct,
                    "grades": len(grades),
                    "fresh_grades": len(grades) - cached,
                    "cached_grades": cached,
                    "user_archive_reviews": len(reviews),
                    "complete_user_archive_reviews": sum(r["complete"] for r in reviews),
                    "max_source_context_tokens": max(contexts),
                    "whole_question_answerer_requests": sum(
                        r["answer"]["notes"]["whole_question_usage"]["requests"]
                        for r in readers.values()
                    ),
                }
            )
            inputs.update({str(p.relative_to(REPO)): sha256_file(p) for p in (rp, gp, up)})
        arms[arm] = {
            "runs": runs,
            "answering_resources_by_model": model_workload(all_records, "answerer"),
            "grading_resources_by_model": model_workload(all_records, "judge"),
        }
    result = {
        "experiment_class": "exposed-dev100-regression-not-unseen-final",
        "primary_gate": gate,
        "arms": arms,
        "inputs": inputs,
        "generator_sha256": sha256_file(Path(__file__)),
        "end_to_end_query_latency": "not captured in this namespace; do not infer from API latency",
        "failed_request_tokens": "provider may omit tokens on failures; unknown, not free",
        "financial_cost": "not inferred from token totals",
        "provider_calls": 0,
        "summary_api_calls_in_this_experiment": 0,
    }
    out = analysis / f"{STEM}.report.json"
    if out.exists() and json.loads(out.read_text(encoding="utf-8")) != result:
        raise ValueError("completed regression report changed")
    if not out.exists():
        save_json(out, result)
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = report()
    print(json.dumps({"pass": result["primary_gate"]["pass"], "arms": result["arms"]}))
