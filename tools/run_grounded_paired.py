"""Run the registered paired gate, retaining each reader before grading it."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]

from analysis_io import rows_by_question, sha256_file  # noqa: E402
from grounded_gate import evaluate  # noqa: E402
from grounded_reader_diagnostic import grade_saved, save_reader  # noqa: E402
from run_grounded_experiment import BASELINE, frozen_identity  # noqa: E402


def combine(instance, reader, grade, identity, arm):
    from llm_long_term_memory.evaluation.harness import QuestionResult

    if (
        reader["question_id"] != instance.question_id
        or grade["question_id"] != instance.question_id
        or reader["inventory_sha256"] != identity
        or grade["inventory_sha256"] != identity
        or reader["cohort"] != arm
        or grade["cohort"] != arm
        or grade["reader_row_sha256"]
        != hashlib.sha256(json.dumps(reader, sort_keys=True).encode()).hexdigest()
    ):
        raise ValueError("grade/reader identity mismatch")
    answer, verdict = reader["answer"], grade["verdict"]
    if type(verdict["correct"]) is not bool:
        raise ValueError("correctness must be boolean")
    expected = {"inventory_sha256": identity, "arm": arm, "answerer": "gemini-3.5-flash-lite"}
    if answer["notes"].get("execution_identity") != expected:
        raise ValueError("answer execution identity mismatch")
    return QuestionResult(
        question_id=instance.question_id,
        question_type=instance.question_type,
        is_abstention=instance.is_abstention,
        correct=verdict["correct"],
        hypothesis=answer["text"],
        gold=instance.answer,
        judge_reason=verdict["reason"],
        context_tokens=answer["context_tokens"],
        prompt_tokens=answer["prompt_tokens"],
        output_tokens=answer["output_tokens"],
        latency_ms=answer["latency_ms"],
        answer_prompt_version=reader["answer_prompt_version"],
        judge_prompt_version=grade["judge_prompt_version"],
        notes=answer["notes"],
    )


class BoundRunner:
    def __init__(self, runner, identity, arm):
        self.runner, self.identity, self.arm = runner, identity, arm
        self.answer_prompt_version = runner.answer_prompt_version

    def answer_request(self, request):
        answer = self.runner.answer_request(request)
        answer.notes["execution_identity"] = {
            "inventory_sha256": self.identity,
            "arm": self.arm,
            "answerer": self.runner.model,
        }
        return answer


def main():
    from dataclasses import asdict

    from llm_long_term_memory import cli
    from llm_long_term_memory.commands.common import manifest_instances
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.harness import _exclusive
    from llm_long_term_memory.evaluation.manifest import load_manifest, require_claim
    from llm_long_term_memory.llm.client import DailyQuotaExhausted
    from llm_long_term_memory.llm.usage import UsageTracker

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "provider_calls": 0,
                    "questions": 100,
                    "repeats_per_arm": 3,
                    "max_reader_calls_before_retries": 1200,
                    "judge_calls": 600,
                }
            )
        )
        return 0
    acceptance_path = REPO / "results/analysis/grounded-reader-v15.acceptance-final.json"
    acceptance = json.loads(acceptance_path.read_text())
    if not acceptance["pass"]:
        raise SystemExit("STOP: selection acceptance has not passed")
    for rel, sha in acceptance["inputs"].items():
        if sha256_file(REPO / rel) != sha:
            raise SystemExit("STOP: acceptance input changed")
    manifest_path = REPO / "results/manifests/dev100.json"
    manifest = load_manifest(manifest_path)
    require_claim(manifest, "regression")
    selected = {i.question_id: i for i in manifest_instances(manifest, Settings())}
    instances = [selected[qid] for qid in manifest.question_ids]
    assert len(instances) == 100
    offline_path = REPO / "results/analysis/grounded-context-v15-r2.train150.json"
    offline = json.loads(offline_path.read_text())
    if not all(offline["offline_gate"].values()) or offline[
        "candidate_source_sha256"
    ] != sha256_file(REPO / "src/llm_long_term_memory/runtime/grounded_answering.py"):
        raise SystemExit("STOP: candidate differs from passing offline replay")
    proof_path = REPO / "results/analysis/grounded-paired-dev100-v15.public-source-audit.json"
    proof = json.loads(proof_path.read_text())
    if proof["status"] != "PASS" or proof["store_sha256"] != sha256_file(REPO / "stores/dev100.db"):
        raise SystemExit("STOP: public source audit/store identity mismatch")
    snapshot = frozen_identity("paired-dev100", manifest_path, "dev100", 15)
    for path in [
        Path(__file__),
        REPO / "tools/grounded_reader_diagnostic.py",
        acceptance_path,
        proof_path,
        offline_path,
        REPO / "results/prereg-grounded-paired-dev100-v15.md",
    ]:
        snapshot["files"][str(path.relative_to(REPO))] = sha256_file(path)
    stem = "grounded-paired-dev100-v15"
    inventory = REPO / f"results/analysis/{stem}.execution.json"
    if inventory.exists():
        if json.loads(inventory.read_text()) != snapshot:
            raise SystemExit("STOP: paired execution identity changed")
    else:
        inventory.write_text(json.dumps(snapshot, indent=2) + "\n")
    identity = sha256_file(inventory)
    paths = {"baseline": [], "candidate": []}
    for rep in (1, 2, 3):
        for arm, variant in [
            ("baseline", BASELINE),
            ("candidate", "two_stage_raw_primary_grounded_v15"),
        ]:
            if any(sha256_file(REPO / rel) != sha for rel, sha in snapshot["files"].items()):
                raise SystemExit("STOP: paired source/config/data identity changed")
            label = f"{stem}-{arm}-rep{rep}"
            readers_path = REPO / f"results/raw/{label}.readers.jsonl"
            grades_path = REPO / f"results/raw/{label}.grades.jsonl"
            usage_path = REPO / f"results/raw/{label}.usage.json"
            combined_path = REPO / f"results/raw/{label}.jsonl"
            with _exclusive(readers_path):
                readers = rows_by_question(readers_path) if readers_path.exists() else {}
                grades = rows_by_question(grades_path) if grades_path.exists() else {}
                if (readers or grades) and not usage_path.exists():
                    raise SystemExit("STOP: missing usage on resume")
                for rows in [readers, grades]:
                    if not set(rows) <= set(selected) or any(
                        r["inventory_sha256"] != identity or r["cohort"] != arm
                        for r in rows.values()
                    ):
                        raise SystemExit("STOP: foreign rows/identity on resume")
                if not set(grades) <= set(readers):
                    raise SystemExit("STOP: grades without saved readers")
                for qid, grade in grades.items():
                    combine(selected[qid], readers[qid], grade, identity, arm)
                _, _, runner, judge, usage = cli._build(
                    variant, "configs/fallback.yaml", "dev100", read_only_store=True
                )
                if usage_path.exists():
                    usage.records[:] = UsageTracker._load_records(usage_path)
                try:
                    with readers_path.open("a") as sink:
                        for instance in instances:
                            if instance.question_id not in readers:
                                readers[instance.question_id] = save_reader(
                                    BoundRunner(runner, identity, arm),
                                    instance,
                                    sink,
                                    identity,
                                    arm,
                                )
                                usage.save(usage_path)
                                print(f"{arm} rep{rep}: readers {len(readers)}/100", flush=True)
                    with grades_path.open("a") as sink:
                        for instance in instances:
                            if instance.question_id not in grades:
                                grades[instance.question_id] = grade_saved(
                                    judge, instance, readers[instance.question_id], sink, identity
                                )
                                usage.save(usage_path)
                                print(f"{arm} rep{rep}: grades {len(grades)}/100", flush=True)
                    joined = [
                        asdict(
                            combine(i, readers[i.question_id], grades[i.question_id], identity, arm)
                        )
                        for i in instances
                    ]
                    if combined_path.exists():
                        existing = list(rows_by_question(combined_path).values())
                        if existing != joined:
                            raise ValueError("combined output changed")
                    else:
                        combined_path.write_text("".join(json.dumps(r) + "\n" for r in joined))
                    paths[arm].append(combined_path)
                except DailyQuotaExhausted as exc:
                    print(json.dumps({"status": "INCOMPLETE", "reason": str(exc)}), flush=True)
                    return 2
                finally:
                    usage.save(usage_path)
                    runner.store.close()
    result = evaluate(
        paths,
        {i.question_id: i.question_type for i in instances},
        identity,
        candidate_prompt="memory-grounded-v15",
    )
    result["execution_inventory"] = str(inventory.relative_to(REPO))
    output = REPO / f"results/analysis/{stem}.gate.json"
    if output.exists():
        raise SystemExit("STOP: gate already exists")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
