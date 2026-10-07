"""Cost, freeze and optionally execute the fixed-reader candidate. Dry run by default."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import rows_by_question, sha256_file  # noqa: E402
from grounded_context_replay import BASELINE, CANDIDATE  # noqa: E402

SOURCES = [
    "src/llm_long_term_memory/runtime/grounded_answering.py",
    "src/llm_long_term_memory/runtime/answer_engine.py",
    "src/llm_long_term_memory/evaluation/runners/memory.py",
    "src/llm_long_term_memory/retrieve/excerpts.py",
    "src/llm_long_term_memory/cli.py",
    "src/llm_long_term_memory/answering.py",
    "src/llm_long_term_memory/evaluation/harness.py",
    "src/llm_long_term_memory/evaluation/judge.py",
    "tools/grounded_gate.py",
    "tools/run_grounded_experiment.py",
    "configs/fallback.yaml",
    "results/prereg-raw-primary-grounded-v1.md",
]


def frozen_identity(stage: str, manifest: Path, store: str, version: int = 1) -> dict:
    from llm_long_term_memory.config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml(REPO / "configs/fallback.yaml")
    if cfg.models.answerer != "gemini-3.5-flash-lite":
        raise ValueError("reader must remain gemini-3.5-flash-lite")
    files = [REPO / name for name in SOURCES] + [manifest]
    if version > 1:
        files.append(REPO / f"results/prereg-raw-primary-grounded-v{version}.md")
        files.extend((REPO / "results").glob(f"prereg-raw-primary-grounded-v{version}-*.md"))
    files.extend((REPO / "src").rglob("*.py"))
    files.extend((REPO / "data").glob("longmemeval_s*.json"))
    for stem in (store + "-index", store + "-turn-key-index"):
        files.extend(
            (REPO / "stores" / stem).with_suffix(suffix) for suffix in (".ids.json", ".npy")
        )
    files.append(REPO / "stores" / f"{store}.db")
    wal = REPO / "stores" / f"{store}.db-wal"
    if wal.exists() and wal.stat().st_size:
        files.append(wal)
    return {
        "stage": stage,
        "version": version,
        "experiment_class": "regression",
        "models": cfg.models.model_dump(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "python": platform.python_version(),
        "temperature": 0,
        "seed": "provider unsupported",
        "files": {str(path.relative_to(REPO)): sha256_file(path) for path in files},
    }


class RegisteredRunner:
    def __init__(self, runner, identity, arm):
        self.runner, self.identity, self.arm = runner, identity, arm

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def answer(self, instance):
        answer = self.runner.answer(instance)
        answer.notes["execution_identity"] = {
            "inventory_sha256": self.identity,
            "arm": self.arm,
            "answerer": self.runner.model,
        }
        return answer


def main() -> int:
    from grounded_gate import evaluate

    from llm_long_term_memory import cli
    from llm_long_term_memory.commands.common import manifest_instances
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.harness import run_eval
    from llm_long_term_memory.evaluation.manifest import load_manifest, require_claim

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("diagnostic", "dev100"), default="diagnostic")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--version",
        type=int,
        choices=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        default=15,
    )
    args = parser.parse_args()
    store = "train150" if args.stage == "diagnostic" else "dev100"
    manifest_path = (
        REPO
        / "results/manifests"
        / ("train150-raw-v1-reasoning-errors.json" if args.stage == "diagnostic" else "dev100.json")
    )
    manifest = load_manifest(manifest_path)
    require_claim(manifest, "regression")
    instances = manifest_instances(manifest, Settings())
    selected = {i.question_id: i for i in instances}
    instances = [selected[q] for q in manifest.question_ids]
    candidate = (
        CANDIDATE if args.version == 1 else f"two_stage_raw_primary_grounded_v{args.version}"
    )
    stages = (
        [("candidate", candidate, 1)]
        if args.stage == "diagnostic"
        else [
            (arm, variant, rep)
            for rep in (1, 2, 3)
            for arm, variant in (("baseline", BASELINE), ("candidate", candidate))
        ]
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "provider_calls": 0,
                    "questions": len(instances),
                    "reader_call_upper_bound": len(instances) * len(stages) * 2,
                    "judge_calls": len(instances) * len(stages),
                    "answerer": "gemini-3.5-flash-lite",
                }
            )
        )
        return 0
    offline = json.loads(
        (REPO / f"results/analysis/grounded-context-v{args.version}.train150.json").read_text(
            encoding="utf-8"
        )
    )
    if not all(offline["offline_gate"].values()):
        raise SystemExit("STOP: registered offline gates failed")
    if args.version >= 3:
        previous = json.loads(
            (
                REPO / f"results/analysis/grounded-context-v{args.version - 1}.train150.json"
            ).read_text(encoding="utf-8")
        )
        if (
            offline["all_gold_turn_coverage"]["candidate"]
            < previous["all_gold_turn_coverage"]["candidate"]
        ):
            raise SystemExit("STOP: offline coverage regressed from previous version")
    if offline["candidate_source_sha256"] != sha256_file(REPO / SOURCES[0]):
        raise SystemExit("STOP: candidate differs from its offline replay")
    snapshot = frozen_identity(args.stage, manifest_path, store, args.version)
    inventory = REPO / f"results/analysis/grounded-{args.stage}-v{args.version}.execution.json"
    if inventory.exists():
        if json.loads(inventory.read_text(encoding="utf-8")) != snapshot:
            raise SystemExit("STOP: execution inventory changed; use a newly registered namespace")
    else:
        inventory.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    identity = sha256_file(inventory)
    paths = {"baseline": [], "candidate": []}
    for arm, variant, rep in stages:
        if frozen_identity(args.stage, manifest_path, store, args.version) != snapshot:
            raise SystemExit("STOP: execution files changed between arms")
        label = f"grounded-{args.stage}-v{args.version}-rep{rep}"
        output = REPO / "results/raw" / f"{variant}.{label}.jsonl"
        if output.exists():
            for row in rows_by_question(output).values():
                if row["notes"].get("execution_identity", {}).get("inventory_sha256") != identity:
                    raise SystemExit("STOP: mixed execution identity on resume")
        _, _, runner, judge, usage = cli._build(
            variant, "configs/fallback.yaml", store, read_only_store=True
        )
        runner.name = f"{variant}.{label}"
        try:
            report = run_eval(
                RegisteredRunner(runner, identity, arm),
                judge,
                instances,
                output,
                usage=usage,
                on_progress=lambda result, report, arm=arm, rep=rep: print(
                    f"{arm} rep{rep}: {report.n}/{len(instances)}", flush=True
                ),
            )
            paths[arm].append(output)
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "rep": rep,
                        "completed": report.completed,
                        "correct": sum(r.correct for r in report.results),
                        "n": report.n,
                    }
                ),
                flush=True,
            )
            if not report.completed:
                return 2
        finally:
            runner.store.close()
    if frozen_identity(args.stage, manifest_path, store, args.version) != snapshot:
        raise SystemExit("STOP: execution files changed during the run")
    if args.stage == "dev100":
        result = evaluate(
            paths,
            {i.question_id: i.question_type for i in instances},
            identity,
            candidate_prompt=f"memory-grounded-v{args.version}",
        )
        result["execution_inventory"] = str(inventory.relative_to(REPO))
        out = REPO / f"results/analysis/grounded-dev100-v{args.version}.gate.json"
        if out.exists():
            raise SystemExit("STOP: gate artifact already exists")
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)
        return 0 if result["pass"] else 1
    else:
        rows = rows_by_question(paths["candidate"][0])
        if set(rows) != set(manifest.question_ids):
            raise SystemExit("STOP: diagnostic covers different questions")
        computed = sum(
            any(c.get("computed") for c in r["notes"].get("grounded_calls", []))
            for r in rows.values()
        )
        result = {
            "experiment_class": "failure-enriched-diagnostic",
            "questions": len(rows),
            "correct": sum(r["correct"] for r in rows.values()),
            "questions_with_validated_computation": computed,
            "provider_usage": usage.summary(),
            "execution_inventory": str(inventory.relative_to(REPO)),
        }
        out = REPO / f"results/analysis/grounded-diagnostic-v{args.version}.result.json"
        if out.exists():
            raise SystemExit("STOP: diagnostic result already exists")
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
