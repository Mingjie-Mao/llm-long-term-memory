"""Grade saved answers with durable bounded recovery, in a NEW result namespace.

Default is a zero-call preview. Never resumes a stopped reader experiment or reruns
readers. The caller preregisters the grading policy before --execute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]

from analysis_io import rows_by_question, sha256_file  # noqa: E402
from grounded_reader_diagnostic import grade_saved  # noqa: E402

from llm_long_term_memory.evaluation.recovery import RecoveringJudge  # noqa: E402


def validate(reader, grade, qid):
    if reader["question_id"] != qid:
        raise ValueError("reader question mismatch")
    if grade is None:
        return
    if (
        grade["question_id"] != qid
        or grade["inventory_sha256"] != reader["inventory_sha256"]
        or grade["cohort"] != reader["cohort"]
        or type(grade["verdict"]["correct"]) is not bool
        or grade["reader_row_sha256"]
        != hashlib.sha256(json.dumps(reader, sort_keys=True).encode()).hexdigest()
    ):
        raise ValueError("saved grade is not bound to the exact original reader")


def grade_pending(judge, instances, readers, grades, sink, checkpoint=lambda: None):
    if not set(grades) <= set(readers) or not set(readers) <= set(instances):
        raise ValueError("foreign reader/grade question")
    for qid, reader in readers.items():
        validate(reader, grades.get(qid), qid)
    for qid, reader in readers.items():
        if qid in grades:
            continue
        grades[qid] = grade_saved(judge, instances[qid], reader, sink, reader["inventory_sha256"])
        checkpoint()
    return grades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--readers", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-recoveries", type=int, default=3)
    parser.add_argument("--max-wait-seconds", type=int, default=180)
    args = parser.parse_args()
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.label):
        raise SystemExit("label must be a new plain result namespace")
    readers = rows_by_question(args.readers)
    if not args.execute:
        print(json.dumps({"provider_calls": 0, "saved_answers": len(readers), "reader_calls": 0}))
        return 0
    from llm_long_term_memory.config import ExperimentConfig, Settings
    from llm_long_term_memory.evaluation.datasets.longmemeval import load
    from llm_long_term_memory.evaluation.harness import _exclusive
    from llm_long_term_memory.evaluation.judge import Judge
    from llm_long_term_memory.llm import Limits, QuotaManager
    from llm_long_term_memory.llm.client import GeminiClient
    from llm_long_term_memory.llm.usage import UsageTracker

    # This execution surface is deliberately confined to the public, already exposed
    # corpus. Reading labels for grading never feeds them back to a product reader.
    settings = Settings()
    source = settings.data_dir / "longmemeval_s_cleaned.json"
    if sha256_file(source) != "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442":
        raise ValueError("public corpus identity mismatch")
    cfg = ExperimentConfig.from_yaml("configs/fallback.yaml")
    # Bind to already audited public reader artifacts, not arbitrary uploaded text.
    relative_reader = str(args.readers.resolve().relative_to(REPO))
    acceptance = json.loads(
        (REPO / "results/analysis/grounded-reader-v15.acceptance-final.json").read_text(
            encoding="utf-8"
        )
    )
    stopped = json.loads(
        (
            REPO
            / "results/analysis/grounded-paired-dev100-v15.stopped-for-prerequisite-repair.json"
        ).read_text(encoding="utf-8")
    )
    allowed_hash = acceptance["inputs"].get(relative_reader)
    if allowed_hash is None:
        allowed_hash = stopped["artifacts"].get(relative_reader, {}).get("sha256")
    if allowed_hash != sha256_file(args.readers):
        raise ValueError("saved reader must have immutable public-source audit provenance")
    inventory = {
        "readers": sha256_file(args.readers),
        "prereg": sha256_file(args.prereg),
        "config": sha256_file(REPO / "configs/fallback.yaml"),
        "models": cfg.models.model_dump(),
        "source": sha256_file(source),
        "max_recoveries": args.max_recoveries,
        "max_wait_seconds": args.max_wait_seconds,
        "tool": sha256_file(Path(__file__)),
        "recovery": sha256_file(REPO / "src/llm_long_term_memory/evaluation/recovery.py"),
        "grade_saved": sha256_file(REPO / "tools/grounded_reader_diagnostic.py"),
    }
    base = REPO / f"results/raw/{args.label}"
    inventory_path = base.with_suffix(".execution.json")
    grades_path = base.with_suffix(".grades.jsonl")
    usage_path = base.with_suffix(".usage.json")
    with _exclusive(grades_path):
        if inventory_path.exists():
            if json.loads(inventory_path.read_text(encoding="utf-8")) != inventory:
                raise ValueError("grading execution identity changed")
        else:
            if grades_path.exists() or usage_path.exists():
                raise ValueError("unbound existing output namespace")
            inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
        usage = UsageTracker()
        if usage_path.exists():
            usage.records[:] = UsageTracker._load_records(usage_path)
        grades = rows_by_question(grades_path) if grades_path.exists() else {}
        if grades and not usage_path.exists():
            raise ValueError("missing usage on resume")
        quota = QuotaManager(
            state_dir=settings.store_dir / "quota",
            default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
        )
        quota.load_learned()
        client = GeminiClient(settings.require_api_key(), quota=quota, usage=usage)
        judge = RecoveringJudge(
            Judge(client, cfg.models.judge),
            usage,
            usage_path,
            base.with_suffix(".recovery.json"),
            sha256_file(inventory_path),
            max_recoveries=args.max_recoveries,
            max_wait_seconds=args.max_wait_seconds,
        )
        instances = {i.question_id: i for i in load("s", settings.data_dir)}
        with grades_path.open("a", encoding="utf-8") as sink:
            grade_pending(judge, instances, readers, grades, sink, lambda: usage.save(usage_path))
        print(
            json.dumps(
                {
                    "saved_answers": len(readers),
                    "grades": len(grades),
                    "reader_calls": 0,
                    "usage": usage.summary(),
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
