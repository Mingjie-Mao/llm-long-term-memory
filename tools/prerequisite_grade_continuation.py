"""One preregistered supplemental grade; immutable original evidence, no readers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from grade_saved_recovering import validate  # noqa: E402
from grounded_reader_diagnostic import grade_saved  # noqa: E402

ORIGINAL = "prerequisite-reader-v18-v9"
LABEL = "prerequisite-grade-continuation-v1"
MISSING = "eaca4986"


def verify_rows(readers, grades):
    from prerequisite_reader_gate import QIDS

    if set(readers) != set(QIDS) or set(readers) - set(grades) != {MISSING}:
        raise ValueError("supplement is restricted to the one missing original grade")
    if not set(grades) < set(readers):
        raise ValueError("foreign supplemental input")
    for qid, reader in readers.items():
        validate(reader, grades.get(qid), qid)


def main():
    from llm_long_term_memory.config import ExperimentConfig, Settings
    from llm_long_term_memory.evaluation.datasets.longmemeval import load
    from llm_long_term_memory.evaluation.harness import _exclusive
    from llm_long_term_memory.evaluation.judge import Judge
    from llm_long_term_memory.llm import Limits, QuotaManager
    from llm_long_term_memory.llm.client import GeminiClient
    from llm_long_term_memory.llm.usage import UsageTracker

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    original = REPO / "results/raw" / ORIGINAL
    source_paths = {
        suffix: original.with_suffix(f".{suffix}")
        for suffix in (
            "execution.json",
            "readers.jsonl",
            "grades.jsonl",
            "usage.json",
            "recovery.json",
        )
    }
    inventory = json.loads(source_paths["execution.json"].read_text(encoding="utf-8"))
    audit_path = REPO / f"results/analysis/{ORIGINAL}.partial-report.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for suffix in ("execution.json", "readers.jsonl", "grades.jsonl"):
        path = source_paths[suffix]
        if audit["inputs"].get(str(path.relative_to(REPO))) != sha256_file(path):
            raise ValueError("audited original artifact changed")
    digest = sha256_file(source_paths["execution.json"])
    for path, expected in inventory["files"].items():
        if sha256_file(REPO / path) != expected:
            raise ValueError("original frozen input changed")
    readers = rows_by_question(source_paths["readers.jsonl"])
    grades = rows_by_question(source_paths["grades.jsonl"])
    verify_rows(readers, grades)
    if any(r["inventory_sha256"] != digest for r in readers.values()):
        raise ValueError("reader inventory changed")
    recovery = json.loads(source_paths["recovery.json"].read_text(encoding="utf-8"))
    if recovery["identity"] != digest or recovery["failures"] <= recovery["max_recoveries"]:
        raise ValueError("original recovery must be exhausted")
    settings = Settings()
    corpus = (settings.data_dir / "longmemeval_s_cleaned.json").resolve()
    if sha256_file(corpus) != inventory["files"][str(corpus.relative_to(REPO))]:
        raise ValueError("public corpus changed")
    identity = {
        "original_label": ORIGINAL,
        "missing_question": MISSING,
        "reader_calls": 0,
        "max_client_attempts": 1,
        "outer_recoveries": 0,
        "inputs": {str(p.relative_to(REPO)): sha256_file(p) for p in source_paths.values()},
        "tool": sha256_file(Path(__file__)),
        "prereg": sha256_file(REPO / f"results/prereg-{LABEL}.md"),
        "public_source_audit": sha256_file(audit_path),
    }
    preview = REPO / f"results/analysis/{LABEL}.preview.json"
    base = REPO / "results/raw" / LABEL
    usage_path = base.with_suffix(".usage.json")
    grades_path = base.with_suffix(".grades.jsonl")
    if not args.execute:
        if preview.exists():
            raise ValueError("preview already exists")
        preview.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"provider_calls": 0, **identity}))
        return
    if json.loads(preview.read_text(encoding="utf-8")) != identity:
        raise ValueError("supplement preview changed")
    with _exclusive(grades_path):
        attempted = base.with_suffix(".attempt.json")
        if attempted.exists() or usage_path.exists() or grades_path.exists():
            raise ValueError("supplement already attempted; no budget reset")
        cfg = ExperimentConfig.from_yaml(REPO / "configs/fallback.yaml")
        if cfg.models.model_dump() != inventory["models"]:
            raise ValueError("judge model changed")
        quota = QuotaManager(
            state_dir=settings.store_dir / "quota",
            default=Limits(rpm=cfg.quota.rpm, tpm=cfg.quota.tpm, rpd=cfg.quota.rpd),
        )
        quota.load_learned()
        usage = UsageTracker()
        client = GeminiClient(
            settings.require_api_key(),
            quota=quota,
            usage=usage,
            max_retries=1,
            max_transport_retries=1,
        )
        judge = Judge(client, cfg.models.judge)
        instance = next(i for i in load("s", settings.data_dir) if i.question_id == MISSING)
        attempted.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
        try:
            with grades_path.open("x", encoding="utf-8") as sink:
                supplemental = grade_saved(judge, instance, readers[MISSING], sink, digest)
            validate(readers[MISSING], supplemental, MISSING)
        finally:
            usage.save(usage_path)
        # Independent evidence view; original readers, five grades and failed usage
        # are never edited. Existing report generator verifies every row binding.
        joined = REPO / "results/raw" / f"{LABEL}-joined"
        joined.with_suffix(".readers.jsonl").write_bytes(source_paths["readers.jsonl"].read_bytes())
        joined.with_suffix(".execution.json").write_bytes(
            source_paths["execution.json"].read_bytes()
        )
        joined.with_suffix(".grades.jsonl").write_text(
            source_paths["grades.jsonl"].read_text(encoding="utf-8")
            + grades_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        combined = UsageTracker()
        combined.records[:] = UsageTracker._load_records(source_paths["usage.json"]) + usage.records
        combined.save(joined.with_suffix(".usage.json"))
        identity["supplemental_grade_sha256"] = sha256_file(grades_path)
        identity["supplemental_usage_sha256"] = sha256_file(usage_path)
        base.with_suffix(".join.json").write_text(
            json.dumps(identity, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {"reader_calls": 0, "grade": supplemental["verdict"], "usage": usage.summary()}
            )
        )


if __name__ == "__main__":
    main()
