"""Complete a new auditable acceptance view; never rewrites the incomplete v9 run."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from grounded_reader_diagnostic import grade_saved  # noqa: E402
from judge_request_cache import ExactRequestJudge  # noqa: E402
from prerequisite_reader_report import report  # noqa: E402


def main():
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.datasets.longmemeval import load
    from llm_long_term_memory.evaluation.judge import Judge

    stem = "prerequisite-cache-acceptance-v1"
    output = REPO / f"results/analysis/{stem}.json"
    if output.exists():
        raise ValueError("acceptance namespace exists")
    replay_path = REPO / "results/analysis/prerequisite-reader-v18-v9.judge-identity-replay-v2.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if not replay["known_mechanism_evidence_complete_with_historical_judge_replay"]:
        raise ValueError("historical identity replay did not pass")
    for path, expected in replay["inputs"].items():
        if sha256_file(REPO / path) != expected:
            raise ValueError("identity replay input changed")
    target = REPO / "results/raw/prerequisite-reader-v18-v9"
    inventory_path = target.with_suffix(".execution.json")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    for path, expected in inventory["files"].items():
        if sha256_file(REPO / path) != expected:
            raise ValueError("candidate frozen source changed")
    old19_path = REPO / "results/analysis/prerequisite-selection-v9.train19.json"
    old19 = json.loads(old19_path.read_text(encoding="utf-8"))
    if not old19["pass"] or old19["questions"] != 19:
        raise ValueError("original19 replay did not pass")
    instances = {i.question_id: i for i in load("s", Settings().data_dir)}
    judge = ExactRequestJudge(Judge(None, inventory["models"]["judge"]))
    qid = "eaca4986"
    paths = [
        Path(__file__),
        REPO / "tools/judge_request_cache.py",
        replay_path,
        old19_path,
        REPO / f"results/prereg-{stem}.md",
    ]
    for version in (2, 3, 6, 8):
        base = REPO / f"results/raw/prerequisite-reader-v18-v{version}"
        rp, gp = base.with_suffix(".readers.jsonl"), base.with_suffix(".grades.jsonl")
        judge.add(instances[qid], rows_by_question(rp)[qid], rows_by_question(gp)[qid], rp, gp)
        paths.extend([rp, gp])
    rp, gp = target.with_suffix(".readers.jsonl"), target.with_suffix(".grades.jsonl")
    readers, grades = rows_by_question(rp), rows_by_question(gp)
    if set(readers) - set(grades) != {qid} or len(grades) != 5:
        raise ValueError("exactly five fresh and one missing grade required")
    sink = io.StringIO()
    cached = grade_saved(judge, instances[qid], readers[qid], sink, sha256_file(inventory_path))
    if judge.hits != 1 or judge.misses:
        raise ValueError("completion must use one verified cache hit, zero provider calls")
    joined = REPO / f"results/raw/{stem}"
    for suffix in ("readers.jsonl", "execution.json", "usage.json"):
        joined.with_suffix(f".{suffix}").write_bytes(target.with_suffix(f".{suffix}").read_bytes())
    joined.with_suffix(".grades.jsonl").write_text(
        gp.read_text(encoding="utf-8") + sink.getvalue(), encoding="utf-8"
    )
    result = report(stem)
    result.update(
        {
            "acceptance_protocol": f"results/prereg-{stem}.md",
            "original_reader_label": "prerequisite-reader-v18-v9",
            "fresh_grade_count": 5,
            "cached_grade_count": 1,
            "new_provider_calls": 0,
            "original19_replay_pass": True,
            "cache_provenance": cached["verdict"]["details"]["judge_cache"],
            "formal_mechanism_acceptance_pass": result[
                "mechanism_pass_under_preregistered_five_plus_one"
            ]
            and result["source_hashes_current"],
            "addendum_inputs": {
                str(p.relative_to(REPO)): sha256_file(p) for p in [*paths, rp, gp, inventory_path]
            },
        }
    )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "formal_mechanism_acceptance_pass",
                    "correct",
                    "fresh_grade_count",
                    "cached_grade_count",
                    "new_provider_calls",
                    "source_hashes_current",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
