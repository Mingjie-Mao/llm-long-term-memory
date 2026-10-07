"""Zero-call comparison with every completed earlier six-question judge run.

Never inserts a synthetic fresh grade into the unfinished v9 experiment.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from grade_saved_recovering import validate  # noqa: E402
from prerequisite_reader_report import report  # noqa: E402


def main():
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.datasets.longmemeval import load
    from llm_long_term_memory.evaluation.judge import JUDGE_SYSTEM, Judge, Verdict

    qid = "eaca4986"
    target_label = "prerequisite-reader-v18-v9"
    target = report(target_label)
    settings = Settings()
    corpus = (settings.data_dir / "longmemeval_s_cleaned.json").resolve()
    instance = next(i for i in load("s", settings.data_dir) if i.question_id == qid)
    paths = [Path(__file__), corpus]
    matches = []
    target_reader = rows_by_question(REPO / f"results/raw/{target_label}.readers.jsonl")[qid]
    target_inventory = json.loads(
        (REPO / f"results/raw/{target_label}.execution.json").read_text(encoding="utf-8")
    )

    def request(reader, inventory):
        for file in ("src/llm_long_term_memory/evaluation/judge.py", "configs/fallback.yaml"):
            if inventory["files"][file] != target_inventory["files"][file]:
                raise ValueError("judge code/config changed; no identity replay")
        if inventory["files"][str(corpus.relative_to(REPO))] != sha256_file(corpus):
            raise ValueError("historical public corpus changed")
        return {
            "model": inventory["models"]["judge"],
            "system": JUDGE_SYSTEM,
            "prompt": Judge._prompt_for(
                instance.question,
                instance.answer,
                reader["answer"]["text"],
                instance.is_abstention,
                instance.question_type,
            ),
            "prompt_version": Judge.prompt_version,
            "schema": Verdict.model_json_schema(),
            "temperature": 0.0,
            "thinking": True,
        }

    expected = request(target_reader, target_inventory)
    # All completed earlier page-selector runs, including negative overall runs.
    # No selection of a favorable single scorer result.
    for version in (2, 3, 6, 8):
        label = f"prerequisite-reader-v18-v{version}"
        base = REPO / "results/raw" / label
        local = [
            base.with_suffix(f".{s}") for s in ("readers.jsonl", "grades.jsonl", "execution.json")
        ]
        paths.extend(local)
        reader, grade = rows_by_question(local[0])[qid], rows_by_question(local[1])[qid]
        inventory = json.loads(local[2].read_text(encoding="utf-8"))
        validate(reader, grade, qid)
        if reader["inventory_sha256"] != sha256_file(local[2]):
            raise ValueError("historical inventory mismatch")
        identical = request(reader, inventory) == expected
        if grade["judge_prompt_version"] != Judge.prompt_version:
            raise ValueError("historical scorer prompt changed")
        matches.append(
            {
                "label": label,
                "identical_judge_request": identical,
                "correct": grade["verdict"]["correct"],
                "grade_sha256": sha256_file(local[1]),
            }
        )
    exact_reference = target_reader["answer"]["text"] == instance.answer
    replay = bool(matches) and all(m["identical_judge_request"] and m["correct"] for m in matches)
    rows = target["rows"]
    complete_mechanism_evidence = (
        target["faithful_temporal_conflict_disclosure"]
        and replay
        and exact_reference
        and all(
            r["review_complete"]
            and r["context_bound"]
            and (not r["user_review"] or r["gold_raw_final_context_complete"])
            and (r["question_id"] in {qid, "gpt4_731e37d7"} or r["judged_correct"] is True)
            for r in rows
        )
    )
    result = {
        "experiment_class": "development-offline-identical-judge-request-replay",
        "provider_calls": 0,
        "target": target,
        "request_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
        "exact_reference_match": exact_reference,
        "all_historical_identical_requests_correct": replay,
        "historical_judge_runs": matches,
        "known_mechanism_evidence_complete_with_historical_judge_replay": (
            complete_mechanism_evidence
        ),
        "fresh_six_grade_gate_complete": False,
        "limitations": (
            "Original v9 has only five fresh grades; no synthetic sixth grade, "
            "no adoption or population accuracy claim."
        ),
        "inputs": {str(p.relative_to(REPO)): sha256_file(p) for p in paths},
    }
    output = REPO / "results/analysis/prerequisite-reader-v18-v9.judge-identity-replay-v2.json"
    if output.exists():
        raise ValueError("refusing to overwrite historical replay")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"target", "inputs", "historical_judge_runs"}
            }
        )
    )


if __name__ == "__main__":
    main()
