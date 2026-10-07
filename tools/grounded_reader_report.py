"""Summarize saved development readers and hash-bound grades without provider calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from analysis_io import rows_by_question, sha256_file, write_report

REPO = Path(__file__).resolve().parent.parent


def summarize(answers_path, grades_path, inventory_path, usage_path):
    answers = rows_by_question(answers_path)
    grades = rows_by_question(grades_path) if grades_path.exists() else {}
    identity = sha256_file(inventory_path)
    if not answers or not set(grades) <= set(answers):
        raise ValueError("empty answers or foreign grade ids")
    cohorts = defaultdict(lambda: {"readers": 0, "graded": 0, "correct": 0})
    rows = []
    for qid, row in answers.items():
        if row["inventory_sha256"] != identity:
            raise ValueError("reader identity mismatch")
        grade = grades.get(qid)
        if grade is not None and (
            grade["inventory_sha256"] != identity
            or grade["reader_row_sha256"]
            != hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
            or grade["cohort"] != row["cohort"]
        ):
            raise ValueError("grade does not match the saved reader")
        answer = row["answer"]
        calls = answer["notes"].get("grounded_calls", [])
        cohort = cohorts[row["cohort"]]
        cohort["readers"] += 1
        cohort["graded"] += grade is not None
        cohort["correct"] += bool(grade and grade["verdict"]["correct"])
        rows.append(
            {
                "question_id": qid,
                "cohort": row["cohort"],
                "answer": answer["text"],
                "correct": grade["verdict"]["correct"] if grade else None,
                "judge_reason": grade["verdict"]["reason"] if grade else None,
                "reader_passes": len(calls),
                "validation_causes": [c.get("calculation", {}).get("cause") for c in calls],
            }
        )
    return {
        "experiment_class": "development/regression; failure-enriched, not overall accuracy",
        "provider_calls": 0,
        "inventory_sha256": identity,
        "inputs": {
            str(p.relative_to(REPO)): sha256_file(p)
            for p in (answers_path, inventory_path, usage_path, grades_path)
            if p.exists()
        },
        "cohorts": dict(cohorts),
        "grading_complete_for_saved_readers": set(grades) == set(answers),
        "usage": json.loads(usage_path.read_text(encoding="utf-8"))["summary"],
        "rows": rows,
    }


def render(result):
    lines = [
        "# Grounded reader development / regression report",
        "",
        "Failure-enriched exposed train150; not overall or unseen-final accuracy.",
        "",
        "| cohort | saved readers | graded | correct |",
        "|---|---:|---:|---:|",
    ]
    for name, stats in result["cohorts"].items():
        lines.append(f"| {name} | {stats['readers']} | {stats['graded']} | {stats['correct']} |")
    lines.extend(
        [
            "",
            f"Grading complete for saved readers: {result['grading_complete_for_saved_readers']}.",
            f"Recorded provider usage: `{json.dumps(result['usage'])}`.",
            "",
            "| question | cohort | grade | answer |",
            "|---|---|---|---|",
        ]
    )
    for row in result["rows"]:
        answer = row["answer"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['question_id']} | {row['cohort']} | {row['correct']} | {answer} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    stem = f"grounded-reader-v{args.version}"
    output = REPO / f"results/analysis/{stem}.report.json"
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise SystemExit("refusing to overwrite an existing report")
    result = summarize(
        REPO / f"results/raw/{stem}.answers.jsonl",
        REPO / f"results/raw/{stem}.grades.jsonl",
        REPO / f"results/analysis/{stem}.execution.json",
        REPO / f"results/raw/{stem}.usage.json",
    )
    result["generator_sha256"] = sha256_file(Path(__file__))
    print(write_report(result, render, json_out=output, md_out=markdown))


if __name__ == "__main__":
    main()
