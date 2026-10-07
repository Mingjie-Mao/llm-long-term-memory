"""Report frozen six-question development gate; inspect exposed train150 only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]
from analysis_io import rows_by_question, sha256_file  # noqa: E402
from grade_saved_recovering import validate  # noqa: E402
from prerequisite_reader_gate import QIDS  # noqa: E402


def report(label):
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.datasets.longmemeval import load
    from llm_long_term_memory.llm.usage import UsageTracker
    from llm_long_term_memory.store import external_session_id

    base = REPO / f"results/raw/{label}"
    reader_path, grades_path = base.with_suffix(".readers.jsonl"), base.with_suffix(".grades.jsonl")
    readers = rows_by_question(reader_path) if reader_path.exists() else {}
    grades = rows_by_question(grades_path) if grades_path.exists() else {}
    inventory_path = base.with_suffix(".execution.json")
    identity = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not set(readers) <= set(QIDS) or not set(grades) <= set(readers):
        raise ValueError("foreign rows")
    digest = sha256_file(inventory_path)
    instances = {i.question_id: i for i in load("s", Settings().data_dir) if i.question_id in QIDS}
    rows = []
    for qid, reader in readers.items():
        if reader["inventory_sha256"] != digest:
            raise ValueError("reader identity changed")
        validate(reader, grades.get(qid), qid)
        instance = instances[qid]
        gold = {
            (s.session_id, n, hashlib.sha256(t.content.encode()).hexdigest())
            for s in instance.sessions
            for n, t in enumerate(s.turns)
            if t.has_answer
        }
        notes = reader["answer"]["notes"]
        review = notes["archive_review"]
        calls = notes.get("grounded_calls", [])
        sources = calls[-1]["evidence"]["sources"] if calls else []
        found = {
            (external_session_id(s["session_id"]), s["turn_index"], s["text_sha256"])
            for s in sources
            if s["kind"] == "raw"
        }
        routed = review.get("role_scope") == ["user"]
        coverage = bool(gold) and gold <= found
        page_contexts = [p["context_tokens"] for p in review["pages"]]
        rows.append(
            {
                "question_id": qid,
                "user_review": routed,
                "review_complete": review["complete"],
                "page_count": len(review["pages"]),
                "gold_raw_final_context_complete": coverage,
                "missing_gold_turns": [
                    {"session_id": s, "turn_index": n} for s, n, _ in sorted(gold - found)
                ],
                "context_bound": reader["answer"]["context_tokens"] <= 6000
                and max(page_contexts, default=0) <= 6000,
                "judged_correct": grades.get(qid, {}).get("verdict", {}).get("correct"),
                "answer": reader["answer"]["text"],
                "calculation": calls[-1].get("calculation") if calls else None,
                "evidence_gap": notes.get("evidence_gap"),
            }
        )
    from decimal import Decimal

    faithful_conflict = False
    conflict_id = "gpt4_731e37d7"
    if conflict_id in readers:
        row = next(r for r in rows if r["question_id"] == conflict_id)
        calculation = row["calculation"] or {}
        gap = row["evidence_gap"] or {}
        if calculation.get("mechanism") == "scoped_payment_conflict":
            original = {
                (s.session_id, n): t.content
                for s in instances[conflict_id].sessions
                for n, t in enumerate(s.turns)
            }
            sources = {
                s["id"]: s
                for s in readers[conflict_id]["answer"]["notes"]["grounded_calls"][-1]["evidence"][
                    "sources"
                ]
            }
            payments = calculation.get("payments", [])
            valid_quotes = bool(payments)
            for payment in payments:
                source = sources.get(payment["source"])
                if not source or source["kind"] != "raw" or source["role"] != "user":
                    valid_quotes = False
                    break
                text = original.get(
                    (external_session_id(source["session_id"]), source["turn_index"]), ""
                )
                if (
                    payment["quote"] not in text
                    or payment["unit"] + payment["amount"] not in payment["quote"]
                ):
                    valid_quotes = False
            faithful_conflict = (
                valid_quotes
                and sorted(p["amount"] for p in payments) == ["20", "200", "500"]
                and Decimal(calculation["reported_total"]) == Decimal("720")
                and not calculation.get("scope_complete", True)
                and sum(Decimal(p["amount"]) for p in payments)
                == Decimal(calculation["reported_total"])
                and set(gap.get("missing_fields", []))
                == {"event_date", "event_year", "report_period"}
                and "cannot confirm" in row["answer"]
                and "not assumed a corrected year" in row["answer"]
            )
    mechanism_pass = (
        len(readers) == len(grades) == 6
        and faithful_conflict
        and all(
            r["review_complete"]
            and r["context_bound"]
            and (not r["user_review"] or r["gold_raw_final_context_complete"])
            and (r["question_id"] == conflict_id or r["judged_correct"] is True)
            for r in rows
        )
    )
    usage = UsageTracker()
    usage.records[:] = UsageTracker._load_records(base.with_suffix(".usage.json"))
    return {
        "experiment_class": "exposed-train150-mechanism-development",
        "reader_label": label,
        "mechanism_pass_under_preregistered_five_plus_one": mechanism_pass,
        "faithful_temporal_conflict_disclosure": faithful_conflict,
        "supported_correct": sum(
            r["judged_correct"] is True for r in rows if r["question_id"] != conflict_id
        ),
        "supported_total": 5,
        "pass": len(readers) == len(grades) == 6
        and all(
            r["judged_correct"] is True
            and r["review_complete"]
            and r["context_bound"]
            and (not r["user_review"] or r["gold_raw_final_context_complete"])
            for r in rows
        ),
        "reader_count": len(readers),
        "grade_count": len(grades),
        "correct": sum(r["judged_correct"] is True for r in rows),
        "rows": rows,
        "usage": usage.summary(),
        "source_hashes_current": all(
            sha256_file(REPO / p) == sha for p, sha in identity["files"].items()
        ),
        "inputs": {
            str(p.relative_to(REPO)): sha256_file(p)
            for p in (reader_path, grades_path, inventory_path)
            if p.exists()
        },
        "limitations": (
            "Six original-history questions; no population accuracy estimate, "
            "no 4x QA, no unseen-final."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-label", required=True)
    args = parser.parse_args()
    output = REPO / f"results/analysis/{args.output_label}.json"
    if output.exists():
        raise ValueError("refusing to overwrite report")
    result = report(args.label)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "pass": result["pass"],
                "readers": result["reader_count"],
                "grades": result["grade_count"],
                "correct": result["correct"],
            }
        )
    )
