"""Zero-call gate for deciding whether v2d should change retrieval.

This reads the already frozen v2c confirmation rows.  It does not rerun questions,
inspect gold answer text, or call a model.  The purpose is only to distinguish losses
before ranking, during ranking/selection, and after evidence was already selected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import read_jsonl, write_report  # noqa: E402


def analyse(path: Path) -> dict:
    rows = read_jsonl(path)
    stages = ("candidates", "ranked", "selected")
    recalled = {
        stage: sum(row.get("notes", {}).get("recall_stages", {}).get(stage) is True for row in rows)
        for stage in stages
    }
    wrong = [row for row in rows if not row.get("correct")]
    wrong_with_selected = [
        row
        for row in wrong
        if row.get("notes", {}).get("recall_stages", {}).get("selected") is True
    ]
    wrong_without_selected = [
        row
        for row in wrong
        if row.get("notes", {}).get("recall_stages", {}).get("selected") is False
    ]
    correct_without_selected = [
        row
        for row in rows
        if row.get("correct")
        and row.get("notes", {}).get("recall_stages", {}).get("selected") is False
    ]
    broad_retrieval = (
        len(wrong_without_selected) >= 3
        or recalled["ranked"] < recalled["candidates"]
        or recalled["selected"] < recalled["ranked"]
    )
    return {
        "source": str(path),
        "questions": len(rows),
        "wrong": len(wrong),
        "recalled": recalled,
        "ranking_losses": recalled["candidates"] - recalled["ranked"],
        "selection_losses": recalled["ranked"] - recalled["selected"],
        "wrong_with_selected_evidence": len(wrong_with_selected),
        "wrong_without_selected_evidence": len(wrong_without_selected),
        "wrong_without_selected_ids": [row["question_id"] for row in wrong_without_selected],
        "correct_without_selected_evidence": len(correct_without_selected),
        "maximum_accuracy_gain_if_every_retrieval_miss_were_fixed": (
            len(wrong_without_selected) / len(rows) if rows else 0.0
        ),
        "decision": "ADD_BROAD_RETRIEVAL" if broad_retrieval else "KEEP_RETRIEVAL_FIXED",
        "reason": (
            "Most wrong answers already had a gold source session selected; no evidence was "
            "lost between candidate generation, ranking, and selection."
            if not broad_retrieval
            else "Multiple errors or stage losses are attributable to retrieval."
        ),
    }


def render(result: dict) -> str:
    q = result["questions"]

    def pct(n: int) -> str:
        return f"{100 * n / q:.1f}%" if q else "0.0%"

    return "\n".join(
        [
            "# v2d retrieval gate (zero model calls)",
            "",
            f"Decision: **{result['decision']}**",
            "",
            f"- candidate / ranked / selected source recall: "
            f"{result['recalled']['candidates']}/{q}, "
            f"{result['recalled']['ranked']}/{q}, "
            f"{result['recalled']['selected']}/{q}",
            f"- losses introduced by ranking: {result['ranking_losses']}",
            f"- losses introduced by selection: {result['selection_losses']}",
            f"- wrong answers with source evidence already selected: "
            f"{result['wrong_with_selected_evidence']}/{result['wrong']}",
            f"- wrong answers without selected source evidence: "
            f"{result['wrong_without_selected_evidence']}/{result['wrong']} "
            f"({', '.join(result['wrong_without_selected_ids']) or 'none'})",
            f"- optimistic ceiling from fixing every observed retrieval miss: "
            f"+{pct(result['wrong_without_selected_evidence'])}",
            "",
            result["reason"],
            "",
            "Therefore v2d does not add broad top-k, reranking, relation-scan, or raw-context "
            "expansion. The existing conditional archive fallback remains available. A new "
            "retrieval mechanism should be reconsidered only on a separately frozen set if "
            "candidate or ranked recall becomes a repeated failure mode.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        type=Path,
        default=Path("results/raw/two_stage_v2c.reasoning48-v2c8.jsonl"),
    )
    parser.add_argument(
        "--json-out", type=Path, default=Path("results/analysis/v2d-retrieval-gate.json")
    )
    parser.add_argument(
        "--md-out", type=Path, default=Path("results/analysis/v2d-retrieval-gate.md")
    )
    args = parser.parse_args()
    result = analyse(args.rows)
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")


if __name__ == "__main__":
    main()
