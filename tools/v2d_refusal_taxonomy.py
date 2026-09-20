"""Classify the deterministic refusals in a v2d-style run, without calling anything.

The gate16 decision hinged on nine refused calculations, and the question that decides
whether a v2e is worth running is *why* they were refused: an operand the model never
selected, a citation label it invented, a date or unit the parser rejected, or evidence
that the schema cannot express. Those need different repairs, and one of them — a
schema the model cannot fill — would mean the v2d design is wrong rather than
under-tuned.

Rows written before `v2d.compute` recorded a `cause` cannot be split this way: they
carry one sentence per refusal and `an item has no valid source label` covers two
different failures. This reports those rows as `unclassifiable_legacy_row` rather than
guessing, because a taxonomy that invents a breakdown is worse than one that admits a
gap.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import read_jsonl, write_report  # noqa: E402

# The sentences the pre-diagnostic runs recorded, and how many distinct causes each of
# them can actually stand for. Anything greater than one cannot be recovered from a row.
LEGACY_REASONS = {
    "an item has no valid source label": ("member_text_empty", "label_not_in_context"),
    "numeric operands are invalid": (
        "operand_arrays_misaligned",
        "numeric_unparseable",
        "label_not_in_context",
        "too_few_operands",
    ),
    "a date operand or citation is invalid": ("date_unparseable", "label_not_in_context"),
    "count operand arrays are misaligned": ("operand_arrays_misaligned",),
    "no members were supplied": ("no_members",),
    "the verdict named a missing operand": ("missing_field_named",),
    "operand units do not match": ("units_incompatible",),
    "percentage baseline is zero": ("percentage_baseline_zero",),
}


def _computation(row: dict) -> dict | None:
    notes = row.get("notes") or {}
    detail = notes.get("synthesis_computation") or notes.get("answer_calculation")
    return detail if isinstance(detail, dict) else None


def _refused(detail: dict) -> bool:
    """A refusal is a computation record that carries a reason instead of a result."""
    return "reason" in detail or "cause" in detail


# The keys `compute` writes when code, not the model, produced the number. A `lookup`
# record has none of them and is not a calculation at all, so counting it as one would
# inflate the very figure this file exists to report: the gate16 run had five records
# without a refusal, but only two of them are arithmetic.
RESULT_KEYS = ("result", "count", "days", "earlier", "correction")


def classify(row: dict) -> dict:
    detail = _computation(row)
    if detail is None:
        return {"outcome": "no_computation_record"}
    if not _refused(detail):
        computed = any(key in detail for key in RESULT_KEYS)
        return {
            "outcome": "computed" if computed else "no_calculation_requested",
            "operation": detail.get("operation"),
        }

    entry = {
        "outcome": "refused",
        "operation": detail.get("operation"),
        "reason": detail.get("reason", ""),
    }
    cause = detail.get("cause")
    if cause:
        entry["cause"] = cause
        entry["evidence"] = {
            key: detail[key]
            for key in (
                "index",
                "member",
                "value",
                "label_supplied",
                "label_resolved",
                "labels_available",
                "missing_field",
                "lengths",
                "endpoints",
                "units_supplied",
                "result_unit",
            )
            if key in detail
        }
        return entry

    candidates = LEGACY_REASONS.get(entry["reason"], ())
    if len(candidates) == 1:
        entry["cause"] = candidates[0]
    else:
        entry["cause"] = "unclassifiable_legacy_row"
        entry["could_be"] = list(candidates)
    return entry


def analyse(path: Path) -> dict:
    rows = read_jsonl(path)
    per_question = {}
    for row in rows:
        entry = classify(row)
        entry["correct"] = bool(row.get("correct"))
        per_question[row["question_id"]] = entry

    refusals = {qid: e for qid, e in per_question.items() if e["outcome"] == "refused"}
    unclassifiable = {
        qid: e for qid, e in refusals.items() if e["cause"] == "unclassifiable_legacy_row"
    }
    return {
        "rows": str(path),
        "questions": len(per_question),
        "computed": sum(e["outcome"] == "computed" for e in per_question.values()),
        "no_calculation_requested": sum(
            e["outcome"] == "no_calculation_requested" for e in per_question.values()
        ),
        "no_computation_record": sum(
            e["outcome"] == "no_computation_record" for e in per_question.values()
        ),
        "refused": len(refusals),
        "refused_and_wrong": sum(not e["correct"] for e in refusals.values()),
        "by_cause": dict(sorted(Counter(e["cause"] for e in refusals.values()).items())),
        "by_operation": dict(sorted(Counter(e["operation"] for e in refusals.values()).items())),
        "unclassifiable": len(unclassifiable),
        "diagnosable": len(refusals) - len(unclassifiable),
        "per_question": per_question,
    }


def render(result: dict) -> str:
    lines = [
        "# v2d refusal taxonomy",
        "",
        f"> Rows: `{result['rows']}`. No model calls; derived from the committed rows alone.",
        "",
        f"- questions: **{result['questions']}**",
        f"- code produced the number: **{result['computed']}**",
        f"- no calculation was requested (`lookup`): **{result['no_calculation_requested']}**",
        f"- no computation record at all: **{result['no_computation_record']}**",
        f"- refused, model prose kept: **{result['refused']}** "
        f"(of which wrong: {result['refused_and_wrong']})",
        f"- refusals classifiable from the row: **{result['diagnosable']}**; "
        f"unclassifiable: **{result['unclassifiable']}**",
        "",
        "## By cause",
        "",
        "| cause | refusals |",
        "|---|---:|",
    ]
    lines += [f"| `{cause}` | {n} |" for cause, n in result["by_cause"].items()]
    lines += ["", "## By operation", "", "| operation | refusals |", "|---|---:|"]
    lines += [f"| `{op}` | {n} |" for op, n in result["by_operation"].items()]
    if result["unclassifiable"]:
        lines += [
            "",
            "## Why some rows cannot be classified",
            "",
            "These rows were written before `v2d.compute` recorded a machine-readable",
            "`cause`. Each carries one sentence that stands for more than one failure, and",
            "the operands the model supplied were not kept, so the breakdown cannot be",
            "recovered without running the questions again.",
            "",
            "| question | reason recorded | could be |",
            "|---|---|---|",
        ]
        lines += [
            f"| `{qid}` | {e['reason']} | {', '.join(f'`{c}`' for c in e['could_be'])} |"
            for qid, e in sorted(result["per_question"].items())
            if e["outcome"] == "refused" and e["cause"] == "unclassifiable_legacy_row"
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows", type=Path, default=Path("results/raw/two_stage_v2d.v2d-gate16.jsonl")
    )
    parser.add_argument(
        "--json-out", type=Path, default=Path("results/analysis/v2d-refusal-taxonomy.json")
    )
    parser.add_argument(
        "--md-out", type=Path, default=Path("results/analysis/v2d-refusal-taxonomy.md")
    )
    args = parser.parse_args()
    result = analyse(args.rows)
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")


if __name__ == "__main__":
    main()
