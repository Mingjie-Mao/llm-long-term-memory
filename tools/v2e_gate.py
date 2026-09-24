"""Judge the v2e run against the gates registered before it ran.

Paired against the committed v2c rows rather than a fresh baseline, which
`results/prereg-v2e-reasoning48.md` registers and explains: the two arms were measured
on different days, so this is a directional development comparison and no significance
is claimed.

Gate 3 needs something the rows do not store — whether the marker actually reached the
reader. The context is not recorded, but it is reproducible: the selected memory ids
are in `notes.retrieval`, the store is on disk, and the renderer is deterministic. So
the context is rebuilt here rather than assumed, because a mechanism that never reached
the prompt would otherwise be scored as if it had.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import rows_by_question, write_report  # noqa: E402

from llm_long_term_memory.stats import paired_outcomes  # noqa: E402

BASELINE = REPO / "results/raw/two_stage_v2c.reasoning48-v2c8.jsonl"
CANDIDATE = REPO / "results/raw/two_stage_v2e.v2e-reasoning48.jsonl"
STORE = REPO / "stores/v2e-reasoning48.db"
MARKER = "states no date"


def _marker_reached(rows: dict[str, dict]) -> dict[str, bool]:
    """Rebuild each question's context and look for the marker."""
    from llm_long_term_memory.evaluation.runners.memory import render_grouped
    from llm_long_term_memory.store import SQLiteMemoryStore

    store = SQLiteMemoryStore(STORE, read_only=True)
    store.initialize()
    try:
        seen = {}
        for qid, row in rows.items():
            memories = [
                memory
                for item in row["notes"]["retrieval"]
                if (memory := store.get(item["memory_id"])) is not None
            ]
            body = render_grouped(memories, True, True, None, True)
            seen[qid] = MARKER in body
        return seen
    finally:
        store.close()


def analyse() -> dict:
    baseline = rows_by_question(BASELINE)
    candidate = rows_by_question(CANDIDATE)
    ids = [qid for qid in baseline if qid in candidate]
    if len(ids) != len(baseline):
        raise SystemExit(
            f"the candidate is incomplete: {len(candidate)} of {len(baseline)} questions"
        )

    paired = paired_outcomes(
        {q: bool(baseline[q]["correct"]) for q in ids},
        {q: bool(candidate[q]["correct"]) for q in ids},
        ids,
    )
    marker = _marker_reached(candidate)
    temporal = [q for q in ids if baseline[q]["question_type"] == "temporal-reasoning"]
    temporal_losses = [
        q for q in temporal if baseline[q]["correct"] and not candidate[q]["correct"]
    ]
    temporal_without_marker = [q for q in temporal if not marker[q]]

    def median(values: list[int]) -> float:
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return (ordered[middle - 1] + ordered[middle]) / 2

    base_context = median([baseline[q]["context_tokens"] for q in ids])
    cand_context = median([candidate[q]["context_tokens"] for q in ids])
    drift = (cand_context - base_context) / base_context if base_context else 0.0

    by_type = {}
    for question_type in sorted({baseline[q]["question_type"] for q in ids}):
        group = [q for q in ids if baseline[q]["question_type"] == question_type]
        by_type[question_type] = {
            "n": len(group),
            "baseline": sum(baseline[q]["correct"] for q in group),
            "candidate": sum(candidate[q]["correct"] for q in group),
        }

    gates = {
        "net_positive": paired["net"] > 0,
        "no_temporal_regression": not temporal_losses,
        "marker_reached_every_temporal_question": not temporal_without_marker,
        "context_within_10_percent": abs(drift) <= 0.10,
    }
    return {
        "experiment": "v2e-reasoning48",
        "class": "development",
        "baseline_rows": str(BASELINE.relative_to(REPO)),
        "candidate_rows": str(CANDIDATE.relative_to(REPO)),
        "questions": len(ids),
        "baseline_correct": paired["old_correct"],
        "candidate_correct": paired["new_correct"],
        "wins": paired["wins"],
        "losses": paired["losses"],
        "ties": paired["ties"],
        "net": paired["net"],
        "exact_mcnemar_p": paired["exact_mcnemar_p"],
        "by_type": by_type,
        "median_context_baseline": base_context,
        "median_context_candidate": cand_context,
        "context_drift": drift,
        "marker_reached": sum(marker.values()),
        "temporal_questions": len(temporal),
        "temporal_losses": temporal_losses,
        "temporal_without_marker": temporal_without_marker,
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "STOP",
    }


def render(result: dict) -> str:
    lines = [
        "# v2e reasoning-48",
        "",
        "> **DEVELOPMENT RESULT.** The 48 questions have been read in full, twice. The",
        "> baseline is the committed v2c run rather than a fresh one, so the two arms were",
        "> measured on different days. Directional only; no significance is claimed.",
        "",
        f"Decision: **{result['decision']}**",
        "",
        f"- baseline: **{result['baseline_correct']}/{result['questions']}**",
        f"- candidate: **{result['candidate_correct']}/{result['questions']}**",
        f"- paired: **{len(result['wins'])} wins / {len(result['losses'])} losses / "
        f"{result['ties']} ties**, net **{result['net']:+d}**",
        f"- exact McNemar p = {result['exact_mcnemar_p']:.4f} "
        "(reported for shape, not as a claim — one run per arm)",
        f"- median context: {result['median_context_baseline']:.0f} -> "
        f"{result['median_context_candidate']:.0f} tokens "
        f"({result['context_drift']:+.1%})",
        f"- the marker reached **{result['marker_reached']}/{result['questions']}** contexts",
        "",
        "## Registered gates",
        "",
    ]
    lines += [
        f"- {'PASS' if passed else 'FAIL'} — `{name}`" for name, passed in result["gates"].items()
    ]
    lines += ["", "## By question type", "", "| type | n | v2c | v2e |", "|---|---:|---:|---:|"]
    lines += [
        f"| {name} | {row['n']} | {row['baseline']} | {row['candidate']} |"
        for name, row in result["by_type"].items()
    ]
    if result["wins"]:
        lines += ["", f"Wins: {', '.join(f'`{q}`' for q in result['wins'])}"]
    if result["losses"]:
        lines += ["", f"Losses: {', '.join(f'`{q}`' for q in result['losses'])}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=REPO / "results/analysis/v2e-gate.json")
    parser.add_argument("--md-out", type=Path, default=REPO / "results/analysis/v2e-gate.md")
    args = parser.parse_args()
    result = analyse()
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0 if result["decision"] == "PASS" else 1


if __name__ == "__main__":
    # `sys.exit`, not a bare call. `main` returns 1 on STOP and discarding it made the
    # process exit 0 while the report said the gate had closed — which is how a closed
    # gate gets walked through by anything reading the status rather than the prose.
    sys.exit(main())
