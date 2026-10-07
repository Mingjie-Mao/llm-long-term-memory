"""Evaluate the registered rules of `results/prereg-raw-primary-heldout100-v1.md`.

    python tools/raw_primary_gate.py

Reads the four control runs and the two candidate runs from `results/raw/`, applies the
rules exactly as registered, and refuses to decide on an incomplete run. No calls.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import write_report  # noqa: E402

RAW = REPO / "results" / "raw"
CONTROLS = (
    "two_stage_hydrated.heldout100",
    "two_stage_hydrated.heldout100-rep2",
    "two_stage_hydrated.heldout100-rep3",
)
FRESH = "two_stage_hydrated.heldout100-rep4"
R = "two_stage_raw_primary.raw-primary-v1"
RAW_ONLY = "two_stage_raw_only.raw-primary-v1"
DRIFT_RANGE = (67, 76)
NET_FLOOR = 5
TYPE_LOSS_LIMIT = 2
CONTEXT_LIMIT = 5500
STATE_TYPES = ("knowledge-update", "temporal-reasoning")


def load(name: str) -> dict[str, dict]:
    path = RAW / f"{name}.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["question_id"]: row for row in rows}


def net(a: dict[str, bool], b: dict[str, bool], ids) -> dict:
    wins = sum(1 for q in ids if a[q] and not b[q])
    losses = sum(1 for q in ids if b[q] and not a[q])
    return {"wins": wins, "losses": losses, "net": wins - losses}


def evaluate() -> dict:
    from llm_long_term_memory.stats import exact_mcnemar

    runs = {name: load(name) for name in (*CONTROLS, FRESH, R, RAW_ONLY)}
    incomplete = {name: len(rows) for name, rows in runs.items() if len(rows) < 100}
    if incomplete:
        raise SystemExit(f"incomplete runs, no rule is evaluated: {incomplete}")
    ids = sorted(runs[R])
    for name, rows in runs.items():
        if sorted(rows) != ids:
            raise SystemExit(f"{name} covers different questions")
    qtype = {q: runs[R][q]["question_type"] for q in ids}
    correct = {name: {q: bool(rows[q]["correct"]) for q in ids} for name, rows in runs.items()}

    fresh_score = sum(correct[FRESH].values())
    drift = not (DRIFT_RANGE[0] <= fresh_score <= DRIFT_RANGE[1])
    control_runs = [FRESH] if drift else [*CONTROLS, FRESH]
    votes = {q: sum(correct[name][q] for name in control_runs) for q in ids}
    # Majority over the control runs; a tie counts as wrong for the control.
    control = {q: votes[q] * 2 > len(control_runs) for q in ids}
    ties = [q for q in ids if votes[q] * 2 == len(control_runs)]

    primary = net(correct[R], control, ids)
    by_type = {
        t: net(correct[R], control, [q for q in ids if qtype[q] == t])
        for t in sorted(set(qtype.values()))
    }
    worst_type_net = min(v["net"] for v in by_type.values())
    median_context = {
        name: statistics.median(runs[name][q]["context_tokens"] for q in ids)
        for name in (FRESH, R, RAW_ONLY)
    }
    rules = {
        "net_at_least_5": primary["net"] >= NET_FLOOR,
        "no_type_loses_more_than_2": worst_type_net >= -TYPE_LOSS_LIMIT,
        "median_context_at_most_5500": median_context[R] <= CONTEXT_LIMIT,
    }
    vs_fresh = net(correct[R], correct[FRESH], ids)
    secondary = net(correct[R], correct[RAW_ONLY], ids)
    state_ids = [q for q in ids if qtype[q] in STATE_TYPES]
    return {
        "scores": {name: sum(correct[name].values()) for name in runs},
        "drift": {"fresh": fresh_score, "range": DRIFT_RANGE, "declared": drift},
        "control_runs_used": control_runs,
        "control_majority_score": sum(control.values()),
        "control_ties": ties,
        "primary_R_vs_control": primary,
        "by_type_R_vs_control": by_type,
        "median_context_tokens": median_context,
        "rules": rules,
        "pass": all(rules.values()),
        "descriptive_R_vs_fresh_control": {
            **vs_fresh,
            "exact_mcnemar_p": exact_mcnemar(vs_fresh["wins"], vs_fresh["losses"]),
        },
        "secondary_R_vs_O": secondary,
        "secondary_R_vs_O_state_types": net(correct[R], correct[RAW_ONLY], state_ids),
        "type_counts": dict(Counter(qtype.values())),
        "flipped_R_vs_control": sorted(q for q in ids if correct[R][q] != control[q]),
        "raw_primary_turns_median": statistics.median(
            runs[R][q]["notes"].get("raw_primary_turns", 0) for q in ids
        ),
    }


def render(result: dict) -> str:
    s = result["scores"]
    p = result["primary_R_vs_control"]
    lines = [
        "# Raw-primary on heldout100 — registered gate",
        "",
        "> Regression-class evidence on an exposed set. Rules as registered in",
        "> `results/prereg-raw-primary-heldout100-v1.md`; not an accuracy claim.",
        "",
        "## Scores (of 100)",
        "",
        "| run | correct |",
        "|---|---:|",
    ]
    lines += [f"| `{name}` | {score} |" for name, score in s.items()]
    d = result["drift"]
    lines += [
        "",
        f"**Drift check:** fresh control {d['fresh']} against {d['range'][0]}-{d['range'][1]} — "
        + (
            "**drift declared**, comparisons use the fresh run only"
            if d["declared"]
            else "no drift"
        ),
        f"Control majority over {len(result['control_runs_used'])} runs: "
        f"{result['control_majority_score']} correct; "
        f"ties counted wrong: {len(result['control_ties'])}.",
        "",
        "## Primary — R against the control majority",
        "",
        f"+{p['wins']} / −{p['losses']}, **net {p['net']:+d}** (floor +{NET_FLOOR})",  # noqa: RUF001
        "",
        "| question type | n | wins | losses | net |",
        "|---|---:|---:|---:|---:|",
    ]
    for t, v in result["by_type_R_vs_control"].items():
        lines.append(
            f"| {t} | {result['type_counts'][t]} | {v['wins']} | {v['losses']} | {v['net']:+d} |"
        )
    mc = result["median_context_tokens"]
    lines += [
        "",
        f"Median answer context: control {mc[FRESH]:,.0f}, R {mc[R]:,.0f}, "
        f"O {mc[RAW_ONLY]:,.0f} tokens "
        f"(R limit {CONTEXT_LIMIT:,}). Median question-found turns in R: "
        f"{result['raw_primary_turns_median']}.",
        "",
        "## Rules",
        "",
    ]
    lines += [f"- {'PASS' if ok else 'FAIL'} — `{name}`" for name, ok in result["rules"].items()]
    lines += ["", f"**Decision: {'PASS' if result['pass'] else 'STOP'}**", ""]
    f = result["descriptive_R_vs_fresh_control"]
    sec, st = result["secondary_R_vs_O"], result["secondary_R_vs_O_state_types"]
    lines += [
        "## Descriptive",
        "",
        f"- R against the fresh control alone: +{f['wins']} / −{f['losses']}, net "  # noqa: RUF001
        f"{f['net']:+d}, exact McNemar p = {f['exact_mcnemar_p']:.4f} (no significance claimed).",
        f"- Secondary, R against O: +{sec['wins']} / −{sec['losses']}, net {sec['net']:+d}; "  # noqa: RUF001
        f"on knowledge-update + temporal-reasoning: net {st['net']:+d}.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate()
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
