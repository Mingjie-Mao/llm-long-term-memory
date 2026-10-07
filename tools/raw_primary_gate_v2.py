"""Evaluate the registered rules of `results/prereg-raw-primary-train150-v2.md`.

    python tools/raw_primary_gate_v2.py

One control run and one candidate run on train150. Refuses to decide on an incomplete
run. Exit 0 on PASS, 1 on STOP. No calls. The heldout100 rules stay in
`raw_primary_gate.py`, unchanged.
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
CONTROL = "two_stage_hydrated.train150-raw-v2-control"
CANDIDATE = "two_stage_raw_primary.train150-raw-v2"
QUESTIONS = 150
NET_FLOOR = 8
TYPE_LOSS_LIMIT = 2
CONTEXT_LIMIT = 6000


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

    runs = {name: load(name) for name in (CONTROL, CANDIDATE)}
    incomplete = {name: len(rows) for name, rows in runs.items() if len(rows) < QUESTIONS}
    if incomplete:
        raise SystemExit(f"incomplete runs, no rule is evaluated: {incomplete}")
    ids = sorted(runs[CANDIDATE])
    if sorted(runs[CONTROL]) != ids:
        raise SystemExit("the two runs cover different questions")
    qtype = {q: runs[CANDIDATE][q]["question_type"] for q in ids}
    correct = {name: {q: bool(rows[q]["correct"]) for q in ids} for name, rows in runs.items()}

    primary = net(correct[CANDIDATE], correct[CONTROL], ids)
    by_type = {
        t: net(correct[CANDIDATE], correct[CONTROL], [q for q in ids if qtype[q] == t])
        for t in sorted(set(qtype.values()))
    }
    median_context = {
        name: statistics.median(runs[name][q]["context_tokens"] for q in ids) for name in runs
    }
    second_calls = {
        name: sum(1 for q in ids if runs[name][q]["notes"].get("fallback_turns")) for name in runs
    }
    rules = {
        "net_at_least_8": primary["net"] >= NET_FLOOR,
        "no_type_loses_more_than_2": min(v["net"] for v in by_type.values()) >= -TYPE_LOSS_LIMIT,
        "median_total_context_at_most_6000": median_context[CANDIDATE] <= CONTEXT_LIMIT,
    }
    return {
        "scores": {name: sum(correct[name].values()) for name in runs},
        "primary": primary,
        "by_type": by_type,
        "type_counts": dict(Counter(qtype.values())),
        "median_context_tokens": median_context,
        "fallback_second_calls": second_calls,
        "rules": rules,
        "pass": all(rules.values()),
        "exact_mcnemar_p": exact_mcnemar(primary["wins"], primary["losses"]),
    }


def render(result: dict) -> str:
    p = result["primary"]
    s = result["scores"]
    mc = result["median_context_tokens"]
    fb = result["fallback_second_calls"]
    lines = [
        "# Raw-primary on train150 — registered gate v2",
        "",
        "> Regression-class evidence (a development check). Rules as registered in",
        "> `results/prereg-raw-primary-train150-v2.md`; not an accuracy claim.",
        "",
        f"| run | correct / {QUESTIONS} | median context | fallback second calls |",
        "|---|---:|---:|---:|",
        f"| control `{CONTROL}` | {s[CONTROL]} | {mc[CONTROL]:,.0f} | {fb[CONTROL]} |",
        f"| R `{CANDIDATE}` | {s[CANDIDATE]} | {mc[CANDIDATE]:,.0f} | {fb[CANDIDATE]} |",
        "",
        f"**Primary:** R against control +{p['wins']} / -{p['losses']}, "
        f"**net {p['net']:+d}** (floor +{NET_FLOOR})",
        "",
        "| question type | n | wins | losses | net |",
        "|---|---:|---:|---:|---:|",
    ]
    for t, v in result["by_type"].items():
        lines.append(
            f"| {t} | {result['type_counts'][t]} | {v['wins']} | {v['losses']} | {v['net']:+d} |"
        )
    lines += ["", "## Rules", ""]
    lines += [f"- {'PASS' if ok else 'FAIL'} — `{name}`" for name, ok in result["rules"].items()]
    lines += [
        "",
        f"**Decision: {'PASS' if result['pass'] else 'STOP'}**",
        "",
        f"Descriptive: exact McNemar p = {result['exact_mcnemar_p']:.4f} "
        "(no significance claimed from a development run).",
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
