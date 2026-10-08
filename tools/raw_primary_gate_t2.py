"""Evaluate the registered rules of `results/prereg-raw-primary-count-latest-v1.md`.

    python tools/raw_primary_gate_t2.py

Three runs of t1 and t2 on heldout100; t2 is t1 plus answer policy `v2_cl`, so the
comparison isolates that policy. Arms are compared on each question's mean correctness
over its three runs; the primary rule is on the two types the policy targets together.
Refuses to decide on an incomplete run. Exit 0 on PASS, 1 on STOP. No calls.
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
REPS = (1, 2, 3)
V1 = [f"two_stage_raw_primary_t1.heldout100-t1-rep{r}" for r in REPS]
V2 = [f"two_stage_raw_primary_t2.heldout100-t1-rep{r}" for r in REPS]
QUESTIONS = 100
TARGETS = ("multi-session", "knowledge-update")
TARGET = "multi-session+knowledge-update"
TARGET_FLOOR = 2.0
MEAN_NET_FLOOR = 0.0
TYPE_LOSS_LIMIT = 2.0
CONTEXT_LIMIT = 6000


def load(name: str) -> dict[str, dict]:
    path = RAW / f"{name}.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["question_id"]: row for row in rows}


def evaluate() -> dict:
    from llm_long_term_memory.stats import exact_mcnemar

    runs = {name: load(name) for name in (*V1, *V2)}
    incomplete = {name: len(rows) for name, rows in runs.items() if len(rows) < QUESTIONS}
    if incomplete:
        raise SystemExit(f"incomplete runs, no rule is evaluated: {incomplete}")
    ids = sorted(runs[V1[0]])
    for name, rows in runs.items():
        if sorted(rows) != ids:
            raise SystemExit(f"{name} covers different questions")
    qtype = {q: runs[V1[0]][q]["question_type"] for q in ids}

    def mean(arm, q):
        return sum(bool(runs[name][q]["correct"]) for name in arm) / len(arm)

    diff = {q: mean(V2, q) - mean(V1, q) for q in ids}
    mean_net = sum(diff.values())
    by_type = {t: sum(diff[q] for q in ids if qtype[q] == t) for t in sorted(set(qtype.values()))}
    target_net = sum(by_type.get(t, 0.0) for t in TARGETS)
    context = statistics.median(runs[name][q]["context_tokens"] for name in V2 for q in ids)
    output = {
        arm: statistics.median(runs[name][q]["output_tokens"] for name in names for q in ids)
        for arm, names in (("t1", V1), ("t2", V2))
    }
    second_calls = {
        arm: sum(1 for name in names for q in ids if runs[name][q]["notes"].get("fallback_turns"))
        / len(names)
        for arm, names in (("t1", V1), ("t2", V2))
    }
    rules = {
        "target_types_mean_net_at_least_2": target_net >= TARGET_FLOOR,
        "overall_mean_net_not_negative": mean_net >= MEAN_NET_FLOOR,
        "no_type_loses_more_than_2": min(by_type.values()) >= -TYPE_LOSS_LIMIT,
        "median_total_context_at_most_6000": context <= CONTEXT_LIMIT,
    }
    up = sum(1 for d in diff.values() if d > 0)
    down = sum(1 for d in diff.values() if d < 0)
    return {
        "scores": {name: sum(bool(r["correct"]) for r in runs[name].values()) for name in runs},
        "mean_scores": {
            "t1": statistics.mean(sum(bool(r["correct"]) for r in runs[n].values()) for n in V1),
            "t2": statistics.mean(sum(bool(r["correct"]) for r in runs[n].values()) for n in V2),
        },
        "mean_net": mean_net,
        "target_mean_net": target_net,
        "by_type_mean_net": by_type,
        "type_counts": dict(Counter(qtype.values())),
        "median_context_t2": context,
        "median_output_tokens": output,
        "fallback_second_calls_per_run": second_calls,
        "questions_up_down": {"up": up, "down": down},
        "sign_test_p": exact_mcnemar(up, down),
        "rules": rules,
        "pass": all(rules.values()),
    }


def render(result: dict) -> str:
    lines = [
        "# Raw-primary t2 (count notes, latest value) against t1 on heldout100 — registered gate",
        "",
        "> Regression-class evidence on an exposed set. Rules as registered in",
        "> `results/prereg-raw-primary-count-latest-v1.md`; not an accuracy claim.",
        "",
        "| run | correct / 100 |",
        "|---|---:|",
    ]
    lines += [f"| `{name}` | {score} |" for name, score in result["scores"].items()]
    m = result["mean_scores"]
    lines += [
        "",
        f"Mean over three runs: t1 {m['t1']:.2f}, t2 {m['t2']:.2f}. "
        f"Mean net {result['mean_net']:+.2f} (floor {MEAN_NET_FLOOR:+.0f}); "
        f"**{TARGET} {result['target_mean_net']:+.2f}** (floor +{TARGET_FLOOR:.0f}).",
        "",
        "| question type | n | mean net |",
        "|---|---:|---:|",
    ]
    for t, v in result["by_type_mean_net"].items():
        lines.append(f"| {t} | {result['type_counts'][t]} | {v:+.2f} |")
    o, f = result["median_output_tokens"], result["fallback_second_calls_per_run"]
    lines += [
        "",
        f"Median total context of t2: {result['median_context_t2']:,.0f} "
        f"(limit {CONTEXT_LIMIT:,}). "
        f"Median output tokens: t1 {o['t1']:,.0f}, t2 {o['t2']:,.0f}. "
        f"Fallback second calls per run: t1 {f['t1']:.1f}, t2 {f['t2']:.1f}.",
        "",
        "## Rules",
        "",
    ]
    lines += [f"- {'PASS' if ok else 'FAIL'} — `{name}`" for name, ok in result["rules"].items()]
    ud = result["questions_up_down"]
    lines += [
        "",
        f"**Decision: {'PASS' if result['pass'] else 'STOP'}**",
        "",
        f"Descriptive: questions whose mean moved up {ud['up']}, down {ud['down']}; "
        f"exact sign test p = {result['sign_test_p']:.4f} (no significance claimed).",
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
