"""Evaluate the registered rules of `results/prereg-raw-primary-t3-t4-heldout100-v1.md`.

    python tools/raw_primary_gate_t3t4.py

Three runs each of v1, t3 and t4 on heldout100; arms are compared on each question's
mean correctness over its three runs. The decision is t4 against v1; t3 against v1
(answering) and t4 against t3 (retrieval) are registered readings that attribute it.
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
V1 = [f"two_stage_raw_primary.heldout100-t3t4-rep{r}" for r in REPS]
T3 = [f"two_stage_raw_primary_t3.heldout100-t3t4-rep{r}" for r in REPS]
V2 = [f"two_stage_raw_primary_t4.heldout100-t3t4-rep{r}" for r in REPS]
QUESTIONS = 100
MEAN_NET_FLOOR = 4.0
TYPE_LOSS_LIMIT = 2.0
CONTEXT_LIMIT = 6500
CONTRIBUTES = 2.0


def load(name: str) -> dict[str, dict]:
    path = RAW / f"{name}.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["question_id"]: row for row in rows}


def evaluate() -> dict:
    from llm_long_term_memory.stats import exact_mcnemar

    runs = {name: load(name) for name in (*V1, *T3, *V2)}
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
    context = statistics.median(runs[name][q]["context_tokens"] for name in V2 for q in ids)
    output = {
        arm: statistics.median(runs[name][q]["output_tokens"] for name in names for q in ids)
        for arm, names in (("v1", V1), ("t3", T3), ("t4", V2))
    }
    second_calls = {
        arm: sum(1 for name in names for q in ids if runs[name][q]["notes"].get("fallback_turns"))
        / len(names)
        for arm, names in (("v1", V1), ("t3", T3), ("t4", V2))
    }
    rules = {
        "t4_mean_net_at_least_4": mean_net >= MEAN_NET_FLOOR,
        "no_type_loses_more_than_2": min(by_type.values()) >= -TYPE_LOSS_LIMIT,
        "median_total_context_at_most_6500": context <= CONTEXT_LIMIT,
    }
    answering = sum(mean(T3, q) - mean(V1, q) for q in ids)
    retrieval = sum(mean(V2, q) - mean(T3, q) for q in ids)
    attribution = {
        "t3_vs_v1_mean_net": answering,
        "t4_vs_t3_mean_net": retrieval,
        "answering_contributes": answering >= CONTRIBUTES,
        "retrieval_contributes": retrieval >= CONTRIBUTES,
        "t3_by_type_vs_v1": {
            t: sum(mean(T3, q) - mean(V1, q) for q in ids if qtype[q] == t)
            for t in sorted(set(qtype.values()))
        },
        "median_context_t3": statistics.median(
            runs[name][q]["context_tokens"] for name in T3 for q in ids
        ),
    }
    up = sum(1 for d in diff.values() if d > 0)
    down = sum(1 for d in diff.values() if d < 0)
    return {
        "scores": {name: sum(bool(r["correct"]) for r in runs[name].values()) for name in runs},
        "mean_scores": {
            "v1": statistics.mean(sum(bool(r["correct"]) for r in runs[n].values()) for n in V1),
            "t3": statistics.mean(sum(bool(r["correct"]) for r in runs[n].values()) for n in T3),
            "t4": statistics.mean(sum(bool(r["correct"]) for r in runs[n].values()) for n in V2),
        },
        "mean_net": mean_net,
        "by_type_mean_net": by_type,
        "type_counts": dict(Counter(qtype.values())),
        "median_context_t4": context,
        "median_output_tokens": output,
        "fallback_second_calls_per_run": second_calls,
        "questions_up_down": {"up": up, "down": down},
        "sign_test_p": exact_mcnemar(up, down),
        "rules": rules,
        "attribution": attribution,
        "pass": all(rules.values()),
    }


def render(result: dict) -> str:
    lines = [
        "# Raw-primary t4 (and t3) against v1 on heldout100 — registered gate",
        "",
        "> Regression-class evidence on an exposed set. Rules as registered in",
        "> `results/prereg-raw-primary-t3-t4-heldout100-v1.md`; not an accuracy claim.",
        "",
        "| run | correct / 100 |",
        "|---|---:|",
    ]
    lines += [f"| `{name}` | {score} |" for name, score in result["scores"].items()]
    m = result["mean_scores"]
    lines += [
        "",
        f"Mean over three runs: v1 {m['v1']:.2f}, t3 {m['t3']:.2f}, t4 {m['t4']:.2f}. "
        f"**t4 mean net {result['mean_net']:+.2f}** (floor +{MEAN_NET_FLOOR:.0f}). "
        f"Attribution: t3 - v1 {result['attribution']['t3_vs_v1_mean_net']:+.2f}, "
        f"t4 - t3 {result['attribution']['t4_vs_t3_mean_net']:+.2f} "
        f"(contributes at +{CONTRIBUTES:.0f}).",
        "",
        "| question type | n | mean net |",
        "|---|---:|---:|",
    ]
    for t, v in result["by_type_mean_net"].items():
        lines.append(f"| {t} | {result['type_counts'][t]} | {v:+.2f} |")
    o, f = result["median_output_tokens"], result["fallback_second_calls_per_run"]
    lines += [
        "",
        f"Median total context of t4: {result['median_context_t4']:,.0f} "
        f"(limit {CONTEXT_LIMIT:,}). "
        f"Median output tokens: v1 {o['v1']:,.0f}, t3 {o['t3']:,.0f}, t4 {o['t4']:,.0f}. "
        f"Fallback second calls per run: v1 {f['v1']:.1f}, t3 {f['t3']:.1f}, t4 {f['t4']:.1f}.",
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
