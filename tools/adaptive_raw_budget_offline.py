"""Question-routed raw-turn budget: does it reach more gold within the cost rule? No calls.

    python tools/adaptive_raw_budget_offline.py                     # train150, decides
    python tools/adaptive_raw_budget_offline.py --store heldout100 --questions heldout100.json \
        --control-rows two_stage_hydrated.heldout100-rep4 --aggregate-only

Registered in `results/prereg-adaptive-raw-budget-offline-v1.md`.
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

HEADER_TOKENS = 100
COVERAGE_MARGIN = 0.03
CONTEXT_LIMIT = 6000
POLICIES = {
    "fixed_4k": lambda kind: 4000,
    "fixed_8k": lambda kind: 8000,
    "adaptive_agg": lambda kind: 8000 if kind == "multi_session_aggregation" else 4000,
    "adaptive_agg_temporal": lambda kind: (
        8000 if kind in ("multi_session_aggregation", "temporal") else 4000
    ),
}
ADAPTIVE = ("adaptive_agg", "adaptive_agg_temporal")


def analyse(store_name: str, manifest_name: str, control_rows: str) -> dict:
    from llm_long_term_memory.config import Settings
    from llm_long_term_memory.evaluation.datasets import longmemeval as lme
    from llm_long_term_memory.evaluation.manifest import load_manifest
    from llm_long_term_memory.evaluation.runners.reasoning import reasoning_kind
    from llm_long_term_memory.retrieve.excerpts import archive_excerpts
    from llm_long_term_memory.store import SQLiteMemoryStore, external_session_id

    settings = Settings()
    manifest = load_manifest(settings.results_dir / "manifests" / manifest_name)
    instances = {i.question_id: i for i in lme.load(manifest.variant, settings.data_dir)}
    control = {
        json.loads(line)["question_id"]: json.loads(line)["context_tokens"]
        for line in (REPO / "results" / "raw" / f"{control_rows}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    }
    store = SQLiteMemoryStore(settings.store_dir / f"{store_name}.db", read_only=True)
    store.initialize()
    rows = []
    try:
        for qid in manifest.question_ids:
            inst = instances[qid]
            kind = reasoning_kind(inst.question)
            gold = {
                (s.session_id, i)
                for s in inst.sessions
                for i, t in enumerate(s.turns)
                if t.has_answer
            }
            row = {"question_id": qid, "type": inst.question_type, "kind": kind, "policies": {}}
            for name, budget_of in POLICIES.items():
                found = archive_excerpts(store, qid, inst.question, budget_of(kind))
                got = {(external_session_id(t.session_id), t.turn_index) for t in found.turns}
                row["policies"][name] = {
                    "all_gold": bool(gold) and gold <= got,
                    "total_context": control[qid] + found.tokens + HEADER_TOKENS,
                }
            row["has_gold"] = bool(gold)
            rows.append(row)
    finally:
        store.close()

    scored = [r for r in rows if r["has_gold"]]
    summary = {}
    for name in POLICIES:
        summary[name] = {
            "all_gold": sum(r["policies"][name]["all_gold"] for r in scored) / len(scored),
            "median_total_context": statistics.median(
                r["policies"][name]["total_context"] for r in rows
            ),
            "by_type": {
                t: sum(r["policies"][name]["all_gold"] for r in scored if r["type"] == t)
                / max(1, sum(1 for r in scored if r["type"] == t))
                for t in sorted({r["type"] for r in scored})
            },
        }
    base = summary["fixed_4k"]["all_gold"]
    qualifying = [
        name
        for name in ADAPTIVE
        if summary[name]["all_gold"] - base >= COVERAGE_MARGIN
        and summary[name]["median_total_context"] <= CONTEXT_LIMIT
    ]
    chosen = max(qualifying, key=lambda n: summary[n]["all_gold"]) if qualifying else "fixed_4k"
    routing = Counter((r["type"], r["kind"]) for r in rows)
    return {
        "store": store_name,
        "questions": len(rows),
        "with_gold": len(scored),
        "summary": summary,
        "routed_share": {
            name: sum(1 for r in rows if POLICIES[name](r["kind"]) == 8000) / len(rows)
            for name in POLICIES
        },
        "routing": {f"{t} -> {k}": n for (t, k), n in sorted(routing.items())},
        "decision": {"qualifying": qualifying, "chosen": chosen},
        "rows": rows,
    }


def render(result: dict) -> str:
    s = result["summary"]
    types = list(next(iter(s.values()))["by_type"])
    lines = [
        f"# Question-routed raw-turn budget — `{result['store']}`",
        "",
        "> Zero calls. Registered in `results/prereg-adaptive-raw-budget-offline-v1.md`.",
        "> Reach, not use. Routing reads the question wording only.",
        "",
        f"{result['with_gold']} of {result['questions']} questions have gold turns.",
        "",
        "| policy | routed to 8k | all-gold coverage | projected median total context |",
        "|---|---:|---:|---:|",
    ]
    for name, row in s.items():
        lines.append(
            f"| `{name}` | {result['routed_share'][name]:.0%} | {row['all_gold']:.1%} | "
            f"{row['median_total_context']:,.0f} |"
        )
    lines += [
        "",
        "| type | " + " | ".join(f"`{n}`" for n in s) + " |",
        "|---|" + "---:|" * len(s),
    ]
    for t in types:
        lines.append(f"| {t} | " + " | ".join(f"{s[n]['by_type'][t]:.0%}" for n in s) + " |")
    d = result["decision"]
    lines += [
        "",
        "Routing (benchmark type -> routed kind): "
        + ", ".join(f"{k}: {v}" for k, v in result["routing"].items()),
        "",
        f"**Decision: `{d['chosen']}`** (qualifying: {', '.join(d['qualifying']) or 'none'})",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="train150")
    parser.add_argument("--questions", default="train150.json")
    parser.add_argument("--control-rows", default="two_stage_hydrated.train150-raw-v2-control")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()
    result = analyse(args.store, args.questions, args.control_rows)
    if args.aggregate_only:
        result["rows"] = []
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
