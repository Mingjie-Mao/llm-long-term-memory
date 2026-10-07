"""Read the small t1/t2 probe as registered in `results/prereg-t-probe-v1.md`. No calls.

    python tools/t_probe_report.py [--json-out P] [--md-out P]

train150 (out of design): v1 rerun, t1, t2 on 19 v1 failures of the target types and
12 controls; the readings compare each arm with the v1 rerun on the same questions.
dev100 (design set): t1 and t2 against v1's three existing runs, reported only.
Refuses to read an incomplete run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import write_report  # noqa: E402

RAW = REPO / "results" / "raw"
MANIFESTS = REPO / "results" / "manifests"
TRAIN = {
    "v1": "two_stage_raw_primary.t-probe-v1",
    "t1": "two_stage_raw_primary_t1.t-probe-v1",
    "t2": "two_stage_raw_primary_t2.t-probe-v1",
}
DEV = {
    "t1": ("two_stage_raw_primary_t1.t-probe-v1-dev", "t-probe-dev100-temporal-v1"),
    "t2": ("two_stage_raw_primary_t2.t-probe-v1-dev", "t-probe-dev100-target-v1"),
}
DEV_V1 = [f"paged-paired-dev100-v18-v1-baseline-rep{r}" for r in (1, 2, 3)]
FLOOR = 2
HARM = 2


def load(name: str) -> dict[str, dict]:
    path = RAW / f"{name}.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    return {r["question_id"]: r for r in rows}


def ids(name: str) -> list[str]:
    return json.loads((MANIFESTS / f"{name}.json").read_text(encoding="utf-8"))["question_ids"]


def evaluate() -> dict:
    groups = json.loads((MANIFESTS / "t-probe-train150-v1.groups.json").read_text("utf-8"))
    every = groups["failures"] + groups["controls"]
    runs = {arm: load(name) for arm, name in TRAIN.items()}
    for arm, rows in runs.items():
        if set(rows) != set(every):
            raise SystemExit(f"train150 {arm}: {len(rows)}/{len(every)} rows, no reading")
    qtype = {q: runs["v1"][q]["question_type"] for q in every}
    temporal = [q for q in groups["failures"] if qtype[q] == "temporal-reasoning"]
    count_ku = [q for q in groups["failures"] if qtype[q] != "temporal-reasoning"]

    def correct(arm, qs):
        return sum(bool(runs[arm][q]["correct"]) for q in qs)

    def wrong(arm, qs):
        return len(qs) - correct(arm, qs)

    train = {
        "temporal_failures": len(temporal),
        "count_ku_failures": len(count_ku),
        "controls": len(groups["controls"]),
        "correct_on_temporal_failures": {a: correct(a, temporal) for a in runs},
        "correct_on_count_ku_failures": {a: correct(a, count_ku) for a in runs},
        "wrong_on_controls": {a: wrong(a, groups["controls"]) for a in runs},
    }
    t1_move = correct("t1", temporal) - correct("v1", temporal)
    t2_move = correct("t2", count_ku) - correct("v1", count_ku)
    harm = {a: wrong(a, groups["controls"]) - wrong("v1", groups["controls"]) for a in ("t1", "t2")}
    readings = {
        "t1_moves_its_target": t1_move >= FLOOR,
        "t2_moves_its_target": t2_move >= FLOOR,
        "t1_harm_signal": harm["t1"] >= HARM,
        "t2_harm_signal": harm["t2"] >= HARM,
    }
    v1_dev = [load(n) for n in DEV_V1]
    dev = {}
    for arm, (name, manifest) in DEV.items():
        rows, qs = load(name), ids(manifest)
        if set(rows) != set(qs):
            raise SystemExit(f"dev100 {arm}: {len(rows)}/{len(qs)} rows, no reading")
        dev[arm] = {
            "questions": len(qs),
            "correct": sum(bool(rows[q]["correct"]) for q in qs),
            "v1_expected": round(
                sum(sum(bool(r[q]["correct"]) for r in v1_dev) / 3 for q in qs), 2
            ),
        }
    return {
        "train150": train,
        "t1_move": t1_move,
        "t2_move": t2_move,
        "harm_vs_v1_rerun": harm,
        "readings": readings,
        "dev100_design_set": dev,
        "proceed_to_heldout100": readings["t1_moves_its_target"]
        and readings["t2_moves_its_target"]
        and not (readings["t1_harm_signal"] or readings["t2_harm_signal"]),
    }


def evaluate_t3() -> dict:
    """`results/prereg-t-probe-v2.md`: t3 against probe v1's v1 rerun (t2 alongside)."""
    groups = json.loads((MANIFESTS / "t-probe-train150-v1.groups.json").read_text("utf-8"))
    every = groups["failures"] + groups["controls"]
    names = {**TRAIN, "t3": "two_stage_raw_primary_t3.t-probe-v1"}
    runs = {arm: load(names[arm]) for arm in ("v1", "t2", "t3")}
    for arm, rows in runs.items():
        if set(rows) != set(every):
            raise SystemExit(f"train150 {arm}: {len(rows)}/{len(every)} rows, no reading")
    qtype = {q: runs["v1"][q]["question_type"] for q in every}
    temporal = [q for q in groups["failures"] if qtype[q] == "temporal-reasoning"]
    count_ku = [q for q in groups["failures"] if qtype[q] != "temporal-reasoning"]

    def correct(arm, qs):
        return sum(bool(runs[arm][q]["correct"]) for q in qs)

    table = {
        "correct_on_temporal_failures": {a: correct(a, temporal) for a in runs},
        "correct_on_count_ku_failures": {a: correct(a, count_ku) for a in runs},
        "wrong_on_controls": {
            a: len(groups["controls"]) - correct(a, groups["controls"]) for a in runs
        },
    }
    moves = {
        "temporal": table["correct_on_temporal_failures"]["t3"]
        - table["correct_on_temporal_failures"]["v1"],
        "count_ku": table["correct_on_count_ku_failures"]["t3"]
        - table["correct_on_count_ku_failures"]["v1"],
        "harm": table["wrong_on_controls"]["t3"] - table["wrong_on_controls"]["v1"],
    }
    readings = {
        "t3_moves_temporal": moves["temporal"] >= FLOOR,
        "t3_moves_count_ku": moves["count_ku"] >= FLOOR,
        "t3_harm_signal": moves["harm"] >= HARM,
    }
    rows, qs = load("two_stage_raw_primary_t3.t-probe-v1-dev"), ids("t-probe-dev100-target-v1")
    if set(rows) != set(qs):
        raise SystemExit(f"dev100 t3: {len(rows)}/{len(qs)} rows, no reading")
    v1_dev = [load(n) for n in DEV_V1]
    modes = {q: runs["t3"][q]["notes"].get("answer_mode") for q in every}
    return {
        "train150": table,
        "moves_vs_v1_rerun": moves,
        "readings": readings,
        "answer_modes": {m: sum(1 for v in modes.values() if v == m) for m in set(modes.values())},
        "dev100_design_set": {
            "questions": len(qs),
            "correct": sum(bool(rows[q]["correct"]) for q in qs),
            "v1_expected": round(
                sum(sum(bool(r[q]["correct"]) for r in v1_dev) / 3 for q in qs), 2
            ),
        },
        "register_t3_on_heldout100": readings["t3_moves_temporal"]
        and readings["t3_moves_count_ku"]
        and not readings["t3_harm_signal"],
    }


def render_t3(r: dict) -> str:
    t = r["train150"]
    lines = [
        "# Small t3 probe — registered reading",
        "",
        "> Diagnostic (`results/prereg-t-probe-v2.md`); one run per arm; v1 rerun and t2",
        "> rows from probe v1. Not an accuracy figure.",
        "",
        "| train150 | v1 rerun | t2 | t3 |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("correct on 8 temporal failures", "correct_on_temporal_failures"),
        ("correct on 11 count/KU failures", "correct_on_count_ku_failures"),
        ("wrong on 12 controls", "wrong_on_controls"),
    ):
        lines.append(f"| {label} | {t[key]['v1']} | {t[key]['t2']} | {t[key]['t3']} |")
    lines += ["", f"Moves against the v1 rerun: {r['moves_vs_v1_rerun']}", ""]
    lines += [f"- {k}: **{v}**" for k, v in r["readings"].items()]
    d = r["dev100_design_set"]
    lines += [
        f"- **register t3 on heldout100: {r['register_t3_on_heldout100']}**",
        f"- t3 answer modes on train150: {r['answer_modes']}",
        f"- dev100 design set (reported only): {d['correct']} / {d['questions']} "
        f"against v1's expected {d['v1_expected']}",
    ]
    return "\n".join(lines) + "\n"


def render(r: dict) -> str:
    t = r["train150"]
    lines = [
        "# Small t1/t2 probe — registered reading",
        "",
        "> Diagnostic (`results/prereg-t-probe-v1.md`); one run per arm. Not an accuracy",
        "> figure; the heldout100 registrations decide t1 and t2.",
        "",
        "## train150, out of design",
        "",
        "| | v1 rerun | t1 | t2 |",
        "|---|---:|---:|---:|",
    ]
    for label, key, n in (
        ("correct on temporal failures", "correct_on_temporal_failures", t["temporal_failures"]),
        ("correct on count/KU failures", "correct_on_count_ku_failures", t["count_ku_failures"]),
        ("wrong on controls", "wrong_on_controls", t["controls"]),
    ):
        v = t[key]
        lines.append(f"| {label} (of {n}) | {v['v1']} | {v['t1']} | {v['t2']} |")
    lines += ["", "## Readings", ""]
    lines += [f"- {k}: **{v}**" for k, v in r["readings"].items()]
    lines += [
        f"- t1 move {r['t1_move']:+d}, t2 move {r['t2_move']:+d} (floor +{FLOOR}); "
        f"harm vs v1 rerun {r['harm_vs_v1_rerun']}",
        f"- **proceed to heldout100: {r['proceed_to_heldout100']}**",
        "",
        "## dev100, design set (reported only)",
        "",
        "| arm | questions | correct | v1 expected (3 runs) |",
        "|---|---:|---:|---:|",
    ]
    for arm, d in r["dev100_design_set"].items():
        lines.append(f"| {arm} | {d['questions']} | {d['correct']} | {d['v1_expected']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--arm", choices=("t1t2", "t3"), default="t1t2")
    args = parser.parse_args()
    if args.arm == "t3":
        result, draw = evaluate_t3(), render_t3
    else:
        result, draw = evaluate(), render
    print(write_report(result, draw, json_out=args.json_out, md_out=args.md_out), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
