"""What can this question set actually detect? Answered before the experiment, not after.

Two mechanism experiments in a row — v2e and v5.0 — were registered against
reasoning-48, run, and returned nets of +0, +0 and -1. Both were then found to be
below the set's resolution. The information needed to predict that was already on
disk: `results/heldout-variance.md` ran one unchanged configuration three times over
100 questions in 2026-08, and six questions disagreed with themselves.

That measurement is not a footnote. Projected onto reasoning-48's composition it
predicts **2.7 questions flipping from run noise alone**, which is the whole range the
three arms moved in. Neither experiment could have detected what it was registered to
detect, and the cost of finding that out was two quota days.

This turns the existing repeat runs into a pre-flight number. It reads whatever
`<variant>.jsonl` plus `<variant>.rep*.jsonl` sets exist, measures how often an
unchanged configuration disagrees with itself per question type, projects that onto a
target set, and says what net difference that set can distinguish from noise.

**The model is deliberately crude and stated rather than hidden.** Each disagreeing
question is treated as contributing an independent ±1 to a paired net, so the net's
noise has standard deviation about `sqrt(N)` where `N` is the expected number of
disagreeing questions. Two standard deviations is the bar a net has to clear before it
is worth interpreting. This is a floor on the requirement, not a significance test:
real flips are not independent, the per-type rates come from one 100-question
measurement, and a set whose questions are enriched for prior failures is exactly the
population most likely to be marginal.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import read_jsonl, rows_by_question, write_report  # noqa: E402

RAW = REPO / "results/raw"


# `heldout100-rep2`, `v2b-gate16-repeat2`, `gate8-v2c3-rep1`. The repeat marker sits
# inside the label rather than after the variant, so grouping on it has to strip a
# suffix rather than split on a separator.
_REPEAT = re.compile(r"-rep(?:eat)?\d+$")


def _repeat_families(minimum: int = 2) -> dict[str, list[Path]]:
    """Runs of one unchanged configuration, grouped by what they are repeats of.

    Two runs are enough to see a question disagree with itself, which is the quantity
    in question; more runs make the rate less noisy, and the report says how many each
    family had so a thin one can be discounted.
    """
    families: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(RAW.glob("*.jsonl")):
        variant, _, label = path.stem.partition(".")
        if not label:
            continue
        families[f"{variant}.{_REPEAT.sub('', label)}"].append(path)
    return {name: paths for name, paths in families.items() if len(paths) >= minimum}


def measure(families: dict[str, list[Path]]) -> dict:
    """How often an unchanged configuration disagrees with itself, per question type."""
    per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    sources = {}
    for name, paths in families.items():
        runs = [rows_by_question(path) for path in paths]
        shared = sorted(set.intersection(*(set(run) for run in runs)))
        if not shared:
            continue
        disagreed = 0
        for qid in shared:
            verdicts = {bool(run[qid]["correct"]) for run in runs}
            kind = runs[0][qid]["question_type"]
            per_type[kind][1] += 1
            if len(verdicts) > 1:
                per_type[kind][0] += 1
                disagreed += 1
        sources[name] = {
            "runs": len(paths),
            "questions": len(shared),
            "disagreed": disagreed,
            "accuracy_per_run": [sum(r[q]["correct"] for q in shared) for r in runs],
        }
    rates = {
        kind: {"unstable": bad, "n": total, "rate": bad / total}
        for kind, (bad, total) in sorted(per_type.items())
        if total
    }
    return {"sources": sources, "rates": rates}


def project(rates: dict, composition: Counter) -> dict:
    """What that noise becomes on a set with this question-type mix."""
    unmeasured = [kind for kind in composition if kind not in rates]
    expected = sum(
        count * rates[kind]["rate"] for kind, count in composition.items() if kind in rates
    )
    sigma = math.sqrt(expected) if expected else 0.0
    return {
        "questions": sum(composition.values()),
        "composition": dict(sorted(composition.items())),
        "types_with_no_repeat_measurement": sorted(unmeasured),
        "expected_disagreeing_questions": expected,
        "net_noise_sd": sigma,
        "smallest_interpretable_net": math.ceil(2 * sigma) if sigma else 0,
    }


def render(result: dict) -> str:
    measured = result["measured"]
    target = result["target"]
    lines = [
        "# Resolution of a question set",
        "",
        "> Derived from repeat runs already on disk. No model calls.",
        "",
        "## Measured run noise",
        "",
        "One unchanged configuration, run more than once. A question that disagrees with",
        "itself here cannot carry a mechanism's signal anywhere else.",
        "",
        "| repeat set | runs | questions | disagreed | accuracy per run |",
        "|---|---:|---:|---:|---|",
    ]
    lines += [
        f"| `{name}` | {s['runs']} | {s['questions']} | **{s['disagreed']}** | "
        f"{', '.join(str(a) for a in s['accuracy_per_run'])} |"
        for name, s in sorted(measured["sources"].items())
    ]
    lines += ["", "| question type | unstable | n | rate |", "|---|---:|---:|---:|"]
    lines += [
        f"| {kind} | {row['unstable']} | {row['n']} | **{row['rate']:.1%}** |"
        for kind, row in sorted(measured["rates"].items(), key=lambda kv: -kv[1]["rate"])
    ]
    lines += [
        "",
        f"## Projected onto `{result['target_name']}`",
        "",
        f"- questions: **{target['questions']}**",
        f"- expected to disagree from run noise alone: "
        f"**{target['expected_disagreeing_questions']:.1f}**",
        f"- standard deviation of the paired net from noise: **±{target['net_noise_sd']:.1f}**",
        f"- **a net below {target['smallest_interpretable_net']} is not distinguishable "
        "from noise with one run per arm**",
    ]
    if target["types_with_no_repeat_measurement"]:
        lines += [
            "",
            "Types with no repeat measurement, counted as perfectly stable and therefore "
            "making this an **under**estimate: "
            + ", ".join(f"`{k}`" for k in target["types_with_no_repeat_measurement"]),
        ]
    lines += [
        "",
        "## Reading this",
        "",
        "The last number is a floor on what the set can register, not a significance",
        "test. Real flips are not independent, the per-type rates come from a single",
        "100-question measurement, and a set enriched for prior failures is exactly the",
        "population most likely to be marginal — so the true requirement is higher than",
        "this, never lower.",
        "",
        "If a mechanism's plausible effect is smaller than the figure above, the choice",
        "is to enrich the set toward the failure it targets, repeat each arm, or not run",
        "it. Deciding that after the run is how two quota days were spent.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        type=Path,
        help="A rows file whose question-type composition the projection uses.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()
    target = args.target if args.target.is_absolute() else (REPO / args.target).resolve()

    families = _repeat_families()
    if not families:
        raise SystemExit(
            "no repeat runs on disk. Run the same configuration at least twice with "
            "`lltm eval repeat` before asking what a set can detect."
        )
    measured = measure(families)
    composition = Counter(row["question_type"] for row in read_jsonl(target))
    result = {
        "measured": measured,
        "target_name": target.stem,
        "target": project(measured["rates"], composition),
    }
    json_out = args.json_out or REPO / f"results/analysis/resolution-{target.stem}.json"
    md_out = args.md_out or REPO / f"results/analysis/resolution-{target.stem}.md"
    print(write_report(result, render, json_out=json_out, md_out=md_out), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
