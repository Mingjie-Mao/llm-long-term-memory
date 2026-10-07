"""Regression probe for the stronger answerer. No calls.

    python tools/strong_regression_probe.py sample      # writes the 20-question manifest
    python tools/strong_regression_probe.py evaluate    # after the strong run

Registered in `results/prereg-strong-answerer-regression-probe-v1.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import write_report  # noqa: E402

RAW = REPO / "results" / "raw"
LITE = "two_stage_raw_primary.train150-raw-v2"
STRONG = "two_stage_raw_primary.strong-regression-v2"
MANIFEST = REPO / "results" / "manifests" / "train150-raw-v1-correct-sample10.json"
SEED = "strong-regression-v2"
QUOTAS = {
    "temporal-reasoning": 2,
    "multi-session": 2,
    "knowledge-update": 2,
    "single-session-preference": 2,
}
# "Numeric / computation": a gold answer with a digit or a number word, drawn from the
# single-session types the quotas above leave out, so all six types are represented.
NUMERIC_TYPES = ("single-session-user", "single-session-assistant")
_NUMBER = re.compile(
    r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|"
    r"hundred)\b",
    re.IGNORECASE,
)
NOISE_RUNS = (
    "two_stage_hydrated.heldout100",
    "two_stage_hydrated.heldout100-rep2",
    "two_stage_hydrated.heldout100-rep3",
    "two_stage_hydrated.heldout100-rep4",
)
FIXES = 5
LITE_CORRECT = 127


def load(name: str) -> dict[str, dict]:
    path = RAW / f"{name}.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["question_id"]: row for row in rows}


def draw(lite: dict[str, dict]) -> list[str]:
    """First k per type among flash-lite's correct answers, by a seeded hash order."""
    chosen = []
    for qtype, k in QUOTAS.items():
        pool = [q for q, r in lite.items() if r["correct"] and r["question_type"] == qtype]
        pool.sort(key=lambda q: hashlib.sha256(f"{SEED}|{q}".encode()).hexdigest())
        if len(pool) < k:
            raise SystemExit(f"{qtype}: only {len(pool)} correct questions, need {k}")
        chosen += pool[:k]
    for qtype in NUMERIC_TYPES:  # one each, so both single-session types appear
        pool = [
            q
            for q, r in lite.items()
            if r["correct"] and r["question_type"] == qtype and _NUMBER.search(str(r["gold"]))
        ]
        pool.sort(key=lambda q: hashlib.sha256(f"{SEED}|{q}".encode()).hexdigest())
        chosen += pool[:1]
    return sorted(chosen)


def flip_rates() -> dict[str, float]:
    """P(wrong in run B | right in run A), per type, over ordered pairs of control runs."""
    runs = [load(name) for name in NOISE_RUNS]
    right = defaultdict(int)
    flipped = defaultdict(int)
    for a, b in itertools.permutations(runs, 2):
        for q, row in a.items():
            if row["correct"]:
                right[row["question_type"]] += 1
                flipped[row["question_type"]] += not b[q]["correct"]
    return {t: flipped[t] / right[t] for t in right}


def evaluate() -> dict:
    ids = json.loads(MANIFEST.read_text(encoding="utf-8"))["question_ids"]
    lite, strong = load(LITE), load(STRONG)
    missing = sorted(set(ids) - set(strong))
    if missing:
        raise SystemExit(f"incomplete strong run, no reading: {len(missing)} missing")
    if not all(lite[q]["correct"] for q in ids):
        raise SystemExit("the sample must contain only flash-lite's correct answers")
    qtype = {q: lite[q]["question_type"] for q in ids}
    regressions = sorted(q for q in ids if not strong[q]["correct"])
    rates = flip_rates()
    expected = sum(rates.get(qtype[q], 0.0) for q in ids)
    r = len(regressions)
    conservative = FIXES - LITE_CORRECT * r / len(ids)
    adjusted = FIXES - LITE_CORRECT * max(0.0, r - expected) / len(ids)
    if r <= 1:
        reading = "regression risk is low — proceed to the full strong vs flash-lite comparison"
    elif r == 2:
        reading = "confirm — draw 10 more by the same rule"
    else:
        reading = "do not replace the answerer globally — analyse the regressions by type"
    by_type = Counter(qtype[q] for q in regressions)
    return {
        "questions": len(ids),
        "regressions": regressions,
        "R": r,
        "by_type": {
            t: {
                "sampled": sum(1 for q in ids if qtype[q] == t),
                "regressed": by_type.get(t, 0),
            }
            for t in sorted(set(qtype.values()))
        },
        "expected_noise_flips": expected,
        "noise_flip_rates": rates,
        "projected_net_conservative": conservative,
        "projected_net_noise_adjusted": adjusted,
        "reading": reading,
    }


def render(result: dict) -> str:
    lines = [
        "# Stronger answerer — regression probe on flash-lite's correct answers",
        "",
        "> Diagnostic, train150. Registered in",
        "> `results/prereg-strong-answerer-regression-probe-v1.md`. Not an accuracy figure.",
        "",
        f"- regressions: **{result['R']} / {result['questions']}** {result['regressions']}",
        "- expected from flash-lite's own run-to-run flips: "
        f"**{result['expected_noise_flips']:.2f}**",
        "- projected net on train150: conservative "
        f"**{result['projected_net_conservative']:+.1f}**, noise-adjusted "
        f"**{result['projected_net_noise_adjusted']:+.1f}** (fixes counted: {FIXES})",
        f"- **reading: {result['reading']}**",
        "",
        "| type | sampled | regressed |",
        "|---|---:|---:|",
    ]
    lines += [f"| {t} | {v['sampled']} | {v['regressed']} |" for t, v in result["by_type"].items()]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("sample", "evaluate"))
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "sample":
        if MANIFEST.exists():
            raise SystemExit(f"{MANIFEST.name} exists; the draw is made once")
        ids = draw(load(LITE))
        MANIFEST.write_text(
            json.dumps(
                {
                    "name": MANIFEST.stem,
                    "variant": "s",
                    "seed": 0,
                    "n": len(ids),
                    "exposure": "regression",
                    "note": "DIAGNOSTIC. 10 train150 questions the raw-primary v1 arm "
                    "(flash-lite) answered right in two_stage_raw_primary.train150-raw-v2: "
                    "2 each of temporal-reasoning, multi-session, knowledge-update and "
                    "preference, and 2 single-session questions with a numeric gold, by a "
                    "seeded hash order (strong-regression-v2). For "
                    "results/prereg-strong-answerer-regression-probe-v1.md. Never an "
                    "accuracy estimate.",
                    "question_ids": ids,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {MANIFEST.relative_to(REPO)}: {len(ids)} questions")
        return 0
    result = evaluate()
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
