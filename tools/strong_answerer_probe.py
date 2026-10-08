"""Evaluate `results/prereg-strong-answerer-probe-v1.md`. No calls.

    python tools/strong_answerer_probe.py

Reads the strong and lite-rerun runs over the 9 reasoning errors and reports the model
effect with the registered reading. Refuses an incomplete run.
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
MANIFEST = REPO / "results" / "manifests" / "train150-raw-v1-reasoning-errors.json"
STRONG = "two_stage_raw_primary.strong-probe-v1"
LITE = "two_stage_raw_primary.lite-rerun-v1"


def load(name: str) -> dict[str, dict]:
    path = RAW / f"{name}.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["question_id"]: row for row in rows}


def reading(effect: int) -> str:
    if effect >= 4:
        return "the answering model is a real lever — a full registered comparison is next"
    if effect <= 1:
        return "the residual is not mainly the model's — stop pursuing the upgrade"
    return "undecided — ask the remaining 14 errors the same way, as a registered extension"


def evaluate() -> dict:
    ids = json.loads(MANIFEST.read_text(encoding="utf-8"))["question_ids"]
    runs = {name: load(name) for name in (STRONG, LITE)}
    missing = {name: sorted(set(ids) - set(rows)) for name, rows in runs.items()}
    if any(missing.values()):
        raise SystemExit(f"incomplete runs, no reading: {missing}")
    fixed = {name: sorted(q for q in ids if runs[name][q]["correct"]) for name in runs}
    effect = len(fixed[STRONG]) - len(fixed[LITE])
    return {
        "questions": len(ids),
        "fixed_by_strong": fixed[STRONG],
        "fixed_by_lite_rerun": fixed[LITE],
        "model_effect": effect,
        "reading": reading(effect),
        "answerer_models": {
            name: sorted({r.get("notes", {}).get("model", "") for r in runs[name].values()} - {""})
            for name in runs
        },
    }


def render(result: dict) -> str:
    return "\n".join(
        [
            "# Stronger answerer on the reasoning errors — minimal probe",
            "",
            "> Diagnostic, failure-enriched, train150. Registered in",
            "> `results/prereg-strong-answerer-probe-v1.md`. Not an accuracy figure.",
            "",
            f"- fixed by `gemini-3.6-flash`: **{len(result['fixed_by_strong'])} / "
            f"{result['questions']}** {result['fixed_by_strong']}",
            f"- fixed by the flash-lite rerun: **{len(result['fixed_by_lite_rerun'])} / "
            f"{result['questions']}** {result['fixed_by_lite_rerun']}",
            f"- **model effect: {result['model_effect']:+d}** — {result['reading']}",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate()
    print(write_report(result, render, json_out=args.json_out, md_out=args.md_out), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
