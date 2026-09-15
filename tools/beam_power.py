"""How small a difference the BEAM split can resolve, before any BEAM answer exists.

Reads the registered split (`results/manifests/beam-split.json`) and the proposed
evaluation declaration (`configs/beam-eval.json`), and writes the minimum detectable effect
of every stratum a decision could be read on to `results/analysis/beam-power.json`.

Nothing has been measured on BEAM, so both inputs a detectable effect depends on are
scenarios, each named with its source:

* **Discordance under the null**: how often two runs of one unchanged arm disagree on a
  question. 0.04 is LongMemEval `heldout100`, where 6 of 100 questions were not identical
  across three runs and a question that flips once disagrees in two of three pairs. 0.19
  is the v4 probe pair, 27 binary disagreements in 142, a noisier instrument.
* **ICC**: how alike the questions of one conversation are. 0 is independence; 0.05 and
  0.15 are assumptions.

`results/gate-protocol.md` forbids registering a gate against an unmeasured noise floor.
These figures size the plan. The development run has to measure both before any gate is
written against them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm_long_term_memory.evaluation.clustered import minimum_detectable_effect

REPO = Path(__file__).resolve().parent.parent
SPLIT = REPO / "results/manifests/beam-split.json"
DECLARATION = REPO / "configs/beam-eval.json"
OUT = REPO / "results/analysis/beam-power.json"

QUESTIONS_PER_CONVERSATION = 20
QUESTIONS_PER_ABILITY = 2
DISCORDANCE = {"heldout100 run-to-run": 0.04, "v4 probes run-to-run": 0.19}
ICC = (0.0, 0.05, 0.15)
SIGMAS = 2.0


def strata(split: dict, declaration: dict) -> list[dict]:
    """Every stratum a registered decision could be read on, with its cluster size."""
    primary = len(declaration["primary"]["abilities"])
    rows = []
    for half in ("dev", "test"):
        conversations = len(split[half]["conversations"])
        for stratum, per_conversation in (
            ("all ten abilities", QUESTIONS_PER_CONVERSATION),
            (f"the {primary} primary abilities", QUESTIONS_PER_ABILITY * primary),
            ("one ability", QUESTIONS_PER_ABILITY),
        ):
            rows.append(
                {
                    "half": half,
                    "stratum": stratum,
                    "questions": conversations * per_conversation,
                    "cluster_size": per_conversation,
                }
            )
    rows.append(
        {
            "half": "test",
            "stratum": "100K conversations, where full context fits",
            "questions": split["test"]["per_scale"].get("100K", 0) * QUESTIONS_PER_CONVERSATION,
            "cluster_size": QUESTIONS_PER_CONVERSATION,
        }
    )
    return rows


def table(split: dict, declaration: dict) -> list[dict]:
    rows = []
    for row in strata(split, declaration):
        cells = {
            f"{label}, icc {icc:g}": round(
                100
                * minimum_detectable_effect(
                    row["questions"], discordance, row["cluster_size"], icc, SIGMAS
                ),
                1,
            )
            for label, discordance in DISCORDANCE.items()
            for icc in ICC
        }
        rows.append({**row, "mde_points": cells})
    return rows


def build(split_path: Path = SPLIT, declaration_path: Path = DECLARATION) -> dict:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    return {
        "name": "beam-power",
        "status": "scenario estimates: BEAM's own discordance and ICC are unmeasured",
        "inputs": {
            "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "declaration_sha256": hashlib.sha256(declaration_path.read_bytes()).hexdigest(),
        },
        "unit": "accuracy points at the declared binary verdict, two-sided, paired arms",
        "sigmas": SIGMAS,
        "discordance_scenarios": DISCORDANCE,
        "icc_scenarios": list(ICC),
        "strata": table(split, declaration),
    }


def main() -> int:
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    columns = list(payload["strata"][0]["mde_points"])
    print(f"minimum detectable effect at {SIGMAS:g} sigma, in accuracy points")
    print("columns: " + " | ".join(columns))
    for row in payload["strata"]:
        cells = "  ".join(f"{value:5.1f}" for value in row["mde_points"].values())
        label = f"{row['half']:4s} {row['stratum']:45s} n={row['questions']:<4d}"
        print(f"{label} m={row['cluster_size']:<3d} {cells}")
    print(f"written: {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
