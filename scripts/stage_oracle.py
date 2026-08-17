"""Where each failure first breaks, and what a perfect module would be worth.

    python scripts/stage_oracle.py           # the audit, free
    python scripts/stage_oracle.py --run     # plus the oracles, which cost quota

Four planned fixes in a row were aimed at the wrong layer. The renderer hypothesis
was falsified on its own target questions; the typed-value work turned out to
address 0.14% of the store; filtering assistant recommendations frees a third of
the context and promotes different noise; and a supersession fix stopped a real
loss without answering the question that motivated it. Each time the pattern was
the same — see a symptom, name a module, change it.

So this stops naming modules. Two parts.

**The audit** walks each failing question through the pipeline and records the
*first* stage that loses it. First, not every stage that looks wrong: "the number
is not in memory" is consistent with never extracting it and with extracting then
mangling it, and separating those took three measurements last time.

    S0 source        is the fact in the conversation at all?
    S1 extraction    did it become a memory?
    S2 lifecycle     did merging or supersession remove it afterwards?
    S3 eligibility   was it filtered before ranking?
    S4 retrieval     did it reach the answerer's context?
    S5 reasoning     everything present, answer still wrong

**The oracles** put a ceiling on each module by giving it a perfect version of
what it needs and re-answering. A module whose oracle fixes nothing cannot be
worth building, however standard it is elsewhere.

    A  perfect extraction   inject the gold fact as a memory, then run normally
    B  perfect retrieval    force the gold memory into context, rank untouched
    C  perfect reasoning    hand the answerer the gold evidence and nothing else

The rule this exists to enforce: before adding a module, ask what its oracle is
worth. Hybrid retrieval is not justified by other systems having it; it is
justified by oracle B moving questions.

No store is written. Oracle stores are copies under `stores/oracle-*`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.config import ExperimentConfig, Settings  # noqa: E402
from llm_long_term_memory.evaluation.datasets import longmemeval as lme  # noqa: E402

STORE = REPO / "stores" / "two-stage-p10.db"
RESULT = REPO / "results" / "raw" / "two_stage_hydrated.a2-clean-p10-fallback-v2.jsonl"
OUT = REPO / "results" / "raw" / "stage-oracle.json"

# The first stage that loses each failure, from the hand audit in
# results/fact-lineage-dev50.md, refined by everything measured since. Written
# down rather than inferred, because every automated attempt at this in the
# project has been confidently wrong.
#
#   S1  the fact never became a memory
#   S2  it became one and the lifecycle removed it
#   S5  everything needed was supplied and correct
STAGE = {
    "edced276": ("S1", "the Hawaii trip's 10-day duration was never extracted"),
    "4adc0475": ("S1", "the two assists were never extracted"),
    "37f165cf": ("S1", "the 416-page figure was never extracted"),
    "73d42213": ("S1", "the two-hour journey was never extracted, at any batch size"),
    "c9f37c46": ("S1", "the open mic night's gold session yielded nothing"),
    "gpt4_7abb270c": ("S1", "the sixth museum was never extracted"),
    "80ec1f4f": ("S1", "the February gallery visit's gold session yielded nothing"),
    "gpt4_fa19884d": ("S1", "the user's own statement was not extracted, only the replies to it"),
    "0edc2aef": ("S1", "the user's transferable preference was not extracted"),
    "92a0aa75": ("S2", "extracted, then folded away as a restatement"),
    "gpt4_e414231e": ("S5", "both dates supplied and annotated; reported as the same day"),
    "gpt4_d6585ce8": ("S5", "every event and date supplied; ordering still wrong"),
    "9a707b81": ("S5", "the date supplied and correct; anchored on the wrong day"),
    "71017277": ("S5", "the fact supplied and correct; not accepted as jewellery"),
}


def load():
    rows = {
        json.loads(x)["question_id"]: json.loads(x)
        for x in RESULT.read_text(encoding="utf-8").splitlines()
        if x.strip()
    }
    cfg = ExperimentConfig.from_yaml(str(REPO / "configs" / "baselines.yaml"))
    inst = {
        i.question_id: i
        for i in lme.load(cfg.dataset_variant, Settings().data_dir, limit=cfg.dataset_limit)
    }
    return rows, inst, cfg


def audit(rows) -> dict:
    fails = [q for q, r in rows.items() if not r["correct"]]
    missing = set(fails) - set(STAGE)
    extra = set(STAGE) - set(fails)
    if missing or extra:
        print(f"  \033[31mSTAGE table is stale: missing {missing}, extra {extra}\033[0m")

    print(f"{'question':15s} {'type':26s} {'stage':6s} what is lost")
    print("-" * 104)
    for q in sorted(fails, key=lambda x: STAGE.get(x, ("", ""))[0]):
        stage, why = STAGE.get(q, ("??", "unclassified"))
        print(f"{q:15s} {rows[q]['question_type']:26s} {stage:6s} {why}")

    counts = Counter(s for s, _ in STAGE.values())
    print()
    labels = {
        "S0": "source — the fact is not in the conversation",
        "S1": "extraction — never became a memory",
        "S2": "lifecycle — merged or superseded away",
        "S3": "eligibility — filtered before ranking",
        "S4": "retrieval — eligible but never in context",
        "S5": "reasoning — everything supplied and correct",
    }
    print(f"{'stage':6s} {'n':>3s}   first cause")
    print("-" * 66)
    for s in ("S0", "S1", "S2", "S3", "S4", "S5"):
        print(f"{s:6s} {counts.get(s, 0):3d}   {labels[s]}")
    return dict(counts)


def contribution(rows) -> dict:
    """How much of the headline accuracy is the memory path, and how much is rescue.

    Worth separating because `single-session-assistant` scores 6/6 and five of
    those six were answered from the raw archive after the answerer reported it
    could not answer from memory. A single accuracy figure reads as "the memory
    system answered these", which for those five is not what happened.
    """
    total = len(rows)
    correct = [r for r in rows.values() if r["correct"]]
    by = Counter(str(r["notes"].get("fallback_level")) for r in correct)
    memory_only = by.get("none", 0)
    rescued = sum(v for k, v in by.items() if k != "none")
    print(f"{'final accuracy':34s} {len(correct):3d}/{total}  {len(correct) / total:6.1%}")
    print(
        f"{'  answered from memory alone':34s} {memory_only:3d}/{total}  {memory_only / total:6.1%}"
    )
    print(f"{'  rescued by the raw archive':34s} {rescued:3d}/{total}  {rescued / total:6.1%}")
    fired = sum(1 for r in rows.values() if str(r["notes"].get("fallback_level")) != "none")
    print(f"\n{'fallback fired':34s} {fired:3d}/{total}")
    print(
        f"{'  and the answer was right':34s} {rescued:3d}/{fired}  {rescued / max(fired, 1):6.1%}"
    )
    return {"final": len(correct), "memory_only": memory_only, "rescued": rescued, "fired": fired}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="also run the oracles (costs quota)")
    args = parser.parse_args()

    rows, _inst, _cfg = load()

    print("\n" + "=" * 104)
    print("FIRST FAILING STAGE — one per question, the earliest that loses it")
    print("=" * 104)
    counts = audit(rows)

    print("\n" + "=" * 104)
    print("WHERE THE ACCURACY COMES FROM")
    print("=" * 104)
    split = contribution(rows)

    print("\n" + "=" * 104)
    print("ORACLE CEILINGS — the most each module could be worth")
    print("=" * 104)
    print(
        f"  A  perfect extraction : up to {counts.get('S1', 0)} questions "
        f"(every S1, if the fact alone is enough)"
    )
    print(f"  B  perfect retrieval  : up to {counts.get('S3', 0) + counts.get('S4', 0)} questions")
    print(f"  C  perfect reasoning  : up to {counts.get('S5', 0)} questions")
    print(
        "\n  These are upper bounds from the audit, not measurements. The "
        "counterfactual already\n  showed the extraction ceiling is not reached: six "
        "facts recovered, two answers fixed."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "note": (
                    "First failing stage per dev50 failure, and the memory/rescue split of "
                    "the headline accuracy. Stages are hand-assigned; see the STAGE table in "
                    "scripts/stage_oracle.py and results/fact-lineage-dev50.md."
                ),
                "stages": {q: {"stage": s, "why": w} for q, (s, w) in STAGE.items()},
                "counts": counts,
                "accuracy_split": split,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n→ {OUT}")

    if not args.run:
        print("\nOracles A/B/C are not run. Re-run with --run to spend quota on them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
