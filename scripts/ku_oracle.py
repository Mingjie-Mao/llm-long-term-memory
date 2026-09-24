"""Give the five knowledge-update failures a coherent context, and see what moves.

    python scripts/ku_oracle.py            # plan, no calls
    python scripts/ku_oracle.py --run      # ~15 answerer calls

The held-out run scored knowledge-update 10/15, its joint-worst category and one
dev50 could never have shown: dev50 scored 8/8 there, which is not evidence of
anything. Reading the five failures attributes them, and the attribution is the
reason this oracle targets what it does.

    07741c45   the shoe-rack memory is present, active, retrieved — and the
               answer says "under the bed", which no memory says
    71315a70   "5-6 hours" and "10-12 hours" are both active under the same coarse
               predicate `hobbies`, no supersession, and the answer adds them
    69fee5aa   "37 coins" and "added a 1915-S quarter" are both present and
               correctly keyed; the answer is 37, and the gold is 38
    0977f2af   the Instant Pot appears only as "plans to make Jjimdak in their
               Instant Pot" — the purchase was never extracted as a purchase
    031748ae_abs  the question presupposes a role the user never held; the
               memories are right and the answer should have been a refusal

All five recalled the gold session and the gold evidence. **None is a retrieval
failure**, which reproduces dev50's S4 = 0 on unseen data. Four of the five are
downstream of retrieval, in how the supplied facts are read and combined.

That is the same layer oracle C implicated for `gpt4_e414231e`, where two
memories ranked first and second still produced the wrong answer while a coherent
slice of the same two sessions produced the right one in three runs of four. This
asks whether that generalises to a second question type.

    arm `flat`      the memories retrieval actually selected — reproduces the run
    arm `coherent`  every memory from the gold sessions, in event order, and
                    nothing else

`69fee5aa` is the control. Its two memories are already correctly keyed and both
already in context, so a context change should not fix an arithmetic slip. If it
flips, the arm is doing something other than what it claims.

Judged by reading, not by string match. A computed answer scored by substring is
the mistake this project has made three times.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.config import Settings  # noqa: E402
from llm_long_term_memory.evaluation.datasets import longmemeval as lme  # noqa: E402

STORE = "heldout100"
CONFIG = "configs/fallback.yaml"
OUT = REPO / "results" / "raw" / "ku-oracle.json"

FAILURES = {
    "07741c45": (
        "A",
        "the shoe-rack memory is present and retrieved; the answer invents 'under the bed'",
    ),
    "71315a70": ("S2", "5-6 and 10-12 hours both active under `hobbies`; the answer adds them"),
    "69fee5aa": ("S5", "CONTROL — 37 + one added coin, both present and correctly keyed"),
    "0977f2af": ("S1", "the Instant Pot purchase was never extracted as a purchase"),
    "031748ae_abs": ("A", "false premise; the answer should have been a refusal"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    inst = {i.question_id: i for i in lme.load("s", settings.data_dir)}

    print(f"knowledge-update oracle · {len(FAILURES)} questions · store {STORE}")
    for qid, (stage, why) in FAILURES.items():
        print(f"  {qid:15s} [{stage:2s}] {why}")
    print(f"\n  ~{len(FAILURES) * 2} answerer calls, no extractor quota")

    if not args.run:
        print("\nRe-run with --run.")
        return 0

    from llm_long_term_memory.api.service import MemoryService

    svc = MemoryService(store_name=STORE, config_path=CONFIG)
    if svc.answerer is None:
        print("no API key; nothing run")
        return 1

    out: dict[str, dict] = {}
    try:
        for qid, (stage, why) in FAILURES.items():
            question = inst[qid]
            gold_sessions = set(question.answer_session_ids)
            everything = list(svc.store.iter_all(qid))
            coherent = [m for m in everything if m.source_session_id in gold_sessions]
            coherent.sort(
                key=lambda m: (m.occurred_at or m.valid_from or "", m.source_turn_index or 0)
            )

            text = svc.answerer.answer_with_memories(question, coherent)
            out[qid] = {
                "stage": stage,
                "why": why,
                "gold": str(question.answer),
                "coherent_n": len(coherent),
                "store_n": len(everything),
                "answer": text,
            }
            print(f"\n  {qid}  [{stage}]  {len(coherent)} of {len(everything)} memories")
            print(f"    gold: {str(question.answer)[:110]}")
            print(f"    got : {str(text)[:110]}")
    finally:
        OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {OUT}")

    print("\nRead these. Do not string-match a computed answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
