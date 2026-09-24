"""Separate the two things the knowledge-update oracle changed at once.

    python scripts/context_arms.py            # plan, no calls
    python scripts/context_arms.py --run      # ~15 answerer calls

[ku_oracle.py](ku_oracle.py) handed the answerer the gold sessions' memories in
event order and two of five failures flipped — including `69fee5aa`, which was
the control and was not supposed to be reachable that way. A control that moves
says the arm is not doing what its name claims.

It changed two variables together:

    session coherence   gold-session scoped, ordered by event time
    size                five to seven memories instead of twenty

This runs a third arm that holds size fixed and drops coherence.

    flat20      the production configuration — top_k 20, retrieval rank order
    flatN       top_k trimmed to the coherent arm's count, still rank order,
                still scattered across sessions
    coherent    gold-session memories in event order

    flatN fixes them   -> the variable is distractor count, and the fix is a
                          smaller k, which costs nothing and needs no new machinery
    only coherent      -> the variable is coherence, and P6 is worth building
    neither            -> ku_oracle's flips came from the gold-session scoping
                          itself, i.e. from knowing the answer

`flat20` is the reproduction check. If it does not reproduce the run's wrong
answers, nothing else here is interpretable — the comparison would be against a
baseline that is not the baseline.

The two questions that matter are `71315a70` and `69fee5aa`. **Their needed
memories were already in the production context** — the run's wrong answers quote
them ("5-6 hours … additional 10-12", "37 … which includes a 1915-S Barber
quarter"). So for these two, precision is not the variable and this comparison is
clean. The other three are carried because they cost nothing and their
non-conversion is itself informative.

Judged by reading. A computed answer scored by substring is the mistake this
project has already made three times.
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

sys.path.insert(0, str(REPO / "scripts"))
from ku_oracle import FAILURES  # noqa: E402

STORE = "heldout100"
CONFIG = "configs/fallback.yaml"
OUT = REPO / "results" / "raw" / "context-arms.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--reps", type=int, default=1, help="repeats per arm; >1 measures run-to-run variance"
    )
    args = parser.parse_args()

    settings = Settings()
    inst = {i.question_id: i for i in lme.load("s", settings.data_dir)}

    print(f"context arms - {len(FAILURES)} questions x 3 arms - store {STORE}")
    print("  flat20    production: top_k 20, rank order")
    print("  flatN     same count as coherent, rank order, scattered")
    print("  coherent  gold-session memories, event order")
    print(f"\n  ~{len(FAILURES) * 3} answerer calls, no extractor quota")

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

            coherent = [m for m in svc.store.iter_all(qid) if m.source_session_id in gold_sessions]
            coherent.sort(
                key=lambda m: (m.occurred_at or m.valid_from or "", m.source_turn_index or 0)
            )
            n = len(coherent)

            ranked20 = [h.memory for h in svc.search(qid, question.question, limit=20).memories]
            rankedN = [h.memory for h in svc.search(qid, question.question, limit=n).memories]

            arms = {"flat20": ranked20, "flatN": rankedN, "coherent": coherent}
            record = {"stage": stage, "why": why, "gold": str(question.answer), "n": n}
            print(f"\n  {qid}  [{stage}]  coherent n={n}")
            print(f"    gold     : {str(question.answer)[:100]}")
            for arm, memories in arms.items():
                texts = [
                    svc.answerer.answer_with_memories(question, memories) for _ in range(args.reps)
                ]
                record[arm] = {"k": len(memories), "answers": texts}
                print(f"    {arm:9s}(k={len(memories):2d}):")
                for j, text in enumerate(texts):
                    print(f"       run{j + 1}: {str(text)[:96]}")
            out[qid] = record
    finally:
        OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {OUT}")

    print("\nRead these. flat20 must reproduce the run's wrong answers or nothing")
    print("else here is interpretable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
