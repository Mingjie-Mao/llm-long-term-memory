"""Can retrieval put a gold session in the top M, without being told which it is?

    python scripts/session_recall.py --store heldout100 --questions heldout100.json

**No LLM calls.** Local encoder, the real retriever, the real store. This is the
gate [the context-shape pre-registration](../results/prereg-context-shape.md) runs
before spending any answerer quota:

> If session recall @ top-M is below 80% on `train150`, `coherent-auto` is not run
> on `dev100` — the development work moves to session selection first, and no
> validation quota is spent on an arm already known to be starved.

The reason to expect it to pass is that the gold session is already being found.
`source_session_recalled` is 94% on `heldout100` and source-session recall has been
93.5-94.0% on every store measured; the question is only whether it survives being
aggregated into a *session* ranking and cut at M.

Those are different things, and the difference is the whole point. A gold memory
ranked 18th of 20 counts for `source_session_recalled` and may still lose its
session to two conversations that each contributed three stronger memories.

## On which set to run it

`train150` is the set this decides on. `heldout100` is spent, so running it there
costs nothing that is not already gone — but its numbers **may not be used to pick
M**. The budget is a `train150` decision by pre-registration, and a threshold
chosen on a spent test set is still a threshold chosen by looking at outcomes.
Use `heldout100` to check that this code does what it says.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_long_term_memory.config import Settings  # noqa: E402
from llm_long_term_memory.evaluation.datasets import longmemeval as lme  # noqa: E402
from llm_long_term_memory.evaluation.manifest import load_manifest  # noqa: E402
from llm_long_term_memory.retrieve import rank_sessions  # noqa: E402

MS = (1, 2, 3, 5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="train150")
    parser.add_argument("--questions", default="train150.json")
    parser.add_argument("--config", default="configs/fallback.yaml")
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from llm_long_term_memory.api.service import MemoryService

    settings = Settings()
    manifest = load_manifest(settings.results_dir / "manifests" / args.questions)
    inst = {i.question_id: i for i in lme.load("s", settings.data_dir)}

    svc = MemoryService(store_name=args.store, config_path=args.config)
    print(f"session recall · store {args.store} · {len(manifest)} questions · no LLM calls\n")

    hits_at = Counter()
    memory_level = 0
    ranks: list[int | None] = []
    skipped: list[str] = []
    rows = []

    try:
        for qid in manifest.question_ids:
            question = inst[qid]
            gold = set(question.answer_session_ids)
            if not gold:
                skipped.append(qid)
                continue

            result = svc.search(qid, question.question, limit=args.candidate_limit)
            hits = list(result.memories)
            if any(h.memory.source_session_id in gold for h in hits):
                memory_level += 1

            ranked = [s for s, _ in rank_sessions(hits)]
            position = next((i + 1 for i, s in enumerate(ranked) if s in gold), None)
            ranks.append(position)
            for m in MS:
                if position is not None and position <= m:
                    hits_at[m] += 1
            rows.append(
                {"question_id": qid, "gold_session_rank": position, "sessions": len(ranked)}
            )
    finally:
        svc.close()

    n = len(ranks)
    if not n:
        print("no questions with gold sessions; nothing measured")
        return 1

    print(f"{'gold session in top M':<26}{'n':>6}{'recall':>9}")
    for m in MS:
        print(f"  M = {m:<21}{hits_at[m]:>6}{hits_at[m] / n:>8.1%}")
    print(
        f"\n  {'any rank at all':<24}{sum(r is not None for r in ranks):>6}"
        f"{sum(r is not None for r in ranks) / n:>8.1%}"
    )
    print(f"  {'memory-level (for contrast)':<24}{memory_level:>6}{memory_level / n:>8.1%}")
    found = [r for r in ranks if r is not None]
    if found:
        found.sort()
        print(f"  median gold-session rank  {found[len(found) // 2]}")
    if skipped:
        print(f"\n  {len(skipped)} question(s) carry no gold session and were excluded")

    gate = hits_at[3] / n
    verdict = "PASS" if gate >= 0.80 else "STOP"
    colour = "\033[32m" if gate >= 0.80 else "\033[31m"
    print(
        f"\n{colour}{verdict}\033[0m  the pre-registered gate is 80% at the chosen M "
        f"(M=3 here: {gate:.1%})"
    )
    if args.store == "heldout100":
        print("\033[33mheldout100 is spent — this run checks the code, and its numbers")
        print("may not be used to choose M. That decision belongs to train150.\033[0m")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "store": args.store,
                    "questions": args.questions,
                    "n": n,
                    "recall_at": {str(m): hits_at[m] / n for m in MS},
                    "memory_level": memory_level / n,
                    "rows": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
