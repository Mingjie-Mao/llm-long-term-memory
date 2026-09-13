"""Does the raw-conversation fallback have the answer just below its cut?

`abstained_with_source` is the largest failure bucket in every readable pool
(44-50%, results/failure-taxonomy.md), and every one of those failures had already
been through the fallback: raw turns were retrieved, shown, and the model still said
it did not know. So the question is not whether to fetch raw text. It is whether the
*right* raw text was inside the three turns the fallback is allowed to show.

The geometry makes this worth asking. `RawFallback` ranks a pool of
``max(15, max_turns * 5)`` turns and then hands over ``ranked[:max_turns]`` — with
the shipped ``max_turns: 3`` it ranks fifteen and shows three. A needed turn sitting
at rank 4-15 was found and then discarded by the cut, which is a config change; one
beyond the pool is a retrieval problem; one absent entirely is an indexing problem.
These need different fixes and the failure looks identical from outside.

No provider call: this replays BM25 over a store already on disk. No sealed set is
opened — `pilot48` and `tune42` are `train150` slices the data protocol allows to be
read without limit.

**The query is an approximation and the result is a lower bound.** The real call
passes ``verdict.source_query or instance.question``, and `source_query` — keywords
the answerer generates when it asks for source — is not recorded in any row. This
probe uses the question text, which is the path taken whenever `source_query` is
empty. If the model's keywords are better than the raw question, the true ranks are
at least this good, so "the turn was in the pool" cannot be an artefact; only
"the turn was missing" could be.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from failure_taxonomy import classify, load  # noqa: E402

from llm_long_term_memory.store import SQLiteMemoryStore, scoped_session_id  # noqa: E402

POOLS = {
    "pilot48 (v3)": "results/sealed/v3-answer-pilot/*.jsonl",
    "tune42 (v3)": "results/sealed/v3-phase*/*.jsonl",
}
# What the shipped config allows the answerer to see, and how deep the ranking that
# feeds it actually goes. Both come from configs/v2.yaml and RawFallback.__init__.
SHOWN = 3
POOL = 15
# Words too common to say anything about whether the answer was visible.
_STOP = {
    "your",
    "yours",
    "that",
    "this",
    "with",
    "from",
    "have",
    "been",
    "were",
    "they",
    "them",
    "about",
    "when",
    "what",
    "which",
    "there",
    "their",
    "would",
    "could",
    "also",
    "into",
    "just",
    "than",
    "then",
    "some",
    "more",
    "most",
    "other",
    "after",
    "before",
    "because",
    "acceptable",
    "including",
    "following",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="stores/train150.db")
    parser.add_argument("--dataset", default="data/longmemeval_s_cleaned.json")
    parser.add_argument("--depth", type=int, default=50)
    args = parser.parse_args()

    dataset = {
        q["question_id"]: q for q in json.loads((REPO / args.dataset).read_text(encoding="utf-8"))
    }
    store = SQLiteMemoryStore(REPO / args.store)

    for pool_name, pattern in POOLS.items():
        failures = [
            row
            for _, row in load(pattern)
            if not row.get("correct") and classify(row) == "abstained_with_source"
        ]
        # One question can fail in several repeats and arms. The rank of the needed
        # turn is a property of the question and the store, not of the run, so
        # counting rows would multiply the same measurement by its repeat count.
        by_question: dict[str, dict] = {}
        for row in failures:
            by_question.setdefault(str(row.get("question_id")), row)

        buckets: Counter[str] = Counter()
        examined = skipped = 0
        for question_id, _row in by_question.items():
            question = dataset.get(question_id)
            if question is None:
                skipped += 1
                continue
            examined += 1
            gold = {
                scoped_session_id(question_id, s) for s in question.get("answer_session_ids") or []
            }
            ranked = store.search_turns(question_id, question["question"], limit=args.depth)
            position = next(
                (i for i, turn in enumerate(ranked) if turn.session_id in gold),
                None,
            )
            # Session-level presence is not the same as the answer being visible.
            # `answer_session_ids` names conversations, and a conversation has many
            # turns; the turn carrying the fact may not be the turn BM25 ranked. So
            # when the gold session was shown, check whether the gold answer's own
            # distinctive words are actually in the text the model received.
            if position is not None and position < SHOWN:
                shown_text = " ".join(turn.content.lower() for turn in ranked[:SHOWN])
                words = {
                    w
                    for w in "".join(
                        c if c.isalnum() else " " for c in str(_row.get("gold", "")).lower()
                    ).split()
                    if len(w) > 3 and w not in _STOP
                }
                if not words:
                    buckets["shown; gold has no matchable words"] += 1
                elif all(w in shown_text for w in words):
                    buckets["shown AND gold words all present"] += 1
                elif any(w in shown_text for w in words):
                    buckets["shown; gold words partly present"] += 1
                else:
                    buckets["shown; gold words absent from the text"] += 1
            elif position is None:
                buckets[f"absent from top {args.depth}"] += 1
            elif position < POOL:
                buckets[f"IN POOL, CUT (rank {SHOWN + 1}-{POOL})"] += 1
            else:
                buckets[f"beyond the pool (rank {POOL + 1}-{args.depth})"] += 1

        print(
            f"\n{pool_name}: {len(failures)} abstention rows -> {examined} distinct questions"
            + (f" ({skipped} unmatched)" if skipped else "")
        )
        for label, count in buckets.most_common():
            print(f"    {label:<40}{count:>3}  {count / max(examined, 1):>6.1%}")

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
