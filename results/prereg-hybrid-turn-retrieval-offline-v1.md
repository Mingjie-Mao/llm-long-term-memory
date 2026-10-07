# Dense + BM25 turn retrieval — offline reach v1 — pre-registration

Date: 2026-09-30 (Australia/Sydney). Written before the tool was run.

## Classification

**DEVELOPMENT, zero calls** — a local embedding model, no provider. `train150` decides;
`heldout100` in aggregate is a direction check.

## Why

Of the raw-primary arm's 23 train150 errors, 5 had no gold turn in context and 9 only
part of it. A larger budget did not fix that
(`results/analysis/adaptive-raw-budget-offline-v1.md`: +3.4 points on train150, +0.0 on
heldout100), so the missing turns are not just below the cut — BM25 does not rank them.
The hand-read cases fit that: "the life event of one of my relatives a week ago" shares no
word with "my cousin's wedding". A dense turn matcher is what Emergence and the
LongMemEval paper use for turns.

This is a retrieval change, which AGENTS.md admits only on evidence that retrieval is
the bottleneck. The evidence is above: for this arm, the first loss of 14 of 23 errors is
turn retrieval.

## Arms (4,000-token budget, whole turns in rank order, skip what overflows)

| arm | ranking |
|---|---|
| `bm25` | BM25 of the question over the namespace's turns (the registered arm) |
| `dense` | cosine between the question and each turn, `all-MiniLM-L6-v2` (the store's encoder) |
| `hybrid` | reciprocal-rank fusion of the two, k = 60 |

Turns longer than the encoder's input window are embedded from their beginning only; the
limitation is stated rather than worked around, because chunking would be a second
change.

## Decision rule

The next paid arm replaces BM25 with the better of `dense` / `hybrid` if, on train150:

1. its all-gold coverage at 4,000 tokens is at least **3 points** above `bm25`; and
2. its paired losses against `bm25` do not exceed its wins;

and on heldout100 its all-gold coverage is not below `bm25`'s. Otherwise BM25 stays.

The budget is unchanged, so the cost rule of the arm is unaffected.

## What this cannot show

Reach, not use.
