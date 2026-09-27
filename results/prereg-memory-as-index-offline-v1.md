# Memory as an index — offline evidence-reach gate v1 — pre-registration

Date: 2026-09-28 (Australia/Sydney). Written before the tool was run.

## Classification

**DEVELOPMENT, zero-call diagnostic.** `train150` is the set the evaluation protocol
reserves for zero-call sweeps of retrieval and context shape, and the only set whose
individual failures may be read (`docs/EVALUATION.md`). No answerer, judge or extractor
is called. Nothing here is an accuracy result.

## Why

On the frozen `test100` run, v2 answered 28 questions wrong. In 25 of them the gold
session had already reached the answer context; in 3 it had not
(`results/final/test100-aggregate.md`). Half of the 14-question gap to full history is in
single-session questions, which full history answered perfectly. v2 hands the answerer a
median of 574 tokens of extracted facts, and the raw fallback adds at most three turns.

The proposal is to stop using extracted memories as the *answer* and use them as the
*index*: retrieval ranks memories as it does now, each memory points to the turn it was
extracted from, and the answerer receives those source turns (plus the facts) under a
larger but bounded budget. External evidence points the same way: LongMemEval's own
analysis found facts help as retrieval keys and hurt as stored values, and a controlled
ablation on LongMemEval-S found verbatim chunks beat extracted artifacts (67.4% vs
45.4%), with artifacts added alongside chunks preserving accuracy.

Before paying for answers, the free question: **at a given token budget, does locating
source turns through memories put the gold evidence in front of the answerer more often
than searching the raw archive with the question directly?** If it does not, the memory
layer is not earning its place as an index, and the paid step should not run.

## Data and system

- Manifest `results/manifests/train150.json` (150 questions), store `stores/train150.db`.
- Retrieval: `configs/v2.yaml` (semantic only, candidate limit 50), temporal filtering on,
  as served. Top 50 memories are ranked; arms consume them in rank order.
- Gold evidence: LongMemEval's own `has_answer` turn flags. Questions without any flagged
  turn are excluded from coverage and counted.
- Tokens: characters / 4.6, the answer runner's own estimate.

## Arms (each filled in rank order up to budget B; a unit that would overflow is skipped
and filling continues)

| arm | unit | order |
|---|---|---|
| `mem_turn` | the whole turn each memory was extracted from | memory rank |
| `mem_window` | that turn and its neighbours (±1 turn) | memory rank |
| `mem_session_bm25` | turns inside the sessions memories came from | BM25 of the question within those sessions |
| `bm25_turns` | any turn in the namespace | BM25 of the question (question-driven raw retrieval) |
| `mem_session_whole` | whole sessions | memory rank |

Budgets B: 1,000 · 2,000 · 4,000 · 8,000 tokens. v2's median context is 574; full
history's is 109,059. Also reported: whether any of the top 20 / top 50 memories is
anchored to a gold turn (the index's own hit rate).

## Primary measure and decision rule

Primary: **all-gold coverage** — the share of questions whose every flagged turn is in
the arm's context — paired per question against `bm25_turns` at the same budget.

The paid step (a registered answer comparison on a development set) is justified only if
**both** hold:

1. at B = 4,000, the better of `mem_turn` / `mem_window` covers all gold turns for at
   least **80%** of questions with gold turns;
2. at B = 4,000 it is **not worse** than `bm25_turns`: paired losses do not exceed wins.

The budget carried forward is the smallest B whose coverage is within 2 points of the
same arm at 8,000. If no memory arm passes, the finding is recorded and the index
direction is not paid for.

## What this cannot show

Coverage is reach, not use. v5.0 delivered three target sentences and converted one
(`results/v5-reasoning48-decision.md`). A pass here permits an answer experiment; it
does not predict its size. Multi-session aggregation questions need several turns at
once, which is exactly what all-gold coverage measures and why it is the primary.
