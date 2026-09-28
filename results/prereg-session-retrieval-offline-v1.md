# Whole sessions found through their turns — offline reach v1 — pre-registration

Date: 2026-09-29 (Australia/Sydney). Written before the tool was run.

## Classification

**DEVELOPMENT, zero calls.** `train150` decides (the set reserved for zero-call sweeps
of context shape); `heldout100` is reported in aggregate only, as a check that the
direction holds, under its protocol.

## Why

Emergence AI reports 82.4% on LongMemEval-S with a public configuration that matches
the question against **turns** but hands the reader **whole sessions**, scoring each
session by how its turns rank among the top hits. This project's registered arm
(`results/prereg-raw-primary-heldout100-v1.md`) hands over the matched turns themselves.
Turns are cheaper; whole sessions keep the context around a turn — the question before
it, the correction after it — which the counting and temporal failures on the batch-8
line appear to need. Before paying for a second arm: does the session-level layer
reach more of the gold evidence at the same budget?

## Arms (whole units in rank order, skip what overflows, as the earlier gate)

| arm | unit | order |
|---|---|---|
| `bm25_turns` | turns | BM25 of the question (the registered raw-primary arm) |
| `bm25_turn_window` | a matched turn with its neighbours (±1) | BM25 rank of the matched turn |
| `bm25_sessions` | whole sessions | Σ over the session's turns in the top 50 of 1/log₂(rank+1) — a DCG over matched turns |

Budgets 2,000 · 4,000 · 8,000 · 16,000 tokens. Primary measure: all-gold coverage (every
`has_answer` turn in context). BM25 only: the store has no turn embeddings, so this is a
lower bound on what an embedding-plus-reranker turn matcher would do.

## Decision rule

The next registered answer arm uses **whole sessions** if, on train150, `bm25_sessions`
beats `bm25_turns` by at least **5 points** of all-gold coverage at any budget up to
8,000, with paired losses not exceeding wins at that budget. Otherwise it stays with
turns. 16,000 is reported and does not decide: it is four times the registered budget
and a sixth of full history's median context.

## What this cannot show

Reach, not use — v5.0 delivered three target sentences and converted one. A session can
carry the gold turn and still bury it.
