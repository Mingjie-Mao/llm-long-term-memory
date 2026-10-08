# Question-routed raw-turn budget — offline reach v1 — pre-registration

Date: 2026-09-30 (Australia/Sydney). Written before the tool was run.

## Classification

**DEVELOPMENT, zero calls**, on `train150`, the set reserved for zero-call sweeps of
context shape. It uses evidence reach only; the answer runs of
`results/prereg-raw-primary-train150-v2.md` are not re-read to choose anything.
`heldout100` is reported in aggregate as a direction check.

## Why

The raw-primary arm passed on train150 (net +29). Of its 23 remaining errors, 9 had
**only part** of the gold evidence in context — 7 of them multi-session totals and counts
whose answer is spread over several conversations — and 5 had none. A fixed 4,000-token
budget is spent the same way on "where did I redeem the coupon?" and "how much did I
spend on luxury items in total?". The second needs more turns.

The product cannot see the benchmark's question type. Routing uses
`reasoning_kind(question)` (`evaluation/runners/reasoning.py`), which reads the question
wording only.

## Policies

| policy | raw-turn budget |
|---|---|
| `fixed_4k` | 4,000 for every question (the registered arm) |
| `fixed_8k` | 8,000 for every question (reference) |
| `adaptive_agg` | 8,000 when routed `multi_session_aggregation`, else 4,000 |
| `adaptive_agg_temporal` | 8,000 when routed `multi_session_aggregation` or `temporal`, else 4,000 |

Turns: question-found BM25 over the namespace, whole turns in rank order, skipping what
overflows — `retrieve.excerpts.archive_excerpts`.

## Measures

- all-gold coverage (every `has_answer` turn in context), overall and by benchmark type;
- **projected total context** per question: that question's recorded context in the
  train150 control run, plus the raw tokens the policy spends, plus 100 for date
  headers; the median of that.

## Decision rule

The next paid arm uses the adaptive policy with the higher overall all-gold coverage,
provided **both**:

1. its all-gold coverage is at least **3 points** above `fixed_4k`;
2. its projected median total context is at most **6,000** tokens.

If neither adaptive policy qualifies, the next arm keeps `fixed_4k`. `fixed_8k` is a
reference and cannot be chosen: it breaks the cost rule by construction.

## What this cannot show

Reach, not use. A pass selects the budget for an answer experiment on another
development set; it predicts nothing about its size.
