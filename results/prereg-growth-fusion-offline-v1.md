# Memory-led turns in the fusion, as history grows — offline v1 — pre-registration

Date: 2026-10-01 (Australia/Sydney). Written before the tool was run.

## Classification

**DEVELOPMENT, zero calls** (local encoder), on scratch copies of `train150` grown with
`history_growth_retention._grow`; the store is not written.

## Why

`results/analysis/history-growth-retention-v1.md`: as each user's history grows to 4x,
a gold-anchored memory stays in the top 20 for 73.3% -> 71.9% of questions, while
question-found turns cover all gold at 4,000 tokens for 78.1% -> 67.8%. The memory layer
holds; the raw locator does not. Zep retrieves facts and episodes together; Mastra keeps
a compressed log that does not grow with the transcript. The cheapest version of that
here: let the turns the retrieved memories point at vote in the fusion.

LongMemEval's time-aware query expansion is the other published remedy. It is **not**
measured here: the simulation dates donor history after the user's own, so a time filter
would remove it for reasons the simulation invented. That is stated, not worked around.

## Arms (4,000 tokens, whole turns in rank order)

| arm | ranking |
|---|---|
| `bm25` | BM25 of the question over the namespace's turns |
| `hybrid` | fused BM25 + dense (the v2 arm's ranking) |
| `hybrid_memory` | fused BM25 + dense + the turns anchored by the top-20 retrieved memories, in memory rank |

Growth factors 1x and 4x. All-gold coverage over the user's own flagged turns.

## Decision rule

The raw-primary ranking adds the memory list if `hybrid_memory` beats `hybrid` by at least
**2 points** at 4x, and is not more than **1 point** below it at 1x.

## What this cannot show

Donor history is other people's: it competes and never contradicts. Reach, not use.
