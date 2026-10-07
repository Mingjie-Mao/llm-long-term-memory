# Facts as keys for raw turns — offline reach v1 — pre-registration

Date: 2026-10-01 (Australia/Sydney). Written before the tool was run.

## Classification

**DEVELOPMENT, zero calls** (local encoder). `train150` decides; `heldout100` in
aggregate is the direction check.

## Why

LongMemEval's own analysis indexes each round under its text **plus the facts extracted
from it**, and returns the round itself: 9.4% better recall@k and 5.4% better accuracy
on average across models ("fact-augmented key expansion"). Here the turns are the value
already (the raw-primary arm); the memories are not yet used to help find them. With
anchors now 96% right (`provenance-v2`), a fact can be attached to the turn it came
from.

## Arms (all fill 4,000 tokens with the raw turns; only the ranking key differs)

| arm | key per turn | ranking |
|---|---|---|
| `plain` | the turn | fused BM25 + dense (the v2 arm's ranking) |
| `expanded` | the turn + the facts anchored to it by `provenance-v2` | fused BM25 + dense over the expanded keys |
| `expanded_stored` | the turn + the facts at their stored (v1) anchors | same — shows what the anchor fix is worth here |

BM25 is computed in Python over each namespace's keys (Okapi, k1 1.2, b 0.75) for every
arm alike, so the arms differ only in their keys; dense vectors are `all-MiniLM-L6-v2`,
re-embedded only for turns whose key changed. Superseded memories are included — a fact
that was later replaced still says what the turn is about.

## Decision rule

The raw-primary ranking adopts expanded keys if, on train150, `expanded` beats `plain` by
at least **2 points** of all-gold coverage at 4,000 tokens with paired losses not above
wins, and on heldout100 `expanded` is not below `plain`.

## What this cannot show

Reach, not use.

## Addendum after the train150 run — 2026-10-01

`expanded` passed on train150 (77.4% -> 81.5%, 7 wins, 1 loss). Before the heldout100
check, one arm is added for implementation rather than for the decision: `expanded_dense_only`
keys BM25 on the plain turn and only the dense side on turn + facts — what a store can do
without a second full-text index. The registered decision stays on `expanded`.
