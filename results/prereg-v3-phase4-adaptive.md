# Pre-registration — v3.2 adaptive source evidence

**Written after the completed tune1 negative result and before any v3.2 provider call.**

## Why this iteration exists

Tune1 tied v2 at 66.7%. Twelve of v3.1's 14 wrong answers covered every labelled
answer session, but compact memories often omitted an exact number, relative date,
event state or list member. Prompt wording alone did not recover that lost detail.

## Candidate and fixed boundary

- Reuse the frozen tune1 `v2-control` rows; do not pay for another baseline call.
- Run exactly one new `v3.2-adaptive` row for each of the same 42 tune questions.
- Store, retrieval ranking, top-k, fallback and answer prompt remain fixed.
- Only temporal, multi-session aggregation and current-state operations receive
  hydrated verbatim spans from already-selected memories, capped at 800 tokens.
- Direct lookups and preference applications do not receive proactive hydration.
- The tune split may guide development. `dev60` remains sealed and unseen.
- No dev60 or test100 question, answer, row or judge reason may be read.

## Tune2 promotion signal

This is a tune diagnostic, not the dev60 gate. Continue to a frozen candidate only if:

| check | requirement |
|---|---|
| time + multi-session | combined accuracy improves by at least 10 percentage points over the frozen v2 tune baseline |
| each target type | neither temporal nor multi-session accuracy is lower than v2 |
| overall | accuracy is not lower than v2's 66.7% |
| ordinary controls | combined ordinary accuracy is not lower than v2 |
| context | hydration is applied only to registered reasoning kinds; median context no more than 2x v2 |
| output cost | answer output tokens no more than 1.5x v2 |

Passing permits one candidate freeze and a separately authorised dev60 run. Failing
is recorded as another negative result; it does not permit repeated dev60/test100 use.
