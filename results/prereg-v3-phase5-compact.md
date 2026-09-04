# Pre-registration — v3.3 compact, session-fair source evidence

**Written after the archived v3.2 tune2 stop-result and before any v3.3 provider call.**

## Why this final tune iteration exists

v3.2 improved tune42 accuracy from 66.7% to 73.8%, temporal accuracy from 40%
to 60%, held multi-session accuracy at 70%, and introduced no other registered
regression. It stopped only because median context was 2.52 times v2, above the
fixed 2 times ceiling. Inspection showed that 28 hydrated questions almost always
consumed the full 760–800-token evidence allowance.

The local, zero-provider-call projection in
`results/analysis/v3-phase5-context-projection.json` replays the same retrieval and
the observed fallback sizes. Exact-sentence, session-fair packing at 425 tokens
projects a 1,124-token median (1.92 times v2). Mean labelled-session hydration is
94.9% versus v3.2's 94.3%, and the full-coverage rate remains 85.7%. This projection
estimates context only; it makes no accuracy claim.

## Candidate and fixed boundary

- Reuse the hash-bound `v2-control` and `v3.2-adaptive` tune42 rows. Do not rerun
  either reference arm.
- If separately authorised, run exactly one new `v3.3-compact` row for each of the
  same 42 inspectable tune questions.
- Store, retrieval, top-k, temporal rendering, fallback, answer prompt, answer model
  and judge remain fixed.
- The only candidate change is source-evidence packing: zero neighbouring sentences,
  a 425-token cap, identical rendered-source deduplication, and round-robin allocation
  across already-selected sessions while preserving rank within each session.
- Direct lookup and preference operations remain unhydrated.
- `dev60` remains sealed and unseen. No dev60 or test100 question, answer, result row
  or judge reason may be read.
- This is the last tune42 iteration for this evidence-packing approach. A failure
  requires retiring or redesigning it on new train-only development data, not repeated
  fitting to these 42 questions.

## Tune3 promotion signal

This tune gate must preserve v3.2's accuracy gain while restoring the original cost
boundary. Every check must pass:

| check | requirement |
|---|---|
| overall accuracy | v3.3 is not below v3.2's 73.8% |
| temporal | not below v3.2's 60% |
| multi-session | not below v3.2's 70% |
| ordinary controls | not below v3.2's combined 88.2% |
| confident errors | no more high-confidence wrong answers than v3.2 |
| targeted hydration | applied only to temporal, aggregation and current-state operations |
| labelled source coverage | mean hydrated labelled-session coverage and full-coverage rate are not below v3.2 |
| median context | no more than 2 times v2 and lower than v3.2 |
| output cost | answer output tokens no more than 1.1 times v3.2 |

Passing permits a new candidate freeze and a separately authorised, one-time dev60
run. Failing is preserved as a stop-result and does not permit opening dev60/test100.
