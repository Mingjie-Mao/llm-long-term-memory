# Pre-registration — frozen v3.3 one-shot dev60 validation

**Written after the archived v3.3 tune42 PASS and before any dev60 question text,
answer, model row or judge reason was read or sent to a provider.**

## Why this document is needed

The phase-3 document registered dev60 while v3 used the same short context as v2,
so its context gate said the candidate median could not exceed v2. Later, still using
only inspectable train/tune data, v3.2 added exact source evidence and registered a
2-times-v2 ceiling; v3.3 then met that ceiling at 1.92 times v2 without losing v3.2's
accuracy. Keeping the obsolete 1-times line would make the already registered evidence
design fail by definition.

This is a prospective amendment, not a result-driven rewrite: dev60 remains unopened,
the v3.3 candidate is already fixed by the archived tune3 result, and the change is
limited to replacing the old 1-times context ceiling with the tune3 ceiling of 2 times
v2. All accuracy, slice, confidence, cost and one-shot rules remain unchanged.

## Fixed execution

- Candidate: the exact `v3.3-compact` source and `configs/v3-phase5-compact.yaml`
  preserved by `results/archive/v3-phase5-tune3/conclusion.json`.
- Data: the existing sealed `v3-reasoning-dev60` manifest; 60 questions disjoint from
  pilot48 and tune42. No `test100` data is used.
- Arms: unchanged `v2-control=two_stage_fallback` and
  `v3.3-compact=two_stage_reasoned_evidence` on the same train150 store.
- Repeats: exactly three per arm, giving 360 complete answer-and-judge rows.
- Rows resume after quota or network interruption, but a durable one-shot ledger
  forbids deleting, refreshing or repeating the run after completion.
- Before the first provider call, the runner, configuration, manifest, store, protocol
  and source are content-hash frozen. Only aggregate results may be reported.
- No candidate change or parameter tuning is allowed after any dev60 provider call.

## Final dev60 gate

Every check must pass:

| check | requirement |
|---|---|
| combined time + multi-session | candidate gains at least 5 percentage points in majority accuracy |
| time questions | candidate majority-correct count is not lower than v2 |
| multi-session questions | candidate majority-correct count is not lower than v2 |
| knowledge updates | candidate majority-correct count is not lower than v2 |
| ordinary controls | candidate majority-correct count is not lower than v2 |
| confident mistakes | no increase in high-confidence majority-wrong answers |
| source-session recall | candidate selected-stage recall is not lower than v2 |
| median context | candidate is no more than 2 times v2 |
| generation cost | candidate answer output tokens are no more than 1.5 times v2 |

A PASS is validation evidence, not permission to reuse `test100`; v3 still needs a
genuinely new hidden final set. Any failed check records a negative result and sends
future work to new train-only data. dev60 is never rerun or used for tuning.

## Aborted first execution — 2026-09-04T19:06:09Z

The first execution of this protocol was started and then aborted **before any row
existed**. It is recorded here rather than in a commit message because a one-shot
protocol whose ledger is deleted must say so in the protocol itself.

What happened: the runner wrote its ledger, issued exactly one answerer request at
19:07:07Z, and then blocked on that request for 31 minutes. The socket stayed
`ESTABLISHED` with both queues empty, the process accumulated 13.7 seconds of CPU
time, and the quota counter never advanced again. `GeminiClient` had constructed the
provider client without `http_options.timeout`, which in google-genai 2.17.0 means no
timeout at all, so a half-open connection blocked forever. Neither retry budget could
fire: both are driven by caught exceptions, and a blocking read raises nothing.

What this cost the protocol: nothing. Zero rows were written, no usage checkpoint was
produced, and **no dev60 question, answer, judge verdict or aggregate was generated or
read by anyone**. The one-shot rule exists to stop a run being repeated after its
result is seen; there was no result.

What changed, and what did not:

| | |
|---|---|
| Changed | `src/llm_long_term_memory/llm/client.py` — a 180-second provider request timeout, plus tests |
| Unchanged | the candidate, `configs/v3-phase5-compact.yaml`, the dev60 manifest, the train150 store, the arms, the repeats, and every gate in the table above |

The freeze was therefore re-captured. The superseded freeze is kept beside the new one
as `freeze.superseded-20260904T190609Z.json`, and the aborted ledger is preserved
verbatim in `results/audit/dev60-transport-hang-abort-20260904T190609Z.json`, so the
`freeze_sha256` that run was bound to remains checkable.

180 seconds is not a free parameter chosen to make a run succeed: the slowest
*successful* answerer call in the archived tune3 run took 101 seconds, and a tighter
timeout would convert a slow provider into missing rows. The timeout is transport
infrastructure, not a candidate parameter, and it affects both arms identically.
