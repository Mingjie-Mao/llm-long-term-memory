# v2b gate16 — quota-minimal mechanism gate

**Registered 2026-09-16 before any batch-8 extraction for these questions.**

This supersedes `prereg-v2b-execution.md` as the next execution step. The earlier
full-`train150` attempt stopped with zero sessions and zero memories because the
daily extractor quota was already exhausted. Its failed-call usage ledger remains
part of the audit trail; no result was observed.

## Funnel

```text
archived offline evidence -> gate16 -> reasoning48 -> one final candidate
          0 calls          conditional    conditional       conditional
```

The 150-question store, three repeats, and a new hidden set are not purchased
until the preceding gate passes.

## Fixed selection

Source pool: `results/manifests/v3-reasoning48.json`.

Question quotas are 4 temporal, 4 multi-session, and 2 each for knowledge-update,
single-session-user, single-session-assistant, and single-session-preference.
Within each type, the selector takes up to half majority-wrong questions under the
three already archived v2-control runs, then majority-right regression controls.
Ties are sorted by:

```text
sha256("20260916-v2b-gate16-v1|<type>|<majority-correct>|<question-id>")
```

The resulting manifest has 9 majority-right controls and 7 majority-wrong targets.
It was generated without any batch-8 store or answer.

This deliberately targeted set is a mechanism diagnostic. Its accuracy must not be
reported as a representative 16-question estimate.

## Spend boundary

Only the batch-8 candidate is ingested. The batch-15 control uses the existing
`train150` store and archived v2-control rows. It is not re-ingested or re-answered.

The 16 questions contain 780 namespace-scoped sessions. At batch 8 they require at
most 106 two-stage chunks / 212 extraction calls before dedup adjudications. This
fits inside one 500-request day unless adjudication demand is unusually high; if it
does not fit, the checkpoint resumes on the next reset.

For scale, the completed batch-15 `train150` ingest used 1,391 adjudications for
18,519 written memories (7.5%). Applying that rate to the held-out batch-8 yield
projects roughly 218 adjudications here, or about **430 extractor-quota requests in
total**. This is a planning estimate, not a cap: the gate should fit in one fresh
500-request day, but two days are allowed if the candidate creates more near
duplicates than the old store.

## Gate order

1. **Offline, zero-call:** re-score archived batch-15/batch-8 extraction fidelity;
   report per-facet gains/losses and lexical-support floor for newly emitted facts.
2. **Ingest:** build `v2b-gate16` under `configs/v2b-batch8.yaml`; require a complete,
   homogeneous store before any answer call.
3. **Evidence gate, zero-call:** compare batch-15 and batch-8 structured-literal
   coverage, source coverage, retrieved context, and context tokens. Do not answer
   questions whose evidence did not change when estimating the mechanism's direct
   conversions.
4. **Answer gate:** run the unchanged v2 answer path once for the batch-8 candidate.
   Reuse the archived three-run batch-15 majority as the control. Save answers;
   deterministic exact checks run before any new judge call.
5. **Judge gate:** judge only rows not decided by deterministic checks, then compute
   paired wins/losses.

## Promotion rule

Expand to all 48 questions only when:

- candidate wins exceed candidate losses;
- memory-only outcomes improve in net;
- final answers do not regress in net;
- newly emitted memories do not show a material fall in the lexical-support floor;
- the evidence comparison supports a mechanism, rather than a score-only change.

An observed `0 wins / >=3 losses` stops v2b immediately. Any smaller ambiguous
result may justify one repeat of changed rows only, never an automatic 48-question
run.

All later raw/coherent/planner candidates must also improve a zero-call evidence
endpoint before answerer quota is allowed.
