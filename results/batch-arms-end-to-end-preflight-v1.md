# Batch 15 / batch 8 / batch 8 + repair — end-to-end preflight

Date: 2026-09-24. Class: development decision, not an experiment result. No provider
calls, tokens, or quota were spent for this preflight. Existing frozen and sealed
artifacts were not changed.

## Decision

**Do not run or publish a three-arm QA comparison on the current 16-question set.**
This is a no-go for this instrument, not evidence that the arms are equally good.

## Why the existing artifacts cannot form the comparison

The `v2b-gate16` store and its repair counterpart cover the same 780 namespaced
sessions. The batch 15 reading, 39.8%, comes from an older build, not a third
same-version gate16 store. Since then extraction, temporal keying, lifecycle and
answer composition have changed. Joining the old batch15 result to these two stores
would confound batch size with code and prompt lineage. The 64.9% first-60 extraction
reading is a disjoint cohort, not a third arm; the
[cohort analysis](analysis/batch8-cohort-attribution-v1.md) verifies only one-session overlap with
gate16. Neither figure predicts a matched three-arm QA gain.

The 16 questions were selected for known failures and have been inspected. Their raw
accuracy is a development diagnostic, not a population estimate or an unseen final.
The [resolution audit](analysis/resolution-batch8-repair-qa-v1.md) projects about one
self-disagreeing item from answerer noise and says a paired net below two cannot be
resolved by one run per arm. A one-answer swing would not justify a product default.
The [repair gate](v2b-gate16-repair-decision.md) measured extraction, **not QA**.

## What a valid future comparison requires

1. Register a new product-representative evaluation set and freeze its questions,
   tenants, gold provenance, baseline/candidate configs, answerer, judge, top-k,
   context budget, repeated-run rule and cost ceiling before any model call. Exposed
   LongMemEval items remain development/regression, never unseen final.
2. Rebuild all three stores from the same namespaced sessions under one source commit,
   extractor/prompt version and temporal schema. Keep batch 15, batch 8, and batch 8
   plus repair in separate new namespaces. Verify tenant isolation and same-session
   extraction *before* QA.
3. Run the identical answer protocol on each arm, with paired per-question outcomes
   and repeats sufficient to measure run noise. Report exact-detail recall, source
   coverage, answer accuracy, context, calls/tokens, latency and failures together;
   preserve negative results. Do not choose a winner from a post-selected subset.
4. Promote a default only if the registered QA benefit clears run noise and the
   operating budget. A measured extraction gain alone is insufficient.

The existing repair build logged 621 extractor/repair/adjudication call attempts,
including 12 failures retried, as recorded in its
[decision](v2b-gate16-repair-decision.md), versus
about 232 extraction/adjudication calls for the batch8 baseline in the earlier gate.
A full same-version three-arm rebuild would therefore require on the order of a
thousand ingestion calls **before** repeated answering and judging. That is a planning
order of magnitude, not a quote or a new usage measurement. The current set cannot
justify that spend.

## What remains established

- Batch size and repair improve the registered exact-detail extraction instrument;
  batch8 plus repair reached 60.5% from 47.5% on the built gate16 store.
- The 16-item delayed paraphrase [regression](analysis/retention-paraphrase-v1.md)
  found all gold source sessions in top 20, but it did not call an answerer and its
  recency weight is zero. It cannot close the QA question.
- The frozen comparable end-to-end gap remains full-history 86% versus memory 72%.
  No number in this report supersedes it.
