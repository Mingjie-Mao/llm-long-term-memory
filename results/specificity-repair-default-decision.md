# Specificity repair default — decision

## Decision

Keep `batch 8 + specificity_repair` as an explicit high-fidelity ingest profile; do not
make it the universal default yet.

## Evidence

- On `v2b-gate16`, store-level specific recall improved from 47.5% to 60.5%, with
  304 paired gains and 0 losses.
- Specific support remained effectively flat (92.7% to 93.0%).
- The profile used roughly three times the ingestion calls because repair fired for
  45% of sessions.
- Existing QA sets cannot resolve the plausible effect, and all LongMemEval questions
  are exposed. No QA improvement is claimed.
- The temporal-transition replay now passes on all 3,299 memories and fixes the known
  Nike/Adidas false supersession. That removes the concrete lifecycle blocker but does
  not supply the missing latency, quota or end-to-end evidence.

## Product policy

- `configs/v2b-batch8-repair.yaml` is the recommended quality-first offline ingest profile.
- The service and ordinary CLI defaults remain cost-first and keep repair disabled.
- A future default change requires an explicit operating budget and a new
  product-representative or unseen QA set. It must not be justified by replaying the
  exhausted 48-question set.

This is a product-default decision, not a negative result about the repair itself. The
repair passed its registered extraction gates.

## 2026-09-24 preflight update

The new [cohort attribution](analysis/batch8-cohort-attribution-v1.md) shows the
first-60 batch8 64.9% and built-store 47.5% readings are almost entirely disjoint
sessions. Reweighting the built store to the first cohort's facet mix gives 46.9%,
so facet mix alone does not explain the difference. It still cannot separate cohort
from pre-/post-dedup effects without same-session extraction artifacts.

The [delayed paraphrase regression](analysis/retention-paraphrase-v1.md) recovered all
gold source sessions in the top 20 on 16 inspected items, but did not score answers.
The [three-arm preflight](batch-arms-end-to-end-preflight-v1.md) finds no comparable
batch15 store or sufficiently resolving QA set. Thus neither new check changes the
cost-first service default or the quality-first profile recommendation. No QA
improvement, 86%/72% gap closure, or production-scale storage claim is made.
