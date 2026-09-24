# Batch 8 cohort attribution v1 — preregistration

Date: 2026-09-24. Class: retrospective development analysis. Every question set and
session cohort named below has been inspected; no result is unseen final evidence.

## Question and inputs

Explain as far as archived evidence permits why batch 8 scores 64.9% on held-out
sessions 1–60, 44.8% on sessions 61–120, and 47.5% in the built v2b gate16 store.
The baseline is each cohort's recorded batch 8 extraction. There is no new candidate.

- Cohort A: `results/raw/extraction-batch-size.extractions.json`, batch8 arm, 60
  sessions; `results/analysis/extraction-batch-size.json` is its frozen historical
  summary.
- Cohort B: `results/raw/specificity-repair-pilot.checkpoint.json`, baseline arm,
  60 sessions at offset 60; `results/analysis/specificity-repair-pilot.json` is its
  historical summary.
- Cohort C: `stores/v2b-gate16.db`, `results/manifests/v2b-gate16.json`, 780
  namespaced sessions; `results/analysis/store-fidelity.v2b-gate16.json` is its
  historical summary.
- Source corpus: local `data/longmemeval_s_cleaned.json`; model in all three runs
  `gemini-3.1-flash-lite`. Historical batch8 prompts remain bound to archived
  fingerprints. Current prompt may differ and will not be used to rescore extraction.

No provider calls, tokens, or quota. Local Python and corpus reading only. Git HEAD
at registration: `4da58681b4e7ae83ed31529f8efc4c738175f6e9` plus existing
uncommitted working tree. New output namespace:
`results/analysis/batch8-cohort-attribution-v1.{json,md}`.

## Analysis fixed before calculation

1. Recompute each cohort's exact-match numerator and denominator with the existing
   `ingest.fidelity` ruler; require equality with the archived totals.
2. Count session-ID overlap, source-character length, user-assertion length, stated
   specifics by facet, and memory volume per session.
3. Reweight A and B's facet recalls to C's facet mix. This diagnoses only facet
   composition; do not label a remaining difference as a causal pipeline effect.
4. Compare B's extraction-only reading to C's built-store reading. If sets do not
   overlap and no pre-dedup C extraction exists, record that cohort and pipeline
   effects cannot be separated from these artifacts.
5. Preserve any discrepancy or missing artifact as a failed/limited finding. Do not
   alter any historical result, threshold, or frozen/sealed artifact.

The endpoint is a reproducible attribution report, not a promotion or accuracy gate.
