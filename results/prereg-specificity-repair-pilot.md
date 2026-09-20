# Specificity repair pilot — pre-registration

Date: 2026-09-19 (Australia/Sydney)

## Classification and data

This is a **DEVELOPMENT mechanism experiment**, not an unseen final evaluation and not
an end-to-end QA result. LongMemEval-S is exhausted; every question has been exposed.
The unit here is a source-text specificity, not a benchmark answer.

- Dataset: LongMemEval-S, the deterministic `split_dev_test(seed=0)` holdout pool.
- Cohort: sessions 61–120 in the same traversal used by `lltm ingest fidelity` (zero-based
  offset 60, 60 sessions). The earlier batch-size experiment used sessions 1–60.
- The ordered session IDs and a SHA-256 digest of the source text are written before the
  first provider request and must match on resume.
- Config: `configs/v2.yaml`, except the baseline batch size is fixed to 8.
- Extractor model: `gemini-3.1-flash-lite`.
- Existing frozen/sealed stores and results are not read or modified.

## Arms

One base extraction supplies both arms; the baseline is never rerun separately.

1. **baseline** — current two-stage extractor, at most 8 sessions per request.
2. **candidate** — the exact baseline memories plus a conditional grounded repair:
   - deterministically compare user assertions with baseline memory contents using the
     existing fidelity facet detector;
   - call the grounded extractor once only for a session with at least one missing
     quantity, duration, money value, date, relative-time expression, or proper noun;
   - accept a repair fact only if it is grounded to an exact source span and at least one
     baseline-missing specificity occurs in both the exact span and the repaired content;
   - candidate construction is additive, so a baseline memory is never removed or edited.

The first 60-session archive is only an offline feasibility check: targeted union raised
fidelity from 64.9% to 82.8% using 31 grounded memories from 15 sessions. It is not part
of the registered outcome below.

## Primary and gates

The primary endpoint is paired specificity retention on the registered cohort.

Promote the mechanism to an ingestion integration candidate only if all hold:

1. candidate overall fidelity improves by at least **10 percentage points** over baseline;
2. candidate gains at least **10** individual specifics and loses **0** (losses should be
   structurally impossible because the candidate is an additive union);
3. every accepted repair fact has a valid turn index and exact source offsets, and every
   specificity credited to repair appears in both its cited span and its content;
4. at most **50% of sessions** require a repair call;
5. added memories average at most **1.0 per session over the full 60-session cohort**;
6. the run finishes without changing any frozen/sealed/archive artifact.

Facet results are descriptive. No significance claim is made; the cohort is too small for
stable per-facet inference. A pass permits implementation plus a separate QA gate, not a
benchmark claim and not a full ingest.

## Cost and stopping

- Baseline: at most 16 successful extraction calls (two-stage, 8 batches).
- Repair: zero to 60 successful calls, paid only for deterministically flagged sessions.
- Expected from the prior cohort: roughly 15 repair calls, so about 31 successful calls.
- Hard ceiling: 76 successful calls; no answerer or judge calls.

The checkpoint is `results/raw/specificity-repair-pilot.checkpoint.json`; results and usage
use the same namespace. Provider/quota failure preserves completed sessions and resumes the
same checkpoint. The run is not restarted under a new label, and a failed gate is retained.
