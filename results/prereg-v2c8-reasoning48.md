# v2c.8 reasoning-48 confirmation preregistration

Date: 2026-09-17 (Australia/Sydney)

## Candidate and data

- Candidate is frozen at `memory-aware-v2c.8`.
- Questions are frozen by `results/manifests/v3-reasoning48.json` (48 items).
- Ingestion uses the unchanged batch-8 two-stage extractor in `configs/v2c.yaml`.
- Run the hybrid `two_stage_v2c` arm once. Do not rerun an old baseline.

## Quota-saving store construction

The completed 16-question gate is a preregistered subset of this same 48-question
manifest and uses the identical extractor/config. Seed a new `v2c-reasoning48` store
with byte copies of the gate16 database, vector index, and checkpoint. Then resume
ingestion against the 48-question manifest; completed sessions must be skipped.

Before ingestion, record that the copied files hash-identically to their gate16
sources. This saves roughly 246 request attempts without changing any memory.

## Confirmation gate

The primary result is descriptive because this is one run, not a publication-grade
replication. Promote to final-set evaluation only if all checks pass:

- overall configured-judge accuracy at least 75%;
- knowledge-update accuracy at least 87.5%;
- temporal-reasoning accuracy at least 75%;
- multi-session accuracy at least 50%;
- each remaining single-session type at least 75%;
- median selected context no more than 900 tokens;
- source-session recall at least 95%;
- `778164c6` is separately audited for conflict-aware behavior and excluded from any
  source-consistent accuracy interpretation, regardless of its judge label.

If ingestion hits the daily quota, stop cleanly and resume the same store/checkpoint
after reset. If the confirmation gate fails, report the failure; do not tune on these
48 rows and rerun them.
