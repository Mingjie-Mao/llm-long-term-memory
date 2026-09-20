# v2d.3 development gate (16 questions)

Date: 2026-09-18 (Australia/Sydney)

This is a **DEVELOPMENT RESULT**, not an unseen final evaluation. All questions come
from the previously exposed `v3-reasoning-tune42` split. The manifest was fixed before
reading any v2d answer or judge outcome.

## Candidate and fixed inputs

- manifest: `results/manifests/v2d-gate16.json`;
- candidate: `memory-aware-v2d.3-arithmetic-complete`;
- ingestion: unchanged batch-8 two-stage extraction from `configs/v2b-batch8.yaml`;
- answer configuration: `configs/v2d.yaml`;
- extractor: `gemini-3.1-flash-lite`;
- answerer: `gemini-3.5-flash-lite`;
- judge: `gemma-4-31b-it`;
- store/result namespace: `v2d-gate16` / `two_stage_v2d.v2d-gate16`;
- git base SHA at registration: `8a56859c724c1134286da3fe53544b22b41fc79b`;
- worktree is intentionally dirty with the candidate implementation and prior v2c work.

The frozen paired baseline is
`results/sealed/v3-phase3-tune1/v2-control.jsonl`; do not rerun it. The candidate runs
once. Do not use `v3-reasoning48` for tuning or rerun it.

## Why these 16

- 13 ordinary count/arithmetic items cover count, grouped count, duration, sum,
  difference, average, and mixed time units;
- `60bf93ed_abs` checks a false premise combined with duration;
- `7e00a6cb` and `0e5e2d1a` check exact assistant-source detail and regression.

Selection is based on question form and mechanism, not baseline correctness.

## Estimated quota

The 16 namespaces contain 775 unique sessions. At eight sessions per extraction batch,
there are 105 Stage-A batches and approximately the same number of Stage-B calls:
about 210 successful extraction calls plus retries. QA adds 16 answerer calls, up to 16
conditional fallback calls, and judge calls/retries. Expected total is roughly 250-330
requests, within one 500-RPD day if provider retries remain moderate.

## Registered decision rule

Promote v2d beyond this development gate only if all hold:

1. candidate overall correct count is at least the frozen v2 baseline;
2. on the 14 count/arithmetic/false-premise items, paired wins exceed paired losses;
3. neither of the two source-detail regression items changes from correct to incorrect;
4. every code-computed answer records valid cited operands; invalid citations must refuse
   computation rather than produce a number;
5. no frozen, sealed, or historical result is modified.

Regardless of outcome, retain and report the run. Do not tune on a failed row and rerun
this same gate under the same label.
