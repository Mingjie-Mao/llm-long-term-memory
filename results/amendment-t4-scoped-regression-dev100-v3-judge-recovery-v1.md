# Uniform judge service recovery amendment v1

2026-10-10 Australia/Sydney. Written before new calls under this amendment.
Parent: `t4-scoped-regression-dev100-v3`. The original registration, inventory,
source snapshot, saved answers, grades and exhausted recovery files remain intact.

The first control repeat stopped after 100 saved readers and 87 grades. The next
grading identity exhausted four recovery rounds with HTTP500/503/500/500. This is
provider availability evidence only; no partial correctness scores or item errors
were inspected to decide this amendment.

## Fixed policy for all six blocks

- Any identity in either arm and any repeat, whose original recovery exhausted
  solely on HTTP500/502/503/504 or connection/timeout errors, receives at most one
  additional identical recovery budget: three recoveries, 180s cooldown cap.
- Only the original saved answer is graded. Model, grading prompt/schema, gold,
  execution identity and answer remain unchanged. No reader or judge cache is used.
- The additional recovery uses a distinct identity and a new
  `recovery-amendment-v1.json` file. The original recovery file is never reset.
- The same extended file is resumed after interruption; restart cannot refresh
  its budget. A second exhaustion ends the identity and leaves the run incomplete.
- Permanent/schema errors stop; daily quota is checkpointed and not bypassed.
- Every invocation is logged before calls with original recovery and reader hashes,
  amendment inventory and outcome; all failures and usage remain in the block ledger.
- Client budgets remain API2/transport2, three shared attempts per invocation.
  The extension can add at most twelve attempts per grading identity: across all
  six blocks, at most7200 additional attempts, not a promise of successful completion.

`tools/t4_judge_amendment.py` records the amended identity and grades the next
unsaved grade only. After success the unchanged original runner resumes; it skips
the newly saved grade and verifies its reader binding. The existing gate rules,
repeat order, question set and original source freeze remain unchanged. A completed
result must disclose this amendment alongside the parent protocol.
