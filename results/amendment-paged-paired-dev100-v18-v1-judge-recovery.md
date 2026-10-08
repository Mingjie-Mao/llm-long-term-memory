# Amendment 1 to `paged-paired-dev100-v18-v1` — one visible judge-recovery extension

2026-10-05 Australia/Sydney. Written before any call it allows. The registration
(`results/prereg-paged-paired-dev100-v18-v1.md`) is unchanged and stays hash-frozen in
the execution inventory; this file adds one rule beside it.

## Why

The run stopped at baseline rep2, 100 answers and 71 grades saved. Question
`ec81a493`'s grading request failed four times with provider server errors
(500, 500, 503, 500; the last two about six hours after the first two), exhausting the
registered judge recovery of three per question
(`results/raw/paged-paired-dev100-v18-v1-baseline-rep2.ec81a493.recovery.json`,
`exhausted: true`). Its answer is saved; the request is ordinary in size. The
registration says exhaustion stops the run and forbids a hidden reset, so the run cannot
resume without a rule, and 600 graded rows are needed for the gate.

## Rule

- Applies uniformly to both arms and every repeat, for the rest of this run.
- Only a grading identity (arm, repeat, question) whose registered recovery is exhausted,
  every recorded failure a transient provider error (HTTP 500/502/503/504, connection or
  timeout), and whose answer row is saved.
- That identity gets **one** further recovery budget, identical to the registered one
  (three recoveries, 180 s total wait), recorded in its own file
  `<arm>-rep<N>.<qid>.recovery-amendment-1.json` under a distinct identity. The original
  recovery file is kept as it is.
- The request is the registered one: same saved answer, same judge model, prompt, schema
  and grading cache. No answer is regenerated; no verdict is read before the grade is
  written; the decision to retry depends only on the provider error, never on content.
- A second exhaustion, or any permanent or schema error, stops that identity for good;
  the run is then recorded as incomplete with the missing grade named.
- Every use is listed in `results/analysis/paged-paired-dev100-v18-v1.judge-amendment.json`
  (identity, original recovery file hash, amendment recovery file, time), without verdicts.

## Execution

`tools/v18_judge_amendment.py` applies the rule to the identity the run's progress file
names. `tools/continue_v18_amended.py` replaces the stopped continuation job for the
rest of this run (that job's files are kept as they are): it resumes answers and grades
with the unchanged `tools/run_paged_paired.py`, calls the amendment tool when a grading
identity is exhausted, and waits for daily quota resets, at most six. It stops when the
run writes its gate; the conditional summary-cost phase is not started by it.

Nothing else in the registration changes: gate, models, prompts, reader recovery,
aggregate/type-only reporting, no per-item tuning.
