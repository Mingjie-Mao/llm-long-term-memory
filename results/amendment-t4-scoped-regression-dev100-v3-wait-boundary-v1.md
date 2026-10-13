# Nonnegative cooldown wait adapter

2026-10-11 Australia/Sydney, before resumed provider calls. The original source,
models, prompts, manifests, gates, grading inputs and durable recovery policies
remain frozen. This is an execution correction, not a new candidate selection.

The previous finite chain saved300 answers and295 grades, then exited1 while
waiting to grade the first unsaved control-rep2 identity `gpt4_c27434e8_abs`.
Traceback: `evaluation/recovery.py:78`, `ValueError: sleep length must be
non-negative`. The cooldown loop calls the clock in its condition and again for
the sleep interval. Crossing the boundary between these calls creates a negative
sleep. The saved state still has failures3, wait_seconds105, max_recoveries3,
max_wait_seconds180; no budget is exhausted or cleared by this adapter.

`t4_nonnegative_wait_resume.py` preserves all parent hashes and frozen code. Its
runtime adapter passes positive sleep intervals through unchanged and skips zero
or negative intervals in RecoveringJudge only. Requests, verdicts, transient
classification, counters, cooldown clocks, quota and recovery budgets are unchanged.
The adapter applies uniformly to dev100 baseline/candidate and original test100
baseline/candidate judges, including the existing server amendment driver.
It does not alter the old v2 reader worker or global time.sleep.

- Record all current raw JSONL byte prefixes plus reader, recovery-state and last
  failed-call SHA for the confirmed first missing grade, and the stopped chain.
- Allow one durable bootstrap of this confirmed timer interruption. It consumes
  no new transport/server allowance and grants zero additional provider attempts
  beyond existing limits. Known completed grades, including False, remain saved.
- The earlier finite controller then applies all unchanged network/server/quota
  stopping rules and starts original test100 only after full dev acceptance.
- A child-launch adapter injects this sleep correction only into four registered
  dev/judge/test drivers. Each child verifies the same adapter/source inventory and
  saved prefixes. No arbitrary executable is accepted by the adapter CLI.
- Unexpected failures, exhausted recovery, or quota still stop. There is no periodic
  evaluation restart. The newly human-authorized hourly completion notification is
  read-only, separate from evaluation execution.

The new source, tests and this amendment are bound by a separate immutable
wait-boundary-v1 execution inventory. All earlier failures and execution receipts
are retained. Completed reporting must disclose this operational adapter and actual
usage. No partial correctness outcomes were inspected to choose this correction.
