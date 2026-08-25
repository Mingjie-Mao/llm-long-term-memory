# Pre-registration — v2 final test and baselines

**Written 2026-08-22 before any `dev100` or `test100` run.**  `test100` remains
sealed.  The exact v2 product arm is conditional on the already-written dev100
decision rule; the baselines below are not conditional on their eventual scores.

## One-shot arms

| order | arm name | runner | purpose |
|---:|---|---|---|
| 1 | `v2` | whichever of `two_stage_coherent` or `two_stage_fallback` survives the registered dev100 rule | product result |
| 2 | `full_context` | `full_context` | reading the entire chat-history baseline |
| 3 | `naive_rag` | `naive_rag` | ordinary raw-session vector-retrieval baseline |
| 4 | `flat_memory_fallback` | `two_stage_fallback` | separates coherent context from the existing memory + raw-source system |

If dev100 selects `two_stage_fallback` as v2, arm 4 is identical to arm 1 and is
omitted rather than spending a second run on the same system.  `coherent-oracle` is
not a product and is not run on test100; its only role is explaining the dev100
decision.

Every retained arm runs exactly once on the same frozen 100 questions, with the
same answerer and judge.  A quota pause resumes the same JSONL; it is not a repeat.
The arm list and order are included in the final content-addressed freeze, and the
one-shot ledger is written before the first answer call.

Implementation guard (2026-08-22, still before `dev100`/`test100` access): formal
freezes and runners reject any omitted, reordered or substituted arm. The v2 variant
must match the hash-bound automatic dev100 decision; formal freezes cannot be
replaced with `--replace`.

The final renderer is separate from the three-repeat dev renderer: it reports one
accuracy per arm and omits repeat variance, majority-vote and agreement fields that
would be meaningless for the registered one-shot protocol.

## Reporting

The first published table contains all retained arms together:

- accuracy and question-type accuracy;
- candidate/ranked/selected source-session recall where the runner provides it;
- context tokens, API requests and tokens, with shared ingestion shown separately;
- raw-fallback use and correct-after-fallback rate;
- aggregate failure stages only, with no question ids or answers;
- exact paired McNemar for each baseline → `v2` on the single frozen verdicts.

The paired p-values are descriptive.  There is no repeat-based majority vote on
test100 because the user required one final run; run-to-run variability is measured
on dev100 instead.  No result is called a win merely because its headline number is
higher.

## Decision boundary

Nothing is changed after test100.  A poor final result is published as a poor final
result.  Deleting the ledger, changing an arm, or re-running after reading the
aggregate would create a new development experiment, not a valid final test.
