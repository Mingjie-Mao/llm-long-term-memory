# Scoped t4 repair: fresh full development regression

2026-10-10 Australia/Sydney. Registered before this experiment's provider calls.
No commits, model replacement, extraction or summary calls.

## Prerequisites and question

The fixed21 acceptance `t4-scoped-selection-train21-v3` must pass. Verify its source
and reader/grade hashes before any new calls. A failed/incomplete acceptance blocks
this comparison. Preserve the two earlier failed t4 known-case gates and failed v18
comparison; neither their rules nor their rows are overwritten.

Question: does the scoped selection repair preserve ordinary t4 performance on the
complete dev100, beyond the enriched 21-case acceptance? This is development/regression
evidence: dev100 has been exposed, including errors inspected by another authorized
task on October 7. No unseen-final claim or tuning of dev item errors in this run.
heldout100 errors are not inspected. No test-set resampling or gold enters answering.

## Design and power limits

Dataset LongMemEval-S; exact `results/manifests/dev100.json`, all 100 IDs in manifest
order. Three fresh repeats each of original t4 (`control`) and scoped repair
(`candidate`), 600 saved readers and 600 original-gold grades. Execute control then
candidate for rep1, rep2, rep3. Save all 100 reader answers of a block before grading
that block. Grading results never change the frozen candidate.

Resolution report `t4-scoped-regression-dev100-v2.resolution.json` projects 10.8
unstable questions and a single-run floor of 7. Expected gains of a few points are
below that floor; three repeats reduce the crude noise floor to about 4, assuming
independent repeats. This is a regression safeguard, not a powered claim of a small
accuracy improvement. Report uncertainty; do not infer significance from a pass.

Original t4: `two_stage_raw_primary_t4`, `memory-aware-v2-clt`; candidate:
`EventCountRepairRunner.from_t4`, `memory-t4-scoped-selection-v2`. Both use identical
store/index/retrieval config, `configs/fallback.yaml`, MiniLM384, answerer
`gemini-3.5-flash-lite`, judge `gemma-4-31b-it`. Extractor remains
`gemini-3.1-flash-lite` but is not called. Temperature 0; provider seed unsupported;
the manifest's split seed is retained unchanged. Record git SHA/Python/config hashes.

Offline preflight validates every stored session/turn against public source and owner:
49,590 raw turns, no missing sessions. Median context control 6,220, candidate 6,344.5.
Stub generation is a composition check only, never reported as accuracy.
The zero-call preflight also checks memory provenance ownership. The execution
entry point verifies every preflight input hash before freezing. The unexecuted
v1 regression registration based on failed selection v2 is withdrawn, preserved
in its own namespace; this is a new candidate and new comparison.

## Registered operational decision

Require all 600 readers and grades with exact ID sets, prompt versions, fixed model,
execution identity and reader SHA binding. Missing rows forbid a gate decision.

- Mean candidate minus control correct count must be >= 0 per 100 questions.
- No question type's mean net may be below -2 questions.
- Candidate median actual supplied context must be <= 6,500 tokens.

Report three scores per arm, mean delta/type deltas, conservative paired majority
confidence interval, API calls/failures/tokens by model, and end-to-end latency.
An operational pass is **not a statistical proof that accuracy is identical**.
Failure blocks adopting this wrapper and summary calls; diagnose only train150 or
separately register an exposed-dev diagnostic, never silently retune the active run.
Pass permits registering a new t4-based recursive-summary experiment that must
measure initial construction, incremental update, navigation and answer **total**
cost alongside fresh paired accuracy. It does not revive the failed v18 summary arm.

## Cost, recovery and artifacts

Nominal 600 answers and 600 grades; fallback may double answers. At most 1,200
nominal answer generations / 3,600 client attempts. Client API retries=2,
transport retries=2, three shared attempts per invocation. Judge max three durable
recoveries, 180-second cooldown cap: at most 7,200 judge attempts. All attempts
consume quota and are recorded. No quota reset bypass, cache-derived fresh answers,
or unregistered recovery after exhaustion. Exhaustion records INCOMPLETE.

Approximately 4–5 million reader-input tokens plus grading; exact offline source
estimate is saved in the preflight. Nominal minimum two 500-request reader quota days;
fallback and service failures may add days. No guaranteed completion date.

Freeze sources/config/data, manifests, preflight/resolution, prerequisite gate,
and this registration in `results/analysis/t4-scoped-regression-dev100-v2.execution.json`
and literal source snapshot. Resume refuses changed inputs. Each arm/rep writes new
`results/raw/t4-scoped-regression-dev100-v2-{control,candidate}-rep{1,2,3}` readers,
grades, usage, combined rows and durable recovery files. The generator writes one
immutable gate only after completion; progress is separately resumable.
