# Fixed-model grounded answering v3 — remaining-gap repair

2026-10-02. New development namespace. No provider requests before registration.
Keep reader gemini-3.5-flash-lite, extractor gemini-3.1-flash-lite, judge gemma-4-31b-it,
and MiniLM embeddings. Baselines: raw-primary v1 and grounded-v2 offline evidence.
Exposed train150 only may guide item-level repairs; dev100 context-only, no gold tuning.

## Observed first failures and repairs

- S4/S4b: v2 context replay covers 127/146 annotated train questions, including seven
  losses relative to lexical raw-primary. Preserve its fused pool, then append omitted
  lexical 4k baseline turns under the same total 6000 estimate; no gold-selected turns.
  Audit appended/dropped source ids. This targets lexical regressions, not every gap.
- S5: numeric validation accepts a value and unit anywhere in a quote, permitting a
  tank's 20-gallon size to be used as 20 fish. v3 requires a local quantity/unit binding
  and a stated unit; no currency conversion. Semantic membership remains model work.
- S5: the calculator supports only two duration endpoints; the train question about
  three books requires six endpoints. v3 supports explicitly identified start/end
  pairs, validates each source expression, and sums days/weeks. Grouped month/year
  durations remain refused because a uniform day-to-month conversion is undefined.
- S5 recovery: a calculation with enough supplied evidence can fail validation; the
  current path skips repair if fallback finds no new turns. v3 permits one same-pool
  structured repair, exposing the exact mechanical refusal. No unconditional retry
  for successful lookup and no third reader call.

v2 original source is preserved in grounded-context-v2.source-snapshot.json. Existing
raw rows, registrations and results are immutable. Default product policy unchanged.

## Verification and zero-call measurement

Require targeted isolation, quote/unit, repeated-event, grouped-duration, reversed/
missing endpoint, approximate-date, retained-pool and repair tests, then full suite
and Ruff. Actual stub replay train150: all-gold-turn coverage, wins/losses against
baseline and v2, context median/p90/max, omissions, exact text checks, source hashes.
Namespace grounded-context-v3.train150. Offline advancement requires coverage >=v2,
no content mismatch and median <=6000; preserving all seven losses is a diagnostic
goal, not a guaranteed gate. Report remaining question ids and first-loss hypotheses.
Stub uses real runtime paths, has zero provider requests/tokens, and never sees gold.
dev100 may use a separate context-only namespace; provider seed unsupported, temp0.

## Model validation and completion boundary

External calls remain subject to the explicit authorization requested after automatic
approval rejected v2's outbound questions/evidence. Do not bypass that rejection.
If authorized, first run the existing exposed nine reasoning-error questions once,
candidate only, new v3 namespace; at most 18 reader and 9 judge calls before internal
provider retries. Freeze identity before calls and record actual usage and failures.
Stop on recurring judge 500/503; incomplete rows are not an accuracy estimate.
Source/model/prompt repair after results requires another registered namespace.

Full dev100 comparison retains three repeats per arm, exact100 ids, mean net >=4,
per-type net >=-2, candidate median context <=6000. Complete mechanical validation
before enlarging; no item-level dev100 tuning. Upper bound 1200 reader/600 judge calls
before retries. This rule is adoption, not statistical significance or unseen-final.

“Complete” cannot mean arbitrary natural-language questions are guaranteed correct.
Acceptance must distinguish supported facts, genuinely missing/ambiguous facts and
benchmark contradictions. For supported cases, require complete evidence, correct
event/operand selections and exact computation; for contradictory/missing cases,
preserve provenance and report uncertainty rather than manufacture a benchmark answer.
Known failures and ordinary-correct regressions must both be measured; mechanisms,
unit tests and evidence coverage alone do not establish that all three goals are met.
