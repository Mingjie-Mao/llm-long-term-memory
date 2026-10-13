# t4 known-selection acceptance before a recursive-summary comparison

2026-10-10 Australia/Sydney. Before this namespace's provider calls. No commits,
model switches, framework migration, or unseen-final claim. This replaces neither
the failed v18 gate nor the unexecuted v18-conditional summary protocol.

## Prerequisite and classification

The completed t3/t4 heldout100 comparison passes its original registered gate.
`tools/audit_t3t4_completion.py` verifies all 11 registered hash prefixes, unique
100-question manifests, nine complete boolean-grade runs, models and prompt versions,
and exact zero-call reproduction of the stored gate. All900 answers/grades retained.
This is exposed-regression evidence, not permission to say 72% unseen-final became87.67%.

Before rebasing summary navigation on t4, verify the fixed union of the original19
selection cases and six prerequisite cases: deduplicated21, entirely exposed train150.
Individual diagnosis is permitted here. No heldout100 item failures are read or used
for tuning. This is named-case mechanism acceptance, not a population estimate or
statistical comparison; it does not need to resolve a noisy small aggregate gain.

## Fixed input and answering

Variant `two_stage_raw_primary_t4`, prompt `memory-aware-v2-clt`, unchanged current
`configs/fallback.yaml`, existing `stores/train150.db` and indexes. Answerer
Gemini3.5flash-lite, extractor Gemini3.1flash-lite (not called), judge Gemma4-31b-it,
local MiniLM384. Only question/date/tenant reaches the answerer, never benchmark gold.
One fresh answer per21 questions, no answer reruns or selection among completions.
All answers saved before any scoring. Instrumentation saves the exact actual systems,
requests, schemas and replies without changing generation. Record whole-question
requests/failures/input/output tokens and end-to-end wall time. No summary API calls.

Fixed IDs are `PILOT` in the frozen driver; exactly21 unique train150 IDs, original19
plus6 prior cases. Source/config/store/manifest/prompt/tool hashes, literal sources,
git SHA and environment frozen before first call. Resume saved answers/grades only.
Do not silently reset durable recovery counters or change frozen code during the run.

## Original scores and faithful missing-source cases

All21 retain the original judge score with original benchmark gold. No disputed gold
is rewritten. The predetermined cases f9e8c073, gpt4_2f8be40d, gpt4_731e37d7 also get
an independent additional original-judge evaluation against these source-derived
requirements, declared before observing this namespace's outputs:

- f9e8c073: acknowledge missing report periods/overlap; do not fabricate a precise
  combined total; request the periods and whether reports overlap.
- gpt4_2f8be40d: exact weddings-this-year count cannot be determined reliably without
  event dates/identities/reporting period; request clarification.
- gpt4_731e37d7: source payments20+200+500 sum to720; distinguish this cited-payment
  sum from a confirmed requested-year total; acknowledge event date/year/reporting
  uncertainty rather than invent a corrected year; request clarification.

Exact auxiliary targets are frozen as `GAP_GOLD` in the driver. The same judge/model/
prompt evaluates all three; separate files/recovery identities from original grading.
These are supplementary mechanism verdicts, never substituted for original scores.

## Acceptance and failure handling

Pass requires all18 supported cases originally judged correct, all3 supplementary
faithful-gap verdicts true, and median source-context<=6500 as the t4 registered bound.
Report every original score and every named-case failure. Full reference-turn text
reach across actual first/fallback prompts is a conservative diagnostic, not a proxy
for truth or an extra after-the-fact gate. Absence of a full turn does not prove the
necessary sentence was absent. No semantic completeness claim from a prompt receipt.

A failed named-case acceptance blocks all summary provider calls: diagnose the first
loss only in train150, preserve failures and register a new fix/acceptance namespace.
A pass permits a NEW t4-based summary pilot/cost protocol, not the old v18 protocol,
and does not permit adoption or a global claim that every selection error is solved.

## Cost and reproducibility

At most42 nominal answerer invocations before retries (two calls per question), at
most126 provider attempts with max2 API attempts plus one additional transport/TPM
attempt. Original grading21 plus3 auxiliary calls nominal; max288 attempts with
max3 durable outer recoveries per request (four invocations, each at most3 attempts).
RecoveringJudge max180s cumulative cooldown, client180s timeout. Permanent errors
or exhausted recovery stop with original saved answers intact; quota pause resumes
without charging a fresh per-question recovery budget. Headroom6 before each reader.
No historical grade cache is used in this namespace. Re-running saved rows is forbidden.

Before calls, existing900 answers and saved old acceptance evidence have been checked:
none measure t4 on this exact fixed21 set or the three faithful-gap targets. This is why
these additional calls are necessary. Per-model accounting separates failed requests;
missing provider token counts are unknown, not proof that a failure cost nothing.

```bash
python tools/t4_selection_acceptance.py
python tools/t4_selection_acceptance.py --execute
```
