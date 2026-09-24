# Extractor model comparison — pre-registration

Date: 2026-09-23 (Australia/Sydney)

## Classification

**DEVELOPMENT mechanism experiment** on the extraction ruler. Not a QA result and not
an unseen final evaluation. Nothing here may be reported as a benchmark number.

## Question

At a fixed batch size, prompt, schema and cohort, does a stronger extractor model retain
more of the user's own specifics — and if so, is the gain extraction quality or simply
more memories written?

## What this measurement can detect

The resolution discipline in `.agents/skills/experiment-guard` applies, but the
instrument here is not the answerer, so `tools/resolution.py` does not. The extraction
ruler has **no repeat measurement anywhere in this repository**: every fidelity figure
on record is a single run of its configuration.

That is why **arm R below replicates the baseline**. Sixteen calls buy the one number
that decides whether any of the other arms mean anything, and spending them is cheaper
than a second retrospective. Effects this ruler has resolved historically:

| comparison | effect | evidence |
|---|---|---|
| batch 15 → 8 | +27.6pp | 41 gains / 4 losses, exact McNemar p = 9.3e-09 |
| conditional specificity repair | +14.5pp | 21 gains / 0 losses |

So effects of roughly ten points with ten or more net specifics have been resolvable.
Anything smaller is unestablished, and arm R is what will say whether it is.

## The ruler's known bias, and the guard against it

`ingest.fidelity` measures **recall only**: whether a specific the user stated survives
into some memory. It cannot see what the extractor invented, and retention rises with
volume almost mechanically — `results/batch-size-result.md` records 13.10 memories per
session at 78.4% against 4.33 at 64.2% for the citation-grounded arm.

So a model that simply writes more will score higher without extracting better.
Memories per session and **retained specifics per memory written** are reported beside
the headline for every arm, and a gain accompanied by proportional volume growth is
recorded as volume, not quality, and does not promote.

## Cohort

The first 60 sessions of the deterministic `split_dev_test(seed=0)` holdout, which is
the cohort `results/batch-size-result.md` used. It is reused deliberately:

- the baseline arm is already extracted and archived
  (`results/raw/extraction-batch-size.extractions.json`, arm `batch8`, fingerprint
  `eddb19a8ba81`), so the comparison costs nothing on that side;
- the ruler has no gold answers — its reference is the source text — so it cannot be
  fitted to the evaluation set;
- but it **can** be overfitted to this cohort's content, which is what stage 2 is for.

## Arms

Everything fixed except the extractor model: `sessions_per_request = 8`, the two-stage
prompt and schema, `temperature = 0`, the same 60 sessions in the same order.

| arm | model | cost |
|---|---|---|
| B — baseline | `gemini-3.1-flash-lite` | free, restored from the archive |
| **R — baseline replicate** | `gemini-3.1-flash-lite`, cache bypassed | 16 calls |
| C1 | `gemini-3.5-flash-lite` | 16 calls |
| ~~C2~~ | ~~`gemini-3.5-flash`~~ | **withdrawn — see amendment 1** |

Roughly 32 successful extraction calls, no answerer and no judge.

### Amendment 1 — C2 withdrawn, 2026-09-23, before any arm was scored

The provider spent the day returning `503 UNAVAILABLE` on most requests. Arm R reached
56 of 60 sessions at a cost of **163 calls**, against a registered estimate of roughly
48 for the whole experiment; the measured success rate moved between 8% and 19% and
then to 0% for three consecutive attempts on the last batch.

At that rate the two candidate arms need somewhere between 170 and 400 calls against a
332-call remainder of the daily cap, so the original design does not fit the day. The
scope is cut to one candidate rather than shrinking the cohort, because 60 sessions is
the archived baseline's own cohort and changing it would forfeit the free baseline and
add a second variable to a single-variable comparison.

This is recorded before any arm is scored. `gemini-3.5-flash` is not measured and no
claim is made about it either way.

## Primary endpoint and gates

Primary: **paired specificity retention against arm B over the cohort's specifics.**

Promote a candidate to a stage-2 confirmation only if all hold:

1. **the gain clears the instrument's own noise** — the candidate's improvement over B
   is at least twice the absolute difference arm R shows against B;
2. **at least 10 percentage points** of overall fidelity over B;
3. **at least 10 net specifics gained**, with gains exceeding losses;
4. **the gain is not volume** — retained specifics per memory written does not fall;
5. frozen, sealed and archived artifacts unmodified.

Gate 1 is the one this experiment exists to make possible. If arm R differs from B by
more than a couple of points, then every single-run fidelity figure in this repository —
including 37.3%, 64.9%, 44.8% and 59.3% — carries an error bar nobody has drawn, and
that finding outranks the model question.

## Stage 2, only on a pass

A candidate that passes is re-run on holdout sessions 61–120, where
`results/analysis/specificity-repair-pilot.json` already records a batch-8 baseline of
44.8% for `gemini-3.1-flash-lite`. Cohort-dependence is established in this project —
batch 8 measured 64.9%, 47.5% and 44.8% on three different cohorts — so a single-cohort
model win is not a model decision.

## Stopping rule

If no candidate passes, the extractor model stays `gemini-3.1-flash-lite` and the result
is retained as a negative. No re-tuning of prompt, batch size or schema on this cohort
afterwards: those are different variables and changing them here would make the model
comparison uninterpretable.

## Limitations, stated before the numbers

- Recall-only ruler; hallucination is invisible to it and is guarded against only
  indirectly, by volume.
- One cohort at stage 1, and cohort dependence is large in this project.
- One run per arm apart from the baseline, which is the point of arm R.
- A stronger model may cost more per call and change latency; that is reported but is
  not a gate, because this experiment is about what is retained.
