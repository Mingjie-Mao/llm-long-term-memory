# Stronger answerer on the raw-primary arm's reasoning errors — minimal probe — pre-registration

Date: 2026-10-01 (Australia/Sydney). Written before any call of this probe. Revised
before any call from a 23-question draft to the 9 questions below, to spend the least
quota that can still decide.

## Classification

**DIAGNOSTIC, failure-enriched, on train150.** Nothing here estimates accuracy: the
questions are chosen because the arm got them wrong.

## Why these 9

Every system scoring 85–95% on LongMemEval-S answers with a stronger model than
`gemini-3.5-flash-lite`, and the largest single gain any of them reports is the answering
model alone (Mastra: 84.2% with gpt-4o, 94.9% with gpt-5-mini). Of the raw-primary v1
arm's 23 train150 errors, 14 lacked some gold turn in context — no model can answer from
evidence it was not given — and **9 had every gold turn in context and were still
answered wrong**. Only those 9 can show a model effect, so only they are asked
(`results/manifests/train150-raw-v1-reasoning-errors.json`).

## Arms (same store, same retrieval, same context)

| arm | variant | config | answerer | label |
|---|---|---|---|---|
| **lite rerun** | `two_stage_raw_primary` | `configs/fallback.yaml` | `gemini-3.5-flash-lite` | `lite-rerun-v1` |
| **strong** | `two_stage_raw_primary` | `configs/fallback-strong-answerer.yaml` | `gemini-3.6-flash` | `strong-probe-v1` |

The lite rerun is the noise control: a failure can flip back to correct on a rerun by
chance, so the strong arm is read against it, not against zero.

## Decision rule

**Model effect = (of the 9, fixed by strong) − (of the 9, fixed by the lite rerun).**

| model effect | reading | next |
|---|---|---|
| **≥ 4** | the answering model is a real lever for the residual | a full, registered comparison — a budget decision on this free tier |
| **≤ 1** | the residual is not mainly the model's | stop pursuing the model upgrade |
| 2–3 | undecided | ask the remaining 14 errors the same way, registered as an extension |

Evaluated by `tools/strong_answerer_probe.py`, which refuses an incomplete run.

## What this cannot show

Regressions: a stronger model could get wrong some of the 127 questions the arm answers
right, and this probe never asks them. That is the full comparison's job.

## Cost

About 10 `gemini-3.6-flash` calls (fallback second calls included), inside one day's 20
on the free tier as last probed; about 10 lite calls; 18 judge calls.
