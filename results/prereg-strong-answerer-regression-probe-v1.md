# Does the stronger answerer break what flash-lite gets right? — regression probe — pre-registration

Date: 2026-10-01 (Australia/Sydney). Written before any call of this probe.

**Revised before any call** from a 20-question draft to the 10 questions below, to spend
the least strong-model quota that can still screen for a clear regression. The 20-question
sample (`train150-raw-v1-correct-sample20.json`, seed `strong-regression-v1`) was drawn and
never run; it is withdrawn and deleted, and this draw uses a new seed.

## Classification

**DIAGNOSTIC, on train150.** Questions the raw-primary v1 arm (flash-lite) answered
**right** in `two_stage_raw_primary.train150-raw-v2`. Not an accuracy figure.

## Why

`results/probes-2026-10-01-decision.md`: on the 9 questions whose evidence was all in
context and still answered wrong, `gemini-3.6-flash` fixed 5 and a flash-lite rerun fixed
0. This asks the cheapest version of the other half: does the stronger model get wrong
what flash-lite gets right?

## Sample (drawn after this was written, deterministically)

From the 127 v1-correct train150 questions, each group taking the first *k* in order of
`sha256("strong-regression-v2|" + question_id)`:

| group | sampled |
|---|---:|
| temporal-reasoning | 2 |
| multi-session (multi-hop) | 2 |
| knowledge-update | 2 |
| single-session-preference | 2 |
| numeric / computation — one single-session-user and one single-session-assistant question with a digit or number word in the gold | 2 |
| **total** | **10** |

The numeric group takes one question from each single-session type the other quotas leave
out, so all six LongMemEval types are represented. Manifest:
`results/manifests/train150-raw-v1-correct-sample10.json`, written by
`tools/strong_regression_probe.py sample`. Nothing about the strong model enters the draw.

## Arm

`two_stage_raw_primary`, `configs/fallback-strong-answerer.yaml` (`gemini-3.6-flash`),
store `train150`, label `strong-regression-v2`. **Flash-lite is not rerun**: its answers
are the existing v1 rows. Retrieval and the raw turns are deterministic for a given store
and question, so the first answer call sees the context flash-lite saw.

## Decision rule

**R** = sampled questions the strong model gets wrong.

| R (of 10) | reading | next |
|---|---|---|
| **0–1** | regression risk is low | proceed to the full strong vs flash-lite comparison |
| **2** | confirm | draw 10 more by the same rule, registered as an extension |
| **≥ 3** | the answerer cannot simply be replaced everywhere | analyse the regressions by type |

Reported alongside, not deciding: R by type; **E**, the regressions expected from
flash-lite's own run-to-run flips (over the four unchanged heldout100 control runs, per
type, P(wrong in run B | right in run A), applied to this sample); and a projected net on
train150 — conservative `+5 − 127 × R / 10`, noise-adjusted `+5 − 127 × max(0, R − E) / 10`
— with the stated caveat that the 14 retrieval-limited errors are untested and credited
with no fix.

Evaluated by `tools/strong_regression_probe.py evaluate`, which refuses an incomplete run.

## What this cannot show

Ten questions are a screen: they can show a large regression cheaply and cannot rule out a
small one. That is what the full comparison is for.

## Cost

About 10–12 `gemini-3.6-flash` answer calls (fallback second calls included) and 10 judge
calls, inside one day's 20 on the free tier. Run after the provider's daily reset.
