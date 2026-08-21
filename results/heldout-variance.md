# Protocol — how much of `70.0%` is the run?

**Pre-committed 2026-08-20, with `rep2` running and no repeat result on disk.**
`results/raw/two_stage_hydrated.heldout100-rep2.jsonl` held zero rows when this
was written. Nothing below was chosen by looking at a repeat.

[context-arms.md](context-arms.md) found the production configuration
disagreeing with itself: `flat20` is the held-out run's exact configuration and
did not reproduce its answers on three of five questions re-run. This measures
that at full scale instead of on five questions chosen for having failed.

## What is being run

`heldout100`, the frozen system, twice more. Same store, same config, same
answerer, same judge, same manifest — **only the run differs.**

    rep2, rep3    lltm eval run two_stage_hydrated --label heldout100-rep{2,3}

137 answerer calls and 126 judge calls per repeat, no extractor quota.

**This deliberately bypasses gate 5 of `heldout_run.py`**, which asserts the
result file does not exist. That gate exists so a re-run is deliberate rather
than accidental, and this is the deliberate case: it measures the instrument, not
the system. The gate is doing its job by making this paragraph necessary.

## What may be concluded, and what may not

The distinction that makes this legitimate rather than a second bite at the
held-out set:

| allowed | not allowed |
|---|---|
| an error bar on the reported `70.0%` | a new headline number |
| the rate at which questions flip verdict between runs | a list of which questions flipped, used to choose what to fix |
| whether the dev50 → held-out gap of -2.0pp is inside run noise | re-reading the failures for new work items |

**`70.0%` stays the reported held-out result.** It is the pre-registered single
shot on the frozen system, and it does not get replaced by a mean that looks
better or worse. The repeats produce a ± beside it, nothing more.

**Per-question stability is not a work-selection signal.** Knowing which
questions are unstable, and then fixing those, is the adaptive use that spent
dev50 — worse here, because it would spend the only clean set the project has.
The per-question data is reported as a count and a rate; the ids stay out of any
decision about what to build.

## What will be reported

1. Accuracy per run, their mean and range.
2. **Verdict agreement**: of 100 questions, how many are correct in all three
   runs, wrong in all three, and how many disagree. The last number is the one
   that matters — it bounds how much any single-run comparison on this scale can
   be trusted.
3. The same split for memory-only accuracy, archive rescue, and fallback trigger
   rate, since a stable total can hide an unstable split.
4. Whether the **-2.0pp** dev50-to-held-out gap and the fallback's **9W-1L
   p = 0.022** survive being read against the measured run noise.

## Registered prediction

- Between **8 and 20** of the 100 questions disagree across three runs. The
  five-question probe saw 4 of 15 arm-cells internally inconsistent, but those
  questions were selected for having failed, which is the population most likely
  to be marginal.
- Per-run accuracy spans **at least 3 points**, i.e. the -2.0pp dev50 gap sits
  inside run noise.
- The fallback trigger rate is **more stable than accuracy**, because the
  answerer's `need_source` verdict is a coarser decision than producing a correct
  answer.
- No repeat changes the conclusion that `knowledge-update` and
  `temporal-reasoning` are the weakest categories, because both gaps are larger
  than the predicted noise band.

If prediction 1 lands above 20, the honest consequence is that **every paired
result in this project on tens of questions is uninterpretable as reported**, and
the repair is repeats everywhere, not a footnote.


---

# Result — the noise is 3 points wide and lives in one category

Three runs, complete. `run1` is the pre-registered single shot; `rep2` and `rep3`
are identical configurations.

| | run1 | rep2 | rep3 | mean | range |
|---|---:|---:|---:|---:|---:|
| **final accuracy** | 70 | 71 | 73 | **71.3** | **3** |
| from structured memory alone | 50 | 52 | 53 | 51.7 | 3 |
| rescued by the raw archive | 20 | 19 | 20 | 19.7 | 1 |
| fallback fired | 36 | 35 | 35 | 35.3 | 1 |

**`70.0%` remains the reported held-out result.** It is the pre-registered shot
and the protocol above says it does not get replaced by a mean. It is worth
noticing that the mean is *higher* — 71.3 — so honouring the commitment costs
this project a point rather than earning it one. That is the direction in which
a pre-commitment is worth something.

## Verdict agreement

| | |
|---|---:|
| correct in all three runs | 69 |
| wrong in all three runs | 25 |
| disagreed | **6** |
| fallback *level* disagreed | 10 |

## The variance is not diffuse

| question type | n | three runs | disagreeing questions |
|---|---:|---|---:|
| **temporal-reasoning** | 27 | 16, 17, 19 | **4 (14.8%)** |
| multi-session | 27 | 19, 19, 19 | 2 (7.4%) |
| knowledge-update | 15 | 10, 10, 10 | 0 |
| single-session-user | 14 | 12, 12, 12 | 0 |
| single-session-assistant | 11 | 9, 9, 9 | 0 |
| single-session-preference | 6 | 4, 4, 4 | 0 |

**Four of six categories are bit-identical across three runs.** Every disagreeing
question is `temporal-reasoning` or `multi-session` — the two types whose answers
must be *derived* rather than looked up. Lookup is deterministic in practice;
arithmetic and aggregation are where sampling shows.

That is a sharper statement than "the answerer is noisy", and it changes what the
repeats protocol has to cost. A comparison that does not touch temporal or
multi-session questions can be read from one run. One that does needs three.

## Predictions, scored

| registered | outcome |
|---|---|
| 8-20 questions disagree | **6 — wrong, and pessimistic.** The five-question probe over-estimated it because those questions were selected for having failed |
| per-run accuracy spans ≥3 points | **3 — correct, at the boundary** |
| fallback trigger rate more stable than accuracy | **split.** As a rate, yes: range 1 against 3. Per question, no: 10 levels disagreed against 6 verdicts |
| knowledge-update and temporal stay weakest | **half wrong.** `knowledge-update` stays weakest at 66.7% and is perfectly stable. `temporal-reasoning` averages **64.2%**, not the 59.3% of the single run, and is no longer the worst |

## What this settles

**The dev50 → held-out gap is not real.** dev50 scored 72.0% and the held-out
mean is 71.3%: a **-0.7pp** difference, comfortably inside a 3-point run band. The
single shot's -2.0pp was a run landing at the bottom of its range. The system
generalises better than the pre-registered number said.

**The fallback's `p = 0.022` does not survive.** It is 9W-1L on 50 questions from
one run. At 6 flips per 100, a 50-question paired comparison expects about 3, and
one flip alone moves it to 8W-2L, `p = 0.109`. The claim that the archive
significantly beats memory-alone is **not established** and needs three runs per
arm to be re-tested. It is not refuted either — the effect size is large — but the
p-value as published overstates what one run can support.

**Five-question probes are worthless for this.** The probe that started all of
this estimated the disagreement rate at roughly 27% of cells. The true rate is
6%. Selecting questions on having failed selects the marginal ones, and the
marginal ones are the unstable ones by construction.
