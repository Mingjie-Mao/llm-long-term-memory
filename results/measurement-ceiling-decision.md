# The 86% / 72% gap — what it would take, and what it is worth

> A decision memo, not a result. It contains no new measurement. It exists because the
> project's most visible number can no longer be improved *or* shown to have improved
> under current conditions, and that should be a stated position rather than something
> a reader works out.

## The situation

The frozen v2 result is 72.0% on 100 unseen LongMemEval-S questions. Handing the model
the whole conversation history instead scores 86.0% on the same questions with 109,059
median context tokens against 574. The gap is the project's headline limitation.

Three things have closed the ordinary routes to moving it.

**The benchmark is spent.** LongMemEval-S's 500 questions are partitioned across five
disjoint sets with none remaining, and `longmemeval_oracle.json` carries the same 500
ids over a different haystack — verified, not assumed (`results/hidden-set-status.md`).
Every set has been read. Nothing on disk can serve as an unseen final.

**The external set was closed for cost.** The BEAM line was adopted, split, judged,
registered and rehearsed; ingestion reached 44% of the development half and stopped on
2026-09-16. Reaching a conclusion needed roughly eighteen quota days.

**The QA instrument cannot resolve what is left.** One unchanged configuration run three
times over 100 questions disagrees with itself on six, concentrated entirely in
`temporal-reasoning` (11.4%) and `multi-session` (4.9%). Projected onto a 48-question
set, a paired net below **4** is indistinguishable from run noise. Three mechanisms
registered against that set returned +0, +0 and −1.

So the position today is: **the gap cannot be closed with the data available, and a
closure could not be demonstrated if it happened.**

## What each route costs

Ingestion at batch 8 with the specificity repair, measured per evaluation set from the
corpus itself. The repair is assumed to fire on 42% of sessions, which is what it did
on its registered cohort. Provider failures are excluded, and over the last three days
they have run at 80–90%, which multiplies every figure below.

| set | questions | unique sessions | extraction calls | repair calls | quota days |
|---|---:|---:|---:|---:|---:|
| `v2c-gate8` | 8 | 374 | 94 | 157 | 0.5 |
| `v2b-gate16` | 16 | 772 | 194 | 324 | 1.0 |
| `v3-reasoning48` | 48 | 2,273 | 570 | 954 | 3.0 |
| `dev50` | 50 | 2,348 | 588 | 986 | 3.1 |
| `test100` | 100 | 4,527 | 1,132 | 1,901 | 6.1 |
| `train150` | 150 | 6,701 | 1,676 | 2,814 | 9.0 |
| whole corpus | 500 | 19,195 | 4,800 | 8,061 | **25.7** |

## The three options

**A. Accept the gap and stop optimising it.** Report 72% as the frozen result with the
86% beside it, and position the project as what it demonstrably is: a working long-term
memory system, and a record of measuring one honestly. Cost: nothing. What it gives up:
the headline number stays where it is.

**B. Buy a fresh evaluation set.** The only route to a defensible new final number. BEAM
was costed at roughly eighteen quota days and abandoned; nothing since has made it
cheaper, and the provider's current failure rate makes it worse. It also needs the
ingestion above on top. Cost: weeks, most of it waiting on a free tier.

**C. Spend on the extraction side only, and claim only extraction.** Re-ingest one set at
batch 8 with the repair and report the fidelity change — that instrument disagrees with
itself on **0 of 134** specifics, so a change there is readable where a QA change is not.
Cost: 0.5 to 6.1 quota days depending on the set. What it gives up: it says nothing about
72%, and must not be written as if it did.

## Recommendation

**A as the position, C as the work.**

The gap is not a defect to be fixed under current conditions; it is a measured property
of a system whose evaluation data is exhausted. Saying so is more useful to a reader than
a fourth mechanism returning a net of ±1.

C is worth doing because it is the one place where effort still converts into evidence:
+7.7 to +27.6pp from batch size depending on cohort, and +14.5pp from the repair at 21
gains and 0 losses, both on an instrument with no measured noise. `v2b-gate16` at 1.0
quota days is the cheapest set that would show it end to end.

B should be reopened only if the project's purpose changes from "a system measured
honestly" to "a number on a leaderboard". That is not a technical judgement and is not
made here.

## Status

**Option A taken, 2026-09-23.** The 86% / 72% gap is closed as a line of work. It stays
reported, with the reason it cannot move stated beside it, and no further mechanism is
registered against it.

Option C is the work that continues: `v2b-gate16` is the set, batch 8 plus the
specificity repair is the change, and the acceptance is on extraction metrics only —
see `results/prereg-v2b-gate16-repair.md`. Option B is not reopened.

**Option C result, 2026-09-23: PASS on all eight gates.** Recall 47.5% -> 60.5% (+13.0 pp,
304 gained, 0 lost) on 780 sessions, at 621 calls over two quota days. An extraction
result only; it does not move, and is not reported against, 72%. See
`results/v2b-gate16-repair-decision.md`.
