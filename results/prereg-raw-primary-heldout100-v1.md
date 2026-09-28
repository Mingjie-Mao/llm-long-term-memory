# Raw turns first, memory for state — heldout100 — pre-registration

Date: 2026-09-28 (Australia/Sydney). Written before any answer or judge call of this
experiment. Git SHA at registration: the commit that adds this file.

## Classification

**A development question, recorded as regression evidence.** `heldout100` was the early
held-out set; it has been run three times and its legacy manifest's exposure is
`regression`, so the CLI records these runs with `--experiment-class regression` and
permits nothing stronger. Nothing here
is an unseen result, a new headline, or evidence about 72%. Under the heldout100
protocol (`results/heldout-variance.md`), questions that flip are reported as data and
are **not** read to choose further changes, and no second mechanism is tuned on this set
afterwards.

## Why

Two zero-call measurements point the same way:

- **Memories are a poor locator.** At 4,000 tokens, searching the raw archive with the
  question put every gold turn in context for 78.1% of train150 questions; following
  memories to their source turns managed 50.0% (9 wins, 50 losses;
  `results/memory-as-index-offline-decision.md`). On heldout100 — a disjoint set, run
  after the budget was fixed — the same comparison is **86.0% against 54.8%**
  (`results/analysis/raw-primary-heldout100-preflight.md`).
- **The first loss is extraction.** 38 of the 39 train150 index misses have no memory
  from the gold turn at all; on the frozen system's test100 run, 25 of 28 wrong answers
  already had the gold session in context.

External work agrees: in a controlled LongMemEval-S ablation, verbatim chunks scored
67.4% against 45.4% for extracted artifacts, and artifacts added *alongside* chunks
kept the accuracy (arXiv 2601.00821).

The question this answers: **with the memory context unchanged, does adding
question-found verbatim turns before the first answer call improve answers — and does
the memory context still add anything once the turns are there?**

## Data and system

- Manifest `results/manifests/heldout100.json`, store `stores/heldout100.db`
  (12,402 memories, extractor `two-stage-p10-v2`), config `configs/fallback.yaml`.
- Answerer `gemini-3.5-flash-lite`, judge `gemma-4-31b-it`, answer prompt
  `memory-aware-v2`, judge prompt `lme-type-aware-v2` — all as in the control runs.

## Arms

| arm | variant | context | runs |
|---|---|---|---|
| **C** control | `two_stage_hydrated` | memories + hydrated spans; conditional raw fallback | 3 on disk (70, 71, 73) + **1 fresh** |
| **R** raw-primary | `two_stage_raw_primary` | C's context unchanged, plus up to 4,000 tokens of question-found turns, oldest first, dated | 1 |
| **O** raw-only | `two_stage_raw_only` | the same turns, no memory context, no memory-located fallback | 1 |

The 4,000-token budget was fixed on train150 (`RAW_PRIMARY_TOKENS`) before heldout100
was looked at. Turns come from `SQLiteMemoryStore.search_turns` (BM25 over the
namespace's turns), whole turns in rank order, skipping any that would overflow. The
answer system prompt is unchanged in every arm.

The fresh control run checks drift: the provider model behind the same id may have
changed since August.

## Resolution

`tools/resolution.py` on the three control runs: 6 of 100 questions disagree with
themselves across unchanged runs; **a paired net below 5 is not distinguishable from
run noise** with one run per arm (`results/analysis/resolution-two_stage_hydrated.heldout100.md`).
The plausible effect is larger than that floor only if a meaningful share of the
control's ~29 failures are evidence-reach failures that the turns fix; the preflight
says reach rises substantially, and v5.0 says reach converts at well below one for one.
This set can detect a large effect and cannot detect a small one. That is stated now.

## Decision rules

**Drift check first.** If the fresh control scores outside 67–76 (the three prior runs'
range widened by 3), drift is declared: R and O are then compared against the fresh run
alone, and the report says so.

**Primary — R against C.** Per question, C's verdict is the majority over the four
control runs (ties counted as wrong for C, which favours no arm: it only matters on
questions the control itself cannot settle, and those are listed). R passes if **all**
hold:

1. paired net (R right & C wrong, minus R wrong & C right) **≥ +5**;
2. no question type loses more than 2 net — the state layer must not be traded away
   on `knowledge-update` or `temporal-reasoning`;
3. median answer context at most **5,500 tokens** (≈ 4,000 turns + the control's
   context; the budget is a ceiling, not a target).

**Secondary — does memory still help? R against O**, paired, one run each. Reported as
net with the same floor of 5, overall and for `knowledge-update` + `temporal-reasoning`
together. No pass/fail: it decides what the next registered candidate is, not whether
this one passed.

An exact McNemar p-value for R against the fresh control is reported descriptively.
Significance is not claimed from one development run on an exposed set whatever it says.

**What a pass permits:** registering R — or O, if the secondary shows memory adds
nothing — as the candidate for a comparison on a **new, unseen** evaluation set. Not a
new accuracy figure, not a claim about the 86% / 72% gap, not a product default.

**What a failure records:** the net, by type, with the turn reach, and that the
extraction-first diagnosis did not convert at this budget.

## Cost

| arm | answer calls | judge calls |
|---|---:|---:|
| C fresh | ≈ 137 (fallback second calls) | 100 |
| R | ≈ 100–137 | 100 |
| O | 100 | 100 |
| **total** | **≈ 340–375** | **300** |

One quota day for each model (500 per day) with no provider failures. R and O carry
≈ 4,000 more input tokens per call: ≈ 0.8M input tokens more than the control in
total, within the per-minute token limit at the configured request rate.

**Stopping rule.** Runs resume from their result files. If all four are not complete
by **2026-10-05**, the experiment is recorded as incomplete and no rule is evaluated.

## Reproduction

```bash
lltm eval run two_stage_hydrated    --config configs/fallback.yaml --store-name heldout100 \
  --questions results/manifests/heldout100.json --label heldout100-rep4
lltm eval run two_stage_raw_primary --config configs/fallback.yaml --store-name heldout100 \
  --questions results/manifests/heldout100.json --label raw-primary-v1
lltm eval run two_stage_raw_only    --config configs/fallback.yaml --store-name heldout100 \
  --questions results/manifests/heldout100.json --label raw-primary-v1
```
