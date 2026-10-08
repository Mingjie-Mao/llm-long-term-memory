# Raw turns first, memory for state — train150 — pre-registration v2

Date: 2026-09-30 (Australia/Sydney). Written before any answer or judge call of this
experiment.

## Classification

**A development check, recorded as regression evidence.** `train150` is the set whose
individual failures may be read; its legacy manifest lets the CLI record runs only with
`--experiment-class regression`. One selection effect is stated rather than hidden: the
4,000-token turn budget was chosen from evidence *reach* on train150
(`results/memory-as-index-offline-decision.md`). No answer of this arm has been scored on
train150. Nothing here is an unseen result or an accuracy figure.

## Why again

`results/raw-primary-heldout100-decision.md`: the same arm scored 83 against a control
of 70–73 on heldout100, net **+14** against the control majority, no question type
negative, and memory still added **+7** over the turns alone. It was stopped by its cost
rule — median context 5,598 against a ceiling of 5,500 that had been set at the expected
value with no margin. The heldout100 protocol forbids rerunning there. This registers
the **same arm, unchanged**, on a different set, with a cost rule that has margin.

## Arms

| arm | variant | context |
|---|---|---|
| **C** control | `two_stage_hydrated` | memories + hydrated spans; conditional raw fallback |
| **R** raw-primary | `two_stage_raw_primary` | C's context plus up to 4,000 tokens of question-found turns, oldest first, dated |

Store `stores/train150.db` (18,519 memories, `two-stage-p10-v2`), config
`configs/fallback.yaml`, manifest `results/manifests/train150.json`, answerer
`gemini-3.5-flash-lite`, judge `gemma-4-31b-it`, prompts `memory-aware-v2` /
`lme-type-aware-v2`. One run each. The raw-only arm is not repeated: heldout100 already
showed the memory context adds to the turns, and history-growth simulation shows the
turns degrade as history grows while memories do not.

## Pre-run estimate of context (zero calls)

A stub client that makes no provider call measured the pre-answer context of both arms
(`two_stage_hydrated` / `two_stage_raw_primary`). Validated first on heldout100, where it
estimated 1,458 / 5,587 against the measured 1,502 / 5,598. On train150 it estimates:

| arm | median | p90 | max |
|---|---:|---:|---:|
| C | 1,554 | 1,644 | 1,807 |
| R | 5,683 | 5,787 | 5,935 |

The conditional fallback adds text on a minority of questions (12 of 100 for R on
heldout100), which moved R's median by 11 tokens there.

## Resolution

`tools/resolution.py` projected onto train150's composition (40 multi-session, 40
temporal-reasoning, 23 knowledge-update, 21 user, 17 assistant, 9 preference): **a net
below 8 is not distinguishable from run noise with one run per arm**
(`results/analysis/resolution-train150.md`). That figure is conservative: the tool
counts a question as unstable if it ever disagreed across the four heldout100 control
runs, and more runs raise that count. The mean pairwise disagreement of those runs,
projected the same way, gives 7. The registered floor is **8**.

## Decision rules

R passes if **all** hold:

1. paired net, R against C (R right & C wrong, minus R wrong & C right), **≥ +8**;
2. no question type loses more than **2** net;
3. median **total** answer context of R — the recorded `context_tokens`, including any
   fallback text — **≤ 6,000** tokens (≈ 300 above the estimate; the heldout100 ceiling
   had none).

Reported, not deciding: per-type nets; exact McNemar p (no significance claimed from a
development run); how often each arm asked for the fallback's second call.

There is no drift check: train150 has no earlier control run to compare with.

**What a pass permits:** naming R the candidate for a comparison on a **new, unseen**
evaluation set, to be registered separately. Not an accuracy figure, not a statement
about 72% or the 86% / 72% gap, not a product default.

**What a failure records:** which rule, by how much, alongside the heldout100 result.
Whatever the outcome, no further arm is tuned on train150 against these runs.

## Cost

| arm | answer calls | judge calls |
|---|---:|---:|
| C | ≈ 200 (150 + fallback second calls) | 150 |
| R | ≈ 170 | 150 |
| **total** | **≈ 370** | **300** |

One answerer quota day (500) and well inside the judge's (1,500), before provider
failures, which on 2026-09-29 retried the judge 206 times across the three heldout100
runs. Runs resume from their result files; if both are not complete by **2026-10-07**,
the experiment is recorded as incomplete and no rule is evaluated.

## Reproduction

```bash
lltm eval run two_stage_hydrated    --config configs/fallback.yaml --store-name train150 \
  --questions results/manifests/train150.json --label train150-raw-v2-control
lltm eval run two_stage_raw_primary --config configs/fallback.yaml --store-name train150 \
  --questions results/manifests/train150.json --label train150-raw-v2
python tools/raw_primary_gate_v2.py
```
