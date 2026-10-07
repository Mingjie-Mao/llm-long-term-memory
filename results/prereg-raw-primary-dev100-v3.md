# Raw-primary v2 against v1 — dev100, three runs per arm — pre-registration

Date: 2026-10-01 (Australia/Sydney). Written before any answer or judge call of this
experiment.

## Classification

**A development comparison, recorded as regression evidence.** `dev100` was drawn on
2026-08-18 and used once, for an aggregate-level arm decision in v2
(`results/validation/dev100-aggregate.md`); no per-question failure of it has been read,
and it played no part in choosing anything below. Its legacy manifest lets the CLI
record runs only as `--experiment-class regression`. Nothing here is an unseen result or
an accuracy figure.

## Why

The raw-primary arm (v1: the control's memory context plus 4,000 tokens of BM25-found
turns) passed on train150 (net +29) after a strong heldout100 result stopped only by a
margin-less cost rule. On train150 its 23 remaining errors split into:

| first loss | errors | change aimed at it |
|---|---:|---|
| gold turn missing or only partly in context | 14 | **fused turn ranking**: BM25 + dense (`all-MiniLM-L6-v2`), reciprocal-rank fusion |
| all gold in context, answered wrong | 9 | **dated headers** (each conversation's distance from the question in days and weeks) and **evidence-first answering** (the answerer lists every relevant line with its date before deciding) |

Zero-call evidence for the ranking change (`results/prereg-hybrid-turn-retrieval-offline-v1.md`):
all-gold coverage at 4,000 tokens 78.1% → 83.6% on train150 (14 wins, 6 losses) and
86.0% → 88.2% on heldout100. A larger budget was tried first and did not help
(`results/analysis/adaptive-raw-budget-offline-v1*.md`). The answering changes have no
offline evidence: they can only be measured by answering.

## Arms

| arm | variant | differs from v1 by |
|---|---|---|
| **v1** | `two_stage_raw_primary` | — (the arm that passed on train150) |
| **v2** | `two_stage_raw_primary_v2` | fused turn ranking; dated headers; `v2_notes` answer policy (`NotedAnswerVerdict`, notes field first) with `max_output_tokens` 1024 instead of 512 |

Store `stores/dev100.db` (12,436 memories, 49,590 turns, `two-stage-p10-v2`), turn index
`stores/dev100-turn-index` (built by `lltm lifecycle build-turn-index`, 49,590 rows, ids
identical to the store's turns), config `configs/fallback.yaml`, manifest
`results/manifests/dev100.json`, answerer `gemini-3.5-flash-lite`, judge
`gemma-4-31b-it`. Answer prompt versions: v1 `memory-aware-v2`, v2
`memory-aware-v2-notes`; judge `lme-type-aware-v2` for both.

**Three changes in one arm, on purpose, and what that costs.** Each change's plausible
effect is a few questions per hundred — the ranking change completes the evidence for 5
of train150's 14 retrieval errors while losing it for 2, and reach converts at well
under one-for-one — which is below what this set can resolve for any one of them. A pass
therefore says the bundle helps and cannot say which part did. A failure cannot say
which part failed.

## Pre-run estimate of context (zero calls)

`tools/context_precheck.py` with a stub client on dev100:

| arm | median | p90 | max |
|---|---:|---:|---:|
| v1 | 5,689 | 5,806 | 5,873 |
| v2 | 5,772 | 5,925 | 6,069 |

## Resolution, and why three runs

`tools/resolution.py` projected onto dev100's composition (27 multi-session, 26
temporal-reasoning, 16 knowledge-update, 14 user, 11 assistant, 6 preference): a net
below **7** is not distinguishable from noise with one run per arm
(`results/analysis/resolution-dev100.md`; the mean pairwise disagreement of the four
heldout100 control runs gives 6). The plausible effect is below that. Per the
experiment guard, that is decided before spending: **each arm is run three times**, and
arms are compared on each question's **mean** correctness over its three runs. Averaging
three runs divides the noise's standard deviation by about √3, so the conservative
floor of 7 for one run becomes **4** for the mean. A majority vote would not do this: a
question that flips at random stays random under a vote.

## Decision rules

v2 passes if **all** hold:

1. mean net — Σ over questions of (v2's mean correctness − v1's) — **≥ +4.0**;
2. no question type's mean net below **−2.0**;
3. median total answer context of v2 over its three runs **≤ 6,000** tokens.

Reported, not deciding: per-run scores; per-type mean nets; median output tokens of each
arm; fallback second calls per run; an exact sign test over questions whose mean moved
(no significance claimed).

**What a pass permits:** v2 replaces v1 as the candidate for a comparison on a **new,
unseen** evaluation set. **What a failure records:** which rule, by how much; v1 remains
the candidate. Whatever the outcome, nothing further is tuned on dev100 against these
runs.

## Cost

| arm | answer calls per run | runs | total | judge calls |
|---|---:|---:|---:|---:|
| v1 | ≈ 112 | 3 | ≈ 340 | 300 |
| v2 | ≈ 112, more output tokens | 3 | ≈ 340 | 300 |
| **total** | | | **≈ 680** | **600** |

Two answerer quota days (500 per day) and well inside the judge's 1,500, before provider
failures. Runs resume from their result files. If all six are not complete by
**2026-10-08**, the experiment is recorded as incomplete and no rule is evaluated.

## Reproduction

```bash
lltm lifecycle build-turn-index --config configs/fallback.yaml --store-name dev100
for rep in 1 2 3; do
  lltm eval run two_stage_raw_primary    --config configs/fallback.yaml --store-name dev100 \
    --questions results/manifests/dev100.json --label dev100-v3-rep$rep
  lltm eval run two_stage_raw_primary_v2 --config configs/fallback.yaml --store-name dev100 \
    --questions results/manifests/dev100.json --label dev100-v3-rep$rep
done
python tools/raw_primary_gate_v3.py
```

## Withdrawn before any call — 2026-10-01

No run of this registration was started. It is superseded by
`results/prereg-raw-primary-dev100-v4.md`, which keeps every change registered here and
adds two that zero-call measurements found in the meantime (memory-led turns in the
fusion; turns keyed by their facts). Withdrawing an unrun registration costs nothing;
running it and then a second arm on the same dev100 would spend the set twice.
