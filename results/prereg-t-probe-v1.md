# Small probe of t1 and t2 before the three-day heldout100 run — pre-registration

Date: 2026-10-07 (Australia/Sydney). Written before any call of this probe. **A
diagnostic, not an accuracy figure and not a decision on t1 or t2** — those stay with
`results/prereg-raw-primary-time-notes-v1.md` and
`results/prereg-raw-primary-count-latest-v1.md`, unchanged. The probe asks one question
with one day's remaining answerer quota (292 calls at registration): do t1 and t2 move
the failures they target at all, before about 1,000 calls are spent on heldout100?

## Sets

1. **Out of design — train150** (`results/manifests/t-probe-train150-v1.json`): every
   temporal-reasoning (8), multi-session (9) and knowledge-update (2) question that
   `two_stage_raw_primary.train150-raw-v2` answered wrong — 19 — plus 12 it answered
   right, by type 3/3/3 and one each of the single-session types, drawn by a seeded hash
   order (`t-probe-v1`; groups in `t-probe-train150-v1.groups.json`). t1 and t2 were not
   designed from these questions; no answer of them was read for it (train150 turn
   annotations were hand-read for correctness, not answers).
2. **Design set — dev100** (`t-probe-dev100-temporal-v1.json`, 9 questions;
   `t-probe-dev100-target-v1.json`, 18): the v1 failures t1 and t2 were built from.
   Reported only; a fix here is expected and proves little.

## Arms and runs

One run each. train150: v1 (`two_stage_raw_primary`, fresh — selecting failures from one
run makes a rerun of v1 recover some by chance, so v1 is rerun as the control), t1, t2 —
93 answers. dev100: t1 on the 9, t2 on the 18 — 27 answers, against v1's three existing
runs (expected correct 2.33 / 9 and 3.0 / 18). About 135–150 answerer calls and 120
judge calls. Store, config, models, prompts as in the two registrations.

## Reading (train150, decided before the runs)

- **t1 moves its target** if, on the 8 temporal failures, t1 correct − v1-rerun correct
  **≥ +2**.
- **t2 moves its target** if, on the 11 multi-session and knowledge-update failures, t2
  correct − v1-rerun correct **≥ +2**.
- **Harm signal** for an arm if, on the 12 controls, its wrong count exceeds v1-rerun's
  by **2 or more**.

Single runs on 8–12 questions: one flaky question is a step of one. These readings say
whether the mechanism does anything, not how much. A move without harm → run heldout100
as registered. No move, or harm → stop and diagnose before the three-day run, and say so.
Either way the probe's rows are not pooled into the heldout100 gates.

## Reproduction

```bash
for v in two_stage_raw_primary two_stage_raw_primary_t1 two_stage_raw_primary_t2; do
  lltm eval run $v --config configs/fallback.yaml --store-name train150 \
    --questions results/manifests/t-probe-train150-v1.json --label t-probe-v1 --experiment-class regression
done
lltm eval run two_stage_raw_primary_t1 --config configs/fallback.yaml --store-name dev100 \
  --questions results/manifests/t-probe-dev100-temporal-v1.json --label t-probe-v1-dev --experiment-class regression
lltm eval run two_stage_raw_primary_t2 --config configs/fallback.yaml --store-name dev100 \
  --questions results/manifests/t-probe-dev100-target-v1.json --label t-probe-v1-dev --experiment-class regression
python tools/t_probe_report.py
```

Source hashes are those of the two registrations (time_notes.py `55d13ca8a02d486f`).
