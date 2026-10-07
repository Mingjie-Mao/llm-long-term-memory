# Small probe of t3 (time questions also answered from notes) — pre-registration

Date: 2026-10-07 (Australia/Sydney). Written before any call of this probe. A
diagnostic like `results/prereg-t-probe-v1.md`, on the same sets, with one more arm.
Not an accuracy figure and not a decision; a pass leads to a new heldout100
registration, not to adoption.

## Why

`results/t-probe-v1-decision.md`: t2 fixed 5 of 8 out-of-design temporal failures, all
of them questions that `asks_for_aggregate` routed to item notes; t1's date notes alone
fixed 1. t3 sends every question about dates, durations or order to notes as well.

## What t3 is

`two_stage_raw_primary_t3` = t2 with answer policy `v2_clt` (prompt
`memory-aware-v2-clt`): v2-cl, plus — for questions `asks_about_time` selects (90 of the
93 temporal-reasoning questions across train150, dev100 and heldout100; also 18–40% of
other types) — notes of each relevant event with its date, then the answer worked out
from those dates. 1,024 output tokens on any noted question. Rendering is t1's.

## Runs

t3 once on `t-probe-train150-v1.json` (31) and on `t-probe-dev100-target-v1.json` (18):
49 answers, about 60 answerer calls (144 left today) and 49 judge calls. Compared with
the v1 rerun and t2 rows already saved by probe v1 on the same questions (same day,
store and models).

## Reading (train150, decided before the run)

- **t3 moves temporal**: on the 8 temporal failures, t3 correct − v1-rerun correct ≥ +2
  (t2 had +5; reported alongside).
- **t3 moves count/KU**: on the 11 multi-session and knowledge-update failures, ≥ +2.
- **Harm signal**: on the 12 controls, t3's wrong count − v1-rerun's ≥ 2.

Both moves and no harm → register t3 against v1 on heldout100 (three runs per arm,
the gate of the t1/t2 registrations), replacing the unrun t1/t2 plan. Otherwise report
and diagnose. dev100: reported only, against v1's three-run expectation (3.0 / 18).

```bash
lltm eval run two_stage_raw_primary_t3 --config configs/fallback.yaml --store-name train150 \
  --questions results/manifests/t-probe-train150-v1.json --label t-probe-v1 --experiment-class regression
lltm eval run two_stage_raw_primary_t3 --config configs/fallback.yaml --store-name dev100 \
  --questions results/manifests/t-probe-dev100-target-v1.json --label t-probe-v1-dev --experiment-class regression
python tools/t_probe_report.py --arm t3
```
