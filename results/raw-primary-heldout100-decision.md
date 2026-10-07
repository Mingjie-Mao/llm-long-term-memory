# Raw turns first, memory for state — heldout100 — decision

Date: 2026-09-30 (Australia/Sydney). Registered in
`results/prereg-raw-primary-heldout100-v1.md` (with its pre-call addendum). Regression-
class evidence on an exposed set; not an accuracy figure and not about 72%. Gate output:
`results/analysis/raw-primary-heldout100-gate.{json,md}` (`tools/raw_primary_gate.py`).

## Verdict: STOP — on the cost rule, by 98 tokens

| rule | registered | measured | |
|---|---|---|---|
| paired net against the control majority | ≥ +5 | **+14** (+18 / −4) | pass |
| no question type loses more than 2 net | — | worst type +1; none negative | pass |
| median answer context | ≤ 5,500 tokens | **5,598** | **fail** |

| run | correct / 100 |
|---|---:|
| control, runs 1–3 (on disk) | 70, 71, 73 |
| control, fresh run 4 | 72 — inside 67–76, no drift |
| control majority over 4 | 69 (4 ties counted wrong) |
| **R** — control context + 4,000 tokens of question-found turns | **83** |
| **O** — the turns alone | **76** |

By type, R against the control majority: temporal-reasoning **+6**, multi-session +3,
single-session-assistant +2, knowledge-update +1, preference +1, user +1.

Descriptive, as registered: R against the fresh control alone +18 / −7, net +11, exact
McNemar p = 0.043 — no significance is claimed from one development run on an exposed
set. Secondary, R against O: **+7 net** (+13 / −6), **+6 on knowledge-update and
temporal-reasoning together** — the memory context adds to the turns rather than being
made redundant by them, as the history-growth simulation also suggested it would over
longer histories.

## Why the cost rule failed

R did not overspend its turn budget: question-found turns used a median of 3,999 of
4,000 tokens, never more. The rule itself had no margin. It was registered as "≈ 4,000
turns + the control's context", and the control's own median on this set is 1,502
tokens; 4,000 + 1,502 plus the conversation date headers is 5,598. Setting the ceiling
at the expected value rather than above it was a registration error, recorded here. It
does not change the verdict: the rule was fixed before the run and is applied as
written.

A cost fact the rule did not measure: with the turns in context the answerer asked for
the raw fallback's second call on 12 questions instead of the control's 35, so R needs
about 112 answer calls per 100 questions to the control's 135. The recorded call counts
(control 196, R 133, O 133) also include provider failures — 31, 10 and 17 — and are not
comparable as cost.

## What follows

- Nothing further is tuned or rerun on heldout100 (its protocol).
- The same arm is registered again on another set with a total-context rule that has
  margin: `results/prereg-raw-primary-train150-v2.md`.
- A pass there would permit a comparison on a new, unseen set; it would not be an
  accuracy claim either.

## Cost

Control run 4: 20 attempts over six hours against provider `500 INTERNAL` and `503
UNAVAILABLE` on the judge; R: one day's remaining answerer quota; O: 6 further
attempts. Answer and judge usage is in the three `results/raw/*.usage.json` files.
