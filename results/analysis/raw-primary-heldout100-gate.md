# Raw-primary on heldout100 — registered gate

> Regression-class evidence on an exposed set. Rules as registered in
> `results/prereg-raw-primary-heldout100-v1.md`; not an accuracy claim.

## Scores (of 100)

| run | correct |
|---|---:|
| `two_stage_hydrated.heldout100` | 70 |
| `two_stage_hydrated.heldout100-rep2` | 71 |
| `two_stage_hydrated.heldout100-rep3` | 73 |
| `two_stage_hydrated.heldout100-rep4` | 72 |
| `two_stage_raw_primary.raw-primary-v1` | 83 |
| `two_stage_raw_only.raw-primary-v1` | 76 |

**Drift check:** fresh control 72 against 67-76 — no drift
Control majority over 4 runs: 69 correct; ties counted wrong: 4.

## Primary — R against the control majority

+18 / −4, **net +14** (floor +5)

| question type | n | wins | losses | net |
|---|---:|---:|---:|---:|
| knowledge-update | 15 | 2 | 1 | +1 |
| multi-session | 27 | 5 | 2 | +3 |
| single-session-assistant | 11 | 2 | 0 | +2 |
| single-session-preference | 6 | 1 | 0 | +1 |
| single-session-user | 14 | 1 | 0 | +1 |
| temporal-reasoning | 27 | 7 | 1 | +6 |

Median answer context: control 1,502, R 5,598, O 4,126 tokens (R limit 5,500). Median question-found turns in R: 20.0.

## Rules

- PASS — `net_at_least_5`
- PASS — `no_type_loses_more_than_2`
- FAIL — `median_context_at_most_5500`

**Decision: STOP**

## Descriptive

- R against the fresh control alone: +18 / −7, net +11, exact McNemar p = 0.0433 (no significance claimed).
- Secondary, R against O: +13 / −6, net +7; on knowledge-update + temporal-reasoning: net +6.
