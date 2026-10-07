# Raw-primary on train150 — registered gate v2

> Regression-class evidence (a development check). Rules as registered in
> `results/prereg-raw-primary-train150-v2.md`; not an accuracy claim.

| run | correct / 150 | median context | fallback second calls |
|---|---:|---:|---:|
| control `two_stage_hydrated.train150-raw-v2-control` | 98 | 1,593 | 44 |
| R `two_stage_raw_primary.train150-raw-v2` | 127 | 5,686 | 16 |

**Primary:** R against control +32 / -3, **net +29** (floor +8)

| question type | n | wins | losses | net |
|---|---:|---:|---:|---:|
| knowledge-update | 23 | 1 | 0 | +1 |
| multi-session | 40 | 16 | 1 | +15 |
| single-session-assistant | 17 | 2 | 0 | +2 |
| single-session-preference | 9 | 2 | 1 | +1 |
| single-session-user | 21 | 1 | 0 | +1 |
| temporal-reasoning | 40 | 10 | 1 | +9 |

## Rules

- PASS — `net_at_least_8`
- PASS — `no_type_loses_more_than_2`
- PASS — `median_total_context_at_most_6000`

**Decision: PASS**

Descriptive: exact McNemar p = 0.0000 (no significance claimed from a development run).
