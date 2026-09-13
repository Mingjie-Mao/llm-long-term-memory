# v4.0 flat — the answerer changes, separated

**Run 2026-09-06. 152 answerer requests (quota 329 → 481). 142 development probes.**
Control reused from [attempt 1](../v4.0-attempt1/) — it was unaffected by the prompt
defect and Gate 0 verified the pairing.

## Gate 0 passed

Retrieval identical across both arms on all 142 probes. Any difference is the answerer.

## What this arm was for

[Attempt 1](../v4.0-attempt1/) bundled three answerer changes and moved in two
directions, so nothing could be attributed. This arm removes exactly one of them —
timeline rendering — and fixes the prompt defect that contaminated 55 of 142 rows.

`current_state` is the operation that decides it: `compute` returns `computed=False`
for it, so deterministic arithmetic never touches it. Its movement can only come from
the prompt or the rendering.

## Result

| operation | n | v3.3 | attempt 1 | **flat** | flat − v3.3 |
|---|---:|---:|---:|---:|---:|
| `current_state` | 49 | 87.8% | 55.1% | **85.7%** | −2.0 |
| `duration` | 30 | 40.0% | 66.7% | **70.0%** | **+30.0** |
| `comparison` | 33 | 54.5% | 24.2% | **57.6%** | +3.0 |
| `count` | 30 | 16.7% | 23.3% | 16.7% | 0.0 |
| **all** | 142 | 54.9% | 43.7% | **61.3%** | **+6.3** |

On the 130 rows with no structure leak: **56.2% → 63.8%**, `duration` **+31.0**,
`comparison` **+10.7**, `current_state` **89.4% → 89.4%** — identical.

Paired: **22 wins, 13 losses, net +9, `p = 0.1755`.** Not statistically conclusive, and
the same order of uncertainty as `dev60`'s `p = 0.1797`. A consistent direction of
uncertain size.

## The three verdicts this bought

**Timeline rendering caused the regression, and is removed.** `current_state` went
87.8 → 55.1 with it and back to 85.7 without — on unleaked rows, exactly 89.4% both
ways. The mechanism that looked like the highest return on investment was the only one
that lost points.

**Deterministic arithmetic holds.** `duration` +31.0 on clean rows, under a fixed prompt
and with the rendering gone. Attempt 1's +34.6 was not an artefact of its contamination.

**`count` did not move, as registered.** v4.0 can fold duplicate members; it cannot
supply members that were never retrieved, and only 57% of count probes have complete
evidence. That is v4.1.

## Stop condition: not triggered

| | |
|---|---:|
| rows that went right → wrong **and** the code had computed a value | 7 |
| rows that went wrong → right | 22 |

Confident-wrong did not outgrow the gain. The seven sit in `count` (4),
`current_state` (2) and `comparison` (1).

Of the 13 regressions, only 3 are structure leaks; the other 10 are real behavioural
differences. The leak fell from 55 rows to 12 and is no longer the dominant contaminant.

## Still open

**`missing_field` fired on 5 rows and those rows went from 40% to 0%.** It is the only
negative signal here. Five rows cannot settle it, and the subgroup is defined by an
outcome of the run, so it is a direction and not a finding.
