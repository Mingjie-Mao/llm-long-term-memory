# Resolution of a question set

> Derived from repeat runs already on disk. No model calls.

## Measured run noise

One unchanged configuration, run more than once. A question that disagrees with
itself here cannot carry a mechanism's signal anywhere else.

| repeat set | runs | questions | disagreed | accuracy per run |
|---|---:|---:|---:|---|
| `two_stage_fallback.v2b-gate16` | 2 | 9 | **1** | 6, 5 |
| `two_stage_hydrated.heldout100` | 3 | 100 | **6** | 71, 73, 70 |
| `two_stage_memory_only.v2b-gate16` | 2 | 9 | **1** | 3, 4 |
| `two_stage_v2c.conflict1-v2c8` | 2 | 1 | **0** | 1, 1 |
| `two_stage_v2c.gate8-v2c3` | 2 | 8 | **0** | 8, 8 |
| `two_stage_v2c_memory_only.gate8-v2c1` | 2 | 8 | **0** | 5, 5 |
| `two_stage_v2c_memory_only.gate8-v2c2` | 2 | 8 | **0** | 5, 5 |

| question type | unstable | n | rate |
|---|---:|---:|---:|
| single-session-preference | 1 | 8 | **12.5%** |
| temporal-reasoning | 4 | 35 | **11.4%** |
| multi-session | 2 | 41 | **4.9%** |
| knowledge-update | 1 | 23 | **4.3%** |
| single-session-assistant | 0 | 17 | **0.0%** |
| single-session-user | 0 | 19 | **0.0%** |

## Projected onto `two_stage_memory_only.v2b-gate16`

- questions: **16**
- expected to disagree from run noise alone: **1.0**
- standard deviation of the paired net from noise: **±1.0**
- **a net below 2 is not distinguishable from noise with one run per arm**

## Reading this

The last number is a floor on what the set can register, not a significance
test. Real flips are not independent, the per-type rates come from a single
100-question measurement, and a set enriched for prior failures is exactly the
population most likely to be marginal — so the true requirement is higher than
this, never lower.

If a mechanism's plausible effect is smaller than the figure above, the choice
is to enrich the set toward the failure it targets, repeat each arm, or not run
it. Deciding that after the run is how two quota days were spent.
