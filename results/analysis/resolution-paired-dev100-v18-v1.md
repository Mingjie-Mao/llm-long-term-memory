# Resolution of a question set

> Derived from repeat runs already on disk. No model calls.

## Measured run noise

One unchanged configuration, run more than once. A question that disagrees with
itself here cannot carry a mechanism's signal anywhere else.

| repeat set | runs | questions | disagreed | accuracy per run |
|---|---:|---:|---:|---|
| `two_stage_fallback.v2b-gate16` | 2 | 9 | **1** | 6, 5 |
| `two_stage_hydrated.heldout100` | 4 | 100 | **12** | 71, 73, 72, 70 |
| `two_stage_memory_only.v2b-gate16` | 2 | 9 | **1** | 3, 4 |
| `two_stage_v2c.conflict1-v2c8` | 2 | 1 | **0** | 1, 1 |
| `two_stage_v2c.gate8-v2c3` | 2 | 8 | **0** | 8, 8 |
| `two_stage_v2c_memory_only.gate8-v2c1` | 2 | 8 | **0** | 5, 5 |
| `two_stage_v2c_memory_only.gate8-v2c2` | 2 | 8 | **0** | 5, 5 |

| question type | unstable | n | rate |
|---|---:|---:|---:|
| single-session-preference | 3 | 8 | **37.5%** |
| temporal-reasoning | 6 | 35 | **17.1%** |
| knowledge-update | 2 | 23 | **8.7%** |
| multi-session | 3 | 41 | **7.3%** |
| single-session-assistant | 0 | 17 | **0.0%** |
| single-session-user | 0 | 19 | **0.0%** |

## Projected onto `grounded-paired-dev100-v15-baseline-rep1`

- questions: **100**
- expected to disagree from run noise alone: **10.1**
- standard deviation of the paired net from noise: **±3.2**
- **a net below 7 is not distinguishable from noise with one run per arm**

## Reading this

The last number is a floor on what the set can register, not a significance
test. Real flips are not independent, the per-type rates come from a single
100-question measurement, and a set enriched for prior failures is exactly the
population most likely to be marginal — so the true requirement is higher than
this, never lower.

If a mechanism's plausible effect is smaller than the figure above, the choice
is to enrich the set toward the failure it targets, repeat each arm, or not run
it. Deciding that after the run is how two quota days were spent.
