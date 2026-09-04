# v3.2 adaptive source-evidence tune42

The candidate adds compact raw spans only for registered multi-step operations.
The v2 baseline is the hash-verified tune1 artifact; it was not rerun.

| arm | accuracy | temporal | multi-session | ordinary | median context | answer output |
|---|---:|---:|---:|---:|---:|---:|
| `v2-control` | 66.7% | 40.0% | 70.0% | 82.4% | 586 | 2,598 |
| `v3.2-adaptive` | 73.8% | 60.0% | 70.0% | 88.2% | 1,476 | 2,699 |

Tune promotion signal: **STOP**

| check | pass |
|---|---:|
| `target_gain` | yes |
| `temporal_no_regression` | yes |
| `multi_session_no_regression` | yes |
| `overall_no_regression` | yes |
| `ordinary_no_regression` | yes |
| `no_new_confident_errors` | yes |
| `hydration_is_targeted` | yes |
| `context_budget` | no |
| `generation_cost` | yes |

Paired difference: +7.1%; exact p=0.3750.
