# v3.3 compact session-fair evidence tune42

v2 and v3.2 are hash-verified references and were not rerun.

| arm | accuracy | temporal | multi-session | ordinary | median context | answer output |
|---|---:|---:|---:|---:|---:|---:|
| `v2-control` | 66.7% | 40.0% | 70.0% | 82.4% | 586 | 2,598 |
| `v3.2-adaptive` | 73.8% | 60.0% | 70.0% | 88.2% | 1,476 | 2,699 |
| `v3.3-compact` | 73.8% | 60.0% | 70.0% | 88.2% | 1,124 | 2,765 |

Tune promotion signal: **PASS**

| check | pass |
|---|---:|
| `overall_no_regression` | yes |
| `temporal_no_regression` | yes |
| `multi_session_no_regression` | yes |
| `ordinary_no_regression` | yes |
| `no_new_confident_errors` | yes |
| `hydration_is_targeted` | yes |
| `mean_gold_coverage_no_regression` | yes |
| `full_gold_coverage_no_regression` | yes |
| `context_within_v2_limit` | yes |
| `context_lower_than_v3_2` | yes |
| `generation_cost` | yes |

Paired v3.3 vs v3.2: +0.0%; exact p=1.0000.
