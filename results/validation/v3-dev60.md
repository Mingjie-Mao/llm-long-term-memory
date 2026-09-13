# Frozen v3.3 dev60 validation

Two frozen policies used the same train150 store; individual content is sealed.

| arm | runs | majority accuracy | target | temporal | multi-session | update | ordinary | selected recall | median context | answer output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v2-control` | 3 | 46.7% | 38.9% | 44.4% | 33.3% | 70.0% | 50.0% | 98.3% | 573 | 11,043 |
| `v3.3-compact` | 3 | 55.0% | 47.2% | 55.6% | 38.9% | 80.0% | 57.1% | 98.3% | 1,115 | 11,399 |

Registered dev60 gate: **PASS**

| check | pass |
|---|---:|
| `target_gain` | yes |
| `temporal_no_regression` | yes |
| `multi_session_no_regression` | yes |
| `knowledge_update_no_regression` | yes |
| `ordinary_no_regression` | yes |
| `no_new_confident_errors` | yes |
| `selected_recall_no_regression` | yes |
| `context_within_2x_v2` | yes |
| `generation_cost` | yes |

Paired majority delta: +8.3%; exact p=0.1797.
