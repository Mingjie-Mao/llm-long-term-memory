# v3 train-only answer pilot

Two answer policies used the same v2 store, retrieval, context and fallback.
No question ids, question text, answers or judge reasons are reported.

| arm | runs | majority accuracy | target accuracy | ordinary accuracy | median context | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| `v2-control` | 3 | 68.8% | 61.1% | 91.7% | 614 | 13,204 |
| `v3-reasoned` | 3 | 72.9% | 63.9% | 100.0% | 605 | 15,783 |

## Pre-registered gate

Overall: **PASS**

| check | pass |
|---|---:|
| `target_gain` | yes |
| `ordinary_regression` | yes |
| `high_confidence_wrong` | yes |
| `context` | yes |
| `generation_cost` | yes |

Paired delta: +4.2%; exact p=0.6875.
