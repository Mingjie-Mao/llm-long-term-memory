# Facts as keys for raw turns — `train150`

> Zero calls (local encoder). Registered in
> `results/prereg-fact-key-expansion-offline-v1.md`. Reach, not use.

146 questions with gold turns. All-gold coverage at 4,000 tokens:

| arm | coverage | paired vs `plain` |
|---|---:|---:|
| `plain` | 77.4% | — |
| `expanded` | 81.5% | +7 / -1 |
| `expanded_stored` | 80.8% | +6 / -1 |
| `expanded_dense_only` | 77.4% | +1 / -1 |

| type | `plain` | `expanded` | `expanded_stored` | `expanded_dense_only` |
|---|---:|---:|---:|---:|
| knowledge-update | 100% | 100% | 100% | 100% |
| multi-session | 49% | 59% | 59% | 51% |
| single-session-assistant | 100% | 100% | 100% | 100% |
| single-session-preference | 67% | 56% | 56% | 56% |
| single-session-user | 95% | 100% | 95% | 95% |
| temporal-reasoning | 78% | 82% | 82% | 78% |

`expanded` qualifies on this set: True.
