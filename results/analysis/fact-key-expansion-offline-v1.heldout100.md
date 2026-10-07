# Facts as keys for raw turns — `heldout100`

> Zero calls (local encoder). Registered in
> `results/prereg-fact-key-expansion-offline-v1.md`. Reach, not use.

93 questions with gold turns. All-gold coverage at 4,000 tokens:

| arm | coverage | paired vs `plain` |
|---|---:|---:|
| `plain` | 84.9% | — |
| `expanded` | 89.2% | +4 / -0 |
| `expanded_stored` | 90.3% | +5 / -0 |
| `expanded_dense_only` | 89.2% | +4 / -0 |

| type | `plain` | `expanded` | `expanded_stored` | `expanded_dense_only` |
|---|---:|---:|---:|---:|
| knowledge-update | 93% | 100% | 100% | 100% |
| multi-session | 79% | 88% | 92% | 88% |
| single-session-assistant | 100% | 100% | 100% | 100% |
| single-session-preference | 67% | 67% | 67% | 67% |
| single-session-user | 92% | 92% | 92% | 92% |
| temporal-reasoning | 81% | 85% | 85% | 85% |

`expanded` qualifies on this set: True.
