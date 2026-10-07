# Question-routed raw-turn budget — `heldout100`

> Zero calls. Registered in `results/prereg-adaptive-raw-budget-offline-v1.md`.
> Reach, not use. Routing reads the question wording only.

93 of 100 questions have gold turns.

| policy | routed to 8k | all-gold coverage | projected median total context |
|---|---:|---:|---:|
| `fixed_4k` | 0% | 86.0% | 5,600 |
| `fixed_8k` | 100% | 88.2% | 9,602 |
| `adaptive_agg` | 30% | 86.0% | 6,042 |
| `adaptive_agg_temporal` | 57% | 87.1% | 9,492 |

| type | `fixed_4k` | `fixed_8k` | `adaptive_agg` | `adaptive_agg_temporal` |
|---|---:|---:|---:|---:|
| knowledge-update | 93% | 93% | 93% | 93% |
| multi-session | 83% | 88% | 83% | 83% |
| single-session-assistant | 100% | 100% | 100% | 100% |
| single-session-preference | 83% | 83% | 83% | 83% |
| single-session-user | 92% | 92% | 92% | 92% |
| temporal-reasoning | 77% | 81% | 77% | 81% |

Routing (benchmark type -> routed kind): knowledge-update -> current_state: 3, knowledge-update -> direct: 2, knowledge-update -> multi_session_aggregation: 8, knowledge-update -> temporal: 2, multi-session -> current_state: 1, multi-session -> direct: 8, multi-session -> multi_session_aggregation: 16, multi-session -> temporal: 2, single-session-assistant -> direct: 8, single-session-assistant -> multi_session_aggregation: 1, single-session-assistant -> temporal: 2, single-session-preference -> direct: 5, single-session-preference -> preference_application: 1, single-session-user -> direct: 11, single-session-user -> temporal: 3, temporal-reasoning -> direct: 4, temporal-reasoning -> multi_session_aggregation: 5, temporal-reasoning -> temporal: 18

**Decision: `fixed_4k`** (qualifying: none)
