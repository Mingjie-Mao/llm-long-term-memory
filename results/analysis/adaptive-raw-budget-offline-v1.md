# Question-routed raw-turn budget — `train150`

> Zero calls. Registered in `results/prereg-adaptive-raw-budget-offline-v1.md`.
> Reach, not use. Routing reads the question wording only.

146 of 150 questions have gold turns.

| policy | routed to 8k | all-gold coverage | projected median total context |
|---|---:|---:|---:|
| `fixed_4k` | 0% | 78.1% | 5,690 |
| `fixed_8k` | 100% | 85.6% | 9,692 |
| `adaptive_agg` | 35% | 81.5% | 6,119 |
| `adaptive_agg_temporal` | 65% | 84.2% | 9,602 |

| type | `fixed_4k` | `fixed_8k` | `adaptive_agg` | `adaptive_agg_temporal` |
|---|---:|---:|---:|---:|
| knowledge-update | 95% | 100% | 95% | 100% |
| multi-session | 54% | 69% | 67% | 69% |
| single-session-assistant | 94% | 100% | 94% | 100% |
| single-session-preference | 67% | 78% | 67% | 67% |
| single-session-user | 95% | 95% | 95% | 95% |
| temporal-reasoning | 80% | 85% | 80% | 82% |

Routing (benchmark type -> routed kind): knowledge-update -> current_state: 2, knowledge-update -> direct: 4, knowledge-update -> multi_session_aggregation: 11, knowledge-update -> temporal: 6, multi-session -> direct: 6, multi-session -> multi_session_aggregation: 29, multi-session -> temporal: 5, single-session-assistant -> direct: 15, single-session-assistant -> multi_session_aggregation: 1, single-session-assistant -> temporal: 1, single-session-preference -> direct: 2, single-session-preference -> preference_application: 7, single-session-user -> direct: 14, single-session-user -> multi_session_aggregation: 3, single-session-user -> temporal: 4, temporal-reasoning -> direct: 3, temporal-reasoning -> multi_session_aggregation: 8, temporal-reasoning -> temporal: 29

**Decision: `fixed_4k`** (qualifying: none)
