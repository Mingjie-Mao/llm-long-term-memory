# Whole sessions through their turns — `train150`

> Zero calls. Registered in `results/prereg-session-retrieval-offline-v1.md`.
> Reach, not use. BM25 turn matching only (no turn embeddings in the store).

146 questions with gold turns. All-gold coverage:

| arm | 2,000 | 4,000 | 8,000 | 16,000 |
|---|---:|---:|---:|---:|
| `bm25_turns` | 71.9% | 78.1% | 85.6% | 89.7% |
| `bm25_turn_window` | 52.7% | 69.9% | 80.1% | 91.1% |
| `bm25_sessions` | 9.6% | 26.0% | 63.0% | 82.2% |

Paired against `bm25_turns` at the same budget (wins / losses):

| arm | 2,000 | 4,000 | 8,000 | 16,000 |
|---|---:|---:|---:|---:|
| `bm25_turn_window` | +2 / -30 | +6 / -18 | +7 / -15 | +5 / -3 |
| `bm25_sessions` | +0 / -91 | +2 / -78 | +5 / -38 | +5 / -16 |

**Decision: stay with turns**
