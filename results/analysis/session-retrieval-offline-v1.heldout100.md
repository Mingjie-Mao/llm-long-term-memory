# Whole sessions through their turns — `heldout100`

> Zero calls. Registered in `results/prereg-session-retrieval-offline-v1.md`.
> Reach, not use. BM25 turn matching only (no turn embeddings in the store).

93 questions with gold turns. All-gold coverage:

| arm | 2,000 | 4,000 | 8,000 | 16,000 |
|---|---:|---:|---:|---:|
| `bm25_turns` | 80.6% | 86.0% | 88.2% | 92.5% |
| `bm25_turn_window` | 55.9% | 75.3% | 92.5% | 94.6% |
| `bm25_sessions` | 9.7% | 33.3% | 75.3% | 91.4% |

Paired against `bm25_turns` at the same budget (wins / losses):

| arm | 2,000 | 4,000 | 8,000 | 16,000 |
|---|---:|---:|---:|---:|
| `bm25_turn_window` | +2 / -25 | +3 / -13 | +4 / -0 | +3 / -1 |
| `bm25_sessions` | +0 / -66 | +1 / -50 | +4 / -16 | +2 / -3 |

**Decision: stay with turns**
