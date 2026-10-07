# Dense + BM25 turn retrieval — `train150`

> Zero calls (local encoder). Registered in
> `results/prereg-hybrid-turn-retrieval-offline-v1.md`. Reach, not use.

146 questions with gold turns; 73,953 turns embedded. All-gold coverage at 4,000 tokens:

| arm | coverage | paired vs `bm25` |
|---|---:|---:|
| `bm25` | 78.1% | — |
| `dense` | 80.8% | +18 / -14 |
| `hybrid` | 83.6% | +14 / -6 |

| type | `bm25` | `dense` | `hybrid` |
|---|---:|---:|---:|
| knowledge-update | 95% | 95% | 100% |
| multi-session | 54% | 64% | 62% |
| single-session-assistant | 94% | 100% | 100% |
| single-session-preference | 67% | 89% | 89% |
| single-session-user | 95% | 95% | 100% |
| temporal-reasoning | 80% | 72% | 80% |

Qualifying on this set: `hybrid`.
