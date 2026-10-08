# Dense + BM25 turn retrieval — `heldout100`

> Zero calls (local encoder). Registered in
> `results/prereg-hybrid-turn-retrieval-offline-v1.md`. Reach, not use.

93 questions with gold turns; 46,848 turns embedded. All-gold coverage at 4,000 tokens:

| arm | coverage | paired vs `bm25` |
|---|---:|---:|
| `bm25` | 86.0% | — |
| `dense` | 83.9% | +7 / -9 |
| `hybrid` | 88.2% | +3 / -1 |

| type | `bm25` | `dense` | `hybrid` |
|---|---:|---:|---:|
| knowledge-update | 93% | 100% | 100% |
| multi-session | 83% | 79% | 88% |
| single-session-assistant | 100% | 100% | 100% |
| single-session-preference | 83% | 67% | 67% |
| single-session-user | 92% | 100% | 92% |
| temporal-reasoning | 77% | 69% | 81% |

Qualifying on this set: none.
